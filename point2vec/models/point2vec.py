from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.loggers import WandbLogger
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from point2vec.modules.EMA import EMA
from point2vec.modules.masking import PointcloudMasking, VariablePointcloudMasking, masked_layer_norm
from point2vec.modules.pointnet import PointcloudTokenizer
from point2vec.modules.transformer import TransformerEncoder, TransformerEncoderOutput
from point2vec.utils import transforms
from profiling_decorator import profile
from sklearn.metrics import classification_report


class Point2Vec(pl.LightningModule):
    def __init__(
        self,
        num_channels: int = 3,
        tokenizer_num_init_groups: int = 64,
        tokenizer_context_length: int = 256,
        tokenizer_group_size: int = 32,
        tokenizer_group_radius: float | None = None,
        tokenizer_upscale_group_size: int | None = None,
        tokenizer_overlap_factor: float | None = None,
        tokenizer_reduction_method: str = 'energy', # method to reduce upscale group size to group size
        tokenizer_normalize_group_centers: bool = False,
        tokenizer_embedding_checkpoint: str | None = None,
        d2v_masking_ratio: float = 0.65,
        d2v_masking_type: str = "rand",  # rand, block
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        encoder_dropout: float = 0,
        encoder_attention_dropout: float = 0.05,
        encoder_drop_path_rate: float = 0.25,
        encoder_add_pos_at_every_layer: bool = True,
        encoder_qkv_bias: bool = True,
        use_relative_features: bool = False,
        decoder: bool = True,
        decoder_depth: int = 4,
        decoder_dropout: float = 0,
        decoder_attention_dropout: float = 0.05,
        decoder_drop_path_rate: float = 0.25,
        decoder_add_pos_at_every_layer: bool = True,
        decoder_qkv_bias: bool = True,
        position_encoder: Optional[str] = "nn",
        embedding_type: str = "mini",
        d2v_target_layers: List[int] = [6, 7, 8, 9, 10, 11],
        d2v_target_layer_part: str = "final",  # ffn, final
        d2v_target_layer_norm: Optional[str] = "layer",  # instance, layer, group, batch
        d2v_target_norm: Optional[str] = "layer",  # instance, layer, group, batch
        d2v_ema_tau_max: Optional[float] = 0.9998,
        d2v_ema_tau_min: Optional[float] = 0.99999,
        d2v_ema_tau_epochs: int = 200,
        d2v_ema_tau_update_every: Optional[int] = None,
        loss: str = "smooth_l1",  # smooth_l1, mse
        learning_rate: float = 1e-3,
        optimizer_adamw_weight_decay: float = 0.05,
        lr_scheduler_linear_warmup_epochs: int = 80,
        lr_scheduler_linear_warmup_start_lr: float = 1e-6,
        lr_scheduler_cosine_eta_min: float = 1e-6,
        lr_scheduler_stepping: str = 'epoch',
        train_transformations: List[str] = [
            # "subsample",
            # "scale",
            # "center",
            # "unit_sphere",
            "rotate",
        ],  # subsample, scale, center, unit_sphere, rotate, translate, height_norm
        val_transformations: List[str] = [],
        transformation_subsample_points: int = 1024,
        transformation_scale_min: float = 0.8,
        transformation_scale_max: float = 1.2,
        transformation_scale_symmetries: Tuple[int, int, int] = (1, 0, 1),
        transformation_rotate_dims: List[int] = [0, 1, 2],
        transformation_rotate_degs: Optional[int] = None,
        transformation_translate: float = 0.2,
        transformation_height_normalize_dim: int = 1,
        svm_validation: Dict[str, pl.LightningDataModule] = {},
        svm_validation_C=0.005,  # C=0.012 copied from Point-M2AE code
        svm_validation_max_tokens: int = 7500,
        fix_estimated_stepping_batches: Optional[int] = None,  # multi GPU bug fix
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        def build_transformation(name: str) -> transforms.Transform:
            if name == "subsample":
                return transforms.PointcloudSubsampling(transformation_subsample_points)
            elif name == "scale":
                return transforms.PointcloudScaling(
                    min=transformation_scale_min, max=transformation_scale_max
                )
            elif name == "center":
                return transforms.PointcloudCentering()
            elif name == "unit_sphere":
                return transforms.PointcloudUnitSphere()
            elif name == "rotate":
                return transforms.PointcloudRotation(
                    dims=transformation_rotate_dims, deg=transformation_rotate_degs
                )
            elif name == "translate":
                return transforms.PointcloudTranslation(transformation_translate)
            elif name == "height_norm":
                return transforms.PointcloudHeightNormalization(
                    transformation_height_normalize_dim
                )
            else:
                raise RuntimeError(f"No such transformation: {name}")

        self.train_transformations = transforms.Compose(
            [build_transformation(name) for name in train_transformations]
        )
        self.val_transformations = transforms.Compose(
            [build_transformation(name) for name in val_transformations]
        )

        self.positional_encoding = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, encoder_dim),
        )
        if use_relative_features:
            assert num_channels > 3, "num_channels must be greater than 3 to use relative features"
            self.feature_encoder = nn.Sequential(
                nn.Linear(num_channels-3, 128),
                nn.GELU(),
                nn.Linear(128, encoder_dim),
            )
        else:
            self.feature_encoder = None

        self.tokenizer = PointcloudTokenizer(
            num_init_groups=tokenizer_num_init_groups,
            context_length=tokenizer_context_length,
            group_size=tokenizer_group_size,
            group_radius=tokenizer_group_radius,
            upscale_group_size=tokenizer_upscale_group_size,
            reduction_method=tokenizer_reduction_method,
            overlap_factor=tokenizer_overlap_factor,
            token_dim=encoder_dim,
            num_channels=num_channels,
            embedding_type=embedding_type,
            use_relative_features=use_relative_features,
            normalize_group_centers=tokenizer_normalize_group_centers,
        )
        if tokenizer_embedding_checkpoint is not None:
            self.tokenizer.load_state_dict(self.tokenizer.extract_model_checkpoint(tokenizer_embedding_checkpoint))
            # freeze tokenizer
            # for param in self.tokenizer.parameters():
                # param.requires_grad = False
            print("❄️  Loaded pretrained tokenizer and did not freeze it.")

        self.masking = VariablePointcloudMasking(
            ratio=d2v_masking_ratio, type=d2v_masking_type
        )

        init_std = 0.02
        self.mask_token = nn.Parameter(torch.zeros(encoder_dim))
        nn.init.trunc_normal_(
            self.mask_token, mean=0, std=init_std, a=-init_std, b=init_std
        )
        # self.cls_token = nn.Parameter(torch.zeros(encoder_dim))
        # nn.init.trunc_normal_(self.cls_token, mean=0, std=init_std, a=-init_std, b=init_std)

        dpr = [
            x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_depth)
        ]
        decoder_dpr = [
            x.item() for x in torch.linspace(0, decoder_drop_path_rate, decoder_depth)
        ]

        print(f'🎲 Encoder qkv bias: {encoder_qkv_bias}')
        print(f'🎲 Decoder qkv bias: {decoder_qkv_bias}')
        self.student = TransformerEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            qkv_bias=encoder_qkv_bias, # usually True
            drop_rate=encoder_dropout,
            attn_drop_rate=encoder_attention_dropout,
            drop_path_rate=dpr,
            add_pos_at_every_layer=encoder_add_pos_at_every_layer,
        )
        if decoder:
            self.decoder = TransformerEncoder(
                embed_dim=encoder_dim,
                depth=decoder_depth,
                num_heads=encoder_heads,
                qkv_bias=decoder_qkv_bias,
                drop_rate=decoder_dropout,
                attn_drop_rate=decoder_attention_dropout,
                drop_path_rate=decoder_dpr,
                add_pos_at_every_layer=decoder_add_pos_at_every_layer,
            )

        if decoder:
            self.regressor = nn.Linear(encoder_dim, encoder_dim)
        else:
            self.regressor = nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim),
                nn.GELU(),
                nn.Linear(encoder_dim, encoder_dim),
            )

        match loss:
            case "mse":
                self.loss_func = nn.MSELoss()
            case "smooth_l1":
                self.loss_func = nn.SmoothL1Loss(beta=2)
            case _:
                raise ValueError(f"Unknown loss: {loss}")

    def setup(self, stage: Optional[str] = None) -> None:
        if self.hparams.d2v_ema_tau_epochs < 500:
            # total # of steps * (tau_epochs / total_epochs)
            tau_steps = (
                self.hparams.fix_estimated_stepping_batches  # type: ignore
                or self.trainer.estimated_stepping_batches
            ) * (self.hparams.d2v_ema_tau_epochs / self.trainer.max_epochs)  # type: ignore
        else:
            tau_steps = self.hparams.d2v_ema_tau_epochs
        print(f"Using tau steps: {tau_steps}")

        self.teacher = EMA(
            self.student,
            tau_min=0
            if self.hparams.d2v_ema_tau_min is None  # type: ignore
            else self.hparams.d2v_ema_tau_min,  # type: ignore
            tau_max=1
            if self.hparams.d2v_ema_tau_max is None  # type: ignore
            else self.hparams.d2v_ema_tau_max,  # type: ignore
            tau_steps=tau_steps,
            update_after_step=0,
            update_every=1
            if self.hparams.d2v_ema_tau_update_every is None  # type: ignore
            else self.hparams.d2v_ema_tau_update_every,  # type: ignore
        )
        # Note: there is a bug in Lightning 1.7.7 that causes `self.trainer.estimated_stepping_batches` to crash when using multiple GPUs
        # see: https://github.com/Lightning-AI/lightning/issues/12317
        # Because of that, we allow to workaround this crash by manually setting this value with `fix_estimated_stepping_batches`.

        svm_validation: Dict[str, pl.LightningDataModule] = self.hparams.svm_validation  # type: ignore
        for dataset_name, datamodule in svm_validation.items():
            datamodule.setup("fit")
            print(f"🍗  Setup {dataset_name} datamodule for SVM validation.")
            for logger in self.loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.define_metric(
                        f"svm_train_acc_{dataset_name}", summary="last,max"
                    )
                    logger.experiment.define_metric(
                        f"svm_val_acc_{dataset_name}", summary="last,max"
                    )

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.watch(self)

    def center_encoding(self, centers: torch.Tensor) -> torch.Tensor:
        pos =  self.positional_encoding(centers[:, :, :3])
        if self.feature_encoder is not None:
            pos = pos + self.feature_encoder(centers[:, :, 3:])
        return pos

    def forward(
        self,
        embeddings: torch.Tensor,
        centers: torch.Tensor,
        masked: torch.Tensor,
        unmasked: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: (B, T, C)
        # centers: (B, T, C)
        # mask: (B, T)
        w_m = masked.unsqueeze(-1)
        w_u = unmasked.unsqueeze(-1)

        corrupted_embeddings = w_u * embeddings + w_m * self.mask_token
        pos = self.center_encoding(centers)

        decoder_mask = w_m.bool().squeeze() | w_u.bool().squeeze()
        if self.hparams.decoder:  # type: ignore
            B, _, C = embeddings.shape

            visible_embeddings = corrupted_embeddings * unmasked.unsqueeze(-1)
            masked_embeddings = corrupted_embeddings * masked.unsqueeze(-1)

            output_embeddings = self.student(
                visible_embeddings, pos, w_u.bool().squeeze()
            ).last_hidden_state # (B, T, C)

            total_embeddings = (output_embeddings * unmasked.unsqueeze(-1)
                                + masked_embeddings * masked.unsqueeze(-1)
            )
            decoder_output_tokens = self.decoder(
                total_embeddings, pos, decoder_mask
            ).last_hidden_state

            masked_output_tokens = decoder_output_tokens[masked]

            predictions = self.regressor(masked_output_tokens)
        else:  # no decoder => like data2vec
            output_embeddings = self.student(
                corrupted_embeddings, pos, decoder_mask
            ).last_hidden_state  # (B, T, C)
            predictions = self.regressor(output_embeddings[masked])

        targets = self.generate_targets(embeddings, pos, decoder_mask)[masked]
        return predictions, targets

    def _perform_step(self, inputs: torch.Tensor, lengths: torch.Tensor, semantic_labels: torch.Tensor | None) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs: (B, N, num_channels)
        tokens, centers, mask, _, _ = self.tokenizer(inputs, lengths, semantic_labels)  # (B, T, C), (B, T, 3)
        masked, unmasked = self.masking(centers, mask.sum(-1))  # (B, T), (B, T)
        return self.forward(tokens, centers, masked, unmasked)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        # inputs: (B, N, num_channels)
        points, lengths, semantic_labels, _ = batch
        points = self.train_transformations(points)
        x, y = self._perform_step(points, lengths, semantic_labels)
        loss = self.loss_func(x, y)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.log("train_pred_std", self.token_std(x), sync_dist=True)  # should always be > 0.01
        self.log("train_target_std", self.token_std(y), sync_dist=True)  # should always be > 0.1
        return loss

    def validation_step(
        self, batch, batch_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # inputs: (B, N, num_channels)
        points, lengths, semantic_labels, _ = batch
        points = self.val_transformations(points)
        x, y = self._perform_step(points, lengths, semantic_labels)
        loss = self.loss_func(x, y)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_pred_std", self.token_std(x), sync_dist=True)
        self.log("val_target_std", self.token_std(y), sync_dist=True)

    def validation_epoch_end(
        self, outputs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        svm_validation: Dict[str, pl.LightningDataModule] = self.hparams.svm_validation  # type: ignore
        for dataset_name, datamodule in svm_validation.items():
            svm_train_acc, svm_val_acc, train_class_scores, val_class_scores = self.svm_validation(datamodule)
            self.log(f"svm_train_acc_{dataset_name}", svm_train_acc, sync_dist=False)
            self.log(f"svm_val_acc_{dataset_name}", svm_val_acc, sync_dist=False)
            for label, score in train_class_scores.items():
                self.log(f"svm_train_class_score_{dataset_name}_{label}", score, sync_dist=True)
            for label, score in val_class_scores.items():
                self.log(f"svm_val_class_score_{dataset_name}_{label}", score, sync_dist=True)

    def svm_validation(self, datamodule: pl.LightningDataModule) -> Tuple[float, float]:
        # Lightning controls the `training` and `grad_enabled` state. Don't want to mess with it, but make sure it's correct.
        assert not self.training
        assert not torch.is_grad_enabled()


        max_tokens: int = self.hparams.svm_validation_max_tokens  # type: ignore
        def xy(dataloader):
            x_list = []
            label_list = []

            total =  max_tokens // (self.tokenizer.grouping.group_size * dataloader.batch_size) if max_tokens is not None else None

            for i, (data, lengths, labels_batch, _) in enumerate(dataloader):
                data = data.cuda()
                lengths = lengths.cuda()
                labels_batch = labels_batch.cuda()
                with torch.no_grad():
                    tokens, centers, mask, semantic_ids, _, _, _ = self.tokenizer(
                        data, lengths, labels_batch, return_point_info=True)
                    pos = self.center_encoding(centers)
                    x = self.generate_targets(tokens, pos, mask).reshape(-1, 384)
                    semantic_ids = semantic_ids.reshape(-1, pos.shape[1])

                    # Vectorized computation to replace the loop
                    N = semantic_ids.shape[0]  # Number of groups
                    D = semantic_ids.shape[1]  # Number of semantic IDs per group

                    group_indices = torch.arange(N, device=semantic_ids.device).unsqueeze(1).expand(-1, D)  # Shape: (N, D)
                    semantic_ids_flat = semantic_ids.reshape(-1)
                    group_indices_flat = group_indices.reshape(-1)
                    valid_mask = semantic_ids_flat != -1
                    semantic_ids_valid = semantic_ids_flat[valid_mask]  # Shape: (K,)
                    group_indices_valid = group_indices_flat[valid_mask]  # Shape: (K,)
                    num_labels = semantic_ids_valid.max().item() + 1
                    counts = torch.zeros((N, num_labels), dtype=torch.int64, device=semantic_ids.device)
                    counts.index_add_(0, group_indices_valid, torch.nn.functional.one_hot(semantic_ids_valid, num_classes=num_labels).to(torch.int64))
                    y = counts.argmax(dim=1)  # Shape: (N,)
                    mask_flat = mask.reshape(-1)
                    x = x[mask_flat]
                    y = y[mask_flat]
                    x_list.append(x.cpu())
                    label_list.append(y.cpu())

                    if total is not None and i >= total:
                        break

            x = torch.cat(x_list, dim=0)[:max_tokens]
            y = torch.cat(label_list, dim=0)[:max_tokens]
            return x, y

        x_train, y_train = xy(datamodule.train_dataloader())  # type: ignore
        x_val, y_val = xy(datamodule.val_dataloader())  # type: ignore

        svm_C: float = self.hparams.svm_validation_C  # type: ignore
        svm = SVC(C=svm_C, kernel="linear", class_weight="balanced")
        svm.fit(x_train, y_train)  # type: ignore
        train_acc: float = svm.score(x_train, y_train)  # type: ignore
        val_acc: float = svm.score(x_val, y_val)  # type: ignore

        train_report = classification_report(y_train, svm.predict(x_train), output_dict=True)
        val_report = classification_report(y_val, svm.predict(x_val), output_dict=True)

        train_class_scores = {label: metrics['f1-score'] for label, metrics in train_report.items() if label.isdigit()}
        val_class_scores = {label: metrics['f1-score'] for label, metrics in val_report.items() if label.isdigit()}

        return train_acc, val_acc, train_class_scores, val_class_scores

    # https://github.com/Lightning-AI/lightning/issues/11688#issuecomment-1026688558
    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.teacher.update()
        self.log("ema_decay", self.teacher.get_current_decay(), sync_dist=True)

    @torch.no_grad()
    def generate_targets(
        self,
        tokens: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.teacher.ema_model is not None  # always false
        self.teacher.ema_model.eval()
        d2v_target_layers: List[int] = self.hparams.d2v_target_layers  # type: ignore
        d2v_target_layer_part: str = self.hparams.d2v_target_layer_part  # type: ignore
        output: TransformerEncoderOutput = self.teacher(
            tokens,
            pos,
            mask,
            return_hidden_states=d2v_target_layer_part == "final",
            return_ffns=d2v_target_layer_part == "ffn",
        )
        if d2v_target_layer_part == "ffn":
            assert output.ffns is not None
            target_layers = output.ffns
        elif d2v_target_layer_part == "final":
            assert output.hidden_states is not None
            target_layers = output.hidden_states
        else:
            raise ValueError()
        target_layers = [
            target_layers[i] for i in d2v_target_layers
        ]  # [(B, T, C)]
        # pre norm

        target_layer_norm = self.hparams.d2v_target_layer_norm  # type: ignore
        if target_layer_norm == 'masked_layer':
            assert mask is not None
            target_layers = [
                masked_layer_norm(target, target.shape[-1], mask)
                for target in target_layers
            ]
        else:
            assert mask is None

        norm_functions = {
            "instance": lambda target: F.instance_norm(target.transpose(1, 2)).transpose(1, 2),
            "layer": lambda target: F.layer_norm(target, target.shape[-1:]),
            "group": lambda target: F.layer_norm(target, target.shape[-2:]),
            "batch": lambda target: F.batch_norm(target.transpose(1, 2),
                running_mean=None,
                running_var=None,
                training=True,
            ).transpose(1, 2),
        }

        if target_layer_norm in norm_functions:
            target_layers = [
                norm_functions[target_layer_norm](target)
                for target in target_layers
            ]
        elif target_layer_norm is not None and target_layer_norm != "masked_layer":
            raise ValueError()

        # Average top K blocks; can do this even with padded tokens because
        # masked_layer_norm zeros out padded tokens across each target
        # layer and do not influence the average for other tokens
        targets = torch.stack(target_layers, dim=0).mean(0)  # (B, T, C)

        # post norm
        target_norm = self.hparams.d2v_target_norm  # type: ignore
        if target_norm == 'masked_layer':
            assert mask is not None
            targets = masked_layer_norm(targets, targets.shape[-1], mask)
        else:
            assert mask is None

        if target_norm in norm_functions:
            targets = norm_functions[target_norm](targets)
        elif target_norm is not None and target_norm != "masked_layer":
            raise ValueError()

        return targets

    @staticmethod
    def token_std(tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, C)
        return tokens.reshape(-1, tokens.shape[-1]).std(0).mean()

    def configure_optimizers(self):
        assert self.trainer is not None

        if self.hparams.fix_estimated_stepping_batches is not None:  # type: ignore
            # check that the correct value for the multi GPU fix was provided
            assert self.trainer.estimated_stepping_batches == self.hparams.fix_estimated_stepping_batches  # type: ignore

        opt = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore
            weight_decay=self.hparams.optimizer_adamw_weight_decay,  # type: ignore
        )

        warmup_epochs = self.hparams.lr_scheduler_linear_warmup_epochs
        if self.hparams.lr_scheduler_stepping == 'step' and self.hparams.lr_scheduler_linear_warmup_epochs < 100: # epochs probably given
            print("❗Performing LR scheduler every step!")
            print('❗Assuming epochs given, not iters')
            steps_per_epoch = self.trainer.num_training_batches
            max_epochs = self.trainer.max_steps if self.trainer.max_steps is not None else self.trainer.max_epochs * steps_per_epoch
            warmup_epochs = self.hparams.lr_scheduler_linear_warmup_epochs * steps_per_epoch
        elif self.hparams.lr_scheduler_stepping == 'step' and self.hparams.lr_scheduler_linear_warmup_epochs >= 100: # iters probably given
            print("❗Performing LR scheduler every step!")
            print('❗Assuming iters given in warmup, not epochs')
            steps_per_epoch = self.trainer.num_training_batches
            max_epochs = self.trainer.max_steps if self.trainer.max_steps is not None else self.trainer.max_epochs * steps_per_epoch
            warmup_epochs = self.hparams.lr_scheduler_linear_warmup_epochs # warmup iters

        for name, val in zip(['max_epochs', 'warmup_epochs'], [max_epochs, warmup_epochs]):
            print(f'\t{name}: {val}')

        # freq = 100
        sched = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=warmup_epochs, # iters if step, epochs otherwise
            max_epochs=max_epochs,       # iters if step, epochs otherwise
            warmup_start_lr=self.hparams.lr_scheduler_linear_warmup_start_lr,  # type: ignore
            eta_min=self.hparams.lr_scheduler_cosine_eta_min,  # type: ignore
        )

        # print(f'Scheduling every {freq} steps...')
        # print('warmup_epochs:', sched.warmup_epochs * freq)
        # print('max_epochs:', sched.max_epochs * freq)

        if self.hparams.lr_scheduler_stepping == 'step':
            sched = {
                "scheduler": sched,
                "interval": "step",
                # "frequency": freq,
            }

        return [opt], [sched]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # This is a bit of a hack. We want to avoid saving the datasets in the svm_validation dict,
        # as this would store the entire dataset inside the checkpoint, blowing it up to multiple GBs.
        checkpoint["hyper_parameters"]["svm_validation"] = {}

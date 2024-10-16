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
from point2vec.modules.pointnet import PointcloudTokenizer, PositionEmbeddingCoordsSine
from point2vec.modules.transformer import TransformerEncoder, TransformerEncoderOutput
from point2vec.utils import transforms

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
        d2v_masking_ratio: float = 0.65,
        d2v_masking_type: str = "rand",  # rand, block
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        encoder_dropout: float = 0,
        encoder_attention_dropout: float = 0.05,
        encoder_drop_path_rate: float = 0.25,
        encoder_add_pos_at_every_layer: bool = True,
        decoder: bool = True,
        decoder_depth: int = 4,
        decoder_dropout: float = 0,
        decoder_attention_dropout: float = 0.05,
        decoder_drop_path_rate: float = 0.25,
        decoder_add_pos_at_every_layer: bool = True,
        position_encoder: Optional[str] = "nn",
        embedding_type: str = "mini",
        d2v_target_layers: List[int] = [6, 7, 8, 9, 10, 11],
        d2v_target_layer_part: str = "final",  # ffn, final
        d2v_target_layer_norm: Optional[str] = "layer",  # instance, layer, group, batch
        d2v_target_norm: Optional[str] = "layer",  # instance, layer, group, batch
        d2v_ema_tau_max: Optional[float] = 0.9998,
        d2v_ema_tau_min: Optional[float] = 0.99999,
        d2v_ema_tau_epochs: int = 200,
        loss: str = "smooth_l1",  # smooth_l1, mse
        learning_rate: float = 1e-3,
        optimizer_adamw_weight_decay: float = 0.05,
        lr_scheduler_linear_warmup_epochs: int = 80,
        lr_scheduler_linear_warmup_start_lr: float = 1e-6,
        lr_scheduler_cosine_eta_min: float = 1e-6,
        train_transformations: List[str] = [
            # "subsample",
            # "scale",
            "center",
            # "unit_sphere",
            "rotate",
        ],  # subsample, scale, center, unit_sphere, rotate, translate, height_norm
        val_transformations: List[str] = ["center"],
        transformation_subsample_points: int = 1024,
        transformation_scale_min: float = 0.8,
        transformation_scale_max: float = 1.2,
        transformation_scale_symmetries: Tuple[int, int, int] = (1, 0, 1),
        transformation_rotate_dims: List[int] = [0, 1, 2],
        transformation_rotate_degs: Optional[int] = None,
        transformation_translate: float = 0.2,
        transformation_height_normalize_dim: int = 1,
        svm_validation: Dict[str, pl.LightningDataModule] = {},
        svm_validation_C=0.012,  # C=0.012 copied from Point-M2AE code
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

        match position_encoder:
            case "nn":
                self.positional_encoding = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, encoder_dim),
            )
            case "sine":
                self.positional_encoding = PositionEmbeddingCoordsSine(
                3, encoder_dim, 1
            )
            case _:
                raise ValueError(f"Unknown position encoder: {position_encoder}")

        self.tokenizer = PointcloudTokenizer(
            num_init_groups=tokenizer_num_init_groups,
            context_length=tokenizer_context_length,
            group_size=tokenizer_group_size,
            group_radius=tokenizer_group_radius,
            upscale_group_size=tokenizer_upscale_group_size,
            overlap_factor=tokenizer_overlap_factor,
            token_dim=encoder_dim,
            num_channels=num_channels,
            embedding_type=embedding_type,
        )

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
        self.student = TransformerEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            qkv_bias=True,
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
                qkv_bias=True,
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
        self.teacher = EMA(
            self.student,
            tau_min=0
            if self.hparams.d2v_ema_tau_min is None  # type: ignore
            else self.hparams.d2v_ema_tau_min,  # type: ignore
            tau_max=1
            if self.hparams.d2v_ema_tau_max is None  # type: ignore
            else self.hparams.d2v_ema_tau_max,  # type: ignore
            tau_steps=(
                self.hparams.fix_estimated_stepping_batches  # type: ignore
                or self.trainer.estimated_stepping_batches
            )
            * (self.hparams.d2v_ema_tau_epochs / self.trainer.max_epochs),  # type: ignore
            update_after_step=0,
            update_every=1,
        )
        # Note: there is a bug in Lightning 1.7.7 that causes `self.trainer.estimated_stepping_batches` to crash when using multiple GPUs
        # see: https://github.com/Lightning-AI/lightning/issues/12317
        # Because of that, we allow to workaround this crash by manually setting this value with `fix_estimated_stepping_batches`.

        # svm_validation: Dict[str, pl.LightningDataModule] = self.hparams.svm_validation  # type: ignore
        # for dataset_name, datamodule in svm_validation.items():
        #     datamodule.setup("fit")
        #     for logger in self.loggers:
        #         if isinstance(logger, WandbLogger):
        #             logger.experiment.define_metric(
        #                 f"svm_train_acc_{dataset_name}", summary="last,max"
        #             )
        #             logger.experiment.define_metric(
        #                 f"svm_val_acc_{dataset_name}", summary="last,max"
        #             )

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.watch(self)

    def forward(
        self,
        embeddings: torch.Tensor,
        centers: torch.Tensor,
        masked: torch.Tensor,
        unmasked: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: (B, T, C)
        # centers: (B, T, 3)
        # mask: (B, T)
        w_m = masked.unsqueeze(-1)
        w_u = unmasked.unsqueeze(-1)
        corrupted_embeddings = w_u * embeddings + w_m * self.mask_token
        pos = self.positional_encoding(centers)
        decoder_mask = w_m.bool().squeeze() | w_u.bool().squeeze()
        if self.hparams.decoder:  # type: ignore
            B, _, C = embeddings.shape

            visible_embeddings = torch.zeros_like(corrupted_embeddings)
            visible_embeddings[unmasked] = corrupted_embeddings[unmasked]

            masked_embeddings = torch.zeros_like(corrupted_embeddings)
            masked_embeddings[masked] = corrupted_embeddings[masked]

            output_embeddings = self.student(
                visible_embeddings, pos, w_u.bool().squeeze()
            ).last_hidden_state # (B, T, C)

            total_embeddings = torch.zeros_like(corrupted_embeddings)
            total_embeddings[unmasked] = output_embeddings[unmasked]
            total_embeddings[masked] = masked_embeddings[masked]

            decoder_output_tokens = self.decoder(
                total_embeddings, pos, decoder_mask
            ).last_hidden_state

            masked_output_tokens = decoder_output_tokens[masked]

            predictions = self.regressor(masked_output_tokens)
        else:  # no decoder => like data2vec
            pos = self.positional_encoding(centers)
            output_embeddings = self.student(
                corrupted_embeddings, pos, decoder_mask
            ).last_hidden_state  # (B, T, C)
            predictions = self.regressor(output_embeddings[masked])

        targets = self.generate_targets(embeddings, pos, decoder_mask)[masked]
        return predictions, targets

    def _perform_step(self, inputs: torch.Tensor, lengths: torch.Tensor, semantic_labels: torch.Tensor | None) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs: (B, N, num_channels)
        tokens, centers, mask, _ = self.tokenizer(inputs, lengths, semantic_labels)  # (B, T, C), (B, T, 3)
        masked, unmasked = self.masking(centers, mask.sum(-1))  # (B, T), (B, T)
        return self.forward(tokens, centers, masked, unmasked)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        # inputs: (B, N, num_channels)
        points, lengths, semantic_labels = batch
        points = self.train_transformations(points)
        x, y = self._perform_step(points, lengths, semantic_labels)
        loss = self.loss_func(x, y)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_pred_std", self.token_std(x))  # should always be > 0.01
        self.log("train_target_std", self.token_std(y))  # should always be > 0.1
        return loss

    def validation_step(
        self, batch, batch_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # inputs: (B, N, num_channels)
        points, lengths, semantic_labels = batch
        points = self.val_transformations(points)
        x, y = self._perform_step(points, lengths, semantic_labels)
        loss = self.loss_func(x, y)
        self.log("val_loss", loss)
        self.log("val_pred_std", self.token_std(x))
        self.log("val_target_std", self.token_std(y))

    def validation_epoch_end(
        self, outputs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        svm_validation: Dict[str, pl.LightningDataModule] = self.hparams.svm_validation  # type: ignore
        for dataset_name, datamodule in svm_validation.items():
            svm_train_acc, svm_val_acc = self.svm_validation(datamodule)
            self.log(f"svm_train_acc_{dataset_name}", svm_train_acc)
            self.log(f"svm_val_acc_{dataset_name}", svm_val_acc)

    def svm_validation(self, datamodule: pl.LightningDataModule) -> Tuple[float, float]:
        # Lightning controls the `training` and `grad_enabled` state. Don't want to mess with it, but make sure it's correct.
        assert not self.training
        assert not torch.is_grad_enabled()

        def xy(
            dataloader: DataLoader,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            x_list = []
            label_list = []
            for (points, label) in iter(dataloader):
                points: torch.Tensor = points.cuda()
                label: torch.Tensor = label.cuda()
                points = self.val_transformations(points)
                embeddings, centers, _ = self.tokenizer(points)
                pos = self.positional_encoding(centers)
                x = self.student(embeddings, pos).last_hidden_state
                x = torch.cat([x.max(dim=1).values, x.mean(dim=1)], dim=-1)
                x_list.append(x.cpu())
                label_list.append(label.cpu())

            x = torch.cat(x_list, dim=0)  # (N, 768)
            y = torch.cat(label_list, dim=0)  # (N,)
            return x, y

        x_train, y_train = xy(datamodule.train_dataloader())  # type: ignore
        x_val, y_val = xy(datamodule.val_dataloader())  # type: ignore

        svm_C: float = self.hparams.svm_validation_C  # type: ignore
        svm = SVC(C=svm_C, kernel="linear")
        svm.fit(x_train, y_train)  # type: ignore
        train_acc: float = svm.score(x_train, y_train)  # type: ignore
        val_acc: float = svm.score(x_val, y_val)  # type: ignore
        return train_acc, val_acc

    # https://github.com/Lightning-AI/lightning/issues/11688#issuecomment-1026688558
    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.teacher.update()
        self.log("ema_decay", self.teacher.get_current_decay())

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

        sched = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=self.hparams.lr_scheduler_linear_warmup_epochs,  # type: ignore
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=self.hparams.lr_scheduler_linear_warmup_start_lr,  # type: ignore
            eta_min=self.hparams.lr_scheduler_cosine_eta_min,  # type: ignore
        )

        return [opt], [sched]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # This is a bit of a hack. We want to avoid saving the datasets in the svm_validation dict,
        # as this would store the entire dataset inside the checkpoint, blowing it up to multiple GBs.
        checkpoint["hyper_parameters"]["svm_validation"] = {}

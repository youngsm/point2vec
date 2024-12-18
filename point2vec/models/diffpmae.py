from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.loggers import WandbLogger
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from point2vec.modules.masking import VariablePointcloudMasking
from point2vec.modules.pointnet import PointcloudTokenizer
from point2vec.modules.transformer import TransformerEncoder, TransformerEncoderOutput
from point2vec.utils import transforms
from pytorch3d.loss import chamfer_distance
from point2vec.modules.loss import KoLeoLoss
from sklearn.metrics import classification_report
from point2vec.modules.diffusion import TimeEmbedding, VarianceSchedule, DiffusionParameters
from point2vec.modules.pointnet import MaskedMiniPointNet
from point2vec.utils.checkpoint import extract_model_checkpoint



class DiffPMAE(pl.LightningModule):
    def __init__(
        self,
        num_channels: int = 3,
        tokenizer_num_init_groups: int = 64,
        tokenizer_context_length: int = 256,
        tokenizer_group_size: int = 32,
        tokenizer_group_radius: float | None = None,
        tokenizer_upscale_group_size: int | None = None,
        tokenizer_overlap_factor: float | None = None,
        tokenizer_reduction_method: str = "fps",
        tokenizer_normalize_group_centers: bool = False,
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
        encoder_freeze: bool = False,
        reuse_parts: bool = False,
        use_relative_features: bool = False,
        decoder_depth: int = 4,
        decoder_dropout: float = 0,
        decoder_attention_dropout: float = 0.05,
        decoder_drop_path_rate: float = 0.25,
        decoder_add_pos_at_every_layer: bool = True,
        decoder_qkv_bias: bool = True,
        diffusion_num_steps: int = 1000,
        diffusion_beta_1: float = 0.0001,
        diffusion_beta_T: float = 0.02,
        diffusion_sched_mode: str = 'linear',
        position_encoder: Optional[str] = "nn",
        embedding_type: str = "mini",
        learning_rate: float = 1e-3,
        apply_loss_to_unmasked: bool = True,
        optimizer_adamw_weight_decay: float = 0.05,
        lr_scheduler_linear_warmup_epochs: int = 80,
        lr_scheduler_linear_warmup_start_lr: float = 1e-6,
        lr_scheduler_cosine_eta_min: float = 1e-6,
        lr_scheduler_stepping: str = 'epoch',
        freeze_last_layer_iters: int = -1,
        encoder_pretrained_ckpt_path: str | None = None,
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
        do_koleo: bool = False,
        do_ae: bool = False,
        # deprecated
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

        if not reuse_parts:
            self.diffusion_positional_encoding = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, encoder_dim),
            )
            self.diffusion_feature_encoder = None
            if use_relative_features:
                assert num_channels > 3, "num_channels must be greater than 3 to use relative features"
                self.diffusion_feature_encoder = nn.Sequential(
                    nn.Linear(num_channels-3, 128),
                    nn.GELU(),
                    nn.Linear(128, encoder_dim),
                )
        else:
            self.diffusion_positional_encoding = self.positional_encoding

            self.diffusion_feature_encoder = None
            if self.feature_encoder is not None:
                self.diffusion_feature_encoder = self.feature_encoder

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

        self.masking = VariablePointcloudMasking(
            ratio=d2v_masking_ratio, type=d2v_masking_type
        )

        # init_std = 0.02
        # self.mask_token = nn.Parameter(torch.zeros(encoder_dim))
        # nn.init.trunc_normal_(
        #     self.mask_token, mean=0, std=init_std, a=-init_std, b=init_std
        # )

        dpr = [
            x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_depth)
        ]
        decoder_dpr = [
            x.item() for x in torch.linspace(0, decoder_drop_path_rate, decoder_depth)
        ]
        self.encoder = TransformerEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            qkv_bias=encoder_qkv_bias,
            drop_rate=encoder_dropout,
            attn_drop_rate=encoder_attention_dropout,
            drop_path_rate=dpr,
            add_pos_at_every_layer=encoder_add_pos_at_every_layer,
        )

        self.diffusion_decoder = TransformerEncoder(
                embed_dim=encoder_dim,
                depth=decoder_depth,
                num_heads=encoder_heads,
                qkv_bias=decoder_qkv_bias,
                drop_rate=decoder_dropout,
                attn_drop_rate=decoder_attention_dropout,
                drop_path_rate=decoder_dpr,
                add_pos_at_every_layer=decoder_add_pos_at_every_layer,
            )
        
        self.diffusion_increase_dim = nn.Sequential(
            nn.Conv1d(encoder_dim, num_channels*tokenizer_group_size, 1),
        )
        if not reuse_parts:
            self.increase_dim = self.diffusion_increase_dim # this will be used in loading ckpt weights

        self.var_scheduler = VarianceSchedule(
            num_steps=diffusion_num_steps,
            beta_1=diffusion_beta_1,
            beta_T=diffusion_beta_T,
            mode=diffusion_sched_mode
        )
        self.time_step = diffusion_num_steps
        self.beta_1 = diffusion_beta_1
        self.beta_T = diffusion_beta_T

        self.diffusion_parameters = DiffusionParameters(
            num_steps=diffusion_num_steps,
            beta_1=diffusion_beta_1,
            beta_T=diffusion_beta_T,
        )

        self.time_encoding = nn.Sequential(
            TimeEmbedding(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU()
        )

        # I think a diffusion-specific patch encoder is probably necessary,
        # as the tokenizer is designed for extracting useful semantic features
        # and not explicit pc reconstruction like in the diffusion decoder.
        self.diffusion_patch_encoder = MaskedMiniPointNet(num_channels, encoder_dim)

    def center_encoding(self, centers: torch.Tensor) -> torch.Tensor:
        pos =  self.positional_encoding(centers[:, :, :3])
        if self.feature_encoder is not None:
            pos = pos + self.feature_encoder(centers[:, :, 3:])
        return pos

    def diffusion_center_encoding(self, centers: torch.Tensor) -> torch.Tensor:
        pos =  self.diffusion_positional_encoding(centers[:, :, :3])
        if self.diffusion_feature_encoder is not None:
            pos = pos + self.diffusion_feature_encoder(centers[:, :, 3:])
        return pos


    def setup(self, stage: Optional[str] = None) -> None:
        # Note: there is a bug in Lightning 1.7.7 that causes `self.trainer.estimated_stepping_batches` to crash when using multiple GPUs
        # see: https://github.com/Lightning-AI/lightning/issues/12317
        # Because of that, we allow to workaround this crash by manually setting this value with `fix_estimated_stepping_batches`.

        if self.hparams.encoder_pretrained_ckpt_path is not None:
            self.load_pretrained_checkpoint(self.hparams.encoder_pretrained_ckpt_path)

        if self.hparams.encoder_freeze:
            print('❄️ Freezing encoder. Everything before the decoder will not be trained.')
            self.encoder.requires_grad_(False)
            self.tokenizer.requires_grad_(False)

            if not self.hparams.reuse_parts:
                self.positional_encoding.requires_grad_(False)
                if self.feature_encoder is not None:
                    self.feature_encoder.requires_grad_(False)
            else:
                print('Reusing positional encoding, so not freezing it.')

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

    def forward_diffusion(self, groups, point_mask, masked, time_steps):
        out_groups = groups.clone()
        masked_groups = groups[masked]
        noise = torch.randn_like(masked_groups)
        sqrt_alpha_bar = self.diffusion_parameters.sqrt_alphas_bar[time_steps]
        sqrt_one_minus_alpha_bar = self.diffusion_parameters.sqrt_one_minus_alphas_bar[time_steps]
        x_t = (
              sqrt_alpha_bar.unsqueeze(-1).unsqueeze(-1) * masked_groups
            + sqrt_one_minus_alpha_bar.unsqueeze(-1).unsqueeze(-1) * noise
        )

        # pad with the first point instead of zeros
        # the mini pointnet is not invariant to padding with zeros, but 
        # invariant to padding with the first point as it is permutation invariant
        x_t = (
              point_mask[masked].unsqueeze(-1) * x_t
            + (~point_mask[masked]).unsqueeze(-1) * x_t[:,0].unsqueeze(1) # use the first point to pad with
        )
        out_groups[masked] = x_t
        return out_groups

    def forward(
        self,
        embeddings: torch.Tensor,
        centers: torch.Tensor,
        groups: torch.Tensor,
        point_mask: torch.Tensor,
        masked: torch.Tensor,
        unmasked: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: (B, T, C)
        # centers: (B, T, 3)
        # mask: (B, T)
        loss_dict = {}
        B, _, C = embeddings.shape

        # Run encoder fwd, just attending to visible tokens
        pos = self.center_encoding(centers)
        encoder_output_embeddings = self.encoder(
            embeddings, pos, unmasked
        ).last_hidden_state # (B, T, C)

        # Run diffusion forward
        time_steps = torch.randint(1, self.time_step, (masked.sum(),), device=groups.device)
        diffused_groups = self.forward_diffusion(groups, point_mask, masked, time_steps)
        B,G,S,C = diffused_groups.shape
        expanded_mask = masked.unsqueeze(-1).expand(-1, -1, S)
        diffused_embeddings = self.diffusion_patch_encoder(
            diffused_groups.reshape(B*G, S, C), expanded_mask.reshape(B*G, 1, S)
        ).reshape(B, G, -1)
        total_embeddings = (
            diffused_embeddings * masked.unsqueeze(-1) + encoder_output_embeddings * unmasked.unsqueeze(-1)
        )
        decoder_pos = self.diffusion_center_encoding(centers)
        time_pos = torch.zeros_like(decoder_pos)
        time_pos[masked] = self.time_encoding(time_steps)
        decoder_mask = masked | unmasked
        decoder_output_embeddings = self.diffusion_decoder(
            total_embeddings, decoder_pos + time_pos, decoder_mask
        ).last_hidden_state

        if not self.hparams.apply_loss_to_unmasked:
            decoder_output_embeddings = decoder_output_embeddings[masked]

        # upscale in fp32
        with torch.amp.autocast('cuda', enabled=False):
            upscaled = self.diffusion_increase_dim(
                decoder_output_embeddings.reshape(-1, self.hparams.encoder_dim).transpose(0, 1)
            ).transpose(0, 1)

        if not self.hparams.apply_loss_to_unmasked:
            groups = groups[masked]
            point_lengths = point_mask[masked].sum(-1)
        else:
            point_lengths = point_mask.sum(-1)

        chamfer_loss, _ = chamfer_distance(
            upscaled.reshape(-1, S, C).float(),
            groups.reshape(-1, S, C).float(),
            x_lengths=point_lengths.view(-1),
            y_lengths=point_lengths.view(-1),
        )
        loss_dict['chamfer'] = chamfer_loss

        # get chamfer loss on masked and unmasked groups
        masked_chamfer_loss = chamfer_loss
        unmasked_chamfer_loss = 0
        if self.hparams.apply_loss_to_unmasked:
            with torch.no_grad():
                masked_groups = groups[masked]
                unmasked_groups = groups[unmasked]
                masked_point_lengths = point_mask[masked].sum(-1)
                unmasked_point_lengths = point_mask[unmasked].sum(-1)

                upscaled_masked = upscaled.reshape(B, G, S, C)[masked]
                upscaled_unmasked = upscaled.reshape(B, G, S, C)[unmasked]

                masked_chamfer_loss, _ = chamfer_distance(
                    upscaled_masked.float(),
                    masked_groups.float(),
                    x_lengths=masked_point_lengths,
                    y_lengths=masked_point_lengths,
                )
                unmasked_chamfer_loss, _ = chamfer_distance(
                    upscaled_unmasked.float(),
                    unmasked_groups.float(),
                    x_lengths=unmasked_point_lengths,
                    y_lengths=unmasked_point_lengths,
                )
        loss_dict['chamfer_masked'] = masked_chamfer_loss
        loss_dict['chamfer_unmasked'] = unmasked_chamfer_loss
        return loss_dict

    def _perform_step(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # inputs: (B, N, num_channels)
        (
            tokens,
            centers,
            embedding_mask,
            semantic_id_groups,
            endpoints_groups,
            group,
            point_mask,
        ) = self.tokenizer(inputs, lengths, return_point_info=True)  # (B, T, C), (B, T, 3), (B, T), (B, T), (B, T), (B, T, 3), (B, T), (B, T), (B, G, K, C), (B, G, K), (B, G, K)
        masked, unmasked = self.masking(centers, embedding_mask.sum(-1))  # (B, T), (B, T)
        return self.forward(tokens, centers, group, point_mask, masked, unmasked)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        # inputs: (B, N, num_channels)
        points, lengths, _, _ = batch
        points = self.train_transformations(points)
        loss_dict = self._perform_step(points, lengths)

        for key, value in loss_dict.items():
            self.log(f"train_loss_{key}", value, on_step=True, on_epoch=True, prog_bar=False)

        loss = loss_dict['chamfer']
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(
        self, batch, batch_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # inputs: (B, N, num_channels)
        points, lengths, _, _ = batch
        points = self.val_transformations(points)
        loss_dict = self._perform_step(points, lengths)

        for key, value in loss_dict.items():
            self.log(f"val_loss_{key}", value, sync_dist=True)
        loss = loss_dict['chamfer']
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def validation_epoch_end(
        self, outputs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        svm_validation: Dict[str, pl.LightningDataModule] = self.hparams.svm_validation  # type: ignore
        for dataset_name, datamodule in svm_validation.items():
            svm_train_acc, svm_val_acc, train_class_scores, val_class_scores = self.svm_validation(datamodule)
            self.log(f"svm_train_acc_{dataset_name}", svm_train_acc, sync_dist=True)
            self.log(f"svm_val_acc_{dataset_name}", svm_val_acc, sync_dist=True)
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
                    x = self.encoder(tokens, pos, mask).last_hidden_state.reshape(-1, self.hparams.encoder_dim)
                    semantic_ids = semantic_ids.reshape(-1, semantic_ids.shape[2])

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

        # ensemble = LinearClassifierEnsemble(input_dim=x_train.shape[1],
        #                                     num_classes=datamodule.num_seg_classes,
        #                                     learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
        #                                     max_iter=100,
        #                                     ).cuda()
        svm_C: float = self.hparams.svm_validation_C  # type: ignore
        svm = SVC(C=svm_C, kernel="linear", class_weight="balanced")
        svm.fit(x_train, y_train)  # type: ignore
        train_acc: float = svm.score(x_train, y_train)  # type: ignore
        val_acc: float = svm.score(x_val, y_val)  # type: ignore

        # x_train, y_train = x_train.cuda(), y_train.cuda()
        # x_val, y_val = x_val.cuda(), y_val.cuda()

        # x_train.requires_grad_(True)
        # x_val.requires_grad_(True)
        # y_train.requires_grad_(True)
        # y_val.requires_grad_(True)

        # classifier, train_acc, val_acc = ensemble.train(x_train, y_train, x_val, y_val)
        train_report = classification_report(y_train, svm.predict(x_train), output_dict=True)
        val_report = classification_report(y_val, svm.predict(x_val), output_dict=True)


        train_class_scores = {datamodule.seg_class_to_category[int(label)]: metrics['f1-score'] for label, metrics in train_report.items() if label.isdigit()}
        val_class_scores = {datamodule.seg_class_to_category[int(label)]: metrics['f1-score'] for label, metrics in val_report.items() if label.isdigit()}
        return train_acc, val_acc, train_class_scores, val_class_scores

    def configure_optimizers(self):
        assert self.trainer is not None

        if self.hparams.fix_estimated_stepping_batches is not None:  # type: ignore
            # check that the correct value for the multi GPU fix was provided
            assert self.trainer.estimated_stepping_batches == self.hparams.fix_estimated_stepping_batches  # type: ignore

        opt = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore
            weight_decay=self.hparams.optimizer_adamw_weight_decay,  # Ensure this line is present
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

    def on_train_epoch_start(self) -> None:
        freeze_last_layer_iters = self.hparams.freeze_last_layer_iters  # type: ignore
        if freeze_last_layer_iters < 0:
            return

        steps_per_epoch = self.trainer.num_training_batches
        freeze_last_layer_epochs = int(freeze_last_layer_iters / steps_per_epoch)  # type: ignore
        if self.trainer.current_epoch == freeze_last_layer_epochs:
            self.increase_dim.requires_grad_(False)
            print(f'  ❄️ Froze last layer at epoch {freeze_last_layer_epochs}!')

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # This is a bit of a hack. We want to avoid saving the datasets in the svm_validation dict,
        # as this would store the entire dataset inside the checkpoint, blowing it up to multiple GBs.
        checkpoint["hyper_parameters"]["svm_validation"] = {}


    def load_pretrained_checkpoint(self, path: str) -> None:
        print(f"Loading pretrained checkpoint from '{path}'.")

        checkpoint = extract_model_checkpoint(path)

        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)  # type: ignore
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

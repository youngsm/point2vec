from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, Precision

from point2vec.modules.loss import SoftmaxFocalLoss, DiceLoss
from point2vec.modules.feature_upsampling import PointNetFeatureUpsampling
from point2vec.modules.pointnet import PointcloudTokenizer, PositionEmbeddingCoordsSine
from point2vec.modules.transformer import TransformerEncoder, TransformerEncoderOutput
from point2vec.utils import transforms
from point2vec.modules.masking import MaskedBatchNorm1d, masked_layer_norm
from point2vec.modules.segmentation import SegmentationHead
from point2vec.utils.checkpoint import extract_model_checkpoint
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

class Point2VecKeyPointGeneration(pl.LightningModule):
    def __init__(
        self,
        num_channels: int = 3,
        tokenizer_num_init_groups: int = 128,
        tokenizer_context_length: int = 256,
        tokenizer_group_size: int = 32,
        tokenizer_group_radius: float | None = None,
        tokenizer_upscale_group_size: int | None = None,
        tokenizer_overlap_factor: float | None = None,
        tokenizer_reduction_method: str = 'fps',
        use_relative_features: bool = True,
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        encoder_dropout: float = 0,
        encoder_attention_dropout: float = 0,
        encoder_drop_path_rate: float = 0.2,
        encoder_add_pos_at_every_layer: bool = True,
        embedding_type: Optional[str] = "mini",
        loss_func: str = "l2",
        encoder_unfreeze_epoch: int = 0,
        seg_head_fetch_layers: List[int] = [3, 7, 11],
        seg_head_dim: int = 512,
        seg_head_dropout: float = 0.5,
        learning_rate: float = 0.001,
        optimizer_adamw_weight_decay: float = 0.05,
        lr_scheduler_linear_warmup_epochs: int = 10,
        lr_scheduler_linear_warmup_start_lr: float = 1e-6,
        lr_scheduler_cosine_eta_min: float = 1e-6,
        pretrained_ckpt_path: str | None = None,
        train_transformations: List[str] = [
            "rotate",
        ],  # scale, center, unit_sphere, rotate, translate, height_norm
        val_transformations: List[str] = [],
        transformation_scale_min: float = 0.8,
        transformation_scale_max: float = 1.2,
        transformation_scale_symmetries: Tuple[int, int, int] = (1, 0, 1),
        transformation_rotate_dims: List[int] = [0,1,2],
        transformation_rotate_degs: Optional[int] = None,
        transformation_translate: float = 0.2,
        transformation_height_normalize_dim: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()


        def build_transformation(name: str) -> transforms.Transform:
            if name == "scale":
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
            reduction_method=tokenizer_reduction_method,
            use_relative_features=use_relative_features,
        )

        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, encoder_dim),
        )
        if use_relative_features:
            assert (
                num_channels > 3
            ), "num_channels must be greater than 3 to use relative features"
            self.feature_encoder = nn.Sequential(
                nn.Linear(num_channels - 3, 128),
                nn.GELU(),
                nn.Linear(128, encoder_dim),
            )
        else:
            self.feature_encoder = None

        dpr = [
            x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_depth)
        ]
        self.encoder = TransformerEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            qkv_bias=True,
            drop_rate=encoder_dropout,
            attn_drop_rate=encoder_attention_dropout,
            drop_path_rate=dpr,
            add_pos_at_every_layer=encoder_add_pos_at_every_layer,
        )

        match loss_func:
            case "cdl1":
                self.loss_func = ChamferDistanceL1()
            case "cdl2":
                self.loss_func = ChamferDistanceL2()
            case _:
                raise ValueError(f"Unknown loss function: {loss_func}")

    def setup(self, stage: Optional[str] = None) -> None:
        point_dim: int = self.hparams.num_channels  # type: ignore
        upsampling_dim: int = self.hparams.encoder_dim  # type: ignore
        self.upsampling = PointNetFeatureUpsampling(in_channel=upsampling_dim + point_dim, mlp=[upsampling_dim, upsampling_dim])  # type: ignore

        self.seg_head = SegmentationHead(
            self.hparams.encoder_dim,
            0, # event-wide label embedding -- 0 for larnet!
            upsampling_dim,
            self.hparams.seg_head_dim,
            self.hparams.seg_head_dropout,
            num_seg_classes=6, # (x1, y1, z1, x2, y2, z2)
        )

        if self.hparams.pretrained_ckpt_path is not None:  # type: ignore
            self.load_pretrained_checkpoint(self.hparams.pretrained_ckpt_path)  # type: ignore

        # freeze everything but the segmentation head
        self.tokenizer.requires_grad_(False)
        self.positional_encoding.requires_grad_(False)
        if self.feature_encoder is not None:
            self.feature_encoder.requires_grad_(False)
        self.encoder.requires_grad_(False)

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                logger.watch(self)
                logger.experiment.define_metric("val_acc", summary="last,max")
                logger.experiment.define_metric("val_macc", summary="last,max")
                logger.experiment.define_metric("val_ins_miou", summary="last,max")
                logger.experiment.define_metric("val_cat_miou", summary="last,max")
                logger.experiment.define_metric("val_precision", summary="last,max")
                logger.experiment.define_metric("val_mprecision", summary="last,max")

    def center_encoding(self, centers: torch.Tensor) -> torch.Tensor:
        pos = self.positional_encoding(centers[:, :, :3])
        if self.feature_encoder is not None:
            pos = pos + self.feature_encoder(centers[:, :, 3:])
        return pos

    def forward(self,
                points: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        # points: (B, N, 3)
        # lengths: (B,)
        B, N, C = points.shape
        point_mask = torch.arange(lengths.max(), device=lengths.device).expand(
            len(lengths), -1
        ) < lengths.unsqueeze(-1)

        tokens: torch.Tensor
        centers: torch.Tensor
        embedding_mask: torch.Tensor
        tokens, centers, embedding_mask, _ = self.tokenizer(points, lengths, None)
        pos_embeddings = self.center_encoding(centers)
        output: TransformerEncoderOutput = self.encoder(
            tokens, pos_embeddings, embedding_mask, return_hidden_states=True
        )

        hidden_states = [
            masked_layer_norm(output.hidden_states[i], output.hidden_states[i].shape[-1], embedding_mask)
            for i in self.hparams.seg_head_fetch_layers]  # type: ignore [(B, T, C)]
        token_features = torch.stack(hidden_states, dim=0).mean(0)  # (B, T, C)
        token_features_max = token_features.max(dim=1).values  # (B, C)
        token_features_mean = token_features.mean(dim=1)  # (B, C)

    
        global_feature = torch.cat(
            [token_features_max, token_features_mean], dim=-1
        )  # (B, 2*C')

        batch_lengths = embedding_mask.sum(dim=1)
        x = self.upsampling(
            points,
            centers[:, :, :3],
            points,
            token_features,
            lengths,
            batch_lengths,
            point_mask,
        )  # (B, N, C)
        x = torch.cat(
            [x, global_feature.unsqueeze(-1).expand(-1, -1, N).transpose(1, 2)], dim=-1
        )  # (B, N, C'); C' = 3*C (+ L if we cared about event-wide embeddings)
        x = self.seg_head(x.transpose(1, 2), point_mask).transpose(1, 2)  # (B, N, 6)
        return x, point_mask

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # points: (B, N_max, 3)
        # lengths: (B,)
        # seg_labels: (B, N_max, 1)
        # endpoints: (B, N_max, 6) (x1, y1, z1, x2, y2, z2)
        points, lengths, seg_labels, endpoints = batch
        points = self.train_transformations(points)

        pred, point_mask = self.forward(points, lengths)
        pred = pred[point_mask].reshape(-1, 2, 3) # (B, N_max, 6) --> (N, 2, 3)
        target = endpoints[point_mask].reshape(-1, 2, 3) # (B, N_max, 6) --> (N, 2, 3)

        # promote to float32
        pred = pred.float()
        target = target.float()
        loss = self.loss_func(pred, target)
        self.log("train_loss", loss, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, List[torch.Tensor]]:
        # points: (B, N_max, 3)
        # lengths: (B,)
        # seg_labels: (B, N_max, 1)
        # endpoints: (B, N_max, 6) (x1, y1, z1, x2, y2, z2)
        points, lengths, seg_labels, endpoints = batch
        points = self.val_transformations(points)

        pred, point_mask = self.forward(points, lengths)
        pred = pred[point_mask].reshape(-1, 2, 3)  # (B, N_max, 6) --> (N, 2, 3)
        target = endpoints[point_mask].reshape(-1, 2, 3)  # (B, N_max, 6) --> (N, 2, 3)
        loss = self.loss_func(pred.float(), target.float())
        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        assert self.trainer is not None

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

    def load_pretrained_checkpoint(self, path: str) -> None:
        print(f"Loading pretrained checkpoint from '{path}'.")

        checkpoint = extract_model_checkpoint(path)

        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)  # type: ignore
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

    def on_train_epoch_start(self) -> None:
        # usually, unfreeze the encoder
        pass

        # if self.trainer.current_epoch == self.hparams.encoder_unfreeze_epoch:  # type: ignore
        #     self.encoder.requires_grad_(True)
        #     print("Unfreeze encoder")
        # self.trainer.model = torch.compile(self.trainer.model, mode='reduce-overhead')

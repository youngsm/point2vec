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
from point2vec.modules.pointnet import PointcloudTokenizer
from point2vec.modules.transformer import TransformerEncoder, TransformerEncoderOutput
from point2vec.utils import transforms
from point2vec.modules.masking import MaskedBatchNorm1d, masked_layer_norm
from point2vec.modules.segmentation import SegmentationHead
from point2vec.utils.checkpoint import extract_model_checkpoint
from point2vec.modules.loss import AggregateLoss
from point2vec.modules.grouping import masked_gather
from pytorch3d.ops import ball_query

@torch.no_grad()
def group_endpoints(
    centers,
    endpoints,
    centers_lengths,
    endpoints_lengths,
    radius,
    K=10,
):
    """
    Finds the endpoints corresponding to single group. I.e., given the group center, out of
    all the endpoints, find the ones that are within the group radius. this is so when we
    regress 10 point proposals, for each group, we can compute the loss based on just
    the endpoints that are within the group radius.

    Args:
        centers: (B, P1, 3)
        endpoints: (B, P2, 3)
    """

    assert centers.dim() == 3, f"centers.dim() = {centers.dim()} != 3"
    assert endpoints.dim() == 3, f"endpoints.dim() = {endpoints.dim()} != 3"

    # ball_query for each group center, get end points within group radius
    distances, idx, _ = ball_query(
        centers[..., :3],  # (B, P1=G, 3) for each center,
        endpoints[..., :3],  # (B, P2=10ish, 3) find the closest endpoints within radius
        radius=radius,
        lengths1=centers_lengths,  # (B,) centers mask
        lengths2=endpoints_lengths,  # (B,) endpoints mask
        K=K,
    )

    endpoints_groups = masked_gather(endpoints, idx)
    endpoints_groups[idx.eq(-1)] = 0.0
    endpoints_mask = idx.ne(-1)

    return endpoints_groups, endpoints_mask


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
        loss_scheduler: str = "cosine",
        loss_scheduler_kwargs: Dict[str, float] = {'start_value': 0.2, 'end_value': 0.05, 'total_steps': 90},
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

        self.loss_func = AggregateLoss(loss_scheduler, **loss_scheduler_kwargs)

        self.ppn_head = nn.Sequential(
            nn.Linear(3 * encoder_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 50 * (3 + 1 + 9)), # (pos + conf + npoint logits (up to 0-8))
        )

    def setup(self, stage: Optional[str] = None) -> None:
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
                logger.experiment.define_metric("val_loss", summary="last,max")
                logger.experiment.define_metric("val_l1", summary="last,max")
                logger.experiment.define_metric("val_ce_active", summary="last,max")
                logger.experiment.define_metric("val_ce_inactive", summary="last,max")

    def center_encoding(self, centers: torch.Tensor) -> torch.Tensor:
        pos = self.positional_encoding(centers[:, :, :3])
        if self.feature_encoder is not None:
            pos = pos + self.feature_encoder(centers[:, :, 3:])
        return pos

    def forward(self,
                points: torch.Tensor,
                lengths: torch.Tensor,
                endpoints: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # points: (B, N, 3)
        # lengths: (B,)
        B, N, C = points.shape
        point_mask = torch.arange(lengths.max(), device=lengths.device).expand(
            len(lengths), -1
        ) < lengths.unsqueeze(-1)

        tokens: torch.Tensor
        centers: torch.Tensor
        embedding_mask: torch.Tensor
        semantic_id_groups: torch.Tensor
        endpoints_groups: torch.Tensor
        group: torch.Tensor
        point_mask: torch.Tensor

        (
            tokens,
            centers,
            embedding_mask,
            semantic_id_groups,
            endpoints_groups,
            group,
            point_mask,
        ) = self.tokenizer(points, lengths, None, None, return_point_info=True)
        pos_embeddings = self.center_encoding(centers)
        output: TransformerEncoderOutput = self.encoder(
            tokens, pos_embeddings, embedding_mask, return_hidden_states=True
        )
        B, T, C = output.hidden_states[0].shape

        hidden_states = [
            masked_layer_norm(output.hidden_states[i], output.hidden_states[i].shape[-1], embedding_mask)
            for i in self.hparams.seg_head_fetch_layers]  # type: ignore [(B, T, C)]
        token_features = torch.stack(hidden_states, dim=0).mean(0)  # (B, T, C)
        token_features_max = token_features.max(dim=1).values  # (B, C)
        token_features_mean = token_features.mean(dim=1)  # (B, C)

    
        x = torch.cat(
            [token_features, token_features_max, token_features_mean], dim=-1
        )  # (B, 3*C')

        x = torch.cat(
            [token_features, token_features_max, token_features_mean], dim=-1
        )  # (B, T, 3*C')

        x = self.ppn_head(x).reshape(B, T, -1, 3 + 1 + 9)

        endpoints_groups, endpoints_mask = group_endpoints(
        centers=centers,
        endpoints=endpoints,
        centers_lengths=embedding_mask.sum(-1),
        endpoints_lengths=(endpoints.ne(0.0).all(-1)).sum(-1),
        radius=self.tokenizer.grouping.group_radius,
        )
        endpoint_offsets = endpoints_groups.clone()
        endpoint_offsets[..., :3] = (endpoints_groups[..., :3] - centers[..., None, :3]) / self.tokenizer.grouping.group_radius * endpoints_mask.unsqueeze(-1)

        pred = x
        target = endpoint_offsets
        return pred, target

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # points: (B, N_max, 3)
        # lengths: (B,)
        # seg_labels: (B, N_max, 1)
        # endpoints: (B, N_max, 6) (x1, y1, z1, x2, y2, z2)
        points, lengths, seg_labels, endpoints = batch
        combined = torch.cat([points, endpoints], dim=1)
        combined = self.train_transformations(combined)
        points, endpoints = combined[:, :points.shape[1], :], combined[:, points.shape[1]:, :]

        pred, target = self.forward(points, lengths, endpoints)
        # promote to float32
        pred = pred.float()
        target = target.float()
        losses = self.loss_func(pred, target)
        loss = losses['l1'] + (losses['ce_active'] + losses['ce_inactive']) / 2

        self.log("train_loss", loss, on_epoch=True)
        for k,v in losses.items():
            self.log(f"train_{k}", v, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, List[torch.Tensor]]:
        # points: (B, N_max, 3)
        # lengths: (B,)
        # seg_labels: (B, N_max, 1)
        # endpoints: (B, N_max, 6) (x1, y1, z1, x2, y2, z2)
        points, lengths, seg_labels, endpoints = batch
        combined = torch.cat([points, endpoints], dim=1)
        combined = self.val_transformations(combined)
        points, endpoints = combined[:, :points.shape[1], :], combined[:, points.shape[1]:, :]

        pred, target = self.forward(points, lengths, endpoints)

        # promote to float32
        pred = pred.float()
        target = target.float()
        losses = self.loss_func(pred, target)
        loss = losses['l1'] + (losses['ce_active'] + losses['ce_inactive']) / 2

        self.log("val_loss", loss)
        for k,v in losses.items():
            self.log(f"val_{k}", v)

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

    def on_train_epoch_end(self, outputs, batch, batch_idx):
        # Update the voxel size scheduler's value
        self.loss_func.size_scheduler.step()

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

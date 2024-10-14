from typing import Tuple

import torch
import torch.nn as nn
import math
from .grouping import PointcloudGrouping
from .masking import MaskedBatchNorm1d

class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(
        self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None
    ):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats
        )

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = nn.functional.pad(pos_emb, (0, self.padding))
        return pos_emb


class MiniPointNet(nn.Module):
    def __init__(self, channels: int, feature_dim: int):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(channels, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, feature_dim, 1),
        )

    def forward(self, points, mask=None) -> torch.Tensor:
        # points: (B, N, C)
        feature = self.first_conv(points.transpose(2, 1))  # (B, 256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True).values  # (B, 256, 1)
        # concating global features to each point features
        feature = torch.cat(
            [feature_global.expand(-1, -1, feature.shape[2]), feature], dim=1
        )  # (B, 512, N)
        feature = self.second_conv(feature)  # (B, feature_dim, N)
        feature_global = torch.max(feature, dim=2).values  # (B, feature_dim)
        return feature_global


class MaskedMiniPointNet(nn.Module):
    def __init__(self, channels: int, feature_dim: int):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(channels, 128, 1, bias=False),
            MaskedBatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1, bias=False),
            MaskedBatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, feature_dim, 1),
        )

    def forward(self, points, mask) -> torch.Tensor:
        # points: (B, N, C)
        # mask: (B, 1, N)

        feature = points.transpose(2, 1)  # (B, C, N)
        for layer in self.first_conv:
            if isinstance(layer, MaskedBatchNorm1d):
                feature = layer(feature, mask)
            else:
                feature = layer(feature)

        # (B, 256, N) --> (B, 256, 1)
        feature_global = torch.max(feature, dim=2, keepdim=True).values  # (B, 256, 1)
        # concating global features to each point features
        feature = torch.cat(
            [feature_global.expand(-1, -1, feature.shape[2]), feature], dim=1
        )  # (B, 512, N)

        for layer in self.second_conv:
            if isinstance(layer, MaskedBatchNorm1d):
                feature = layer(feature, mask)
            else:
                feature = layer(feature)

        # (B, feature_dim, N) --> (B, feature_dim)
        feature_global = torch.max(feature, dim=2).values  # (B, feature_dim)
        return feature_global

class LargePointNet(nn.Module):
    def __init__(self, channels: int, feature_dim: int):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(channels, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1),
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(2048, 2048, 1, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, feature_dim, 1),
        )

    def forward(self, points, mask=None) -> torch.Tensor:
        # points: (B, N, C)
        feature = self.first_conv(points.transpose(2, 1))  # (B, 256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True).values  # (B, 256, 1)
        # concating global features to each point features
        feature = torch.cat(
            [feature_global.expand(-1, -1, feature.shape[2]), feature], dim=1
        )  # (B, 512, N)
        feature = self.second_conv(feature)  # (B, feature_dim, N)
        feature_global = torch.max(feature, dim=2).values  # (B, feature_dim)
        return feature_global


class PointcloudTokenizer(nn.Module):
    def __init__(
        self,
        num_init_groups: int,
        context_length: int,
        group_size: int,
        group_radius: float | None,
        upscale_group_size: int | None,
        overlap_factor: float | None,
        token_dim: int,
        num_channels: int,
        embedding_type: str = "mini",
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.grouping = PointcloudGrouping(
            num_groups=num_init_groups,
            group_size=group_size,
            group_radius=group_radius,
            upscale_group_size=upscale_group_size,
            overlap_factor=overlap_factor,
            context_length=context_length,
        )

        if embedding_type == "mini":
            self.embedding = MiniPointNet(num_channels, token_dim)
        elif embedding_type == "large":
            self.embedding = LargePointNet(num_channels, token_dim)
        elif embedding_type == "masked_mini":
            self.embedding = MaskedMiniPointNet(num_channels, token_dim)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

    def forward(self,
                points: torch.Tensor,
                lengths: torch.Tensor,
                semantic_id: torch.Tensor | None = None,
                return_group: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # points: (B, N, num_channels)
        # lengths: (B,)
        group: torch.Tensor
        group_center: torch.Tensor
        tokens: torch.Tensor
        lengths: torch.Tensor
        semantic_id_groups: torch.Tensor | None

        group, group_center, embedding_mask, point_mask, semantic_id_groups = self.grouping(points, lengths, semantic_id)  # (B, G, K, C), (B, G, 3), (B, G, K)
        B, G, S, C = group.shape
        tokens = self.embedding(group.reshape(B * G, S, C), point_mask.reshape(B * G, 1, S)).reshape(
            B, G, self.token_dim
        )  # (B, G, C')
        if return_group:
            return tokens, group_center, embedding_mask, semantic_id_groups, group
        else:
            return tokens, group_center, embedding_mask, semantic_id_groups
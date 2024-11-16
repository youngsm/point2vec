from typing import Tuple

import torch
import torch.nn as nn
import math
from .grouping import PointcloudGrouping
from .masking import MaskedBatchNorm1d

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
        reduction_method: str = 'energy',
        use_relative_features: bool = False,
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
            reduction_method=reduction_method,
            use_relative_features=use_relative_features,
        )

        if embedding_type == "mini":
            self.embedding = MiniPointNet(num_channels, token_dim)
        elif embedding_type == "large":
            self.embedding = LargePointNet(num_channels, token_dim)
        elif embedding_type == "masked_mini":
            self.embedding = MaskedMiniPointNet(num_channels, token_dim)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

    def forward(
        self,
        points: torch.Tensor,
        lengths: torch.Tensor,
        semantic_id: torch.Tensor | None = None,
        endpoints: torch.Tensor | None = None,
        return_point_info: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # points: (B, N, num_channels)
        # lengths: (B,)
        group: torch.Tensor
        group_center: torch.Tensor
        tokens: torch.Tensor
        lengths: torch.Tensor
        semantic_id_groups: torch.Tensor | None

        (group, group_center, embedding_mask,
        point_mask, semantic_id_groups, endpoints_groups) = self.grouping(
            points, lengths, semantic_id, endpoints)  # (B, G, K, C), (B, G, 3), (B, G, K)
        B, G, S, C = group.shape
        tokens = self.embedding(group.reshape(B * G, S, C), point_mask.reshape(B * G, 1, S)).reshape(
            B, G, self.token_dim
        )  # (B, G, C')
        if return_point_info:
            return (
                tokens,
                group_center,
                embedding_mask,
                semantic_id_groups,
                endpoints_groups,
                group,
                point_mask,
            )
        else:
            return (
                tokens,
                group_center,
                embedding_mask,
                semantic_id_groups,
                endpoints_groups,
            )

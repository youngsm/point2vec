from typing import Tuple

import torch
import torch.nn as nn
from pytorch3d.ops import ball_query, knn_gather, knn_points, sample_farthest_points
from pytorch3d.ops.utils import masked_gather
from torch import nn
import math

def fill_empty_indices(idx: torch.Tensor) -> torch.Tensor:
    """
    replaces all empty indices (-1) with the first index from its group
    """
    B, G, K = idx.shape
    mask = idx == -1
    first_idx = idx[:, :, 0].unsqueeze(-1).expand(-1, -1, K)
    idx[mask] = first_idx[mask]  # replace -1 index with first index
    # print(f"DEBUG: {(len(idx[mask].view(-1)) / len(idx.view(-1))) * 100:.1f}% of ball query indices are empty")

    return idx


def select_topk_by_energy(
    points: torch.Tensor,
    idx: torch.Tensor,
    K_new: int,
    invalid_idx_mask: torch.Tensor,
    energies_idx: int = 3,
) -> torch.Tensor:
    """
    Select the top K_new indices based on energies for each group.

    Args:
        points: Tensor of shape (B, N, C) containing point cloud data.
        idx: Tensor of shape (B, G, K_original) containing indices from ball_query.
        K_new: Desired number of top points to select per group.
        invalid_idx_mask: Boolean tensor of shape (B, G, K_original), where True indicates invalid indices.
        energies_idx: Index in `points` where the energy value is stored.

    Returns:
        topk_idx: Tensor of shape (B, G, K_new) containing indices of the top K_new energies per group.
    """
    B, G, K_original = idx.shape

    # Clamp idx to handle negative indices for gathering
    idx_clamped = idx.clamp(min=0)  # Shape: (B, G, K_original)

    # Extract energies from points
    points_energies = points[..., energies_idx]  # Shape: (B, N)

    # Expand points_energies to match idx_clamped's shape for gathering
    points_energies_expanded = points_energies.unsqueeze(1).expand(
        -1, G, -1
    )  # Shape: (B, G, N)

    # Gather energies using idx_clamped
    energies = torch.gather(
        points_energies_expanded, dim=2, index=idx_clamped
    )  # Shape: (B, G, K_original)

    # Set energies of invalid indices to -infinity
    energies[invalid_idx_mask] = -float("inf")

    # Select top K_new energies
    topk_energies, topk_indices = energies.topk(K_new, dim=2)  # Shapes: (B, G, K_new)

    # Gather corresponding indices
    topk_idx = torch.gather(idx, 2, topk_indices)  # Shape: (B, G, K_new)

    # Set invalid indices back to -1
    mask_valid = topk_energies != -float("inf")  # Shape: (B, G, K_new)
    topk_idx[~mask_valid] = -1

    return topk_idx


class PointcloudGrouping(nn.Module):
    def __init__(
        self,
        num_groups: int,
        group_size: int,
        group_radius: float | None,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size  # Desired final group size (K_new)
        self.group_radius = group_radius

    def forward(
        self,
        points: torch.Tensor,
        lengths: torch.Tensor,
        semantic_id: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # points: (B, N, C)
        # lengths: (B,)

        # Sample farthest points using lengths
        group_centers, _ = sample_farthest_points(
            points[:, :, :3].float(),
            K=self.num_groups,
            lengths=lengths,
            random_start_point=True,
        )  # (B, G, 3)
        semantic_id_groups = None

        if self.group_radius is None:
            # Use KNN grouping
            _, idx, _ = knn_points(
                group_centers.float(),
                points[:, :, :3].float(),
                lengths1=None,
                lengths2=lengths,
                K=self.group_size,
                return_sorted=False,
                return_nn=False,
            )  # idx: (B, G, K)
            groups = knn_gather(points, idx)  # (B, G, K, C)
            if semantic_id is not None:
                semantic_id_groups = knn_gather(semantic_id, idx)  # (B, G, K)
        else:
            # Use ball query with energy-based selection
            K_original = 5 * self.group_size  # Initial K

            dists, idx, _ = ball_query(
                group_centers.float(),
                points[:, :, :3].float(),
                K=K_original,
                radius=self.group_radius,
                lengths1=None,  # Lengths of group centers (None since all have num_groups)
                lengths2=lengths,
                return_nn=False,
            )  # idx: (B, G, K_original)

            # Create invalid index mask
            invalid_idx_mask = idx == -1  # Shape: (B, G, K_original)

            # Energy-based selection
            idx = select_topk_by_energy(
                points=points,
                idx=idx,
                K_new=self.group_size,
                invalid_idx_mask=invalid_idx_mask,
                energies_idx=3,  # Assuming energy is at index 3
            )  # idx: (B, G, K_new)

            if semantic_id is not None:
                semantic_id_groups = masked_gather(semantic_id, idx)  # (B, G, K_new)
                semantic_id_groups[idx.eq(-1)] = -1

            groups = masked_gather(points, fill_empty_indices(idx))  # (B, G, K_new, C)

        # Normalize group coordinates
        groups[:, :, :, :3] = groups[:, :, :, :3] - group_centers.unsqueeze(2)
        if self.group_radius is not None:
            groups = groups / self.group_radius  # Normalize by group radius

        return (
            groups,
            group_centers,
            semantic_id_groups,
        )  # (B, G, K_new, C), (B, G, 3), (B, G, K_new)


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

    def forward(self, points) -> torch.Tensor:
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

    def forward(self, points) -> torch.Tensor:
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
        num_groups: int,
        group_size: int,
        group_radius: float | None,
        token_dim: int,
        num_channels: int,
        embedding_type: str = "mini",
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.grouping = PointcloudGrouping(
            num_groups=num_groups, group_size=group_size, group_radius=group_radius
        )

        if embedding_type == "mini":
            self.embedding = MiniPointNet(num_channels, token_dim)
        elif embedding_type == "large":
            self.embedding = LargePointNet(num_channels, token_dim)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

    def forward(self, points: torch.Tensor, semantic_id: torch.Tensor | None = None, return_group: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # points: (B, N, num_channels)
        group: torch.Tensor
        group_center: torch.Tensor
        tokens: torch.Tensor
        semantic_id_groups: torch.Tensor | None


        group, group_center, semantic_id_groups = self.grouping(points, semantic_id)  # (B, G, K, C), (B, G, 3), (B, G, K)
        B, G, S, C = group.shape
        tokens = self.embedding(group.reshape(B * G, S, C)).reshape(
            B, G, self.token_dim
        )  # (B, G, C')
        if return_group:
            return tokens, group_center, semantic_id_groups, group
        else:
            return tokens, group_center, semantic_id_groups

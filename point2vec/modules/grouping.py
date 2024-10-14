from pytorch3d.ops import ball_query, knn_gather, knn_points, sample_farthest_points
import torch
from typing import Tuple
import torch.nn as nn
from cnms import cnms

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

def masked_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Helper function for torch.gather to collect the points at
    the given indices in idx where some of the indices might be -1 to
    indicate padding. These indices are first replaced with 0.
    Then the points are gathered after which the padded values
    are set to 0.0.

    Args:
        points: (N, P, D) float32 tensor of points
        idx: (N, K) or (N, P, K) long tensor of indices into points, where
            some indices are -1 to indicate padding

    Returns:
        selected_points: (N, K, D) float32 tensor of points
            at the given indices
    """
    if len(idx) != len(points):
        raise ValueError("points and idx must have the same batch dimension")

    N, P, D = points.shape

    # Replace -1 values with 0 before expanding
    idx = idx.clone()
    idx[idx.eq(-1)] = 0

    if idx.ndim == 3:
        # Case: KNN, Ball Query where idx is of shape (N, P', K)
        # where P' is not necessarily the same as P as the
        # points may be gathered from a different pointcloud.
        K = idx.shape[2]
        # Match dimensions for points and indices
        idx_expanded = idx[..., None].expand(-1, -1, -1, D)
        points = points[:, :, None, :].expand(-1, -1, K, -1)
    elif idx.ndim == 2:
        # Farthest point sampling where idx is of shape (N, K)
        idx_expanded = idx[..., None].expand(-1, -1, D)
    else:
        raise ValueError(f"idx format is not supported {idx.shape}")

    # idx_expanded_mask = idx_expanded.eq(-1)
    # Gather points
    selected_points = points.gather(dim=1, index=idx_expanded)

    # SY 10/7/24: this takes a while and doesn't seem to be necessary because
    # we don't use the invalid indices for anything
    # Replace padded values
    # selected_points[idx_expanded_mask] = 0.0
    return selected_points

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
        group_radius: float,
        upscale_group_size: int = 512,
        overlap_factor: float = 0.7,
        context_length: int = 256
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size  # Desired final group size (K_new)
        self.group_radius = group_radius
        self.upscale_group_size = upscale_group_size
        self.overlap_factor = overlap_factor
        self.context_length = context_length

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

        # Sphere culling
        if self.overlap_factor is not None:
            retain = cnms(group_centers, overlap_factor=self.overlap_factor, radius=self.group_radius)
            lengths1 = retain.sum(dim=1)
        else:
            lengths1 = None

        if self.upscale_group_size is None:
            self.upscale_group_size = self.num_groups

        if self.group_radius is None:
            # KNN
            _, idx, _ = knn_points(
                group_centers.float(),
                points[:, :, :3].float(),
                lengths1=lengths1,
                lengths2=lengths,
                K=self.upscale_group_size,
                return_sorted=False,
                return_nn=False,
            )  # (B, G, K_big)
        else:
            # Ball query
            _, idx, _ = ball_query(
                group_centers.float(),
                points[:, :, :3].float(),
                K=self.upscale_group_size,
                radius=self.group_radius,
                lengths1=lengths1,
                lengths2=lengths,
                return_nn=False,
            )  # idx: (B, G, K_big)

        # Create invalid index mask
        invalid_idx_mask = idx == -1  # Shape: (B, G, K_original)

        # Energy-based selection (K_big --> K by taking top K energies)
        idx = select_topk_by_energy(
            points=points,
            idx=idx,
            K_new=self.group_size,
            invalid_idx_mask=invalid_idx_mask,
            energies_idx=3,  # Assuming energy is at index 3
        )  # idx: (B, G, K)

        # Gather semantic ids
        if semantic_id is not None:
            semantic_id_groups = masked_gather(
                    semantic_id, idx
            )  # (B, G, K, 1)
            semantic_id_groups[idx.eq(-1)] = -1
        # Create point mask with shape (B, G, K)
        point_lengths = (~idx.eq(-1)).sum(2)  # (B, G)
        groups = masked_gather(points, fill_empty_indices(idx))  # (B, G, K, C)
        point_mask = torch.arange(self.group_size, device=idx.device).expand(groups.size(0), self.context_length, -1) < point_lengths[:, :self.context_length].unsqueeze(-1)  # (B, G, K)

        # Create embedding mask (i.e. which groups/embeddings to ignore in transformer)
        B, G, K, C = groups.shape
        group_lengths = (~idx.eq(-1)).all(2).sum(1) # (B,)
        embedding_mask = torch.arange(G, device=points.device).repeat(B, 1) < group_lengths.unsqueeze(1)

        # Normalize group coordinates
        groups[:, :, :, :3] = groups[:, :, :, :3] - group_centers.unsqueeze(2)
        if self.group_radius is not None:
            groups[:, :, :, :3] = (
                groups[:, :, :, :3] / self.group_radius
            )  # proposed by PointNeXT to make relative coordinates less small

        # G (max groups) --> T (context length)
        groups = groups[:, :self.context_length] # (B, G, K, C) --> (B, T, K, C)
        group_centers = group_centers[:, :self.context_length] # (B, G, 3) --> (B, T, 3)
        embedding_mask = embedding_mask[:, :self.context_length] # (B, G) --> (B, T)
        if semantic_id_groups is not None:
            semantic_id_groups = semantic_id_groups[:, :self.context_length] # (B, G, K) --> (B, T, K)

        return (
            groups,
            group_centers,
            embedding_mask,
            point_mask,
            semantic_id_groups,
        )  # (B, T, K, C), (B, T, 3), (B, T), (B, T, K)

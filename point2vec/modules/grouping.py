from pytorch3d.ops import ball_query, knn_gather, knn_points, sample_farthest_points
from pytorch3d.ops.utils import masked_gather
import torch
from typing import Tuple
import torch.nn as nn
from greedy_reduction import greedy_reduction

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


def cull_spheres_cuda(sphere_centers, overlap_factor=0.7, K=512, radius=25 / 760):
    """
    Perform sphere culling using CUDA to remove overlapping spheres.

    Args:
        sphere_centers: Tensor of shape (N, P, 3) containing the coordinates of sphere centers.
        overlap_factor: Factor to determine the query radius for overlap checking.
        K: Number of neighbors to consider in ball query.
        radius: Radius of the spheres.

    Returns:
        retain: Boolean tensor of shape (N, P) indicating which spheres to retain.
    """
    device = sphere_centers.device
    p1 = sphere_centers
    p2 = sphere_centers

    query_radius = 2 * radius * overlap_factor  # Example: 2.4 for R=1.0

    dists, idx, nn = ball_query(
        p1=p1,
        p2=p2,
        K=K,
        radius=query_radius,
        return_nn=True,
    )

    N, P, K, D = nn.shape
    device = nn.device

    # Compute overlap counts by excluding self (assuming self is always included)
    # Create a tensor of point indices
    point_indices = (
        torch.arange(P, device=device).view(1, P, 1).repeat(N, 1, K)
    )  # (N, P, K)
    # Compare with idx to exclude self
    mask = idx != point_indices
    overlap_counts = mask.sum(dim=-1)  # (N, P)

    # Sort overlap counts in descending order
    sorted_overlap_counts, sorted_indices = overlap_counts.sort(
        dim=-1, descending=True
    )  # Both: (N, P)

    retain = greedy_reduction(sorted_indices, idx)

    return retain


class PointcloudGrouping(nn.Module):
    def __init__(
        self,
        num_groups: int,
        group_size: int,
        group_radius: float | None = None,
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

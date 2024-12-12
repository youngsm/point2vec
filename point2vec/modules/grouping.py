from pytorch3d.ops import ball_query, knn_points
import torch
from typing import Tuple
import torch.nn as nn
from cnms import cnms
from pytorch3d import _C
from typing import Optional, List, Union

def masked_mean(group, point_mask):
    valid_elements = point_mask.sum(-1).float() + 1e-10
    return (group * point_mask.unsqueeze(-1)).sum(-2) / valid_elements.unsqueeze(-1)

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


def sample_farthest_points(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List, torch.Tensor] = 50,
    random_start_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative farthest point sampling algorithm [1] to subsample a set of
    K points from a given pointcloud. At each iteration, a point is selected
    which has the largest nearest neighbor distance to any of the
    already selected points.

    Farthest point sampling provides more uniform coverage of the input
    point cloud compared to uniform random sampling.

    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.

    Args:
        points: (N, P, D) array containing the batch of pointclouds
        lengths: (N,) number of points in each pointcloud (to support heterogeneous
            batches of pointclouds)
        K: samples required in each sampled point cloud (this is typically << P). If
            K is an int then the same number of samples are selected for each
            pointcloud in the batch. If K is a tensor is should be length (N,)
            giving the number of samples to select for each element in the batch
        random_start_point: bool, if True, a random point is selected as the starting
            point for iterative sampling.

    Returns:
        selected_points: (N, K, D), array of selected values from points. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            0.0 for batch elements where k_i < max(K).
        selected_indices: (N, K) array of selected indices. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            -1 for batch elements where k_i < max(K).
    """
    N, P, D = points.shape
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)
    else:
        if lengths.shape != (N,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("A value in lengths was too large.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # Check dtypes are correct and convert if necessary
    if not (points.dtype == torch.float32):
        points = points.to(torch.float32)
    if not (lengths.dtype == torch.int64):
        lengths = lengths.to(torch.int64)
    if not (K.dtype == torch.int64):
        K = K.to(torch.int64)

    # Generate the starting indices for sampling
    start_idxs = torch.zeros_like(lengths)
    if random_start_point:
        for n in range(N):
            # pyre-fixme[6]: For 1st param expected `int` but got `Tensor`.
            start_idxs[n] = torch.randint(high=lengths[n], size=(1,)).item()

    with torch.no_grad():
        # pyre-fixme[16]: `pytorch3d_._C` has no attribute `sample_farthest_points`.
        idx = _C.sample_farthest_points(points[:, :, :3], lengths, K, start_idxs)
    sampled_points = masked_gather(points, idx)
    return sampled_points, idx


def select_topk_by_energy(
    points: torch.Tensor,
    idx: torch.Tensor,
    K: int,
    energies_idx: int = 3,
) -> torch.Tensor:
    """
    Select the top K indices based on energies for each group.

    Args:
        points: Tensor of shape (B, N, C) containing point cloud data.
        idx: Tensor of shape (B, G, K_original) containing indices from ball_query.
        K: Desired number of top points to select per group.
        energies_idx: Index in `points` where the energy value is stored.

    Returns:
        topk_idx: Tensor of shape (B, G, K) containing indices of the top K energies per group.
    """
    B, G, K_original = idx.shape
    invalid_idx_mask = idx == -1

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

    # Select top K energies
    topk_energies, topk_indices = energies.topk(K, dim=2)  # Shapes: (B, G, K)

    # Gather corresponding indices
    topk_idx = torch.gather(idx, 2, topk_indices)  # Shape: (B, G, K)

    # Set invalid indices back to -1
    mask_valid = topk_energies != -float("inf")  # Shape: (B, G, K)
    topk_idx[~mask_valid] = -1

    return topk_idx

def select_topk_by_fps(points: torch.Tensor, idx: torch.Tensor, K: int) -> torch.Tensor:
    """
    Args:
        points: Tensor of shape (B, N, C) containing point cloud data.
        idx: Tensor of shape (B, G, K_original) containing indices from ball_query.
        K: Desired number of points to select per group.
    """
    B, N, C = points.shape
    B, G, K_original = idx.shape

    # 1. index points by idx
    points_grouped = masked_gather(points, idx)  # (B, G, K, C)
    # 2. reshape points to (B*G, K, C)
    points_grouped = points_grouped.view(B*G, K_original, C)  # (B*G, K, C)
    # 3. run fps on the reshaped points
    _, idx_fps = sample_farthest_points(
        points_grouped,
        lengths=(~idx.eq(-1)).sum(2).view(B*G),
        K=K,
    )  # (B*G, K)
    # 4. reshape the fps indices back to to (B, G, K)
    idx_fps = idx_fps.view(B, G, K)

    invalid_idx_mask = idx_fps == -1
    idx_fps = idx_fps.clamp(min=0)
    idx_fps = torch.gather(idx, 2, idx_fps)
    idx_fps[invalid_idx_mask] = -1

    # 5. return the fps indices
    return idx_fps

class PointcloudGrouping(nn.Module):
    def __init__(
        self,
        num_groups: int,
        group_size: int,
        group_radius: float,
        upscale_group_size: int = 512,
        overlap_factor: float = 0.7,
        context_length: int = 256,
        reduction_method: str = "energy",  # energy or fps
        use_relative_features: bool = False,
        normalize_group_centers: bool = False,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size  # Desired final group size (K_new)
        self.group_radius = group_radius
        self.upscale_group_size = upscale_group_size
        self.overlap_factor = overlap_factor
        self.context_length = context_length
        self.reduction_method = reduction_method
        self.use_relative_features = use_relative_features
        self.normalize_group_centers = normalize_group_centers

    @torch.no_grad()
    def forward(
        self,
        points: torch.Tensor,
        lengths: torch.Tensor,
        semantic_id: torch.Tensor | None = None,
        endpoints: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # points: (B, N, C)
        # lengths: (B,)

        # Sample farthest points using lengths
        group_centers, _ = sample_farthest_points(
            points,
            K=self.num_groups,
            lengths=lengths,
            random_start_point=False,
        )  # (B, G, 3)

        semantic_id_groups = None
        endpoints_groups = None

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
                group_centers[:, :, :3].float(),
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
                group_centers[:, :, :3].float(),
                points[:, :, :3].float(),
                K=self.upscale_group_size,
                radius=self.group_radius,
                lengths1=lengths1,
                lengths2=lengths,
                return_nn=False,
            )  # idx: (B, G, K_big)

        # Energy-based selection (K_big --> K by taking top K energies)
        if self.reduction_method == 'energy':
            idx = select_topk_by_energy(
                points=points,
                idx=idx,
                K=self.group_size,
                energies_idx=3,  # Assuming energy is at index 3
            )  # idx: (B, G, K)
        elif self.reduction_method == 'fps':
            idx = select_topk_by_fps(
                points=points,
                idx=idx,
                K=self.group_size,
            )  # idx: (B, G, K)

        # Gather semantic ids
        if semantic_id is not None:
            semantic_id_groups = masked_gather(
                    semantic_id, idx
            )  # (B, G, K, 1)
            semantic_id_groups[idx.eq(-1)] = -1

        if endpoints is not None:
            endpoints_groups = masked_gather(
                endpoints, idx
            )  # (B, G, K, 6)
            endpoints_groups[idx.eq(-1)] = -1

        # Create point mask with shape (B, G, K)
        point_lengths = (~idx.eq(-1)).sum(2)  # (B, G)
        groups = masked_gather(points, fill_empty_indices(idx))  # (B, G, K, C)
        point_mask = torch.arange(self.group_size, device=idx.device).expand(groups.size(0), self.context_length, -1) < point_lengths[:, :self.context_length].unsqueeze(-1)  # (B, G, K)

        # Create embedding mask (i.e. which groups/embeddings to ignore in transformer)
        B, G, K, C = groups.shape
        group_lengths = (~idx.eq(-1)).all(2).sum(1) # (B,)
        embedding_mask = torch.arange(G, device=points.device).repeat(B, 1) < group_lengths.unsqueeze(1)


        # G (max groups) --> T (context length)
        # we are implicitly assuming that the number of non-padded groups is less than the context length. if 
        # this is not the case, some valid groups will be ignored, which is terrible. be careful!
        groups = groups[:, :self.context_length] # (B, G, K, C) --> (B, T, K, C)
        group_centers = group_centers[:, :self.context_length] # (B, G, 3) --> (B, T, 3)
        embedding_mask = embedding_mask[:, :self.context_length] # (B, G) --> (B, T)
        if semantic_id_groups is not None:
            semantic_id_groups = semantic_id_groups[:, :self.context_length] # (B, G, K) --> (B, T, K)
        if endpoints_groups is not None:
            endpoints_groups = endpoints_groups[:, :self.context_length] # (B, G, K, 6) --> (B, T, K, 6)

        if self.normalize_group_centers:
            group_centers = masked_mean(groups, point_mask)

        # Normalize group coordinates
        if self.use_relative_features:
            groups = groups - group_centers[:, :, None, :]
        else:
            groups[:, :, :, :3] = groups[:, :, :, :3] - group_centers[:, :, None, :3]

        if self.group_radius is not None:
            groups[:, :, :, :3] = (
                groups[:, :, :, :3] / self.group_radius
            )  # proposed by PointNeXT to make relative coordinates less small

        return (
            groups,
            group_centers,
            embedding_mask,
            point_mask,
            semantic_id_groups,
            endpoints_groups,
        )  # (B, T, K, C), (B, T, 3), (B, T), (B, T, K), (B, T, K, 6)

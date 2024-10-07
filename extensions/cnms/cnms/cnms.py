from pytorch3d.ops import ball_query
import torch
from cnms import _ext

def cnms(centroids, radius, overlap_factor):
    """
    Perform Centrality-Based Non-Maximum Suppression (C-NMS).

    Args:
        centroids: Tensor of shape (N, P, 3) containing the coordinates of centroids.
        overlap_factor: Factor to determine the query radius for overlap checking.
        radius: Radius of the spheres.

    Returns:
        retain: Boolean tensor of shape (N, P) indicating which spheres to retain.
    """
    N, P, _ = centroids.shape
    device = centroids.device

    query_radius = 2 * radius * overlap_factor
    _, idx, _ = ball_query(
        p1=centroids,
        p2=centroids,
        K=P,
        radius=query_radius,
        return_nn=False,
    )

    # Compute overlap counts by excluding self (assuming self is always included)
    point_indices = (
        torch.arange(P, device=device).view(1, P, 1).repeat(N, 1, P) # <-- last P is K
    )  # (N, P, K)

    # Compare with idx to exclude self
    mask = idx != point_indices
    overlap_counts = mask.sum(dim=-1)  # (N, P)

    # Sort overlap counts in descending order
    _, sorted_indices = overlap_counts.sort(
        dim=-1, descending=True
    )  # Both: (N, P)

    # Dispatch to C++/CUDA
    retain = _ext.greedy_reduction(sorted_indices, idx)

    return retain
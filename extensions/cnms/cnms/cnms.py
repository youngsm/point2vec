from pytorch3d.ops import ball_query
from cnms import _ext
import torch

@torch.no_grad()
def cnms(centroids, radius, overlap_factor, K=None, lengths=None):
    """
    Perform Centrality-Based Non-Maximum Suppression (C-NMS).

    Args:
        centroids: Tensor of shape (N, P, 3) containing the coordinates of centroids.
        overlap_factor: Factor to determine the query radius for overlap checking. (0.7 = 70% overlap of diameters)
        radius: Radius of the spheres (fixed).
        K: Number of points to consider for overlap checking. Default is P (all centroids).
        lengths: Tensor of shape (N,) containing the number of points in each cloud. Default is P for all clouds.
    Returns:
        retain: Boolean tensor of shape (N, P) indicating which spheres to retain.
    """
    N, P, _ = centroids.shape

    if lengths is None:
        # Create a dummy lengths tensor with shape (N,) and
        # all entries = P
        lengths = torch.full(
            (N,),
            fill_value=P,
            dtype=torch.int64,
            device=centroids.device
        )

    query_radius = 2 * radius * overlap_factor
    _, idx, _ = ball_query(
        p1=centroids,
        p2=centroids,
        K=P if K is None else K,
        radius=query_radius,
        lengths1=lengths,
        lengths2=lengths,
        return_nn=False,
    )

    overlap_counts = (~idx.eq(-1)).sum(-1)
    _, sorted_indices = overlap_counts.sort(
        dim=-1, descending=True
    )  # Both: (N, P)

    # Dispatch to C++/CUDA
    retain = _ext.greedy_reduction(sorted_indices, idx, lengths)

    return retain
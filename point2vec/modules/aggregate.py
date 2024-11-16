import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from math import cos, pi

class VoxelClustering(nn.Module):
    def __init__(self, size_scheduler):
        super(VoxelClustering, self).__init__()
        self.size_scheduler = size_scheduler

    def forward(self, pred, active_mask):
        B, G, P, F = pred.shape
        device = pred.device

        # Step 1: Identify active groups (groups with at least one valid target point)
        active_pred = pred[active_mask]  # Shape: (N_active, P, F)

        # Create batch_id and group_id for each point
        batch_ids = torch.arange(B, device=device).view(B, 1, 1).expand(B, G, P)
        group_ids = torch.arange(G, device=device).view(1, G, 1).expand(B, G, P)

        # Concatenate batch_id, group_id with the point proposals
        batch_group_ids = torch.cat(
            (batch_ids.unsqueeze(-1), group_ids.unsqueeze(-1)), dim=-1
        )  # Shape: (B, G, P, 2)

        batch_group_ids = batch_group_ids[active_mask]  # Shape: (N_active, P, 2)

        # Flatten batch_group_ids and active_pred
        batch_group_ids = batch_group_ids.reshape(-1, 2)
        offsets_flattened = active_pred.reshape(-1, F)  # Shape: (N_active * P, F)
        offsets = offsets_flattened[:, :3]  # Shape: (N_active * P, 3)

        # Compute voxel indices
        voxel_indices = torch.floor(offsets / self.size_scheduler.value).long()  # Shape: (N_active * P, 3)

        # Stack batch_group_ids and voxel_indices to create cluster keys
        cluster_keys = torch.cat([batch_group_ids, voxel_indices], dim=1)  # Shape: (N_active * P, 5)

        # Get unique clusters and cluster indices
        unique_keys, cluster_indices = torch.unique(cluster_keys, dim=0, return_inverse=True)

        return unique_keys, cluster_indices

class PredictionAggregator(nn.Module):
    def __init__(self, size_scheduler):
        super(PredictionAggregator, self).__init__()
        self.voxel_clustering = VoxelClustering(size_scheduler)

    def forward(self, pred, active_mask):
        B, G, P, F = pred.shape
        device = pred.device
        unique_keys, cluster_indices = self.voxel_clustering(pred, active_mask)

        pred_flat = pred[active_mask].reshape(-1, F)  # Shape: (N_active * P, F)
        confidences_logits = pred_flat[:, 3]  # Shape: (N_active * P,)
        logits = pred_flat[:, 4:]  # Shape: (N_active * P, num_classes)

        # Apply sigmoid to confidences
        confidences = torch.sigmoid(confidences_logits)  # Shape: (N_active * P,)

        # Aggregate using scatter operations
        max_weighted_logits, _ = scatter_max(logits, cluster_indices, dim=0)
        max_confidences, _ = scatter_max(confidences, cluster_indices, dim=0)
        centroids = scatter_mean(
            pred_flat[:, :3] * confidences.unsqueeze(1), cluster_indices, dim=0
        )

        agg_pred = torch.cat(
            [centroids, max_confidences.unsqueeze(1), max_weighted_logits], dim=1
        )  # Shape: (N_clusters, F_agg)

        batch_ids, group_ids = unique_keys[:, 0], unique_keys[:, 1]

        # Constants
        num_features = agg_pred.shape[1]

        # Compute combined group indices
        group_indices = batch_ids * G + group_ids  # Shape: (N_clusters,)

        # Sort group_indices and agg_pred accordingly
        sorted_indices = group_indices.argsort()
        agg_pred_sorted = agg_pred[sorted_indices]
        group_indices_sorted = group_indices[sorted_indices]
        batch_ids_sorted = batch_ids[sorted_indices]
        group_ids_sorted = group_ids[sorted_indices]

        # Compute counts per group
        unique_group_indices, counts = torch.unique_consecutive(
            group_indices_sorted, return_counts=True
        )

        N_max = counts.max().item()  # Maximum number of predictions per group

        # Compute positions within each group
        cumsum_counts = torch.cumsum(torch.cat([torch.tensor([0], device=device), counts[:-1]]), dim=0)
        group_positions = torch.arange(len(agg_pred_sorted), device=device) - torch.repeat_interleave(
            cumsum_counts, counts
        )

        # Create padded tensor for aggregated predictions
        agg_pred_padded = torch.zeros((B, G, N_max, num_features), device=device)

        # Compute flat indices for assignment
        flat_indices = batch_ids_sorted * (G * N_max) + group_ids_sorted * N_max + group_positions

        # Flatten the padded tensor for assignment
        agg_pred_padded_flat = agg_pred_padded.view(-1, num_features)

        # Assign aggregated predictions to the appropriate positions
        agg_pred_padded_flat[flat_indices.long()] = agg_pred_sorted

        return agg_pred_padded  # Shape: (B, G, N_max, num_features)


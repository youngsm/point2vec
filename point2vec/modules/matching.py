import torch
import torch.nn as nn
from torch_linear_assignment import batch_linear_assignment

class HungarianMatcher(nn.Module):
    def __init__(self, p=1):
        super(HungarianMatcher, self).__init__()
        self.p = p

    def forward(self, pred, target, pred_lens, target_lens):
        # Compute the cost matrices
        cost_matrices = torch.cdist(pred, target, p=self.p)  # Shape: (B, N_pred, N_target)

        # Mask the cost matrices based on the valid lengths
        max_pred_len = pred.size(1)
        max_target_len = target.size(1)
        batch_size = pred.size(0)
        device = pred.device

        # Create masks for predictions and targets
        pred_mask = torch.arange(max_pred_len, device=device).unsqueeze(0) < pred_lens.unsqueeze(1)
        target_mask = torch.arange(max_target_len, device=device).unsqueeze(0) < target_lens.unsqueeze(1)

        # Expand masks to match cost matrices
        pred_mask_expanded = pred_mask.unsqueeze(-1).expand(-1, -1, max_target_len)
        target_mask_expanded = target_mask.unsqueeze(1).expand(-1, max_pred_len, -1)
        valid_mask = pred_mask_expanded & target_mask_expanded

        # Set invalid entries in the cost matrices to a high value
        cost_matrices = cost_matrices.masked_fill(~valid_mask, 1e32)

        # Run batch linear assignment
        transposed = max_target_len < max_pred_len
        assignments = batch_linear_assignment(
            cost_matrices if not transposed else cost_matrices.transpose(1, 2)
        )

        # Set invalid assignments to -1
        assignments[~(pred_mask if not transposed else target_mask)] = -1

        # Convert assignments to indices
        valid_batch_indices, valid_pred_indices, valid_target_indices = self.assignment_to_indices(assignments)
        if transposed:
            valid_pred_indices, valid_target_indices = valid_target_indices, valid_pred_indices

        return valid_batch_indices, valid_pred_indices, valid_target_indices

    @staticmethod
    def assignment_to_indices(assignment):
        """
        Convert assignment tensor to pred and target indices in a vectorized manner.
        """
        B, W = assignment.shape
        device = assignment.device

        # Create batch indices and prediction indices
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, W)
        pred_indices = torch.arange(W, device=device).unsqueeze(0).expand(B, W)

        # Flatten the tensors to 1D
        flat_batch_indices = batch_indices.flatten()
        flat_pred_indices = pred_indices.flatten()
        flat_target_indices = assignment.flatten()

        # Create a mask for valid assignments (target indices >= 0)
        mask = flat_target_indices >= 0

        # Apply the mask to filter out invalid assignments
        valid_batch_indices = flat_batch_indices[mask]
        valid_pred_indices = flat_pred_indices[mask]
        valid_target_indices = flat_target_indices[mask]

        return valid_batch_indices, valid_pred_indices, valid_target_indices

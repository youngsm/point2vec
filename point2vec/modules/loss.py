import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
from typing import Optional
from torch import Tensor
from point2vec.modules.scheduler import build_scheduler
from point2vec.modules.aggregate import PredictionAggregator
from point2vec.modules.matching import HungarianMatcher

class SoftmaxFocalLoss(_WeightedLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        gamma: float = 2,
        ignore_index: int = -1,
    ):
        super().__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        flattened_logits = logits.reshape(-1, logits.shape[-1])
        flattened_labels = labels.view(-1)

        p_t = flattened_logits.softmax(dim=-1)
        ce_loss = F.cross_entropy(
            flattened_logits,
            flattened_labels,
            reduction="none",
            ignore_index=self.ignore_index,
        )  # -log(p_t)

        alpha_t = self.weight
        # alpha_t = labels.ne(-1).sum() / labels[labels.ne(-1)].bincount()
        loss = (
            alpha_t[flattened_labels]
            * ((1 - p_t[torch.arange(p_t.shape[0]), flattened_labels]) ** self.gamma)
            * ce_loss
        )

        if self.reduction == "mean":
            loss = loss.sum() / labels.ne(self.ignore_index).sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
        return loss
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Computes the Dice Loss for multi-class segmentation or classification tasks.

        Args:
            inputs: Tensor of shape (N, C, ...) where C = number of classes.
                    This tensor contains raw scores (logits) for each class.
            targets: Tensor of shape (N, ...) with integer values in the range [0, C-1].
                     This tensor contains the ground truth class indices.

        Returns:
            dice_loss: Scalar tensor representing the Dice Loss.
        """
        # Apply softmax to get class probabilities
        inputs = F.softmax(inputs, dim=1)  # Shape: (N, C, ...)

        # Ensure targets have the same spatial dimensions as inputs
        if inputs.dim() > 2:
            targets = targets.unsqueeze(1)  # Shape: (N, 1, ...)
            targets = targets.expand(
                -1, inputs.size(2), *([-1] * (inputs.dim() - 2))
            )  # Shape: (N, C, ...)

        # Flatten inputs and targets to compute per-pixel/per-element Dice coefficients
        inputs = inputs.contiguous().view(
            inputs.size(0), inputs.size(1), -1
        )  # Shape: (N, C, D)
        targets = targets.contiguous().view(targets.size(0), -1)  # Shape: (N, D)

        # Convert targets to one-hot encoding
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets.clamp(0), num_classes)  # Shape: (N, D, C)
        targets_one_hot = targets_one_hot.permute(0, 2, 1).float()  # Shape: (N, C, D)

        # Handle ignore_index if specified
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index  # Shape: (N, D)
            valid_mask = valid_mask.unsqueeze(1).expand_as(inputs)  # Shape: (N, C, D)
            inputs = inputs * valid_mask
            targets_one_hot = targets_one_hot * valid_mask

        # Calculate Dice coefficient
        intersection = (inputs * targets_one_hot).sum(2)  # Shape: (N, C)
        inputs_sum = inputs.sum(2)  # Shape: (N, C)
        targets_sum = targets_one_hot.sum(2)  # Shape: (N, C)

        dice_coeff = (2.0 * intersection + self.smooth) / (
            inputs_sum + targets_sum + self.smooth
        )  # Shape: (N, C)
        dice_loss = 1 - dice_coeff  # Shape: (N, C)

        # Mean over classes and batch
        dice_loss = dice_loss.mean()

        return dice_loss


class AggregateLoss(nn.Module):
    def __init__(self, scheduler_type, **scheduler_kwargs):
        super(AggregateLoss, self).__init__()
        self.size_scheduler = build_scheduler(scheduler_type, **scheduler_kwargs)
        self.aggregator = PredictionAggregator(self.size_scheduler)
        self.matcher = HungarianMatcher()

    def forward(self, pred, target, voxel_size=None):
        """
        pred: (B, G, P, 50, 13): (batch, group, point, proposal, features) features: (x,y,z,conf,9 logits)
        target: (B, G, P, 10, 4): (batch, group, point, target, (x,y,z,cls))
        """
        if voxel_size is not None:
            self.aggregator.voxel_size = voxel_size
        assert pred.shape[:2] == target.shape[:2], 'batch and group dimensions must match'
        active_mask = target[..., -1].gt(1e-8).any(dim=-1)  # Shape: (B, G) active groups
        agg_pred = self.aggregator(pred, active_mask)
        l1, ce_active = self._compute_active_loss(agg_pred, target, active_mask)
        ce_inactive = self._compute_inactive_loss(pred, ~active_mask)
        return {'l1': l1, 'ce_active': ce_active, 'ce_inactive': ce_inactive}

    def _compute_active_loss(self, pred: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        pred = pred[active_mask]
        target = target[active_mask]
        assert pred.shape[0] == target.shape[0], 'batch dimensions must match'

        # split pred and target
        offset_pred, conf_pred, logit_pred = torch.split(pred, [3, 1, 9], dim=-1)
        offset_pred_lens = offset_pred.ne(0.0).any(-1).sum(-1)
        offset_target, cls_target = torch.split(target, [3, 1], dim=-1)
        offset_target_lens = cls_target.ne(0.0).any(-1).sum(-1)

        # hungarian match offset pred and target based on l1 distances
        valid_batch_indices, valid_pred_indices, valid_target_indices = self.matcher(
            offset_pred, offset_target, offset_pred_lens, offset_target_lens
        )

        # Gather matched predictions and targets
        matched_pred_offsets = offset_pred[valid_batch_indices, valid_pred_indices]
        matched_pred_logits = logit_pred[valid_batch_indices, valid_pred_indices]
        matched_target_offsets = offset_target[valid_batch_indices, valid_target_indices]
        matched_target_classes = cls_target[valid_batch_indices, valid_target_indices]

        # compute l1 loss and ce loss
        regression_loss = F.l1_loss(matched_pred_offsets, matched_target_offsets)
        classification_loss = F.cross_entropy(matched_pred_logits, matched_target_classes.squeeze(-1).long())
        return regression_loss, classification_loss

    def _compute_inactive_loss(self, pred: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        # """compute just classification loss for all argmax(probs)=0"""
        pred = pred[~active_mask].reshape(-1, pred.shape[-1])
        logit_pred = pred[:, 4:]
        classification_loss = F.cross_entropy(logit_pred, torch.zeros(logit_pred.shape[0], dtype=torch.long, device=logit_pred.device))
        return classification_loss
    
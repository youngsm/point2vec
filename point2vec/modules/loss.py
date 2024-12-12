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
    def __init__(
        self,
        ce_weights,
        bce_weights,
        scheduler_type,
        **scheduler_kwargs
    ):
        super(AggregateLoss, self).__init__()

        # register weights
        self.ce_weights = ce_weights
        self.bce_weights = bce_weights

        # build scheduler
        self.size_scheduler = build_scheduler(scheduler_type, **scheduler_kwargs)

        # build aggregator and matcher
        self.aggregator = PredictionAggregator(self.size_scheduler)
        self.matcher = HungarianMatcher()

    def to(self, device):
        self.ce_weights = self.ce_weights.to(device)
        self.bce_weights = self.bce_weights.to(device)
        return super().to(device)

    def cuda(self):
        self.ce_weights = self.ce_weights.cuda()
        self.bce_weights = self.bce_weights.cuda()
        return super().cuda()

    def forward(self, pred, target, voxel_size=None):
        """
        pred: (B, G, P, F): (batch, group, point, features)
        target: (B, G, T, 4): (batch, group, target_points, (x,y,z,cls))
        """
        if voxel_size is not None:
            self.aggregator.voxel_size = voxel_size
        assert pred.shape[:2] == target.shape[:2], 'batch and group dimensions must match'
        if self.ce_weights.device != pred.device:
            self.ce_weights = self.ce_weights.to(pred.device)
            self.bce_weights = self.bce_weights.to(pred.device)

        B, G = pred.shape[:2]

        # Determine active and inactive groups
        # Active groups have at least one target keypoint
        active_mask = target[..., -1].gt(0).any(dim=-1)  # Shape: (B, G)

        # Aggregate predictions
        agg_pred = self.aggregator(pred, active_mask)

        # Compute losses
        agg_pred_active = agg_pred[active_mask]
        target_active = target[active_mask]
        l1_loss, ce_loss, bce_active_loss = self._compute_active_loss(agg_pred_active, target_active)

        agg_pred_inactive = agg_pred[~active_mask]
        bce_inactive_loss = self._compute_inactive_loss(agg_pred_inactive)

        total_loss = {
            'l1_loss': l1_loss,
            'ce_loss': ce_loss,
            'bce_active_loss': bce_active_loss,
            'bce_inactive_loss': bce_inactive_loss
        }

        return total_loss

    def _compute_active_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute losses for active groups (groups with keypoints):
        - L1 Loss for matched predictions
        - Cross-Entropy Loss for matched predictions
        - Binary Cross-Entropy Loss for confidences (both matched and unmatched)
        """
        # Select active groups
        N_active_groups = pred.shape[0]
        device = pred.device

        # Flatten predictions and targets
        N_max = pred.shape[1]
        pred = pred.reshape(-1, pred.shape[-1])  # Shape: (N_active_groups * N_max, F)
        conf_pred = pred[:, 3]  # Confidence logits
        offset_pred = pred[:, :3]  # Positions
        logit_pred = pred[:, 4:]  # Class logits

        # Prepare target confidences
        # First, we need to identify matched and unmatched predictions

        # Reshape targets
        T = target.shape[1]
        offset_target = target[:, :, :3]  # Positions
        cls_target = target[:, :, 3]  # Class labels

        # Flatten targets
        offset_target = offset_target.reshape(-1, 3)  # Shape: (N_active_groups * T, 3)
        cls_target = cls_target.reshape(-1)  # Shape: (N_active_groups * T,)

        # Prepare lengths for matching
        offset_pred_lens = torch.full((N_active_groups,), N_max, dtype=torch.long, device=device)
        offset_target_lens = torch.full((N_active_groups,), T, dtype=torch.long, device=device)

        # Reshape predictions for matching
        offset_pred_reshaped = pred[:, :3].reshape(N_active_groups, N_max, 3)
        offset_target_reshaped = target[:, :, :3]  # Shape: (N_active_groups, T, 3)

        # Perform matching
        valid_batch_indices, valid_pred_indices, valid_target_indices = self.matcher(
            offset_pred_reshaped, offset_target_reshaped, offset_pred_lens, offset_target_lens
        )

        # Create a mask for matched and unmatched predictions
        total_preds = N_active_groups * N_max
        matched_pred_mask = torch.zeros(total_preds, dtype=torch.bool, device=device)
        matched_pred_indices_flat = valid_batch_indices * N_max + valid_pred_indices
        matched_pred_mask[matched_pred_indices_flat] = True

        # Target confidences for all predictions
        target_confidences = torch.zeros(total_preds, device=device)
        target_confidences[matched_pred_indices_flat] = 1.0  # Set target confidence to 1 for matched predictions

        # Compute BCE Loss on confidences for all predictions
        pos_weight = self.bce_weights[1] / self.bce_weights[0] # equiv to freq(0)/freq(1)
        bce_loss = F.binary_cross_entropy_with_logits(conf_pred, target_confidences, pos_weight=pos_weight)
        # bce_loss = focal_loss(conf_pred, target_confidences, pos_weight=pos_weight)

        # Compute L1 and CE Losses for matched predictions
        if matched_pred_indices_flat.numel() > 0:
            matched_pred_offsets = offset_pred[matched_pred_mask]
            matched_pred_logits = logit_pred[matched_pred_mask]
            matched_target_offsets = offset_target[valid_batch_indices * T + valid_target_indices]
            matched_target_classes = cls_target[valid_batch_indices * T + valid_target_indices].long()

            # Compute Smooth L1 Loss
            l1_loss = F.smooth_l1_loss(matched_pred_offsets, matched_target_offsets)

            # Compute Cross-Entropy Loss
            ce_loss = F.cross_entropy(matched_pred_logits, matched_target_classes, weight=self.ce_weights)
        else:
            raise ValueError('No matched predictions!')
            l1_loss = torch.tensor(0.0, device=device)
            ce_loss = torch.tensor(0.0, device=device)

        return l1_loss, ce_loss, bce_loss

    def _compute_inactive_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute BCE Loss on confidences for inactive groups (groups without keypoints)
        """
        if pred.numel() == 0:
            # No inactive groups
            return torch.tensor(0.0, device=pred.device)

        # Flatten predictions
        pred = pred.reshape(-1, pred.shape[-1])  # Shape: (N_inactive_groups * N_max, F)
        conf_pred = pred[:, 3]  # Confidence logits

        # Target confidences are all zeros
        target_confidences = torch.zeros_like(conf_pred)

        # Compute BCE Loss
        pos_weight = self.bce_weights[1] / self.bce_weights[0]  # equiv to freq(0)/freq(1)
        bce_loss = F.binary_cross_entropy_with_logits(conf_pred, target_confidences, pos_weight=pos_weight)
        # bce_loss = focal_loss(conf_pred, target_confidences, pos_weight=pos_weight)

        return bce_loss

def focal_loss(inputs, targets, pos_weight=None, alpha=0.25, gamma=2):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=pos_weight)
    pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean()

# https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, embeddings, eps=1e-8):
        """
        Args:
            embeddings (BxD): backbone output of encoder
        """
        with torch.amp.autocast('cuda', enabled=False):
            embeddings = F.normalize(embeddings, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(embeddings)  # noqa: E741
            distances = self.pdist(embeddings, embeddings[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss

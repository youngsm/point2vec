import torch
import torch.nn as nn
from pytorch3d.ops import knn_points, sample_farthest_points
from torch import nn


class PointcloudMasking(nn.Module):
    def __init__(self, ratio: float, type: str):
        super().__init__()
        self.ratio = ratio

        if type == "rand":
            self.forward = self._mask_center_rand
        elif type == "block":
            self.forward = self._mask_center_block
        elif type == "fps":
            self.forward = self._mask_center_fps
        else:
            raise ValueError(f"No such masking type: {type}")

    def _mask_center_fps(self, centers: torch.Tensor) -> torch.Tensor:
        # centers: (B, G, 3)
        if self.ratio == 0:
            return torch.zeros(centers.shape[:2]).bool()

        B, G, D = centers.shape
        assert D == 3
        num_mask = int(self.ratio * G)


        _, idx = sample_farthest_points(centers, torch.full((B,), G, dtype=torch.int64), K=num_mask, random_start_point=True)

        mask = torch.zeros(B, G, device=centers.device)
        mask.scatter_(dim=1, index=idx, value=1.0)
        mask = mask.to(torch.bool)
        return mask

    def _mask_center_rand(self, centers: torch.Tensor) -> torch.Tensor:
        # centers: (B, G, 3)
        if self.ratio == 0:
            return torch.zeros(centers.shape[:2]).bool()

        B, G, _ = centers.shape

        num_mask = int(self.ratio * G)

        mask = (
            torch.cat(
                [
                    torch.zeros(G - num_mask, device=centers.device),
                    torch.ones(num_mask, device=centers.device),
                ]
            )
            .to(torch.bool)
            .unsqueeze(0)
            .expand(B, -1)
        )  # (B, G)
        # TODO: profile if this loop is slow
        for i in range(B):
            mask[i, :] = mask[i, torch.randperm(mask.shape[1])]

        return mask  # (B, G)

    def _mask_center_block(self, centers: torch.Tensor) -> torch.Tensor:
        # centers: (B, G, 3)
        if self.ratio == 0:
            return torch.zeros(centers.shape[:2]).bool()

        B, G, D = centers.shape
        assert D == 3

        num_mask = int(self.ratio * G)

        # random center
        center = torch.empty((B, 1, D), device=centers.device)
        for i in range(B):
            center[i, 0, :] = centers[i, torch.randint(0, G, (1,)), :]

        # center's nearest neighbors
        _, knn_idx, _ = knn_points(
            center.float(), centers.float(), K=num_mask, return_sorted=False
        )  # (B, 1, K)
        knn_idx = knn_idx.squeeze(1)  # (B, K)

        mask = torch.zeros([B, G], device=centers.device)
        mask.scatter_(dim=1, index=knn_idx, value=1.0)
        mask = mask.to(torch.bool)
        return mask

class VariablePointcloudMasking(nn.Module):
    def __init__(self, ratio: float, type: str):
        super().__init__()
        self.ratio = ratio

        if type == "rand":
            self.forward = self._mask_center_rand
        elif type == "block":
            raise NotImplementedError('Block masking is not implemented for variable group masking')
            self.forward = self._mask_center_block
        elif type == "fps":
            raise NotImplementedError('FPS masking is not implemented for variable group masking')
            self.forward = self._mask_center_fps
        else:
            raise ValueError(f"No such masking type: {type}")

    def _mask_center_fps(self, centers: torch.Tensor) -> torch.Tensor:
        # centers: (B, G, 3)
        if self.ratio == 0:
            return torch.zeros(centers.shape[:2]).bool()

        B, G, D = centers.shape
        assert D == 3
        num_mask = int(self.ratio * G)

        _, idx = sample_farthest_points(
            centers,
            torch.full((B,), G, dtype=torch.int64),
            K=num_mask,
            random_start_point=True,
        )

        mask = torch.zeros(B, G, device=centers.device)
        mask.scatter_(dim=1, index=idx, value=1.0)
        mask = mask.to(torch.bool)
        return mask

    # def _mask_center_rand(
    #     self, centers: torch.Tensor, lengths: torch.Tensor
    # ) -> torch.Tensor:
    #     # centers: (B, G, 3)
    #     if self.ratio == 0:
    #         return torch.zeros(centers.shape[:2]).bool()

    #     B, G, _ = centers.shape

    #     masked = torch.zeros(B, G, device=centers.device, dtype=torch.bool)
    #     not_masked = torch.zeros(B, G, device=centers.device, dtype=torch.bool)

    #     for i in range(B):
    #         num_mask = (self.ratio * lengths[i]).int()
    #         perm = torch.randperm(lengths[i], device=centers.device)
    #         masked[i, perm[:num_mask]] = True
    #         not_masked[i, perm[num_mask:]] = True

    #     return masked, not_masked  # (B, G)

    def _mask_center_rand(
        self, centers: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        # centers: (B, G, 3)
        if self.ratio == 0:
            return torch.zeros(centers.shape[:2], device=centers.device, dtype=torch.bool)

        B, G, _ = centers.shape
        device = centers.device

        # Generate random scores
        random_scores = torch.rand(B, G, device=device)

        # Create a mask for valid positions (positions within lengths)
        valid_positions_mask = torch.arange(G, device=device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)  # Shape: (B, G)

        # Set random_scores for invalid positions to infinity so they are sorted to the end
        random_scores[~valid_positions_mask] = float("inf")

        # Sort the random scores to simulate random permutations
        sorted_scores, sorted_indices = torch.sort(random_scores, dim=1)

        # Compute the number of tokens to mask per batch
        num_mask = (self.ratio * lengths).int()  # Shape: (B,)

        # Create indices for batch and sequence positions
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, G)
        seq_indices = torch.arange(G, device=device).unsqueeze(0).expand(B, G)

        # Create a mask indicating which positions should be masked
        mask = seq_indices < num_mask.unsqueeze(1)
        mask = mask & valid_positions_mask  # Ensure we only mask valid positions

        # Initialize masked and not_masked tensors
        masked = torch.zeros(B, G, device=device, dtype=torch.bool)
        not_masked = torch.zeros(B, G, device=device, dtype=torch.bool)

        # Assign masked and not_masked positions using advanced indexing
        masked[batch_indices, sorted_indices] = mask
        not_masked[batch_indices, sorted_indices] = (~mask) & valid_positions_mask

        return masked, not_masked  # (B, G)

    def _mask_center_block(self, centers: torch.Tensor) -> torch.Tensor:
        # centers: (B, G, 3)
        if self.ratio == 0:
            return torch.zeros(centers.shape[:2]).bool()

        B, G, D = centers.shape
        assert D == 3

        num_mask = int(self.ratio * G)

        # random center
        center = torch.empty((B, 1, D), device=centers.device)
        for i in range(B):
            center[i, 0, :] = centers[i, torch.randint(0, G, (1,)), :]

        # center's nearest neighbors
        _, knn_idx, _ = knn_points(
            center.float(), centers.float(), K=num_mask, return_sorted=False
        )  # (B, 1, K)
        knn_idx = knn_idx.squeeze(1)  # (B, K)

        mask = torch.zeros([B, G], device=centers.device)
        mask.scatter_(dim=1, index=knn_idx, value=1.0)
        mask = mask.to(torch.bool)
        return mask

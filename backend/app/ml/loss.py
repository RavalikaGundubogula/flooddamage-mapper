from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary semantic segmentation.
    Penalizes disparity in spatial overlap, robust to extreme background/foreground class imbalance.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: Predicted probabilities in [0, 1], shape (B, 1, H, W) or (B, H, W)
            targets: Binary ground truth labels {0, 1}, matching shape
        """
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1).float()

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class CombinedSegmentationLoss(nn.Module):
    """
    Joint loss function combining Binary Cross Entropy (BCE) with logits and Soft Dice Loss.
    Ensures both smooth calibrated pixel probabilities and optimal boundary overlap.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized network output logits, shape (B, 1, H, W)
            targets: Binary ground truth {0, 1}, shape (B, 1, H, W)
        """
        targets_float = targets.float()
        bce_loss = self.bce(logits, targets_float)
        probs = torch.sigmoid(logits)
        dice_loss = self.dice(probs, targets_float)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

"""Differentiable Haversine distance and loss for geographic coordinates."""

import math
import torch
import torch.nn as nn

EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute great-circle distance between predicted and target coordinates.

    Args:
        pred: Predicted (lat, lon) in degrees, shape (..., 2).
        target: Target (lat, lon) in degrees, shape (..., 2).

    Returns:
        Distance in km, shape (...,).
    """
    lat1 = pred[..., 0] * (math.pi / 180.0)
    lon1 = pred[..., 1] * (math.pi / 180.0)
    lat2 = target[..., 0] * (math.pi / 180.0)
    lon2 = target[..., 1] * (math.pi / 180.0)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        torch.sin(dlat / 2.0) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2
    )
    # Clamp for numerical stability
    a = torch.clamp(a, min=0.0, max=1.0)
    c = 2.0 * torch.asin(torch.sqrt(a))

    return EARTH_RADIUS_KM * c


class HaversineLoss(nn.Module):
    """Differentiable Haversine loss (mean great-circle distance in km)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 2) predicted (lat, lon) in degrees.
            target: (B, 2) ground-truth (lat, lon) in degrees.
        """
        dist = haversine_distance(pred, target)
        if self.reduction == "mean":
            return dist.mean()
        elif self.reduction == "sum":
            return dist.sum()
        return dist

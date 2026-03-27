"""Geographic evaluation metrics for geolocation models."""

import torch
from .haversine import haversine_distance


class GeoMetrics:
    """Accumulates and computes geographic evaluation metrics.

    Tracks:
        - Mean/median haversine distance (km)
        - Accuracy at distance thresholds (1km, 25km, 200km, 750km, 2500km)
        - Country-level and region-level accuracy (if labels provided)
    """

    THRESHOLDS_KM = [1, 25, 200, 750, 2500]

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all accumulated predictions."""
        self._distances: list[torch.Tensor] = []
        self._count = 0

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        """Add a batch of predictions.

        Args:
            pred: Predicted (lat, lon), shape (B, 2).
            target: Ground-truth (lat, lon), shape (B, 2).
        """
        dist = haversine_distance(pred.detach().cpu(), target.detach().cpu())
        self._distances.append(dist)
        self._count += dist.shape[0]

    def compute(self) -> dict[str, float]:
        """Compute all metrics from accumulated predictions.

        Returns:
            Dictionary with metric names and values.
        """
        if self._count == 0:
            return {"mean_km": 0.0, "median_km": 0.0}

        all_dist = torch.cat(self._distances, dim=0)
        results = {
            "mean_km": all_dist.mean().item(),
            "median_km": all_dist.median().item(),
            "count": self._count,
        }

        # Accuracy at thresholds
        for thresh in self.THRESHOLDS_KM:
            acc = (all_dist <= thresh).float().mean().item() * 100.0
            results[f"acc@{thresh}km"] = acc

        return results

    def __repr__(self) -> str:
        metrics = self.compute()
        parts = [f"GeoMetrics(n={metrics.get('count', 0)})"]
        if metrics.get("count", 0) > 0:
            parts.append(f"  mean={metrics['mean_km']:.1f}km")
            parts.append(f"  median={metrics['median_km']:.1f}km")
            for thresh in self.THRESHOLDS_KM:
                parts.append(f"  acc@{thresh}km={metrics[f'acc@{thresh}km']:.1f}%")
        return "\n".join(parts)

"""Multi-task loss for GeoSaccade with 6 components."""

import torch
import torch.nn as nn

from ..utils.haversine import HaversineLoss


class GeoSaccadeLoss(nn.Module):
    """Combined loss for GeoSaccade training.

    Components:
        1. L_haversine: Great-circle distance loss on final prediction.
        2. L_progressive: Haversine loss on intermediate steps (weighted by step).
        3. L_entropy: Attention entropy regularization (encourage focused attention).
        4. L_diversity: Encourage diverse attention across steps (KL between steps).
        5. L_ior: Inhibition-of-return reward (penalize attention overlap).
        6. L_level: Geographic hierarchy level routing regularization.

    Args:
        w_haversine: Weight for haversine loss (default: 1.0).
        w_progressive: Weight for progressive loss (default: 0.5).
        w_entropy: Weight for entropy regularization (default: 0.01).
        w_diversity: Weight for diversity loss (default: 0.1).
        w_ior: Weight for IOR loss (default: 0.05).
        w_level: Weight for level routing loss (default: 0.01).
    """

    def __init__(
        self,
        w_haversine: float = 1.0,
        w_progressive: float = 0.5,
        w_entropy: float = 0.01,
        w_diversity: float = 0.1,
        w_ior: float = 0.05,
        w_level: float = 0.01,
    ):
        super().__init__()
        self.w_haversine = w_haversine
        self.w_progressive = w_progressive
        self.w_entropy = w_entropy
        self.w_diversity = w_diversity
        self.w_ior = w_ior
        self.w_level = w_level

        self.haversine = HaversineLoss(reduction="mean")

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model output dict with keys:
                - 'pred_coords': (B, 2) final prediction.
                - 'step_coords': (B, T, 2) per-step predictions.
                - 'step_attentions': (B, T, N) per-step attention weights.
                - 'level_weights': (B, T, L) per-step routing weights.
            targets: Ground-truth (lat, lon), shape (B, 2).

        Returns:
            Dictionary with 'total' and individual component losses.
        """
        pred_coords = outputs["pred_coords"]
        step_coords = outputs["step_coords"]
        step_attn = outputs["step_attentions"]
        level_weights = outputs["level_weights"]

        B, T, N = step_attn.shape
        losses = {}

        # 1. Haversine loss on final prediction
        losses["haversine"] = self.haversine(pred_coords, targets)

        # 2. Progressive loss: weighted haversine across steps
        prog_loss = torch.tensor(0.0, device=pred_coords.device)
        for t in range(T):
            weight = (t + 1) / T  # Later steps weighted more
            prog_loss = prog_loss + weight * self.haversine(
                step_coords[:, t, :], targets
            )
        losses["progressive"] = prog_loss / T

        # 3. Entropy regularization (encourage low entropy = focused attention)
        eps = 1e-8
        entropy = -(step_attn * (step_attn + eps).log()).sum(dim=-1)  # (B, T)
        losses["entropy"] = entropy.mean()

        # 4. Diversity loss: encourage different attention patterns across steps
        if T > 1:
            div_loss = torch.tensor(0.0, device=pred_coords.device)
            for t1 in range(T):
                for t2 in range(t1 + 1, T):
                    # Cosine similarity between attention distributions
                    sim = torch.nn.functional.cosine_similarity(
                        step_attn[:, t1, :], step_attn[:, t2, :], dim=-1
                    )
                    div_loss = div_loss + sim.mean()
            n_pairs = T * (T - 1) / 2
            losses["diversity"] = div_loss / n_pairs
        else:
            losses["diversity"] = torch.tensor(0.0, device=pred_coords.device)

        # 5. IOR loss: penalize revisiting same patches
        cumulative = torch.zeros(B, N, device=pred_coords.device)
        ior_penalty = torch.tensor(0.0, device=pred_coords.device)
        for t in range(T):
            overlap = (cumulative * step_attn[:, t, :]).sum(dim=-1).mean()
            ior_penalty = ior_penalty + overlap
            cumulative = cumulative + step_attn[:, t, :]
        losses["ior"] = ior_penalty / T

        # 6. Level routing regularization (encourage non-uniform routing)
        level_entropy = -(
            level_weights * (level_weights + eps).log()
        ).sum(dim=-1).mean()
        losses["level"] = level_entropy

        # Total weighted loss
        losses["total"] = (
            self.w_haversine * losses["haversine"]
            + self.w_progressive * losses["progressive"]
            + self.w_entropy * losses["entropy"]
            + self.w_diversity * losses["diversity"]
            + self.w_ior * losses["ior"]
            + self.w_level * losses["level"]
        )

        return losses

"""Glimpse Extractor: multi-resolution feature aggregation per saccade step."""

import torch
import torch.nn as nn


class GlimpseExtractor(nn.Module):
    """Extracts a multi-resolution glimpse from patch features using attention.

    Three components per glimpse:
        1. Focus: weighted sum of top-k attended patches (fine detail)
        2. Broad: weighted sum of all patches (global context)
        3. Peripheral: mean of non-top-k patches (background)

    Each step t has its own value projection W_v^(t).

    Args:
        D: Input patch feature dimension (default: 1024).
        D_v: Value/output dimension per component (default: 512).
        T: Number of saccade steps (default: 5).
        top_k: Number of top patches for focus (default: 16).
    """

    def __init__(
        self,
        D: int = 1024,
        D_v: int = 512,
        T: int = 5,
        top_k: int = 16,
    ):
        super().__init__()
        self.D = D
        self.D_v = D_v
        self.T = T
        self.top_k = top_k

        # Per-step value projections
        self.W_v = nn.ModuleList([nn.Linear(D, D_v) for _ in range(T)])

        # Fusion of 3 components into single glimpse vector
        self.fusion = nn.Sequential(
            nn.Linear(D_v * 3, D_v * 2),
            nn.GELU(),
            nn.Linear(D_v * 2, D),  # Output matches input dim for GRU
        )

    def forward(
        self,
        F_patches: torch.Tensor,
        alpha: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """
        Args:
            F_patches: Patch features, shape (B, N, D).
            alpha: Attention weights, shape (B, N).
            step: Current saccade step index (0-indexed).

        Returns:
            glimpse: Aggregated glimpse vector, shape (B, D).
        """
        B, N, _ = F_patches.shape

        # Value projection for this step
        V = self.W_v[step](F_patches)  # (B, N, D_v)

        # --- Focus: top-k attended patches ---
        topk_vals, topk_idx = torch.topk(alpha, self.top_k, dim=-1)  # (B, top_k)
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # Gather top-k value vectors
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, self.D_v)
        V_topk = torch.gather(V, 1, topk_idx_exp)  # (B, top_k, D_v)
        focus = (V_topk * topk_weights.unsqueeze(-1)).sum(dim=1)  # (B, D_v)

        # --- Broad: attention-weighted sum of all patches ---
        broad = (V * alpha.unsqueeze(-1)).sum(dim=1)  # (B, D_v)

        # --- Peripheral: mean of non-top-k patches ---
        mask = torch.ones(B, N, device=alpha.device)
        mask.scatter_(1, topk_idx, 0.0)
        n_periph = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        peripheral = (V * mask.unsqueeze(-1)).sum(dim=1) / n_periph  # (B, D_v)

        # Fuse all components
        concat = torch.cat([focus, broad, peripheral], dim=-1)  # (B, 3*D_v)
        glimpse = self.fusion(concat)  # (B, D)

        return glimpse

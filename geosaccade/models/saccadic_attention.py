"""Saccadic Attention module with geographic gating, IOR, and adaptive temperature."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SaccadicAttention(nn.Module):
    """Sequential attention mechanism inspired by human saccadic eye movements.

    Features:
        - Query generation from GRU hidden state
        - Feature gate modulated by geographic context
        - Spatial bias from positional encoding
        - Inhibition of Return (IOR) to discourage revisiting
        - Adaptive temperature for attention sharpness

    Args:
        D_h: Hidden state dimension (default: 1024).
        D_k: Query/key dimension (default: 256).
        D_g: Geographic context dimension (default: 512).
    """

    def __init__(
        self,
        D_h: int = 1024,
        D_k: int = 256,
        D_g: int = 512,
    ):
        super().__init__()
        self.D_h = D_h
        self.D_k = D_k
        self.D_g = D_g

        # Query generation from hidden state
        self.W_q = nn.Linear(D_h, D_k)

        # Key projection from patch features
        self.W_k = nn.Linear(D_h, D_k)

        # Feature gate conditioned on geographic context
        self.feature_gate = nn.Sequential(
            nn.Linear(D_g, D_h),
            nn.Sigmoid(),
        )

        # Spatial bias MLP
        self.spatial_bias = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Adaptive temperature from hidden state
        self.temp_net = nn.Sequential(
            nn.Linear(D_h, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # Ensure positive temperature
        )

        # IOR decay parameter (learnable)
        self.ior_gamma = nn.Parameter(torch.tensor(0.9))

    def forward(
        self,
        h_t: torch.Tensor,
        F_patches: torch.Tensor,
        g_t: torch.Tensor,
        positions: torch.Tensor,
        ior_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: Hidden state, shape (B, D_h).
            F_patches: Patch features, shape (B, N, D_h).
            g_t: Geographic context, shape (B, D_g).
            positions: Normalized 2D patch positions, shape (B, N, 2) or (N, 2).
            ior_map: Accumulated attention from prior steps, shape (B, N) or None.

        Returns:
            alpha: Attention weights, shape (B, N).
            ior_map_updated: Updated IOR map, shape (B, N).
        """
        B, N, _ = F_patches.shape

        # Apply feature gate
        gate = self.feature_gate(g_t).unsqueeze(1)  # (B, 1, D_h)
        F_gated = F_patches * gate  # (B, N, D_h)

        # Query-key attention
        q = self.W_q(h_t).unsqueeze(1)  # (B, 1, D_k)
        k = self.W_k(F_gated)  # (B, N, D_k)
        logits = (q * k).sum(dim=-1) / (self.D_k ** 0.5)  # (B, N)

        # Spatial bias
        if positions.dim() == 2:
            positions = positions.unsqueeze(0).expand(B, -1, -1)
        s_bias = self.spatial_bias(positions).squeeze(-1)  # (B, N)
        logits = logits + s_bias

        # Inhibition of Return
        if ior_map is not None:
            logits = logits - ior_map

        # Adaptive temperature
        tau = self.temp_net(h_t)  # (B, 1)
        tau = tau.squeeze(-1).unsqueeze(-1)  # (B, 1)
        tau = torch.clamp(tau, min=0.1)  # Floor to avoid division issues
        logits = logits / tau

        # Softmax attention
        alpha = F.softmax(logits, dim=-1)  # (B, N)

        # Update IOR map
        if ior_map is None:
            ior_map_updated = alpha.detach()
        else:
            ior_map_updated = self.ior_gamma * ior_map + alpha.detach()

        return alpha, ior_map_updated

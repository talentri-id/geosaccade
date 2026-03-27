"""GeoSaccade: Full model assembling vision encoder, saccadic attention loop, and GNN querier."""

import torch
import torch.nn as nn

from .vision import VisionEncoder
from .geo_gru import GeoGRUCell
from .saccadic_attention import SaccadicAttention
from .glimpse import GlimpseExtractor
from .gnn_querier import GNNQuerier


class GeoSaccade(nn.Module):
    """Sequential attention model with geographic gating for image geolocation.

    Architecture:
        1. VisionEncoder extracts N patch features from an input image.
        2. For T saccade steps:
           a. SaccadicAttention computes attention over patches.
           b. GlimpseExtractor aggregates a multi-resolution glimpse.
           c. GNNQuerier produces a geographic context vector.
           d. GeoGRUCell updates the hidden state.
        3. Final hidden state is decoded into (lat, lon) prediction.

    Args:
        T: Number of saccade steps (default: 5).
        D: Patch feature / hidden state dimension (default: 1024).
        D_k: Attention key dimension (default: 256).
        D_v: Value / GNN node dimension (default: 512).
        D_g: Geographic context dimension (default: 512).
        L: Number of geographic hierarchy levels (default: 3).
        top_k: Number of focus patches per step (default: 16).
        backbone_name: DINOv2 variant (default: 'dinov2_vitb14').
        freeze_backbone: Freeze vision backbone (default: True).
    """

    def __init__(
        self,
        T: int = 5,
        D: int = 1024,
        D_k: int = 256,
        D_v: int = 512,
        D_g: int = 512,
        L: int = 3,
        top_k: int = 16,
        backbone_name: str = "dinov2_vitb14",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.T = T
        self.D = D

        # Sub-modules
        self.vision_encoder = VisionEncoder(
            backbone_name=backbone_name,
            output_dim=D,
            freeze_backbone=freeze_backbone,
        )
        self.geo_gru = GeoGRUCell(
            input_dim=D,
            hidden_dim=D,
            geo_dim=D_g,
        )
        self.saccadic_attention = SaccadicAttention(
            D_h=D,
            D_k=D_k,
            D_g=D_g,
        )
        self.glimpse_extractor = GlimpseExtractor(
            D=D,
            D_v=D_v,
            T=T,
            top_k=top_k,
        )
        self.gnn_querier = GNNQuerier(
            D_v=D_v,
            D_h=D,
            D_g=D_g,
            L=L,
        )

        # Initial hidden state projection (from global average of patches)
        self.h0_proj = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
        )

        # Per-step coordinate prediction heads
        self.coord_heads = nn.ModuleList([
            nn.Linear(D, 2) for _ in range(T)
        ])

    def _constrain_coords(self, raw: torch.Tensor) -> torch.Tensor:
        """Map raw predictions to valid (lat, lon) ranges."""
        lat = 90.0 * torch.tanh(raw[:, 0:1])
        lon = 180.0 * torch.tanh(raw[:, 1:2])
        return torch.cat([lat, lon], dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        graph_nodes: torch.Tensor | None = None,
        level_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Input images, shape (B, 3, H, W).
            graph_nodes: Optional graph node features, shape (B, M, D_v).
            level_masks: Optional level masks, shape (B, L, M).

        Returns:
            Dictionary with:
                - 'pred_coords': Final (lat, lon) prediction, shape (B, 2).
                - 'step_coords': Per-step predictions, shape (B, T, 2).
                - 'step_attentions': Per-step attention maps, shape (B, T, N).
                - 'level_weights': Per-step GNN routing, shape (B, T, L).
        """
        B = images.shape[0]

        # 1. Extract patch features
        F_patches, positions = self.vision_encoder(images)  # (B, N, D), (N, 2)
        N = F_patches.shape[1]

        # 2. Initialize hidden state from global average
        h_t = self.h0_proj(F_patches.mean(dim=1))  # (B, D)

        # Storage for outputs
        step_coords = []
        step_attentions = []
        level_weights_all = []
        ior_map = None

        # 3. Saccade loop
        for t in range(self.T):
            # Get geographic context
            g_t, level_w = self.gnn_querier(h_t, graph_nodes, level_masks)
            level_weights_all.append(level_w)

            # Saccadic attention
            alpha, ior_map = self.saccadic_attention(
                h_t, F_patches, g_t, positions, ior_map,
            )
            step_attentions.append(alpha)

            # Extract glimpse
            x_t = self.glimpse_extractor(F_patches, alpha, t)

            # Update hidden state
            h_t = self.geo_gru(x_t, h_t, g_t)

            # Per-step coordinate prediction
            raw_coords = self.coord_heads[t](h_t)
            coords = self._constrain_coords(raw_coords)
            step_coords.append(coords)

        # Stack outputs
        step_coords = torch.stack(step_coords, dim=1)  # (B, T, 2)
        step_attentions = torch.stack(step_attentions, dim=1)  # (B, T, N)
        level_weights_all = torch.stack(level_weights_all, dim=1)  # (B, T, L)

        return {
            "pred_coords": step_coords[:, -1, :],  # Final step prediction
            "step_coords": step_coords,
            "step_attentions": step_attentions,
            "level_weights": level_weights_all,
        }

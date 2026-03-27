"""GNN Querier: queries a geographic knowledge graph for context vectors."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNQuerier(nn.Module):
    """Queries a hierarchical geographic graph to produce context vectors.

    Uses soft level routing to aggregate information across geographic
    hierarchy levels (e.g., continent -> country -> region -> city).

    Args:
        D_v: Node feature dimension (default: 512).
        D_h: Hidden state dimension for routing (default: 1024).
        D_g: Output geographic context dimension (default: 512).
        L: Number of hierarchy levels (default: 3).
        num_gnn_layers: Number of message-passing layers per level (default: 2).
    """

    def __init__(
        self,
        D_v: int = 512,
        D_h: int = 1024,
        D_g: int = 512,
        L: int = 3,
        num_gnn_layers: int = 2,
    ):
        super().__init__()
        self.D_v = D_v
        self.D_h = D_h
        self.D_g = D_g
        self.L = L

        # Soft level routing from hidden state
        self.level_router = nn.Sequential(
            nn.Linear(D_h, 128),
            nn.ReLU(),
            nn.Linear(128, L),
        )

        # Per-level message passing (simplified as MLPs for portability)
        self.level_encoders = nn.ModuleList()
        for _ in range(L):
            layers = []
            in_dim = D_v
            for _ in range(num_gnn_layers):
                layers.extend([
                    nn.Linear(in_dim, D_v),
                    nn.ReLU(),
                ])
                in_dim = D_v
            self.level_encoders.append(nn.Sequential(*layers))

        # Context aggregation
        self.context_proj = nn.Sequential(
            nn.Linear(D_v, D_g),
            nn.LayerNorm(D_g),
        )

        # Coordinate prediction head (lat, lon from geographic context)
        self.coord_head = nn.Linear(D_g, 2)

    def forward(
        self,
        h_t: torch.Tensor,
        node_features: torch.Tensor | None = None,
        level_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: Hidden state for routing, shape (B, D_h).
            node_features: Graph node features, shape (B, M, D_v) or None.
                If None, uses a learned default context.
            level_masks: Binary masks per level, shape (B, L, M) or None.

        Returns:
            g_t: Geographic context vector, shape (B, D_g).
            level_weights: Soft routing weights, shape (B, L).
        """
        B = h_t.shape[0]

        # Soft level routing
        level_logits = self.level_router(h_t)  # (B, L)
        level_weights = F.softmax(level_logits, dim=-1)  # (B, L)

        if node_features is not None:
            # Process each level
            level_contexts = []
            for l in range(self.L):
                encoded = self.level_encoders[l](node_features)  # (B, M, D_v)

                if level_masks is not None:
                    mask = level_masks[:, l, :].unsqueeze(-1)  # (B, M, 1)
                    encoded = encoded * mask
                    ctx = encoded.sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
                else:
                    ctx = encoded.mean(dim=1)  # (B, D_v)

                level_contexts.append(ctx)

            # Stack and weight
            level_stack = torch.stack(level_contexts, dim=1)  # (B, L, D_v)
            weighted = (level_stack * level_weights.unsqueeze(-1)).sum(dim=1)  # (B, D_v)
        else:
            # No graph available — use projected hidden state as fallback
            weighted = torch.zeros(B, self.D_v, device=h_t.device)

        g_t = self.context_proj(weighted)  # (B, D_g)

        return g_t, level_weights

    def predict_coordinates(self, g_t: torch.Tensor) -> torch.Tensor:
        """Predict (lat, lon) from geographic context.

        Args:
            g_t: Geographic context, shape (B, D_g).

        Returns:
            coords: Predicted (lat, lon) in degrees, shape (B, 2).
        """
        raw = self.coord_head(g_t)  # (B, 2)
        # Constrain to valid ranges: lat in [-90, 90], lon in [-180, 180]
        lat = 90.0 * torch.tanh(raw[:, 0:1])
        lon = 180.0 * torch.tanh(raw[:, 1:2])
        return torch.cat([lat, lon], dim=-1)

"""GeoGRU Cell with geographic context gating (V_z, V_r)."""

import torch
import torch.nn as nn


class GeoGRUCell(nn.Module):
    """GRU cell augmented with geographic context gates.

    Standard GRU gates are modulated by geographic embedding g_t via
    learned projection matrices V_z and V_r, allowing the hidden state
    dynamics to be conditioned on the current geographic hypothesis.

    Args:
        input_dim: Dimension of input features (default: 1024).
        hidden_dim: Dimension of hidden state (default: 1024).
        geo_dim: Dimension of geographic context vector (default: 512).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 1024,
        geo_dim: int = 512,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.geo_dim = geo_dim

        # Standard GRU weights
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Geographic context gates
        self.V_z = nn.Linear(geo_dim, hidden_dim, bias=False)
        self.V_r = nn.Linear(geo_dim, hidden_dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        g_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t: Input features at step t, shape (B, input_dim).
            h_prev: Previous hidden state, shape (B, hidden_dim).
            g_t: Geographic context vector, shape (B, geo_dim).

        Returns:
            h_t: Updated hidden state, shape (B, hidden_dim).
        """
        combined = torch.cat([x_t, h_prev], dim=-1)  # (B, input_dim + hidden_dim)

        # Update gate with geographic modulation
        z_t = torch.sigmoid(self.W_z(combined) + self.V_z(g_t))

        # Reset gate with geographic modulation
        r_t = torch.sigmoid(self.W_r(combined) + self.V_r(g_t))

        # Candidate hidden state
        combined_r = torch.cat([x_t, r_t * h_prev], dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_r))

        # Final hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        return h_t

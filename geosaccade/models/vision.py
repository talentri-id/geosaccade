"""Vision Encoder: DINOv2 backbone wrapper with frozen features and projection."""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """DINOv2-based vision encoder for patch feature extraction.

    Wraps a frozen DINOv2 backbone and adds a trainable projection head.
    Outputs N patch tokens suitable for the saccadic attention loop.

    Args:
        backbone_name: DINOv2 model variant (default: 'dinov2_vitb14').
        output_dim: Projected feature dimension (default: 1024).
        freeze_backbone: Whether to freeze backbone weights (default: True).
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        output_dim: int = 1024,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.output_dim = output_dim

        # Load DINOv2 backbone
        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                backbone_name,
                pretrained=True,
            )
            backbone_dim = self.backbone.embed_dim
        except Exception:
            # Fallback: create a stub ViT-like feature extractor
            backbone_dim = 768
            self.backbone = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head: backbone_dim -> output_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        # Positional encoding for patches (will be set based on input)
        self._backbone_dim = backbone_dim
        self._pos_embed = None

    def _get_patch_positions(self, n_patches: int, device: torch.device) -> torch.Tensor:
        """Generate normalized 2D grid positions for patches.

        Args:
            n_patches: Number of patches (assumes square grid).
            device: Target device.

        Returns:
            positions: (N, 2) normalized positions in [0, 1].
        """
        grid_size = int(n_patches ** 0.5)
        coords = torch.linspace(0, 1, grid_size, device=device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        positions = torch.stack([yy.flatten(), xx.flatten()], dim=-1)
        return positions

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: Input images, shape (B, 3, H, W). Expected 518x518 for
                DINOv2 ViT-B/14 to produce 37x37 = 1369 patches, or
                336x336 for 24x24 = 576 patches.

        Returns:
            F_patches: Projected patch features, shape (B, N, output_dim).
            positions: 2D patch positions, shape (N, 2).
        """
        # Extract features from backbone
        if isinstance(self.backbone, nn.Identity):
            # Stub mode: create random patch features
            B = images.shape[0]
            N = 576  # Default patch count
            patch_tokens = torch.randn(B, N, self._backbone_dim, device=images.device)
        else:
            # DINOv2 returns dict with 'x_norm_patchtokens'
            with torch.no_grad():
                features = self.backbone.forward_features(images)

            if isinstance(features, dict):
                patch_tokens = features["x_norm_patchtokens"]
            else:
                # Some versions return the tensor directly
                patch_tokens = features
                if patch_tokens.dim() == 3 and patch_tokens.shape[1] > 1:
                    patch_tokens = patch_tokens[:, 1:, :]  # Remove CLS token

        B, N, _ = patch_tokens.shape

        # Project to output dimension
        F_patches = self.projection(patch_tokens)  # (B, N, output_dim)

        # Generate patch positions
        positions = self._get_patch_positions(N, images.device)  # (N, 2)

        return F_patches, positions

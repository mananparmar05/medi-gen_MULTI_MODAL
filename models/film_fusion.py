"""
FiLM Fusion Layer — Feature-wise Linear Modulation for conditioning 
visual features with structured metadata.

Mechanism:
    output = γ ⊙ visual_features + β   (channel-wise affine transform)
    
    Where γ (scale) and β (shift) are generated from the metadata embedding
    via small linear layers. This lets the metadata directly modulate which
    visual feature channels are amplified or suppressed.

Example:
    If metadata says "Pneumonia: 1", FiLM learns to amplify channels that
    respond to consolidation/opacity patterns and suppress irrelevant channels.

Initialization:
    γ weights → small, γ bias → 1.0 (identity scale)
    β weights → small, β bias → 0.0 (zero shift)
    This ensures output ≈ input at the start of training.

Inspired by:
    - Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)
    - QAHNet's C1 quality-conditioned feature modulation
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FiLMFusionLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    Metadata embedding → γ (scale) and β (shift) → affine transform on
    visual feature channels.
    
    Args:
        conditioning_dim: Dimension of metadata embedding input (default 256).
        feature_channels: Number of channels in visual features (default 1024).
        use_residual: If True, add residual connection (output += input).
        identity_init: If True, initialize γ→1, β→0 for stability.
    """

    def __init__(
        self,
        conditioning_dim: int = 256,
        feature_channels: int = 1024,
        use_residual: bool = True,
        identity_init: bool = True,
    ):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.use_residual = use_residual
        
        # Generate γ (scale) from metadata embedding
        self.gamma_generator = nn.Linear(conditioning_dim, feature_channels)
        
        # Generate β (shift) from metadata embedding
        self.beta_generator = nn.Linear(conditioning_dim, feature_channels)
        
        # Optional: additional non-linear processing of the conditioning signal
        self.conditioning_transform = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.ReLU(inplace=True),
        )
        
        # Post-FiLM batch norm for stability
        self.post_bn = nn.BatchNorm2d(feature_channels)
        
        # Initialize for identity transform
        if identity_init:
            self._identity_init()
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"FiLMFusionLayer: cond_dim={conditioning_dim}, "
            f"feat_channels={feature_channels}, "
            f"residual={use_residual} ({total_params:,} params)"
        )

    def _identity_init(self) -> None:
        """
        Initialize so that FiLM output ≈ input at the start.
        γ ≈ 1 (scale), β ≈ 0 (shift).
        """
        # Gamma: small weights, bias = 1.0
        nn.init.zeros_(self.gamma_generator.weight)
        nn.init.ones_(self.gamma_generator.bias)
        
        # Beta: small weights, bias = 0.0
        nn.init.zeros_(self.beta_generator.weight)
        nn.init.zeros_(self.beta_generator.bias)
        
        logger.info("FiLM initialized with identity transform (γ=1, β=0)")

    def forward(
        self,
        visual_features: torch.Tensor,
        metadata_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning to visual features.
        
        Args:
            visual_features: [B, C, H, W] spatial feature map 
                             (e.g., [B, 1024, 7, 7] from vision encoder).
            metadata_embedding: [B, conditioning_dim] dense metadata embedding
                                (e.g., [B, 256] from MetadataEmbeddingMLP).
                                
        Returns:
            modulated_features: [B, C, H, W] conditioned visual features.
        """
        # Transform conditioning signal
        cond = self.conditioning_transform(metadata_embedding)  # [B, cond_dim]
        
        # Generate scale and shift
        gamma = self.gamma_generator(cond)  # [B, C]
        beta = self.beta_generator(cond)    # [B, C]
        
        # Reshape for spatial broadcasting: [B, C] → [B, C, 1, 1]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        
        # Apply FiLM: output = γ ⊙ features + β
        modulated = gamma * visual_features + beta  # [B, C, H, W]
        
        # Post-normalization
        modulated = self.post_bn(modulated)
        
        # Residual connection
        if self.use_residual:
            modulated = modulated + visual_features
        
        return modulated

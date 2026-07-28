"""
Metadata Embedding MLP — converts 14-dimensional binary structured findings 
into dense embeddings for downstream conditioning.

Architecture:
    14 → Linear → ReLU → Dropout → 128
    128 → Linear → ReLU → Dropout → 256
    256 → LayerNorm → output
    
The output embedding is used by:
    - FiLM layer (to produce γ and β conditioning vectors)
    - Cross-attention bridge (as additional context)

Design rationale:
    Raw binary labels are sparse and hard for networks to reason over.
    This MLP learns a smooth, continuous representation that captures 
    co-occurrence patterns (e.g., Cardiomegaly + Edema often co-occur).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MetadataEmbeddingMLP(nn.Module):
    """
    Two-layer MLP that embeds 14-dim binary findings into a dense vector.
    
    Args:
        input_dim: Dimension of input label vector (default 14).
        hidden_dim: Hidden layer dimension (default 128).
        output_dim: Output embedding dimension (default 256).
        dropout: Dropout probability (default 0.3).
        label_smoothing: If > 0, smooth binary labels (1 → 1-ε, 0 → ε).
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 128,
        output_dim: int = 256,
        dropout: float = 0.3,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.label_smoothing = label_smoothing
        
        self.mlp = nn.Sequential(
            # Layer 1: 14 → 128
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            # Layer 2: 128 → 256
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        
        # LayerNorm for stable downstream conditioning (FiLM γ/β)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MetadataEmbeddingMLP: {input_dim} → {hidden_dim} → {output_dim} "
            f"({total_params:,} params, dropout={dropout})"
        )

    def _init_weights(self) -> None:
        """Initialize with Kaiming for ReLU activations."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Embed structured findings labels.
        
        Args:
            labels: [B, 14] binary label vector (or soft probabilities).
            
        Returns:
            embedding: [B, 256] dense metadata embedding.
        """
        # Optional label smoothing
        if self.training and self.label_smoothing > 0:
            eps = self.label_smoothing
            labels = labels * (1 - eps) + eps / 2  # smooth 0→ε/2, 1→1-ε/2
        
        embedding = self.mlp(labels)          # [B, 256]
        embedding = self.layer_norm(embedding)  # [B, 256]
        
        return embedding

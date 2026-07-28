"""
Dual-View Vision Encoder — DenseNet-121 backbone with shared weights
for frontal and lateral chest X-ray views.

Architecture:
    - DenseNet-121 pretrained on ImageNet (features only, no classifier)
    - Shared weights across frontal and lateral views (Siamese-style)
    - Dual-view merge: concatenation with projection OR average pooling
    - Optional freezing of early dense blocks to prevent overfitting
    
Output:
    Spatial feature grid [B, C, 7, 7] where C = feature_dim (default 1024)
    Each of the 49 spatial positions encodes a region of the input X-ray.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


class DualViewVisionEncoder(nn.Module):
    """
    DenseNet-121 based dual-view vision encoder for chest X-rays.
    
    Processes frontal and lateral views through a shared DenseNet backbone,
    then merges the feature maps via concatenation (with projection) or
    average pooling.
    
    Args:
        pretrained: Load ImageNet-pretrained weights.
        feature_dim: Output feature dimension per spatial position (default 1024).
        dual_view_mode: 'concat' or 'avg' for merging frontal + lateral features.
        freeze_blocks: List of DenseNet block names to freeze 
                       (e.g., ['denseblock1', 'denseblock2']).
    """

    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 1024,
        dual_view_mode: str = "concat",
        freeze_blocks: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.dual_view_mode = dual_view_mode
        
        # Load DenseNet-121 backbone
        if pretrained:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            densenet = models.densenet121(weights=weights)
            logger.info("Loaded DenseNet-121 with ImageNet pretrained weights")
        else:
            densenet = models.densenet121(weights=None)
            logger.info("Loaded DenseNet-121 without pretrained weights")
        
        # Extract feature layers (remove classifier)
        # DenseNet-121 .features outputs [B, 1024, 7, 7] for 224×224 input
        self.features = densenet.features
        self.densenet_out_channels = 1024  # DenseNet-121 final feature channels
        
        # Adaptive pooling to ensure 7×7 output regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Batch norm after feature extraction (from DenseNet's final norm)
        self.final_bn = nn.BatchNorm2d(self.densenet_out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        
        # Dual-view merge layers
        if dual_view_mode == "concat":
            # Concatenation doubles channels → project back to feature_dim
            self.concat_projection = nn.Sequential(
                nn.Conv2d(
                    self.densenet_out_channels * 2,
                    feature_dim,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
            )
            logger.info(
                f"Dual-view mode: concat "
                f"({self.densenet_out_channels * 2} → {feature_dim})"
            )
        else:
            # Average mode: output channels = densenet channels
            if feature_dim != self.densenet_out_channels:
                self.avg_projection = nn.Sequential(
                    nn.Conv2d(
                        self.densenet_out_channels,
                        feature_dim,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(inplace=True),
                )
            else:
                self.avg_projection = nn.Identity()
            logger.info(f"Dual-view mode: avg → {feature_dim}")
        
        # Freeze early blocks if specified
        if freeze_blocks:
            self._freeze_blocks(freeze_blocks)
        
        # Log parameter counts
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Vision Encoder — "
            f"Total params: {total:,}, Trainable: {trainable:,}"
        )

    def _freeze_blocks(self, block_names: List[str]) -> None:
        """Freeze specified DenseNet blocks."""
        frozen_count = 0
        for name, param in self.features.named_parameters():
            for block_name in block_names:
                if block_name == "features" or block_name in name:
                    param.requires_grad = False
                    frozen_count += 1
                    break
        logger.info(
            f"Froze {frozen_count} parameters in blocks: {block_names}"
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from a single image.
        
        Args:
            x: [B, 3, H, W] input image tensor.
            
        Returns:
            features: [B, C, 7, 7] spatial feature grid.
        """
        features = self.features(x)      # [B, 1024, H', W']
        features = self.final_bn(features)
        features = self.final_relu(features)
        features = self.adaptive_pool(features)  # [B, 1024, 7, 7]
        return features

    def forward(
        self,
        frontal_img: torch.Tensor,
        lateral_img: torch.Tensor,
        lateral_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process dual-view X-rays through shared encoder and merge.
        
        Args:
            frontal_img: [B, 3, 224, 224] frontal view X-ray.
            lateral_img: [B, 3, 224, 224] lateral view X-ray 
                         (zero tensor if missing).
            lateral_mask: [B] binary mask (1=lateral exists, 0=missing).
            
        Returns:
            fused_features: [B, feature_dim, 7, 7] merged spatial features.
        """
        # Extract features from both views (shared weights)
        frontal_features = self._extract_features(frontal_img)  # [B, 1024, 7, 7]
        lateral_features = self._extract_features(lateral_img)  # [B, 1024, 7, 7]
        
        # Mask out lateral features where lateral view is missing
        # lateral_mask: [B] → [B, 1, 1, 1] for broadcasting
        mask = lateral_mask.view(-1, 1, 1, 1).float()
        lateral_features = lateral_features * mask
        
        if self.dual_view_mode == "concat":
            # Concatenate along channel dimension
            merged = torch.cat(
                [frontal_features, lateral_features], dim=1
            )  # [B, 2048, 7, 7]
            fused = self.concat_projection(merged)  # [B, feature_dim, 7, 7]
        else:
            # Average pooling (accounting for missing laterals)
            # When lateral exists: avg = (frontal + lateral) / 2
            # When lateral missing: avg = frontal (/ 1)
            denominator = 1.0 + mask.squeeze(-1).squeeze(-1).squeeze(-1)
            # denominator shape: [B]
            fused = (frontal_features + lateral_features) / denominator.view(
                -1, 1, 1, 1
            )
            fused = self.avg_projection(fused)  # [B, feature_dim, 7, 7]
        
        return fused

    def get_output_shape(self) -> Tuple[int, int, int]:
        """Return expected output shape (C, H, W)."""
        return (self.feature_dim, 7, 7)

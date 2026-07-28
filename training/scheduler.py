"""
Scheduler and Optimization Helpers.

Provides:
    - Curriculum lambda scheduler to linearly ramp consistency weight λ.
    - Parameter group selectors to support discriminative learning rates
      (lower learning rate for pretrained weights, higher for new blocks).
"""

import logging
from typing import Dict, List, Optional

import torch
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class CurriculumLambdaScheduler:
    """
    Manages the curriculum training schedule for the factual consistency weight λ.
    
    Gradually ramps λ from 0 to λ_max during training to allow the generator
    to learn spelling and syntax structure before being penalized for factual errors.
    
    Formula:
        λ(epoch) = 0,                                 for epoch < ramp_start
        λ(epoch) = λ_max * (epoch - ramp_start) / (ramp_end - ramp_start),
                                                      for ramp_start <= epoch < ramp_end
        λ(epoch) = λ_max,                             for epoch >= ramp_end
    """

    def __init__(
        self,
        lambda_max: float = 0.5,
        ramp_start: int = 5,
        ramp_end: int = 15,
    ):
        self.lambda_max = lambda_max
        self.ramp_start = ramp_start
        self.ramp_end = ramp_end
        
        logger.info(
            f"CurriculumLambdaScheduler initialized: "
            f"λ_max={lambda_max}, ramp={ramp_start}→{ramp_end} epochs"
        )

    def get_lambda(self, epoch: int) -> float:
        """
        Get the factual consistency loss weight for the current epoch.
        
        Args:
            epoch: Current epoch number (0-indexed).
            
        Returns:
            λ value (float).
        """
        if epoch < self.ramp_start:
            return 0.0
        
        if epoch >= self.ramp_end:
            return self.lambda_max
        
        # Linear interpolation
        fraction = (epoch - self.ramp_start) / (self.ramp_end - self.ramp_start)
        return self.lambda_max * fraction


def get_optimizer_params(
    model: torch.nn.Module,
    base_lr: float,
    discriminative_factor: float = 0.1,
    weight_decay: float = 0.01,
) -> List[Dict]:
    """
    Split model parameters into groups for discriminative learning rates and weight decay.
    
    Groups:
        1. Pretrained weights with weight decay (lower LR)
        2. Pretrained weights without weight decay (biases/norms, lower LR)
        3. New weights with weight decay (base LR)
        4. New weights without weight decay (biases/norms, base LR)
        
    Args:
        model: Full report generator model.
        base_lr: Base learning rate.
        discriminative_factor: Learning rate scale factor for pretrained blocks.
        weight_decay: Weight decay value.
        
    Returns:
        List of parameter dictionary groups.
    """
    decay_params_pretrained = []
    no_decay_params_pretrained = []
    decay_params_new = []
    no_decay_params_new = []
    
    # We define norm and bias types that should not receive weight decay
    no_decay_types = (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if parameter belongs to pretrained layers (vision encoder or GPT-2 decoder)
        is_pretrained = "vision_encoder.features" in name or "decoder.gpt2" in name
        
        # Check if parameter should have weight decay
        has_decay = not any(nd in name for nd in ["bias", "LayerNorm", "BatchNorm", "bn"])
        
        if is_pretrained:
            if has_decay:
                decay_params_pretrained.append(param)
            else:
                no_decay_params_pretrained.append(param)
        else:
            if has_decay:
                decay_params_new.append(param)
            else:
                no_decay_params_new.append(param)
                
    groups = [
        # Pretrained groups (lower learning rate)
        {
            "params": decay_params_pretrained,
            "lr": base_lr * discriminative_factor,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params_pretrained,
            "lr": base_lr * discriminative_factor,
            "weight_decay": 0.0,
        },
        # New layers groups (base learning rate)
        {
            "params": decay_params_new,
            "lr": base_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params_new,
            "lr": base_lr,
            "weight_decay": 0.0,
        },
    ]
    
    # Count totals
    total_pretrained = len(decay_params_pretrained) + len(no_decay_params_pretrained)
    total_new = len(decay_params_new) + len(no_decay_params_new)
    
    logger.info(
        f"Optimizing {total_pretrained} pretrained parameters (LR={base_lr*discriminative_factor:.2e}) "
        f"and {total_new} new parameters (LR={base_lr:.2e})"
    )
    
    return groups

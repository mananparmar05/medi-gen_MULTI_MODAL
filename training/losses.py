"""
Loss Functions for the Multimodal Medical Report Generation system.

Implements the Consistency-Weighted Loss:
    L_total = L_gen + λ(t)·L_consist + α·L_align
    
    Where:
    - L_gen: standard language modeling Cross-Entropy loss on token logits.
    - L_consist: contradiction probability penalty scored by NLI scorer.
    - L_align: weak supervision alignment loss (KL divergence between cross-attention
      weights and keyword-based target maps).
    - λ(t): consistency weight ramped dynamically over epochs (curriculum learning).
    - α: scaling factor for the alignment loss (default 0.1).
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConsistencyWeightedLoss(nn.Module):
    """
    Combines generator cross-entropy, NLI contradiction, and attention alignment losses.
    
    Args:
        config: Hyperparameter configuration DotDict.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Generation loss
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Hyperparameters
        self.alpha = config.cross_attention_bridge.get("alignment_loss_weight", 0.1)
        self.finding_labels = config.data.finding_labels
        self.finding_templates = config.data.finding_templates
        
        logger.info(
            f"ConsistencyWeightedLoss initialized with alignment loss weight α={self.alpha}"
        )

    def forward(
        self,
        generator_outputs: Dict[str, torch.Tensor],
        target_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer,
        nli_scorer,
        current_lambda: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task combined loss.
        
        Args:
            generator_outputs: Output dict from report generator forward pass.
            target_ids: [B, seq_len] shifted target token IDs.
            labels: [B, 14] binary structured findings.
            tokenizer: ReportTokenizer.
            nli_scorer: FactualConsistencyScorer.
            current_lambda: Ramped loss multiplier λ for factual consistency.
            device: Compute device.
            
        Returns:
            total_loss: Scalar combined loss tensor.
            loss_components: Dict of floats for logging.
        """
        # 1. Generation Loss (Cross-Entropy)
        # logits shape: [B, seq_len, vocab_size]
        logits = generator_outputs["logits"]
        vocab_size = logits.size(-1)
        
        gen_loss = self.gen_loss_fct(
            logits.view(-1, vocab_size),
            target_ids.view(-1),
        )
        
        # 2. Alignment Loss (BCE / KL from cross-attention bridge)
        # Included in the model's output as bridge alignment loss
        align_loss = generator_outputs.get("alignment_loss", torch.tensor(0.0, device=device))
        
        # 3. Consistency Loss (NLI Contradiction Scorer)
        consist_loss = torch.tensor(0.0, device=device)
        
        if current_lambda > 0:
            # Generate temporary sentences from generator to evaluate them online
            # We use the current logits to do a differentiable approximation or greedy decoding.
            # In curriculum NLI scoring, we decode sentences and feed them to the frozen NLI scorer.
            generated_ids = logits.argmax(dim=-1)  # [B, seq_len]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Compute average contradiction probability per batch element
            contradiction_probs = nli_scorer.get_contradiction_scores(
                finding_labels=labels,
                generated_texts=generated_texts,
                finding_label_names=self.finding_labels,
                finding_templates=self.finding_templates,
                device=device,
            )
            
            # Mean contradiction rate across the batch
            consist_loss = contradiction_probs.mean()
            
        # Combine losses: L = L_gen + λ·L_consist + α·L_align
        total_loss = gen_loss + (current_lambda * consist_loss) + (self.alpha * align_loss)
        
        loss_details = {
            "loss_total": total_loss.item(),
            "loss_generation": gen_loss.item(),
            "loss_alignment": align_loss.item(),
            "loss_consistency": consist_loss.item() if isinstance(consist_loss, torch.Tensor) else consist_loss,
            "lambda": current_lambda,
        }
        
        return total_loss, loss_details

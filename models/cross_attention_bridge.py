"""
Cross-Attention Bridge — Novel Contribution #1

Creates explicit, inspectable links between each structured finding and 
specific text spans in the generated report. This serves dual purposes:

1. Interpretability: Shows which words in the report correspond to each finding
2. Hallucination detection: If a finding's query produces no strong attention 
   anywhere in the text, it signals an omission-type hallucination

Architecture:
    - 14 learned finding query vectors (one per diagnostic category)
    - Multi-head cross-attention: queries attend over decoder hidden states
    - Alignment loss: weakly supervised by keyword matching in ground-truth
    - Omission detection: flags findings with max attention < threshold

Supervision strategy:
    At training time, ground-truth keyword positions in the report provide 
    weak supervision targets for the attention distributions. This is not 
    hard labels — it's a soft BCE loss that nudges attention toward the 
    right places without constraining it too rigidly.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossAttentionBridge(nn.Module):
    """
    Maps structured findings to text spans via learned cross-attention.
    
    Each of the 14 findings has a learned query embedding that attends
    over the decoder's hidden states to identify which generated words
    correspond to that finding.
    
    Args:
        num_findings: Number of diagnostic categories (default 14).
        query_dim: Dimension of finding query embeddings (default 256).
        decoder_hidden_dim: Dimension of decoder hidden states (default 768).
        num_heads: Number of attention heads (default 8).
        dropout: Attention dropout probability (default 0.1).
        omission_threshold: Max attention threshold for omission detection.
    """

    def __init__(
        self,
        num_findings: int = 14,
        query_dim: int = 256,
        decoder_hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        omission_threshold: float = 0.1,
    ):
        super().__init__()
        
        self.num_findings = num_findings
        self.query_dim = query_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_heads = num_heads
        self.omission_threshold = omission_threshold
        
        # Learned query embeddings — one per finding
        # These are the "questions" each finding asks of the generated text
        self.finding_queries = nn.Embedding(num_findings, query_dim)
        
        # Project finding queries to match attention dimensions
        self.query_projection = nn.Linear(query_dim, decoder_hidden_dim)
        
        # Project decoder hidden states for key/value
        self.key_projection = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.value_projection = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=decoder_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection: finding contexts back to query_dim
        self.output_projection = nn.Sequential(
            nn.Linear(decoder_hidden_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.ReLU(inplace=True),
        )
        
        # Finding-conditioned gate: scales attention by finding label
        # (inactive findings should produce near-zero attention)
        self.label_gate = nn.Sequential(
            nn.Linear(1, query_dim),
            nn.Sigmoid(),
        )
        
        # Initialize
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"CrossAttentionBridge: {num_findings} findings, "
            f"query_dim={query_dim}, heads={num_heads} "
            f"({total_params:,} params)"
        )

    def _init_weights(self) -> None:
        """Initialize finding queries with small random values."""
        nn.init.normal_(self.finding_queries.weight, mean=0.0, std=0.02)
        for module in [self.query_projection, self.key_projection, 
                       self.value_projection]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        finding_labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        alignment_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute finding-to-text alignments and optional alignment loss.
        
        Args:
            decoder_hidden: [B, seq_len, 768] hidden states from decoder.
            finding_labels: [B, 14] binary finding labels.
            attention_mask: [B, seq_len] mask for padding tokens.
            alignment_targets: [B, 14, seq_len] weak supervision targets
                               (from keyword matching). None at inference.
        
        Returns:
            Dict containing:
                - context_vectors: [B, 14, query_dim] finding-specific contexts
                - alignment_maps: [B, 14, seq_len] attention distributions
                - alignment_loss: scalar (only during training with targets)
                - omission_flags: [B, 14] boolean (only at inference)
        """
        B, seq_len, _ = decoder_hidden.shape
        device = decoder_hidden.device
        
        # Get finding query embeddings: [14, query_dim]
        finding_indices = torch.arange(
            self.num_findings, device=device
        )
        queries = self.finding_queries(finding_indices)  # [14, query_dim]
        
        # Expand queries for batch: [B, 14, query_dim]
        queries = queries.unsqueeze(0).expand(B, -1, -1)
        
        # Apply label-conditioned gating
        # finding_labels: [B, 14] → [B, 14, 1]
        gate_input = finding_labels.unsqueeze(-1)  # [B, 14, 1]
        gate = self.label_gate(gate_input)          # [B, 14, query_dim]
        queries = queries * gate                     # [B, 14, query_dim]
        
        # Project queries: [B, 14, query_dim] → [B, 14, hidden_dim]
        Q = self.query_projection(queries)
        
        # Project decoder hidden states for keys and values
        K = self.key_projection(decoder_hidden)    # [B, seq_len, hidden_dim]
        V = self.value_projection(decoder_hidden)  # [B, seq_len, hidden_dim]
        
        # Create key padding mask (True = ignore) for attention
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: [B, seq_len], 1=valid, 0=pad
            # MultiheadAttention wants True=ignore
            key_padding_mask = (attention_mask == 0)
        
        # Multi-head cross-attention
        # Q: [B, 14, hidden_dim], K: [B, seq_len, hidden_dim]
        context, attention_weights = self.attention(
            query=Q,
            key=K,
            value=V,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,  # average across heads
        )
        # context: [B, 14, hidden_dim]
        # attention_weights: [B, 14, seq_len]
        
        # Project back to query_dim
        context_vectors = self.output_projection(context)  # [B, 14, query_dim]
        
        # Alignment maps (attention distributions)
        alignment_maps = attention_weights  # [B, 14, seq_len]
        
        result = {
            "context_vectors": context_vectors,
            "alignment_maps": alignment_maps,
        }
        
        # Compute alignment loss during training
        if alignment_targets is not None and self.training:
            alignment_loss = self._compute_alignment_loss(
                alignment_maps, alignment_targets, finding_labels
            )
            result["alignment_loss"] = alignment_loss
        else:
            result["alignment_loss"] = torch.tensor(0.0, device=device)
        
        # Omission detection at inference
        if not self.training:
            omission_flags = self._detect_omissions(
                alignment_maps, finding_labels
            )
            result["omission_flags"] = omission_flags
        
        return result

    def _compute_alignment_loss(
        self,
        alignment_maps: torch.Tensor,
        alignment_targets: torch.Tensor,
        finding_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weak supervision loss on alignment maps.
        
        Uses binary cross-entropy between predicted attention distribution
        and keyword-based target distribution, weighted by finding labels
        (only active findings contribute to the loss).
        
        Args:
            alignment_maps: [B, 14, seq_len] predicted attention.
            alignment_targets: [B, 14, seq_len] target attention.
            finding_labels: [B, 14] binary labels (weight mask).
            
        Returns:
            Scalar alignment loss.
        """
        # KL divergence between predicted and target distributions
        # Add small epsilon for numerical stability
        eps = 1e-8
        pred = alignment_maps + eps
        target = alignment_targets + eps
        
        # Normalize target to be a valid distribution
        target = target / target.sum(dim=-1, keepdim=True).clamp(min=eps)
        
        # KL(target || pred) per finding per sample
        kl = target * (target.log() - pred.log())  # [B, 14, seq_len]
        kl = kl.sum(dim=-1)  # [B, 14]
        
        # Weight by finding labels (only active findings matter)
        weighted_kl = kl * finding_labels  # [B, 14]
        
        # Mean over active findings
        num_active = finding_labels.sum().clamp(min=1.0)
        loss = weighted_kl.sum() / num_active
        
        return loss

    def _detect_omissions(
        self,
        alignment_maps: torch.Tensor,
        finding_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Detect findings that should be mentioned but have weak attention.
        
        A finding is flagged as omitted if:
            1. Its label is 1 (finding is present)
            2. Its maximum attention weight < threshold
        
        Args:
            alignment_maps: [B, 14, seq_len] attention distributions.
            finding_labels: [B, 14] binary labels.
            
        Returns:
            omission_flags: [B, 14] boolean tensor (True = omitted).
        """
        # Max attention weight per finding: [B, 14]
        max_attention = alignment_maps.max(dim=-1).values
        
        # Flag: label=1 AND max_attention < threshold
        omission_flags = (
            (finding_labels > 0.5) & 
            (max_attention < self.omission_threshold)
        )
        
        return omission_flags

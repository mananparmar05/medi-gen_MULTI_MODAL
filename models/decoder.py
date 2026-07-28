"""
Medical Report Decoder — GPT-2 small with injected cross-attention layers 
for visual grounding.

Key modifications to standard GPT-2:
    1. Cross-attention layers inserted after every self-attention block
       to attend to visual context (fused image+metadata features)
    2. Visual context projection: 1024-dim spatial features → 768-dim 
       to match GPT-2's hidden dimension
    3. Causal masking maintained for autoregressive generation
    4. Exposes intermediate hidden states for cross-attention bridge

Training strategy:
    - Epochs 1-3: Freeze GPT-2 body, train only cross-attention + projection
    - Epochs 4+: Unfreeze all with discriminative LR (pretrained gets lower LR)
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config

logger = logging.getLogger(__name__)


# ============================================================================
# Cross-Attention Layer (injected into GPT-2)
# ============================================================================

class VisualCrossAttentionLayer(nn.Module):
    """
    Cross-attention layer that lets the decoder attend to visual features.
    
    Inserted after each GPT-2 self-attention block. The decoder's hidden 
    states serve as queries; visual spatial features serve as keys/values.
    
    Args:
        hidden_dim: GPT-2 hidden dimension (768).
        num_heads: Number of attention heads (12).
        dropout: Attention dropout (0.1).
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attend from text hidden states to visual context.
        
        Args:
            hidden_states: [B, seq_len, 768] from self-attention.
            visual_context: [B, num_visual_tokens, 768] projected visual features.
            
        Returns:
            output: [B, seq_len, 768] cross-attended hidden states.
        """
        # Pre-norm cross-attention
        normed = self.layer_norm(hidden_states)
        attended, _ = self.attention(
            query=normed,
            key=visual_context,
            value=visual_context,
        )
        hidden_states = hidden_states + attended
        
        # Pre-norm feedforward
        normed = self.ff_layer_norm(hidden_states)
        hidden_states = hidden_states + self.feedforward(normed)
        
        return hidden_states


# ============================================================================
# Medical Report Decoder
# ============================================================================

class MedicalReportDecoder(nn.Module):
    """
    GPT-2 small fine-tuned for medical report generation with visual grounding.
    
    Architecture:
        1. Visual context projection: [B, 1024, 7, 7] → [B, 49, 768]
        2. GPT-2 with interleaved cross-attention:
           For each transformer block:
               hidden = self_attention(hidden) 
               hidden = cross_attention(hidden, visual_context)
        3. LM head for next-token prediction
    
    Args:
        model_name: HuggingFace GPT-2 model identifier.
        visual_feature_dim: Input visual feature channels (default 1024).
        visual_grid_size: Spatial grid size (default 7, giving 49 visual tokens).
        vocab_size: Vocabulary size (set after adding special tokens).
        max_length: Maximum generation length.
        num_cross_attn_layers: How many GPT-2 layers get cross-attention.
                               If < num_layers, only the last N layers get it.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        visual_feature_dim: int = 1024,
        visual_grid_size: int = 7,
        vocab_size: Optional[int] = None,
        max_length: int = 128,
        num_cross_attn_layers: Optional[int] = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.visual_feature_dim = visual_feature_dim
        self.num_visual_tokens = visual_grid_size ** 2  # 49
        self.max_length = max_length
        
        # Load pretrained GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        config = self.gpt2.config
        self.hidden_dim = config.n_embd       # 768
        self.num_layers = config.n_layer       # 12
        self.num_heads = config.n_head         # 12
        
        logger.info(
            f"Loaded {model_name}: {self.num_layers} layers, "
            f"hidden_dim={self.hidden_dim}, heads={self.num_heads}"
        )
        
        # Resize embeddings if vocab changed (special tokens added)
        if vocab_size is not None and vocab_size != config.vocab_size:
            self.gpt2.resize_token_embeddings(vocab_size)
            logger.info(
                f"Resized embeddings: {config.vocab_size} → {vocab_size}"
            )
        
        # Visual context projection: [B, C, H, W] → [B, 49, 768]
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Learned positional embeddings for visual tokens
        self.visual_position_embedding = nn.Embedding(
            self.num_visual_tokens, self.hidden_dim
        )
        
        # Cross-attention layers (one per GPT-2 layer, or subset)
        if num_cross_attn_layers is None:
            num_cross_attn_layers = self.num_layers
        num_cross_attn_layers = min(num_cross_attn_layers, self.num_layers)
        
        # Insert cross-attention for the last N layers
        self.cross_attention_layers = nn.ModuleDict()
        start_layer = self.num_layers - num_cross_attn_layers
        for i in range(start_layer, self.num_layers):
            self.cross_attention_layers[str(i)] = VisualCrossAttentionLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
            )
        
        logger.info(
            f"Added {len(self.cross_attention_layers)} cross-attention layers "
            f"(layers {start_layer}-{self.num_layers - 1})"
        )
        
        # Log parameter counts
        pretrained_params = sum(
            p.numel() for p in self.gpt2.parameters()
        )
        new_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "gpt2" not in name
        )
        logger.info(
            f"Decoder — Pretrained: {pretrained_params:,}, "
            f"New (cross-attn + projection): {new_params:,}"
        )

    def freeze_pretrained(self) -> None:
        """Freeze GPT-2 pretrained parameters (for initial training epochs)."""
        for param in self.gpt2.parameters():
            param.requires_grad = False
        logger.info("GPT-2 pretrained parameters frozen")

    def unfreeze_pretrained(self) -> None:
        """Unfreeze all GPT-2 parameters."""
        for param in self.gpt2.parameters():
            param.requires_grad = True
        logger.info("GPT-2 pretrained parameters unfrozen")

    def get_parameter_groups(
        self,
        base_lr: float,
        discriminative_factor: float = 0.1,
    ) -> list:
        """
        Create parameter groups with discriminative learning rates.
        
        Pretrained GPT-2 params get base_lr * discriminative_factor.
        New cross-attention and projection params get base_lr.
        
        Args:
            base_lr: Base learning rate for new parameters.
            discriminative_factor: Multiplier for pretrained params.
            
        Returns:
            List of parameter group dicts for optimizer.
        """
        pretrained_params = []
        new_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "gpt2" in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)
        
        return [
            {"params": pretrained_params, "lr": base_lr * discriminative_factor},
            {"params": new_params, "lr": base_lr},
        ]

    def _prepare_visual_context(
        self,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project spatial visual features into decoder's embedding space.
        
        Args:
            visual_features: [B, C, H, W] (e.g., [B, 1024, 7, 7])
            
        Returns:
            visual_context: [B, 49, 768] projected visual tokens.
        """
        B, C, H, W = visual_features.shape
        
        # Flatten spatial dimensions: [B, C, H, W] → [B, H*W, C]
        visual_flat = visual_features.flatten(2).permute(0, 2, 1)  # [B, 49, 1024]
        
        # Project to hidden dim: [B, 49, 1024] → [B, 49, 768]
        visual_context = self.visual_projection(visual_flat)
        
        # Add positional embeddings
        positions = torch.arange(
            self.num_visual_tokens, device=visual_features.device
        )
        visual_context = visual_context + self.visual_position_embedding(positions)
        
        return visual_context

    def forward(
        self,
        visual_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with visual grounding.
        
        Args:
            visual_features: [B, C, H, W] fused visual-metadata features.
            input_ids: [B, seq_len] input token IDs.
            attention_mask: [B, seq_len] attention mask (1=valid, 0=pad).
            labels: [B, seq_len] target token IDs for loss (optional).
            
        Returns:
            Dict with:
                - logits: [B, seq_len, vocab_size] 
                - hidden_states: [B, seq_len, 768] (last layer, for cross-attn bridge)
                - loss: scalar (if labels provided)
        """
        B, seq_len = input_ids.shape
        
        # Prepare visual context
        visual_context = self._prepare_visual_context(visual_features)
        
        # Get GPT-2 transformer blocks
        transformer = self.gpt2.transformer
        
        # Token + position embeddings
        inputs_embeds = transformer.wte(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = transformer.drop(hidden_states)
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1,
        ).bool()
        
        # Attention mask for padding
        if attention_mask is not None:
            # Expand to [B, 1, 1, seq_len] for broadcasting
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask.float()) * -1e4
        else:
            extended_mask = None
        
        # Pass through each GPT-2 transformer block
        for i, block in enumerate(transformer.h):
            # Standard GPT-2 self-attention + FFN
            outputs = block(
                hidden_states,
                attention_mask=extended_mask,
            )
            hidden_states = outputs[0]
            
            # Cross-attention to visual context (if this layer has one)
            if str(i) in self.cross_attention_layers:
                hidden_states = self.cross_attention_layers[str(i)](
                    hidden_states, visual_context
                )
        
        # Final layer norm
        hidden_states = transformer.ln_f(hidden_states)
        
        # LM head
        logits = self.gpt2.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "loss": loss,
        }

    @torch.no_grad()
    def generate_greedy(
        self,
        visual_features: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple greedy decoding for inference.
        
        Args:
            visual_features: [B, C, H, W] visual context.
            tokenizer: ReportTokenizer instance.
            max_length: Maximum generation length.
            
        Returns:
            (generated_ids, hidden_states_all)
            generated_ids: [B, gen_len] generated token IDs.
            hidden_states_all: [B, gen_len, 768] hidden states for bridge.
        """
        if max_length is None:
            max_length = self.max_length
        
        B = visual_features.size(0)
        device = visual_features.device
        
        # Start with BOS token
        input_ids = torch.full(
            (B, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
        )
        
        all_hidden_states = []
        
        for step in range(max_length - 1):
            outputs = self.forward(
                visual_features=visual_features,
                input_ids=input_ids,
            )
            
            # Get next token (greedy)
            next_logits = outputs["logits"][:, -1, :]  # [B, vocab_size]
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            
            # Collect hidden state for this position
            all_hidden_states.append(outputs["hidden_states"][:, -1:, :])
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if all sequences have generated EOS
            if (next_token.squeeze(-1) == tokenizer.eos_token_id).all():
                break
        
        # Stack hidden states
        hidden_states_all = torch.cat(all_hidden_states, dim=1)
        
        return input_ids, hidden_states_all

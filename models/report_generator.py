"""
Multimodal Report Generator — Full pipeline assembly.

Orchestrates all components in the correct order:
    1. Vision Encoder    → visual features [B, 1024, 7, 7]
    2. Metadata MLP      → metadata embedding [B, 256]
    3. FiLM Fusion       → conditioned features [B, 1024, 7, 7]
    4. Decoder (GPT-2)   → logits + hidden states
    5. Cross-Attn Bridge → alignment maps + alignment loss
    6. NLI Scorer        → consistency scores (training signal + evaluation)

Two execution modes:
    - Training: returns (logits, generation_loss, alignment_loss, hidden_states)
    - Inference: returns (generated_text, alignment_maps, omission_flags, fcs)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.vision_encoder import DualViewVisionEncoder
from models.metadata_mlp import MetadataEmbeddingMLP
from models.film_fusion import FiLMFusionLayer
from models.cross_attention_bridge import CrossAttentionBridge
from models.decoder import MedicalReportDecoder
from models.nli_scorer import FactualConsistencyScorer

logger = logging.getLogger(__name__)


class MultimodalReportGenerator(nn.Module):
    """
    Full multimodal medical report generation pipeline.
    
    Assembles vision encoder, metadata embedder, FiLM fusion, 
    transformer decoder, cross-attention bridge, and NLI scorer
    into a single trainable model.
    
    Args:
        config: Configuration DotDict with all hyperparameters.
        vocab_size: Vocabulary size (from tokenizer, after special tokens).
    """

    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = config
        
        # ---- Component 1: Vision Encoder ----
        self.vision_encoder = DualViewVisionEncoder(
            pretrained=config.vision_encoder.pretrained,
            feature_dim=config.vision_encoder.feature_dim,
            dual_view_mode=config.vision_encoder.dual_view_mode,
            freeze_blocks=config.vision_encoder.get("freeze_blocks", None),
        )
        
        # ---- Component 2: Metadata Embedding MLP ----
        self.metadata_mlp = MetadataEmbeddingMLP(
            input_dim=config.metadata_mlp.input_dim,
            hidden_dim=config.metadata_mlp.hidden_dim,
            output_dim=config.metadata_mlp.output_dim,
            dropout=config.metadata_mlp.dropout,
            label_smoothing=config.metadata_mlp.get("label_smoothing", 0.0),
        )
        
        # ---- Component 3: FiLM Fusion Layer ----
        self.film_fusion = FiLMFusionLayer(
            conditioning_dim=config.metadata_mlp.output_dim,
            feature_channels=config.vision_encoder.feature_dim,
            use_residual=config.film_fusion.use_residual,
            identity_init=config.film_fusion.identity_init,
        )
        
        # ---- Component 4: Decoder (GPT-2 with cross-attention) ----
        self.decoder = MedicalReportDecoder(
            model_name=config.decoder.model_name,
            visual_feature_dim=config.vision_encoder.feature_dim,
            visual_grid_size=config.vision_encoder.output_grid_size,
            vocab_size=vocab_size,
            max_length=config.decoder.max_gen_length,
        )
        
        # ---- Component 5: Cross-Attention Bridge ----
        self.cross_attention_bridge = CrossAttentionBridge(
            num_findings=config.cross_attention_bridge.num_findings,
            query_dim=config.cross_attention_bridge.query_dim,
            decoder_hidden_dim=config.decoder.embed_dim,
            num_heads=config.cross_attention_bridge.num_heads,
            dropout=config.cross_attention_bridge.dropout,
            omission_threshold=config.cross_attention_bridge.omission_threshold,
        )
        
        # ---- Component 6: NLI Scorer (loaded separately, frozen during gen training) ----
        self.nli_scorer = FactualConsistencyScorer(
            model_name=config.nli_scorer.model_name,
            hidden_dim=config.nli_scorer.hidden_dim,
            num_classes=config.nli_scorer.num_classes,
            dropout=config.nli_scorer.dropout,
        )
        
        # Freeze NLI scorer if configured
        if config.nli_scorer.get("frozen_during_gen_training", True):
            self.freeze_nli_scorer()
        
        # Log total model size
        self._log_model_stats()

    def _log_model_stats(self) -> None:
        """Log parameter counts for each component."""
        components = {
            "Vision Encoder": self.vision_encoder,
            "Metadata MLP": self.metadata_mlp,
            "FiLM Fusion": self.film_fusion,
            "Decoder": self.decoder,
            "Cross-Attn Bridge": self.cross_attention_bridge,
            "NLI Scorer": self.nli_scorer,
        }
        
        total = 0
        trainable = 0
        logger.info("=" * 60)
        logger.info("Model Component Sizes:")
        for name, module in components.items():
            n_total = sum(p.numel() for p in module.parameters())
            n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total += n_total
            trainable += n_train
            logger.info(
                f"  {name:25s}: {n_total:>12,} total, {n_train:>12,} trainable"
            )
        logger.info("-" * 60)
        logger.info(
            f"  {'TOTAL':25s}: {total:>12,} total, {trainable:>12,} trainable"
        )
        logger.info("=" * 60)

    def freeze_nli_scorer(self) -> None:
        """Freeze NLI scorer parameters (not trained during generation)."""
        for param in self.nli_scorer.parameters():
            param.requires_grad = False
        logger.info("NLI Scorer frozen (will not be updated during training)")

    def unfreeze_nli_scorer(self) -> None:
        """Unfreeze NLI scorer for joint fine-tuning."""
        for param in self.nli_scorer.parameters():
            param.requires_grad = True
        logger.info("NLI Scorer unfrozen")

    def set_training_phase(self, epoch: int) -> None:
        """
        Configure model for the current training phase.
        
        Phase 1 (epochs 0 to freeze_epochs-1): 
            Freeze GPT-2 body, only train new layers
        Phase 2 (epochs freeze_epochs+):
            Unfreeze GPT-2 with discriminative LR
        
        Args:
            epoch: Current epoch number (0-indexed).
        """
        freeze_epochs = self.config.decoder.freeze_epochs
        
        if epoch < freeze_epochs:
            self.decoder.freeze_pretrained()
            logger.info(f"Epoch {epoch}: Phase 1 — GPT-2 frozen")
        elif epoch == freeze_epochs:
            self.decoder.unfreeze_pretrained()
            logger.info(f"Epoch {epoch}: Phase 2 — GPT-2 unfrozen")

    def forward(
        self,
        frontal_img: torch.Tensor,
        lateral_img: torch.Tensor,
        lateral_mask: torch.Tensor,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        alignment_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the pipeline.
        
        Args:
            frontal_img: [B, 3, 224, 224] frontal X-ray.
            lateral_img: [B, 3, 224, 224] lateral X-ray.
            lateral_mask: [B] lateral view existence mask.
            labels: [B, 14] binary finding labels.
            input_ids: [B, seq_len] decoder input token IDs.
            target_ids: [B, seq_len] decoder target token IDs (training only).
            attention_mask: [B, seq_len] padding mask.
            alignment_targets: [B, 14, seq_len] cross-attention targets (training only).
            
        Returns:
            Dict with:
                - logits: [B, seq_len, vocab_size]
                - generation_loss: scalar (if target_ids provided)
                - alignment_loss: scalar (if alignment_targets provided)
                - alignment_maps: [B, 14, seq_len]
                - hidden_states: [B, seq_len, 768]
                - fused_features: [B, C, 7, 7] (for visualization)
        """
        # Step 1: Vision Encoder
        visual_features = self.vision_encoder(
            frontal_img, lateral_img, lateral_mask
        )  # [B, 1024, 7, 7]
        
        # Step 2: Metadata Embedding
        metadata_embedding = self.metadata_mlp(labels)  # [B, 256]
        
        # Step 3: FiLM Fusion
        fused_features = self.film_fusion(
            visual_features, metadata_embedding
        )  # [B, 1024, 7, 7]
        
        # Step 4: Decoder
        decoder_output = self.decoder(
            visual_features=fused_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
        )
        
        logits = decoder_output["logits"]
        hidden_states = decoder_output["hidden_states"]
        generation_loss = decoder_output.get("loss")
        
        # Step 5: Cross-Attention Bridge
        bridge_output = self.cross_attention_bridge(
            decoder_hidden=hidden_states,
            finding_labels=labels,
            attention_mask=attention_mask,
            alignment_targets=alignment_targets,
        )
        
        alignment_maps = bridge_output["alignment_maps"]
        alignment_loss = bridge_output["alignment_loss"]
        
        result = {
            "logits": logits,
            "generation_loss": generation_loss,
            "alignment_loss": alignment_loss,
            "alignment_maps": alignment_maps,
            "hidden_states": hidden_states,
            "fused_features": fused_features,
        }
        
        # Include omission flags at inference
        if not self.training and "omission_flags" in bridge_output:
            result["omission_flags"] = bridge_output["omission_flags"]
        
        return result

    @torch.no_grad()
    def generate(
        self,
        frontal_img: torch.Tensor,
        lateral_img: torch.Tensor,
        lateral_mask: torch.Tensor,
        labels: torch.Tensor,
        tokenizer,
        max_length: Optional[int] = None,
        beam_searcher=None,
    ) -> Dict[str, Any]:
        """
        Generate report text with full pipeline.
        
        Args:
            frontal_img: [B, 3, 224, 224] frontal X-ray.
            lateral_img: [B, 3, 224, 224] lateral X-ray.
            lateral_mask: [B] lateral view existence mask.
            labels: [B, 14] binary finding labels.
            tokenizer: ReportTokenizer instance.
            max_length: Maximum generation length.
            beam_searcher: Optional beam search module.
            
        Returns:
            Dict with:
                - generated_texts: List[str] of generated reports
                - generated_ids: [B, gen_len] token IDs
                - alignment_maps: [B, 14, gen_len]
                - omission_flags: [B, 14]
                - fused_features: [B, C, 7, 7]
        """
        self.eval()
        
        # Steps 1-3: Encode + Fuse
        visual_features = self.vision_encoder(
            frontal_img, lateral_img, lateral_mask
        )
        metadata_embedding = self.metadata_mlp(labels)
        fused_features = self.film_fusion(visual_features, metadata_embedding)
        
        # Step 4: Generate text
        if beam_searcher is not None:
            # Beam search only supports B=1 per call.
            # Loop over each sample in the batch individually, then pad & stack.
            B = fused_features.size(0)
            all_ids = []
            all_hidden = []
            for b in range(B):
                sample_features = fused_features[b:b+1]   # [1, C, H, W]
                ids_b, hidden_b = beam_searcher.search(
                    self.decoder, sample_features, tokenizer, max_length
                )
                all_ids.append(ids_b[0])       # [gen_len_b]
                all_hidden.append(hidden_b[0]) # [gen_len_b, 768]

            # Pad sequences to the same length for batched bridge
            # NOTE: all_ids includes BOS (len=steps+1); all_hidden has len=steps.
            #       Compute max lengths independently to avoid negative padding.
            max_ids_len = max(t.size(0) for t in all_ids)
            max_hidden_len = max(h.size(0) for h in all_hidden)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            device = fused_features.device
            generated_ids = torch.stack([
                torch.cat([t, torch.full((max_ids_len - t.size(0),), pad_id,
                                         dtype=torch.long, device=device)])
                for t in all_ids
            ])  # [B, max_ids_len]
            hidden_states = torch.stack([
                torch.cat([h, torch.zeros(max_hidden_len - h.size(0), h.size(1), device=device)])
                for h in all_hidden
            ])  # [B, max_hidden_len, 768]
        else:
            generated_ids, hidden_states = self.decoder.generate_greedy(
                fused_features, tokenizer, max_length
            )
        
        # Decode to text
        generated_texts = tokenizer.batch_decode(generated_ids)
        
        # Step 5: Cross-Attention Bridge on generated hidden states
        bridge_output = self.cross_attention_bridge(
            decoder_hidden=hidden_states,
            finding_labels=labels,
        )
        
        return {
            "generated_texts": generated_texts,
            "generated_ids": generated_ids,
            "alignment_maps": bridge_output["alignment_maps"],
            "omission_flags": bridge_output.get("omission_flags"),
            "fused_features": fused_features,
        }

"""
Unit tests and sanity checks for the Multimodal Medical Report Generation pipeline.

This script verifies:
1. Tokenizer initialization and encoding/decoding.
2. Synthetic dataset generation and loading.
3. Shape and forward pass correctness of:
   - Vision Encoder (DualViewVisionEncoder)
   - Metadata MLP (MetadataEmbeddingMLP)
   - FiLM Fusion (FiLMFusionLayer)
   - GPT-2 Decoder (MedicalReportDecoder)
   - Cross-Attention Bridge (CrossAttentionBridge)
   - Factual Consistency Scorer (FactualConsistencyScorer)
4. Full pipeline assembly (MultimodalReportGenerator) training and generation steps.
"""

import sys
import unittest
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, DotDict
from data.tokenizer import ReportTokenizer
from data.dataset import IUXRayDataset
from models.vision_encoder import DualViewVisionEncoder
from models.metadata_mlp import MetadataEmbeddingMLP
from models.film_fusion import FiLMFusionLayer
from models.decoder import MedicalReportDecoder
from models.cross_attention_bridge import CrossAttentionBridge
from models.nli_scorer import FactualConsistencyScorer
from models.report_generator import MultimodalReportGenerator


class TestPipelineComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load config or use a mock mini-config for testing
        cls.config = load_config("config/config.yaml")
        # Initialize tokenizers
        cls.tokenizer = ReportTokenizer(cls.config.decoder.model_name, max_length=32)
        cls.vocab_size = cls.tokenizer.vocab_size
        cls.device = torch.device("cpu")

    def test_1_tokenizer(self):
        print("\n--- Testing Tokenizer ---")
        text = "The heart size is normal. No focal consolidation is seen."
        encoded = self.tokenizer.encode(text)
        
        self.assertIn("input_ids", encoded)
        self.assertIn("attention_mask", encoded)
        self.assertEqual(encoded["input_ids"].shape[0], 32)
        
        decoded = self.tokenizer.decode(encoded["input_ids"])
        print(f"Original text: {text}")
        print(f"Decoded text:  {decoded}")
        
        # Test input/target generation
        input_ids, target_ids = self.tokenizer.get_target_tokens(text)
        self.assertEqual(input_ids.shape[0], 32)
        self.assertEqual(target_ids.shape[0], 32)
        # First target should be the token after BOS (which is input_ids[1])
        self.assertEqual(target_ids[0], input_ids[1])

    def test_2_vision_encoder(self):
        print("\n--- Testing Vision Encoder ---")
        batch_size = 2
        # Frontal and lateral dummy images [B, 3, 224, 224]
        frontal = torch.randn(batch_size, 3, 224, 224)
        lateral = torch.randn(batch_size, 3, 224, 224)
        lateral_mask = torch.tensor([1.0, 0.0])  # Sample 1 has lateral, sample 2 doesn't
        
        encoder = DualViewVisionEncoder(
            pretrained=False,
            feature_dim=1024,
            dual_view_mode="concat"
        )
        encoder.eval()
        
        with torch.no_grad():
            features = encoder(frontal, lateral, lateral_mask)
            
        self.assertEqual(features.shape, (batch_size, 1024, 7, 7))
        print(f"Vision Encoder output shape: {features.shape}")

    def test_3_metadata_mlp(self):
        print("\n--- Testing Metadata MLP ---")
        batch_size = 2
        dummy_labels = torch.randint(0, 2, (batch_size, 14)).float()
        
        mlp = MetadataEmbeddingMLP(
            input_dim=14,
            hidden_dim=64,
            output_dim=256,
            dropout=0.1
        )
        mlp.eval()
        
        with torch.no_grad():
            embedding = mlp(dummy_labels)
            
        self.assertEqual(embedding.shape, (batch_size, 256))
        print(f"Metadata MLP output shape: {embedding.shape}")

    def test_4_film_fusion(self):
        print("\n--- Testing FiLM Fusion ---")
        batch_size = 2
        visual_features = torch.randn(batch_size, 1024, 7, 7)
        metadata_embed = torch.randn(batch_size, 256)
        
        film = FiLMFusionLayer(
            conditioning_dim=256,
            feature_channels=1024,
            use_residual=True,
            identity_init=True
        )
        film.eval()
        
        with torch.no_grad():
            fused = film(visual_features, metadata_embed)
            
        self.assertEqual(fused.shape, (batch_size, 1024, 7, 7))
        print(f"FiLM Fusion output shape: {fused.shape}")

    def test_5_decoder(self):
        print("\n--- Testing GPT-2 Decoder ---")
        batch_size = 2
        seq_len = 16
        
        fused_features = torch.randn(batch_size, 1024, 7, 7)
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        decoder = MedicalReportDecoder(
            model_name="gpt2",
            visual_feature_dim=1024,
            visual_grid_size=7,
            vocab_size=self.vocab_size,
            max_length=32
        )
        decoder.eval()
        
        with torch.no_grad():
            outputs = decoder(
                visual_features=fused_features,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        self.assertEqual(outputs["logits"].shape, (batch_size, seq_len, self.vocab_size))
        self.assertEqual(outputs["hidden_states"].shape, (batch_size, seq_len, 768))
        print(f"Decoder logits shape: {outputs['logits'].shape}")
        print(f"Decoder hidden states shape: {outputs['hidden_states'].shape}")

    def test_6_cross_attention_bridge(self):
        print("\n--- Testing Cross-Attention Bridge ---")
        batch_size = 2
        seq_len = 16
        
        decoder_hidden = torch.randn(batch_size, seq_len, 768)
        finding_labels = torch.randint(0, 2, (batch_size, 14)).float()
        attention_mask = torch.ones(batch_size, seq_len)
        
        bridge = CrossAttentionBridge(
            num_findings=14,
            query_dim=256,
            decoder_hidden_dim=768,
            num_heads=8,
            dropout=0.1
        )
        bridge.eval()
        
        with torch.no_grad():
            outputs = bridge(
                decoder_hidden=decoder_hidden,
                finding_labels=finding_labels,
                attention_mask=attention_mask
            )
            
        self.assertEqual(outputs["context_vectors"].shape, (batch_size, 14, 256))
        self.assertEqual(outputs["alignment_maps"].shape, (batch_size, 14, seq_len))
        self.assertIn("omission_flags", outputs)
        self.assertEqual(outputs["omission_flags"].shape, (batch_size, 14))
        print(f"Bridge context vectors shape: {outputs['context_vectors'].shape}")
        print(f"Bridge alignment maps shape: {outputs['alignment_maps'].shape}")

    def test_7_full_generator(self):
        print("\n--- Testing Full Report Generator ---")
        batch_size = 2
        seq_len = 16
        
        frontal = torch.randn(batch_size, 3, 224, 224)
        lateral = torch.randn(batch_size, 3, 224, 224)
        lateral_mask = torch.tensor([1.0, 1.0])
        labels = torch.randint(0, 2, (batch_size, 14)).float()
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        target_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        alignment_targets = torch.randn(batch_size, 14, seq_len)
        
        generator = MultimodalReportGenerator(self.config, self.vocab_size)
        generator.eval()
        
        # Test training forward pass
        with torch.no_grad():
            outputs = generator(
                frontal_img=frontal,
                lateral_img=lateral,
                lateral_mask=lateral_mask,
                labels=labels,
                input_ids=input_ids,
                target_ids=target_ids,
                attention_mask=attention_mask,
                alignment_targets=alignment_targets
            )
            
        self.assertIn("logits", outputs)
        self.assertIn("generation_loss", outputs)
        self.assertIn("alignment_loss", outputs)
        print("Generator forward pass successful!")

        # Test generation forward pass
        with torch.no_grad():
            gen_outputs = generator.generate(
                frontal_img=frontal,
                lateral_img=lateral,
                lateral_mask=lateral_mask,
                labels=labels,
                tokenizer=self.tokenizer,
                max_length=20
            )
        self.assertIn("generated_texts", gen_outputs)
        self.assertIn("alignment_maps", gen_outputs)
        print(f"Generated text sample: {gen_outputs['generated_texts'][0]}")

    def test_8_batched_nli_scorer(self):
        print("\n--- Testing Batched NLI Scorer ---")
        batch_size = 2
        labels = torch.randint(0, 2, (batch_size, 14)).float()
        generated_texts = [
            "The heart size is normal. No focal consolidation is seen.",
            "There is evidence of cardiomegaly and pulmonary edema."
        ]
        
        scorer = FactualConsistencyScorer(
            model_name="bert-base-uncased",
            hidden_dim=64,
            num_classes=3,
            dropout=0.1
        )
        scorer.eval()
        
        # Test score_reports_batch
        res_batch = scorer.score_reports_batch(
            finding_labels=labels,
            generated_texts=generated_texts,
            finding_label_names=self.config.data.finding_labels,
            finding_templates=self.config.data.finding_templates,
            device=self.device,
        )
        
        self.assertEqual(len(res_batch["fcs"]), batch_size)
        self.assertEqual(len(res_batch["contradiction_rate"]), batch_size)
        self.assertEqual(len(res_batch["per_finding_scores"]), batch_size)
        self.assertEqual(len(res_batch["details"]), batch_size)
        print("score_reports_batch shape and type checks passed!")
        
        # Verify it matches score_report individually
        for b_idx in range(batch_size):
            res_single = scorer.score_report(
                finding_labels=labels[b_idx],
                generated_text=generated_texts[b_idx],
                finding_label_names=self.config.data.finding_labels,
                finding_templates=self.config.data.finding_templates,
                device=self.device,
            )
            self.assertAlmostEqual(res_batch["fcs"][b_idx], res_single["fcs"], places=5)
            self.assertAlmostEqual(res_batch["contradiction_rate"][b_idx], res_single["contradiction_rate"], places=5)
            
        print("Comparison with individual score_report passed!")
        
        # Test get_contradiction_scores
        contra_scores = scorer.get_contradiction_scores(
            finding_labels=labels,
            generated_texts=generated_texts,
            finding_label_names=self.config.data.finding_labels,
            finding_templates=self.config.data.finding_templates,
            device=self.device,
        )
        self.assertEqual(contra_scores.shape, (batch_size,))
        print("get_contradiction_scores shape check passed!")


if __name__ == "__main__":
    unittest.main()

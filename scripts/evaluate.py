"""
Evaluation script for the Multimodal Medical Report Generation system.

Usage:
    python scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/best.pt

Orchestrates:
    1. Loads test DataLoader.
    2. Loads trained model from checkpoint (best.pt).
    3. Runs autoregressive beam search report generation on the test set.
    4. Computes NLG metrics (BLEU-1..4, ROUGE-L).
    5. Computes Factual Consistency Score (FCS) & Contradiction Rate.
    6. Displays sample ground-truth vs. generated reports.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, setup_logging, get_device, set_seed
from data.tokenizer import ReportTokenizer
from data.dataset import create_dataloaders
from models.report_generator import MultimodalReportGenerator
from evaluation.metrics import compute_nlg_metrics
from evaluation.factual_scorer import FactualConsistencyEvaluator

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Report Generator")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--num-samples-to-show", type=int, default=5,
        help="Number of sample report comparisons to display"
    )
    args = parser.parse_args()

    # 1. Setup logging & seed
    config = load_config(args.config)
    setup_logging(log_dir=config.training.log_dir, experiment_name="evaluation")
    set_seed(config.training.seed)
    device = get_device()

    # 2. Initialize tokenizer & dataloaders
    logger.info("Initializing Tokenizer and DataLoaders...")
    tokenizer = ReportTokenizer(
        model_name=config.decoder.model_name,
        max_length=config.data.max_report_length
    )
    vocab_size = tokenizer.vocab_size
    _, _, test_loader = create_dataloaders(config, tokenizer)

    # 3. Assemble model & load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}!")
        sys.exit(1)

    logger.info(f"Loading trained model checkpoint from {checkpoint_path}...")
    model = MultimodalReportGenerator(config, vocab_size).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Successfully loaded checkpoint from epoch {ckpt.get('epoch', 0) + 1}.")

    # 4. Generate reports on test set
    logger.info("=" * 60)
    logger.info("Running autoregressive report generation on test set...")
    logger.info("=" * 60)

    all_hypotheses = []
    all_references = []
    all_finding_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Evaluating Test Set]"):
            frontal_img = batch["frontal_img"].to(device)
            lateral_img = batch["lateral_img"].to(device)
            lateral_mask = batch["lateral_mask"].to(device)
            labels = batch["labels"].to(device)
            gt_texts = batch["report_text"]

            # Autoregressive generation
            gen_outputs = model.generate(
                frontal_img=frontal_img,
                lateral_img=lateral_img,
                lateral_mask=lateral_mask,
                labels=labels,
                tokenizer=tokenizer,
                max_length=config.decoder.max_gen_length,
            )

            generated_texts = gen_outputs["generated_texts"]

            all_hypotheses.extend(generated_texts)
            all_references.extend(gt_texts)
            all_finding_labels.append(labels.cpu())

    all_finding_labels = torch.cat(all_finding_labels, dim=0)

    # 5. Compute NLG Metrics (BLEU & ROUGE)
    logger.info("Computing NLG Evaluation Metrics (BLEU-1..4, ROUGE-L)...")
    nlg_metrics = compute_nlg_metrics(all_hypotheses, all_references)

    # 6. Compute Factual Consistency Metrics (FCS & Contradiction Rate)
    logger.info("Evaluating Factual Consistency Score (FCS) via NLI...")
    fcs_evaluator = FactualConsistencyEvaluator(
        nli_scorer=model.nli_scorer,
        finding_labels=config.data.finding_labels,
        finding_templates=config.data.finding_templates,
        device=device,
    )
    fcs_results = fcs_evaluator.evaluate_reports(
        generated_texts=all_hypotheses,
        finding_labels_batch=all_finding_labels,
    )

    # 7. Print Official Results Table
    logger.info("=" * 60)
    logger.info("OFFICIAL TEST SET EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  BLEU-1                   : {nlg_metrics['bleu_1']:.4f}")
    logger.info(f"  BLEU-2                   : {nlg_metrics['bleu_2']:.4f}")
    logger.info(f"  BLEU-3                   : {nlg_metrics['bleu_3']:.4f}")
    logger.info(f"  BLEU-4                   : {nlg_metrics['bleu_4']:.4f}  (Target: ≥ 0.10)")
    logger.info(f"  ROUGE-L                  : {nlg_metrics['rouge_l']:.4f}  (Target: ≥ 0.25)")
    logger.info(f"  Factual Consistency (FCS): {fcs_results['factual_consistency_score']:.4f}")
    logger.info(f"  Contradiction Rate (CR)  : {fcs_results['contradiction_rate']:.4f}")
    logger.info("=" * 60)

    # 8. Print Sample Report Comparisons
    num_samples = min(args.num_samples_to_show, len(all_hypotheses))
    logger.info(f"\nShowing {num_samples} Sample Report Comparisons:\n")

    for i in range(num_samples):
        logger.info(f"--- Sample {i+1} ---")
        logger.info(f"Ground Truth : {all_references[i]}")
        logger.info(f"Generated    : {all_hypotheses[i]}")
        logger.info("")


if __name__ == "__main__":
    main()

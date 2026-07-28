"""
Pretrain NLI Scorer Entry Script.

Usage:
    python scripts/pretrain_nli.py --config config/config.yaml

This script runs the standalone NLI pretraining process to initialize the
FactualConsistencyScorer using synthetic entailment/contradiction pairs
extracted from the dataset reports.
"""

import argparse
import logging
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, setup_logging, get_device, set_seed
from models.nli_scorer import FactualConsistencyScorer
from training.nli_pretrainer import build_synthetic_nli_pairs, pretrain_nli_scorer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pretrain NLI Scorer Head")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # Setup loggers
    setup_logging()
    
    # Load config and enforce parameters
    config = load_config(args.config)
    set_seed(config.training.seed)
    
    device = get_device()
    
    logger.info("Starting NLI Pretraining setup...")
    
    # Initialize the NLI model
    model = FactualConsistencyScorer(
        model_name=config.nli_scorer.model_name,
        hidden_dim=config.nli_scorer.hidden_dim,
        num_classes=config.nli_scorer.num_classes,
        dropout=config.nli_scorer.dropout
    )
    
    # Generate path and files
    annotations_file = Path(config.data.data_dir) / "processed" / "annotations.json"
    if not annotations_file.exists():
        # Check if dummy sample needs to be created
        logger.warning(f"Annotations file not found at {annotations_file}. Creating synthetic sample data first...")
        from scripts.download_data import create_sample_data
        create_sample_data(config.data.data_dir, num_samples=60)
        
    finding_labels = config.data.finding_labels
    finding_templates = config.data.finding_templates
    
    # Generate synthetic NLI pairs
    pairs = build_synthetic_nli_pairs(
        annotations_file=annotations_file,
        finding_labels=finding_labels,
        finding_templates=finding_templates
    )
    
    if not pairs:
        logger.error("No NLI pairs generated! Make sure annotations file contains reports.")
        sys.exit(1)
        
    # Split into train/validation NLI pairs
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.15, random_state=42)
    logger.info(f"NLI pretraining size — Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Setup checkpoint path
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    nli_save_path = str(checkpoint_dir / "nli_scorer_best.pt")
    
    # Train
    pretrain_nli_scorer(
        model=model,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        epochs=config.nli_scorer.pretrain_epochs,
        lr=float(config.nli_scorer.pretrain_lr),
        batch_size=config.nli_scorer.pretrain_batch_size,
        device=device,
        save_path=nli_save_path
    )
    
    logger.info(f"NLI Pretraining completed successfully. Best checkpoint saved to {nli_save_path}")


if __name__ == "__main__":
    main()

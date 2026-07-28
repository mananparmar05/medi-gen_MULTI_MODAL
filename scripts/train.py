"""
Main Training Script for the Multimodal Medical Report Generation system.

Usage:
    python scripts/train.py --config config/config.yaml

Orchestrates:
    - Loading configurations and setting random seeds.
    - Initializing the ReportTokenizer and building DataLoaders.
    - Assembling the MultimodalReportGenerator network.
    - Loading the pretrained Factual Consistency Scorer (NLI) checkpoint.
    - Training the generator under the Consistency-Weighted curriculum loss.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, setup_logging, get_device, set_seed
from data.tokenizer import ReportTokenizer
from data.dataset import create_dataloaders
from models.report_generator import MultimodalReportGenerator
from training.trainer import ReportGenerationTrainer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Report Generator")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint file to resume training from (e.g. checkpoints/checkpoint_latest.pt)"
    )
    args = parser.parse_args()
    
    # 1. Logging setup
    config = load_config(args.config)
    setup_logging(log_dir=config.training.log_dir, experiment_name="main_training")
    
    # 2. Set seed
    set_seed(config.training.seed)
    
    device = get_device()
    
    # 3. Initialize tokenizer
    logger.info("Initializing custom Report Tokenizer...")
    tokenizer = ReportTokenizer(
        model_name=config.decoder.model_name,
        max_length=config.data.max_report_length
    )
    vocab_size = tokenizer.vocab_size
    
    # 4. Create dataloaders
    logger.info("Creating dataset and dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, tokenizer)
    
    # 5. Initialize the generator model
    logger.info("Assembling Multimodal Report Generator pipeline...")
    model = MultimodalReportGenerator(config, vocab_size)
    
    # 6. Load pretrained NLI scorer weights
    checkpoint_dir = Path(config.training.checkpoint_dir)
    nli_path = checkpoint_dir / "nli_scorer_best.pt"
    
    if nli_path.exists():
        logger.info(f"Loading pre-trained NLI Scorer checkpoint from {nli_path}...")
        nli_ckpt = torch.load(nli_path, map_location="cpu")
        model.nli_scorer.load_state_dict(nli_ckpt["model_state_dict"])
        logger.info("NLI Scorer loaded successfully.")
    else:
        logger.warning(
            f"Pre-trained NLI Scorer checkpoint not found at {nli_path}! "
            f"Factual consistency penalty will be computed using random weights. "
            f"It is highly recommended to run scripts/pretrain_nli.py first."
        )
        
    # 7. Initialize Trainer
    trainer = ReportGenerationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
        device=device
    )
    
    # 8. Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.resume_from_checkpoint(args.resume)
    
    # 9. Start training process
    logger.info("=" * 60)
    logger.info("Starting generator model training loop...")
    logger.info("=" * 60)
    
    try:
        trainer.train(start_epoch=start_epoch)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Exiting gracefully.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during training: {e}")
        raise e
        
    logger.info("Generator model training complete.")


if __name__ == "__main__":
    main()

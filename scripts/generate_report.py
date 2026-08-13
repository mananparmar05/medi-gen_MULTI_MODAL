"""
Single Image Report Generation & Inference Script.

Usage:
    python scripts/generate_report.py --sample-idx 0
    python scripts/generate_report.py --image path/to/frontal.png

Orchestrates:
    1. Loads trained model checkpoint (best.pt).
    2. Takes an input X-ray image + finding labels.
    3. Runs model.generate() to produce a medical radiology report.
    4. Evaluates factual consistency score on the generated report.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, setup_logging, get_device, set_seed
from data.tokenizer import ReportTokenizer
from data.dataset import create_dataloaders
from data.augmentation import get_eval_transforms
from models.report_generator import MultimodalReportGenerator

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate Radiology Report for an X-Ray")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--sample-idx", type=int, default=0,
        help="Sample index from the test set to run inference on"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Optional path to a custom frontal chest X-ray image"
    )
    args = parser.parse_args()

    # 1. Setup
    config = load_config(args.config)
    setup_logging()
    device = get_device()

    # 2. Tokenizer & Dataloaders
    tokenizer = ReportTokenizer(
        model_name=config.decoder.model_name,
        max_length=config.data.max_report_length
    )
    vocab_size = tokenizer.vocab_size

    # 3. Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}!")
        sys.exit(1)

    logger.info(f"Loading trained model from {checkpoint_path}...")
    model = MultimodalReportGenerator(config, vocab_size).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.image:
        # Custom image input
        img_path = Path(args.image)
        if not img_path.exists():
            logger.error(f"Image not found at {img_path}!")
            sys.exit(1)

        transform = get_eval_transforms(config.data.image_size)
        raw_img = Image.open(img_path).convert("RGB")
        frontal_img = transform(raw_img).unsqueeze(0).to(device)
        lateral_img = torch.zeros_like(frontal_img).to(device)
        lateral_mask = torch.tensor([0.0], device=device)
        labels = torch.zeros(1, 14, device=device)  # Default zero labels for unannotated custom image
        gt_report = "N/A (Custom Image)"
    else:
        # Select sample from test dataset
        _, _, test_loader = create_dataloaders(config, tokenizer)
        test_dataset = test_loader.dataset
        sample_idx = min(args.sample_idx, len(test_dataset) - 1)
        sample = test_dataset[sample_idx]

        frontal_img = sample["frontal_img"].unsqueeze(0).to(device)
        lateral_img = sample["lateral_img"].unsqueeze(0).to(device)
        lateral_mask = sample["lateral_mask"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)
        gt_report = sample["report_text"]

    # 4. Run Generation
    with torch.no_grad():
        gen_outputs = model.generate(
            frontal_img=frontal_img,
            lateral_img=lateral_img,
            lateral_mask=lateral_mask,
            labels=labels,
            tokenizer=tokenizer,
            max_length=config.decoder.max_gen_length,
        )

        generated_text = gen_outputs["generated_texts"][0]

        # Evaluate NLI factual consistency on this single report
        nli_results = model.nli_scorer.score_reports_batch(
            finding_labels=labels,
            generated_texts=[generated_text],
            finding_label_names=config.data.finding_labels,
            finding_templates=config.data.finding_templates,
            device=device,
        )

        fcs = nli_results["fcs"][0]
        cr = nli_results["contradiction_rate"][0]

    # 5. Display Output
    print("\n" + "=" * 60)
    print("RADIOLOGY REPORT GENERATION INFERENCE")
    print("=" * 60)
    print(f"Ground Truth Report : {gt_report}")
    print("-" * 60)
    print(f"Generated Report    : {generated_text}")
    print("-" * 60)
    print(f"Factual Consistency Score (FCS) : {fcs:.4f}")
    print(f"Contradiction Rate (CR)         : {cr:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

"""
Ingest the IU X-Ray archive data into the pipeline's expected format.

Reads:
    - archive (1) 2/indiana_reports.csv      — report text + MeSH tags
    - archive (1) 2/indiana_projections.csv   — image filenames + view type
    - archive (1) 2/images/images_normalized/ — actual PNG images

Produces:
    - data/iu_xray/images/<filename>.png      — copied image files
    - data/iu_xray/processed/annotations.json — pipeline-ready annotations

Usage:
    python scripts/ingest_archive.py --config config/config.yaml
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, setup_logging

logger = logging.getLogger(__name__)

# ============================================================================
# Archive directory name (relative to project root)
# ============================================================================
ARCHIVE_DIR = "archive (1) 2"


def parse_mesh_tags(mesh_string: str) -> list:
    """
    Parse the MeSH column from indiana_reports.csv.

    Format examples:
        "normal"
        "Cardiomegaly/borderline;Pulmonary Artery/enlarged"
        "Pulmonary Disease, Chronic Obstructive;Bullous Emphysema"

    Returns a list of cleaned MeSH term strings.
    """
    if not mesh_string or str(mesh_string).strip().lower() == "nan":
        return []

    tags = []
    for part in str(mesh_string).split(";"):
        part = part.strip()
        if not part:
            continue
        # Take the base term before any slash-qualified modifiers
        # e.g. "Cardiomegaly/borderline" → "Cardiomegaly"
        base_term = part.split("/")[0].strip()
        if base_term:
            tags.append(base_term)
    return tags


def load_reports(csv_path: Path) -> dict:
    """
    Load indiana_reports.csv into a dict keyed by uid.

    Returns:
        {uid: {findings, impression, report, mesh_tags, problems}}
    """
    reports = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = str(row.get("uid", "")).strip()
            if not uid:
                continue

            findings = (row.get("findings") or "").strip()
            impression = (row.get("impression") or "").strip()
            report = f"{findings} {impression}".strip()

            mesh_tags = parse_mesh_tags(row.get("MeSH", ""))

            reports[uid] = {
                "findings": findings,
                "impression": impression,
                "report": report,
                "mesh_tags": mesh_tags,
                "indication": (row.get("indication") or "").strip(),
            }
    return reports


def load_projections(csv_path: Path) -> dict:
    """
    Load indiana_projections.csv and group images by uid.

    Returns:
        {uid: {"frontal": [filename, ...], "lateral": [filename, ...]}}
    """
    projections = defaultdict(lambda: {"frontal": [], "lateral": []})
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = str(row.get("uid", "")).strip()
            filename = (row.get("filename") or "").strip()
            projection = (row.get("projection") or "").strip().lower()

            if not uid or not filename:
                continue

            if "lateral" in projection:
                projections[uid]["lateral"].append(filename)
            else:
                # Default to frontal for "Frontal" or anything else
                projections[uid]["frontal"].append(filename)
    return dict(projections)


def ingest(config) -> None:
    """Main ingestion pipeline."""
    project_root = Path(__file__).parent.parent
    archive_path = project_root / ARCHIVE_DIR
    data_dir = Path(config.data.data_dir)

    # Source paths
    reports_csv = archive_path / "indiana_reports.csv"
    projections_csv = archive_path / "indiana_projections.csv"
    source_images_dir = archive_path / "images" / "images_normalized"

    # Destination paths
    dest_images_dir = data_dir / "images"
    dest_processed_dir = data_dir / "processed"

    # Validate source files exist
    for p in [reports_csv, projections_csv, source_images_dir]:
        if not p.exists():
            logger.error(f"Missing required path: {p}")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("IU X-Ray Archive Ingestion")
    logger.info("=" * 60)

    # Step 1: Load CSVs
    logger.info(f"Loading reports from {reports_csv}...")
    reports = load_reports(reports_csv)
    logger.info(f"  → {len(reports)} reports loaded")

    logger.info(f"Loading projections from {projections_csv}...")
    projections = load_projections(projections_csv)
    logger.info(f"  → {len(projections)} studies with image mappings")

    # Step 2: Prepare destination directories
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_processed_dir.mkdir(parents=True, exist_ok=True)

    # Clear old synthetic images (if any)
    old_images = list(dest_images_dir.glob("CXR*_frontal.png")) + \
                 list(dest_images_dir.glob("CXR*_lateral.png"))
    if old_images:
        logger.info(f"Removing {len(old_images)} old synthetic images...")
        for img in old_images:
            img.unlink()

    # Step 3: Build annotations and copy images
    logger.info("Building annotations and copying images...")
    annotations = []
    images_copied = 0
    images_skipped = 0
    reports_skipped_no_text = 0
    reports_skipped_no_frontal = 0

    # Build set of available source images
    available_images = set()
    for img_file in source_images_dir.iterdir():
        if img_file.suffix.lower() == ".png":
            available_images.add(img_file.name)

    logger.info(f"  → {len(available_images)} source images available")

    for uid in sorted(reports.keys(), key=lambda x: int(x) if x.isdigit() else x):
        report_data = reports[uid]

        # Skip reports with no text content
        if not report_data["report"]:
            reports_skipped_no_text += 1
            continue

        # Get image filenames for this study
        proj = projections.get(uid, {"frontal": [], "lateral": []})

        # Find a valid frontal image
        frontal_file = ""
        for fname in proj["frontal"]:
            if fname in available_images:
                frontal_file = fname
                break

        if not frontal_file:
            reports_skipped_no_frontal += 1
            continue

        # Find a valid lateral image (optional)
        lateral_file = ""
        for fname in proj["lateral"]:
            if fname in available_images:
                lateral_file = fname
                break

        # Copy images to destination
        for fname in [frontal_file, lateral_file]:
            if not fname:
                continue
            src = source_images_dir / fname
            dst = dest_images_dir / fname
            if not dst.exists():
                shutil.copy2(src, dst)
                images_copied += 1
            else:
                images_skipped += 1

        # Build annotation entry
        annotation = {
            "report_id": f"uid_{uid}",
            "frontal_image": f"images/{frontal_file}",
            "lateral_image": f"images/{lateral_file}" if lateral_file else "",
            "findings": report_data["findings"],
            "impression": report_data["impression"],
            "report": report_data["report"],
            "mesh_tags": report_data["mesh_tags"],
        }
        annotations.append(annotation)

    # Step 4: Save annotations.json
    output_file = dest_processed_dir / "annotations.json"
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=2)

    # Also remove any old split files so the dataset class re-splits
    for split_file in dest_processed_dir.glob("*_annotations.json"):
        split_file.unlink()
        logger.info(f"  Removed stale split file: {split_file.name}")

    # Step 5: Report statistics
    logger.info("=" * 60)
    logger.info("Ingestion Summary")
    logger.info("=" * 60)
    logger.info(f"  Total reports in CSV:        {len(reports)}")
    logger.info(f"  Skipped (no text):           {reports_skipped_no_text}")
    logger.info(f"  Skipped (no frontal image):  {reports_skipped_no_frontal}")
    logger.info(f"  Final annotations:           {len(annotations)}")
    logger.info(f"  Images copied:               {images_copied}")
    logger.info(f"  Images already present:      {images_skipped}")
    logger.info(f"  Annotations saved to:        {output_file}")

    # Count lateral availability
    has_lateral = sum(1 for a in annotations if a["lateral_image"])
    logger.info(f"  Studies with lateral view:    {has_lateral} / {len(annotations)} "
                f"({100*has_lateral/max(len(annotations),1):.1f}%)")

    # Print label distribution
    _print_label_stats(annotations)

    logger.info("=" * 60)
    logger.info("Done! Next steps:")
    logger.info("  1. python scripts/pretrain_nli.py --config config/config.yaml")
    logger.info("  2. python scripts/train.py --config config/config.yaml")
    logger.info("=" * 60)


def _print_label_stats(annotations: list) -> None:
    """Print finding label distribution across the real dataset."""
    from data.dataset import extract_labels_from_report

    finding_names = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices",
    ]

    counts = [0] * 14
    for ann in annotations:
        labels = extract_labels_from_report(
            ann["report"], ann.get("mesh_tags", [])
        )
        for i in range(14):
            counts[i] += int(labels[i].item())

    logger.info("")
    logger.info("=== Label Distribution ===")
    for name, count in zip(finding_names, counts):
        pct = 100 * count / max(len(annotations), 1)
        bar = "█" * int(pct / 2)
        logger.info(f"  {name:30s}: {count:5d} ({pct:5.1f}%) {bar}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ingest IU X-Ray archive data into pipeline format"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ingest(config)


if __name__ == "__main__":
    main()

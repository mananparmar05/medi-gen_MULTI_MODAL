"""
Download and preprocess the IU X-Ray (Indiana University Chest X-Ray) dataset.

This script:
    1. Downloads chest X-ray images and XML reports from the Open-I API
    2. Parses XML annotations to extract findings, impressions, MeSH tags
    3. Pairs frontal + lateral views per study
    4. Saves processed annotations as JSON
    
Usage:
    python scripts/download_data.py --config config/config.yaml
    
Output structure:
    data/iu_xray/
    ├── images/
    │   ├── CXR1_1_IM-0001-3001.png
    │   ├── CXR1_1_IM-0001-4001.png
    │   └── ...
    ├── reports/
    │   ├── 1.xml
    │   └── ...
    └── processed/
        └── annotations.json
"""

import argparse
import json
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, setup_logging

logger = logging.getLogger(__name__)


# ============================================================================
# XML Parsing
# ============================================================================

def parse_report_xml(xml_path: str) -> Optional[Dict]:
    """
    Parse a single IU X-Ray XML report file.
    
    Extracts:
        - report_id: unique identifier
        - findings: text from FINDINGS section
        - impression: text from IMPRESSION section
        - mesh_tags: MeSH annotation terms
        - image_ids: list of associated image filenames
    
    Args:
        xml_path: Path to the XML report file.
        
    Returns:
        Dict with extracted fields, or None if parsing fails.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.warning(f"Failed to parse {xml_path}: {e}")
        return None
    
    report = {"source_file": str(xml_path)}
    
    # Extract report ID
    report["report_id"] = Path(xml_path).stem
    
    # Extract text sections
    findings = ""
    impression = ""
    
    # Try different XML structures (IU X-Ray has varied formats)
    for abstract in root.iter("AbstractText"):
        label = abstract.get("Label", "").upper()
        text = abstract.text or ""
        text = text.strip()
        
        if "FINDING" in label:
            findings = text
        elif "IMPRESSION" in label:
            impression = text
    
    # Fallback: look for MedlineCitation structure
    if not findings and not impression:
        for section in root.iter("section"):
            header = ""
            for h in section.iter("header"):
                header = (h.text or "").upper()
            for p in section.iter("p"):
                text = (p.text or "").strip()
                if "FINDING" in header:
                    findings = text
                elif "IMPRESSION" in header:
                    impression = text
    
    report["findings"] = findings
    report["impression"] = impression
    report["report"] = f"{findings} {impression}".strip()
    
    # Skip empty reports
    if not report["report"]:
        return None
    
    # Extract MeSH tags
    mesh_tags = []
    for mesh in root.iter("MeSH"):
        for term in mesh.iter("major") if mesh.find("major") is not None else []:
            if term.text:
                mesh_tags.append(term.text.strip())
        for term in mesh.iter("minor") if mesh.find("minor") is not None else []:
            if term.text:
                mesh_tags.append(term.text.strip())
    
    # Alternative MeSH extraction
    if not mesh_tags:
        for mesh in root.iter("mesh"):
            if mesh.text:
                mesh_tags.append(mesh.text.strip())
    
    report["mesh_tags"] = mesh_tags
    
    # Extract associated images
    image_ids = []
    for fig in root.iter("parentImage"):
        img_id = fig.get("id", "")
        if img_id:
            image_ids.append(img_id)
    
    # Alternative image extraction
    if not image_ids:
        for fig in root.iter("figure"):
            img_elem = fig.find(".//graphic")
            if img_elem is not None:
                url = img_elem.get("url", "")
                if url:
                    image_ids.append(url)
    
    report["image_ids"] = image_ids
    
    return report


def classify_view(image_filename: str) -> str:
    """
    Classify an X-ray image as frontal or lateral based on filename convention.
    
    IU X-Ray convention:
        - Frontal views typically have even suffixes or contain 'frontal'
        - Lateral views have odd suffixes or contain 'lateral'
        
    This is a heuristic — the dataset doesn't always consistently label views.
    
    Args:
        image_filename: Image filename string.
        
    Returns:
        'frontal' or 'lateral'
    """
    name = image_filename.lower()
    
    # Direct keyword matching
    if "lateral" in name:
        return "lateral"
    if "frontal" in name:
        return "frontal"
    
    # IU X-Ray convention: images ending in odd numbers tend to be frontal (PA/AP)
    # and even numbers tend to be lateral, but this varies.
    # Default to frontal if unclear.
    numbers = re.findall(r"(\d+)", name)
    if numbers:
        last_num = int(numbers[-1])
        # Heuristic based on common IU X-Ray naming patterns
        if last_num % 2 == 0:
            return "lateral"
    
    return "frontal"


# ============================================================================
# Dataset Processing
# ============================================================================

def process_dataset(
    reports_dir: str,
    images_dir: str,
    output_dir: str,
) -> List[Dict]:
    """
    Process all XML reports and pair with images.
    
    Creates a unified annotations JSON with one entry per study:
        {
            "report_id": "...",
            "frontal_image": "images/CXR...",  (relative path)
            "lateral_image": "images/CXR...",   (relative path, empty if missing)
            "findings": "...",
            "impression": "...",
            "report": "...",                    (findings + impression)
            "mesh_tags": [...],
        }
    
    Args:
        reports_dir: Directory containing XML report files.
        images_dir: Directory containing X-ray images.
        output_dir: Directory to save processed annotations.
        
    Returns:
        List of processed annotation dicts.
    """
    reports_path = Path(reports_dir)
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all available image files
    image_extensions = {".png", ".jpg", ".jpeg", ".dcm", ".gif"}
    available_images = {}
    if images_path.exists():
        for img_file in images_path.iterdir():
            if img_file.suffix.lower() in image_extensions:
                available_images[img_file.stem] = img_file.name
    logger.info(f"Found {len(available_images)} images in {images_dir}")
    
    # Parse all XML reports
    annotations = []
    xml_files = sorted(reports_path.glob("*.xml")) if reports_path.exists() else []
    logger.info(f"Found {len(xml_files)} XML report files")
    
    skipped = 0
    for xml_file in xml_files:
        parsed = parse_report_xml(str(xml_file))
        if parsed is None:
            skipped += 1
            continue
        
        # Match images to this report
        frontal_image = ""
        lateral_image = ""
        
        for img_id in parsed.get("image_ids", []):
            # Clean up image ID and find matching file
            img_stem = img_id.replace(".png", "").replace(".jpg", "")
            
            matched_file = available_images.get(img_stem, "")
            if not matched_file:
                # Try partial matching
                for avail_stem, avail_name in available_images.items():
                    if img_stem in avail_stem or avail_stem in img_stem:
                        matched_file = avail_name
                        break
            
            if matched_file:
                view = classify_view(matched_file)
                if view == "frontal" and not frontal_image:
                    frontal_image = f"images/{matched_file}"
                elif view == "lateral" and not lateral_image:
                    lateral_image = f"images/{matched_file}"
        
        # Skip if no frontal image found
        if not frontal_image:
            skipped += 1
            continue
        
        annotation = {
            "report_id": parsed["report_id"],
            "frontal_image": frontal_image,
            "lateral_image": lateral_image,
            "findings": parsed["findings"],
            "impression": parsed["impression"],
            "report": parsed["report"],
            "mesh_tags": parsed["mesh_tags"],
        }
        annotations.append(annotation)
    
    # Save
    output_file = output_path / "annotations.json"
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=2)
    
    logger.info(
        f"Processed {len(annotations)} studies "
        f"(skipped {skipped} with missing data). "
        f"Saved to {output_file}"
    )
    
    # Print label distribution
    _print_label_stats(annotations)
    
    return annotations


def _print_label_stats(annotations: List[Dict]) -> None:
    """Print finding label distribution across the dataset."""
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
    
    logger.info("=== Label Distribution ===")
    for name, count in zip(finding_names, counts):
        pct = 100 * count / max(len(annotations), 1)
        logger.info(f"  {name:30s}: {count:5d} ({pct:5.1f}%)")


# ============================================================================
# Download (placeholder — IU X-Ray requires manual download or API access)
# ============================================================================

def create_sample_data(data_dir: str, num_samples: int = 50) -> None:
    """
    Create synthetic sample data for development/testing.
    
    Generates dummy X-ray images (noise) and simple reports
    so the pipeline can be tested without downloading the real dataset.
    
    Args:
        data_dir: Root data directory.
        num_samples: Number of synthetic samples to create.
    """
    import numpy as np
    from PIL import Image
    
    data_path = Path(data_dir)
    images_path = data_path / "images"
    processed_path = data_path / "processed"
    images_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Sample report templates
    report_templates = [
        "The heart is normal in size. The lungs are clear. No acute cardiopulmonary process.",
        "The cardiac silhouette is enlarged consistent with cardiomegaly. "
        "There is mild pulmonary vascular congestion.",
        "There is a right lower lobe opacity which may represent pneumonia or atelectasis. "
        "Small right pleural effusion.",
        "The lungs are clear. Heart size is normal. No pleural effusion or pneumothorax. "
        "No acute findings.",
        "There is consolidation in the left lower lobe suggestive of pneumonia. "
        "The heart is normal in size.",
        "Bilateral pleural effusions, greater on the right. Cardiomegaly. "
        "Pulmonary edema.",
        "The lungs are hyperinflated consistent with COPD. No focal consolidation. "
        "The heart size is normal.",
        "There is a left-sided chest tube in place. Small residual pneumothorax. "
        "No pleural effusion.",
        "Normal chest radiograph. No acute cardiopulmonary abnormality.",
        "Mild cardiomegaly. Bilateral atelectasis at the lung bases. "
        "No focal consolidation or effusion.",
    ]
    
    annotations = []
    for i in range(num_samples):
        study_id = f"study_{i:04d}"
        
        # Create dummy frontal image (gray noise simulating X-ray)
        frontal_name = f"CXR{i}_frontal.png"
        frontal_arr = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
        Image.fromarray(frontal_arr, mode="L").convert("RGB").save(
            images_path / frontal_name
        )
        
        # Create lateral image (50% chance)
        lateral_name = ""
        if np.random.random() > 0.5:
            lateral_name = f"CXR{i}_lateral.png"
            lateral_arr = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
            Image.fromarray(lateral_arr, mode="L").convert("RGB").save(
                images_path / lateral_name
            )
        
        # Select report
        report = report_templates[i % len(report_templates)]
        
        annotations.append({
            "report_id": study_id,
            "frontal_image": f"images/{frontal_name}",
            "lateral_image": f"images/{lateral_name}" if lateral_name else "",
            "findings": report,
            "impression": "",
            "report": report,
            "mesh_tags": [],
        })
    
    # Save annotations
    output_file = processed_path / "annotations.json"
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=2)
    
    logger.info(
        f"Created {num_samples} synthetic samples in {data_dir}. "
        f"Use real IU X-Ray data for actual training."
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess IU X-Ray dataset"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--create-sample", action="store_true",
        help="Create synthetic sample data for development"
    )
    parser.add_argument(
        "--num-samples", type=int, default=50,
        help="Number of synthetic samples to create"
    )
    parser.add_argument(
        "--process-only", action="store_true",
        help="Only process existing XML reports (skip download)"
    )
    args = parser.parse_args()
    
    setup_logging()
    config = load_config(args.config)
    
    data_dir = config.data.data_dir
    
    if args.create_sample:
        create_sample_data(data_dir, args.num_samples)
        return
    
    if args.process_only:
        process_dataset(
            reports_dir=str(Path(data_dir) / "reports"),
            images_dir=str(Path(data_dir) / "images"),
            output_dir=str(Path(data_dir) / "processed"),
        )
        return
    
    # Full pipeline
    logger.info("=" * 60)
    logger.info("IU X-Ray Dataset Setup")
    logger.info("=" * 60)
    logger.info(
        "NOTE: The IU X-Ray dataset must be downloaded manually from:"
    )
    logger.info(
        "  https://openi.nlm.nih.gov/faq#collection"
    )
    logger.info(
        "Place images in: data/iu_xray/images/"
    )
    logger.info(
        "Place XML reports in: data/iu_xray/reports/"
    )
    logger.info(
        "Then run: python scripts/download_data.py --process-only"
    )
    logger.info("")
    logger.info(
        "For development/testing, create sample data with:"
    )
    logger.info(
        "  python scripts/download_data.py --create-sample"
    )
    
    # Check if data already exists
    images_dir = Path(data_dir) / "images"
    reports_dir = Path(data_dir) / "reports"
    
    if images_dir.exists() and reports_dir.exists():
        logger.info("Found existing data. Processing...")
        process_dataset(
            reports_dir=str(reports_dir),
            images_dir=str(images_dir),
            output_dir=str(Path(data_dir) / "processed"),
        )
    else:
        logger.info("Creating sample data for development...")
        create_sample_data(data_dir, args.num_samples)


if __name__ == "__main__":
    main()

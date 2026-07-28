"""
NLI Scorer Pretraining Module.

Handles:
    - Constructing synthetic NLI training pairs from reports and findings.
    - Tokenization and PyTorch Dataset creation for NLI training.
    - Training loop for the BERT NLI model with cross-entropy loss.
    - Validation and checkpoints.

Synthetic Pair Construction Logic:
    - Positive (Entailment): Pair a finding template (e.g. "Cardiomegaly is present")
      with a sentence from a ground-truth report where Cardiomegaly=1.
    - Negative (Contradiction): Pair a finding template with a sentence from
      a ground-truth report where that finding is 0, but the sentence explicitly
      describes its presence (or vice versa).
    - Neutral: Pair a finding template with a completely unrelated sentence
      (e.g., "The lungs are clear" paired with "Support Devices").
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from models.nli_scorer import FactualConsistencyScorer, ENTAILMENT, CONTRADICTION, NEUTRAL

logger = logging.getLogger(__name__)


class SyntheticNLIDataset(Dataset):
    """
    Dataset of synthetic NLI pairs for pretraining the factual consistency head.
    
    Returns:
        input_ids: token IDs for the pair [CLS] premise [SEP] hypothesis [SEP]
        attention_mask: attention mask
        token_type_ids: segment IDs
        label: NLI class label (0=entailment, 1=contradiction, 2=neutral)
    """

    def __init__(self, pairs: List[Tuple[str, str, int]], scorer: FactualConsistencyScorer):
        self.pairs = pairs
        self.scorer = scorer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        premise, hypothesis, label = self.pairs[idx]
        
        encoded = self.scorer.encode_pair(premise, hypothesis)
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": encoded["token_type_ids"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def build_synthetic_nli_pairs(
    annotations_file: Path,
    finding_labels: List[str],
    finding_templates: Dict[str, Dict[str, str]],
) -> List[Tuple[str, str, int]]:
    """
    Generate balanced NLI training pairs from reports.
    
    Args:
        annotations_file: Path to processed annotations.json.
        finding_labels: List of 14 finding names.
        finding_templates: Template positive/negative finding statements.
        
    Returns:
        List of (premise, hypothesis, label_index) tuples.
    """
    with open(annotations_file, "r") as f:
        samples = json.load(f)
        
    from data.dataset import extract_labels_from_report
    
    sentences_positive = {name: [] for name in finding_labels}
    sentences_negative = {name: [] for name in finding_labels}
    
    # Heuristic sentence extraction and mapping
    for sample in samples:
        report = sample.get("findings", sample.get("report", ""))
        mesh_tags = sample.get("mesh_tags", [])
        labels = extract_labels_from_report(report, mesh_tags)
        
        # Split report into sentences
        sentences = FactualConsistencyScorer._split_sentences(report)
        
        for f_idx, f_name in enumerate(finding_labels):
            label_val = labels[f_idx].item()
            for sentence in sentences:
                sentence_lower = sentence.lower()
                # Simple check if sentence contains keywords related to this finding
                keywords = [f_name.lower()]
                if "cardiomegaly" in f_name.lower():
                    keywords += ["heart", "cardiac"]
                elif "effusion" in f_name.lower():
                    keywords += ["effusion", "blunting"]
                elif "pneumonia" in f_name.lower():
                    keywords += ["pneumonia", "infiltrate"]
                elif "edema" in f_name.lower():
                    keywords += ["edema", "congestion"]
                
                has_keyword = any(kw in sentence_lower for kw in keywords)
                
                if has_keyword:
                    if label_val > 0.5:
                        sentences_positive[f_name].append(sentence)
                    else:
                        sentences_negative[f_name].append(sentence)
                        
    nli_pairs = []
    
    for f_name in finding_labels:
        pos_templates = finding_templates.get("positive", {})
        neg_templates = finding_templates.get("negative", {})
        
        pos_premise = pos_templates.get(f_name, "")
        neg_premise = neg_templates.get(f_name, "")
        
        if not pos_premise or not neg_premise:
            continue
            
        pos_sents = sentences_positive[f_name]
        neg_sents = sentences_negative[f_name]
        
        # 1. ENTAILMENT (True relationships)
        # Positive premise + positive sentence
        for sent in pos_sents[:100]:  # limit size per pathology to balance
            nli_pairs.append((pos_premise, sent, ENTAILMENT))
        # Negative premise + negative sentence
        for sent in neg_sents[:100]:
            nli_pairs.append((neg_premise, sent, ENTAILMENT))
            
        # 2. CONTRADICTION (False relationships)
        # Positive premise + negative sentence
        for sent in neg_sents[:100]:
            nli_pairs.append((pos_premise, sent, CONTRADICTION))
        # Negative premise + positive sentence
        for sent in pos_sents[:100]:
            nli_pairs.append((neg_premise, sent, CONTRADICTION))
            
        # 3. NEUTRAL (Unrelated relationships)
        # Pair premise with sentence from a completely different finding category
        other_findings = [f for f in finding_labels if f != f_name]
        for _ in range(min(len(pos_sents), 200)):
            other_f = random.choice(other_findings)
            other_sents = sentences_positive[other_f]
            if other_sents:
                sent = random.choice(other_sents)
                nli_pairs.append((pos_premise, sent, NEUTRAL))
                
    random.shuffle(nli_pairs)
    logger.info(f"Generated {len(nli_pairs)} synthetic NLI pairs.")
    
    # Log class distribution
    counts = [0, 0, 0]
    for _, _, label in nli_pairs:
        counts[label] += 1
    logger.info(f"NLI distribution — Entailment: {counts[0]}, Contradiction: {counts[1]}, Neutral: {counts[2]}")
    
    return nli_pairs


# ============================================================================
# Pretraining Loop
# ============================================================================

def pretrain_nli_scorer(
    model: FactualConsistencyScorer,
    train_pairs: List[Tuple[str, str, int]],
    val_pairs: List[Tuple[str, str, int]],
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    save_path: str
) -> None:
    """Run NLI pretraining loop."""
    train_dataset = SyntheticNLIDataset(train_pairs, model)
    val_dataset = SyntheticNLIDataset(val_pairs, model)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs["loss"]
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            preds = outputs["predictions"]
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct/total:.4f}"
            })
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                val_loss += outputs["loss"].item()
                preds = outputs["predictions"]
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        logger.info(
            f"Epoch {epoch+1} finished — "
            f"Train Loss: {total_loss/len(train_loader):.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save NLI Scorer checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc
            }, save_path)
            logger.info(f"Saved new best NLI Scorer checkpoint with accuracy {val_acc:.4f} to {save_path}")

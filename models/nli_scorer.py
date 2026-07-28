"""
Factual Consistency Scorer — Novel Contribution #2

BERT-based 3-way NLI (Natural Language Inference) classifier that checks 
whether generated report sentences are factually consistent with the 
structured findings.

Classification:
    Input:  (premise, hypothesis) pair
            premise   = structured finding text template
            hypothesis = generated sentence from report
    Output: [entailment_prob, contradiction_prob, neutral_prob]

Roles:
    1. Offline evaluation: Factual Consistency Score (FCS) = % entailed
    2. Online training signal: contradiction probability penalizes the 
       generator via the consistency-weighted loss

Architecture:
    BERT-base → [CLS] token → MLP head → 3-class softmax
    
    With an additional sentence-pair encoding where the finding text 
    and generated sentence are separated by [SEP].
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

logger = logging.getLogger(__name__)


# NLI class indices
ENTAILMENT = 0
CONTRADICTION = 1
NEUTRAL = 2

NLI_LABELS = ["entailment", "contradiction", "neutral"]


class FactualConsistencyScorer(nn.Module):
    """
    BERT-based NLI classifier for factual consistency scoring.
    
    Takes (finding_text, generated_sentence) pairs and classifies their 
    relationship as entailment, contradiction, or neutral.
    
    Args:
        model_name: HuggingFace BERT model name (default 'bert-base-uncased').
        hidden_dim: Classifier hidden dimension (default 256).
        num_classes: Number of NLI classes (default 3).
        dropout: Classifier dropout (default 0.1).
        max_length: Max token length for BERT inputs (default 128).
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
        max_length: int = 128,
    ):
        super().__init__()
        
        self.max_length = max_length
        self.num_classes = num_classes
        
        # Load BERT encoder
        self.bert = BertModel.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size  # 768
        
        # NLI classification head
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # BERT tokenizer for encoding finding-sentence pairs
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Initialize classifier
        self._init_classifier()
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"FactualConsistencyScorer: {model_name}, "
            f"classifier {bert_hidden}→{hidden_dim}→{num_classes} "
            f"({total_params:,} params)"
        )

    def _init_classifier(self) -> None:
        """Initialize classifier head with Kaiming init."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def encode_pair(
        self,
        premise: str,
        hypothesis: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a (premise, hypothesis) pair for BERT.
        
        Format: [CLS] premise [SEP] hypothesis [SEP]
        
        Args:
            premise: Finding text template.
            hypothesis: Generated sentence.
            
        Returns:
            Dict with 'input_ids', 'attention_mask', 'token_type_ids'.
        """
        encoding = self.bert_tokenizer(
            premise,
            hypothesis,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoding

    def encode_batch(
        self,
        premises: List[str],
        hypotheses: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of (premise, hypothesis) pairs.
        
        Args:
            premises: List of finding text templates.
            hypotheses: List of generated sentences.
            
        Returns:
            Dict with batched 'input_ids', 'attention_mask', 'token_type_ids'.
        """
        encoding = self.bert_tokenizer(
            premises,
            hypotheses,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoding

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for NLI classification.
        
        Args:
            input_ids: [B, seq_len] BERT input IDs.
            attention_mask: [B, seq_len] attention mask.
            token_type_ids: [B, seq_len] segment IDs.
            labels: [B] ground truth NLI labels (optional, for training).
            
        Returns:
            Dict with:
                - logits: [B, 3] raw class logits
                - probs: [B, 3] class probabilities (softmax)
                - loss: scalar (if labels provided)
                - predictions: [B] predicted class indices
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        
        # Classification
        logits = self.classifier(cls_output)  # [B, 3]
        probs = F.softmax(logits, dim=-1)     # [B, 3]
        predictions = logits.argmax(dim=-1)   # [B]
        
        result = {
            "logits": logits,
            "probs": probs,
            "predictions": predictions,
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss
        
        return result

    @torch.no_grad()
    def score_report(
        self,
        finding_labels: torch.Tensor,
        generated_text: str,
        finding_label_names: List[str],
        finding_templates: Dict[str, Dict[str, str]],
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Score factual consistency of a generated report against findings.
        
        For each active finding, constructs the premise text, pairs it 
        with each sentence of the generated report, and classifies.
        
        Args:
            finding_labels: [14] binary finding labels.
            generated_text: Full generated report text.
            finding_label_names: List of 14 finding names.
            finding_templates: Dict with 'positive'/'negative' templates.
            device: Compute device.
            
        Returns:
            Dict with:
                - fcs: Factual Consistency Score (% entailed)
                - contradiction_rate: % contradicted
                - per_finding_scores: Dict of per-finding scores
                - details: List of (finding, sentence, prediction) tuples
        """
        self.eval()
        
        # Split report into sentences
        sentences = self._split_sentences(generated_text)
        if not sentences:
            return {
                "fcs": 0.0,
                "contradiction_rate": 0.0,
                "per_finding_scores": {},
                "details": [],
            }
        
        all_entail = 0
        all_contradict = 0
        all_total = 0
        per_finding = {}
        details = []
        
        for f_idx, f_name in enumerate(finding_label_names):
            label_val = finding_labels[f_idx].item()
            
            # Get premise text based on label
            if label_val > 0.5:
                premise = finding_templates.get("positive", {}).get(f_name, "")
            else:
                premise = finding_templates.get("negative", {}).get(f_name, "")
            
            if not premise:
                continue
            
            finding_entail = 0
            finding_total = 0
            
            for sentence in sentences:
                if len(sentence.strip()) < 5:
                    continue
                
                # Encode pair
                encoding = self.encode_pair(premise, sentence)
                encoding = {k: v.to(device) for k, v in encoding.items()}
                
                # Classify
                output = self.forward(**encoding)
                pred = output["predictions"].item()
                probs = output["probs"][0].cpu().tolist()
                
                if pred == ENTAILMENT:
                    finding_entail += 1
                    all_entail += 1
                elif pred == CONTRADICTION:
                    all_contradict += 1
                
                finding_total += 1
                all_total += 1
                
                details.append({
                    "finding": f_name,
                    "label": label_val,
                    "sentence": sentence,
                    "prediction": NLI_LABELS[pred],
                    "probs": {
                        "entailment": probs[0],
                        "contradiction": probs[1],
                        "neutral": probs[2],
                    },
                })
            
            if finding_total > 0:
                per_finding[f_name] = finding_entail / finding_total
        
        fcs = all_entail / max(all_total, 1)
        cr = all_contradict / max(all_total, 1)
        
        return {
            "fcs": fcs,
            "contradiction_rate": cr,
            "per_finding_scores": per_finding,
            "details": details,
        }

    @torch.no_grad()
    def score_reports_batch(
        self,
        finding_labels: torch.Tensor,
        generated_texts: List[str],
        finding_label_names: List[str],
        finding_templates: Dict[str, Dict[str, str]],
        device: torch.device,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Score factual consistency of a batch of generated reports against findings.
        Batches premise-hypothesis pairs across reports to perform fast batched inference.
        """
        self.eval()
        B = len(generated_texts)
        
        # 1. Compile all pairs across all reports in the batch
        pairs = []  # list of dicts containing pair info
        report_details = [[] for _ in range(B)]
        report_pair_counts = [0] * B
        
        for b_idx, gen_text in enumerate(generated_texts):
            sentences = self._split_sentences(gen_text)
            if not sentences:
                continue
                
            for f_idx, f_name in enumerate(finding_label_names):
                label_val = finding_labels[b_idx, f_idx].item()
                
                # Get premise text based on label
                if label_val > 0.5:
                    premise = finding_templates.get("positive", {}).get(f_name, "")
                else:
                    premise = finding_templates.get("negative", {}).get(f_name, "")
                
                if not premise:
                    continue
                    
                for sentence in sentences:
                    if len(sentence.strip()) < 5:
                        continue
                    
                    pairs.append({
                        "premise": premise,
                        "hypothesis": sentence,
                        "report_idx": b_idx,
                        "finding_idx": f_idx,
                        "finding_name": f_name,
                        "label_val": label_val,
                        "sentence": sentence
                    })
                    report_pair_counts[b_idx] += 1
        
        if not pairs:
            return {
                "fcs": [0.0] * B,
                "contradiction_rate": [0.0] * B,
                "per_finding_scores": [{} for _ in range(B)],
                "details": report_details,
            }
            
        # 2. Process pairs in sub-batches
        all_preds = []
        all_probs = []
        
        for i in range(0, len(pairs), batch_size):
            sub_pairs = pairs[i : i + batch_size]
            premises = [p["premise"] for p in sub_pairs]
            hypotheses = [p["hypothesis"] for p in sub_pairs]
            
            encoding = self.encode_batch(premises, hypotheses)
            encoding = {k: v.to(device) for k, v in encoding.items()}
            
            output = self.forward(**encoding)
            preds = output["predictions"].cpu().tolist()
            probs = output["probs"].cpu().tolist()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            
        # 3. Aggregate results per report
        report_fcs_counts = [{"entail": 0, "contradict": 0, "total": 0} for _ in range(B)]
        
        for pair_idx, pair in enumerate(pairs):
            r_idx = pair["report_idx"]
            f_name = pair["finding_name"]
            label_val = pair["label_val"]
            sentence = pair["sentence"]
            pred = all_preds[pair_idx]
            probs = all_probs[pair_idx]
            
            report_fcs_counts[r_idx]["total"] += 1
            if pred == ENTAILMENT:
                report_fcs_counts[r_idx]["entail"] += 1
            elif pred == CONTRADICTION:
                report_fcs_counts[r_idx]["contradict"] += 1
                
            report_details[r_idx].append({
                "finding": f_name,
                "label": label_val,
                "sentence": sentence,
                "prediction": NLI_LABELS[pred],
                "probs": {
                    "entailment": probs[0],
                    "contradiction": probs[1],
                    "neutral": probs[2],
                },
            })
            
        fcs_scores = []
        contra_rates = []
        per_finding_scores_list = []
        
        for r_idx in range(B):
            counts = report_fcs_counts[r_idx]
            total = counts["total"]
            fcs = counts["entail"] / max(total, 1) if total > 0 else 0.0
            cr = counts["contradict"] / max(total, 1) if total > 0 else 0.0
            fcs_scores.append(fcs)
            contra_rates.append(cr)
            
            # Compute finding_entail / finding_total for each finding in this report
            finding_counts = {name: {"entail": 0, "total": 0} for name in finding_label_names}
            for d in report_details[r_idx]:
                f_name = d["finding"]
                finding_counts[f_name]["total"] += 1
                if d["prediction"] == "entailment":
                    finding_counts[f_name]["entail"] += 1
            
            per_finding_scores = {}
            for f_name in finding_label_names:
                cnts = finding_counts[f_name]
                if cnts["total"] > 0:
                    per_finding_scores[f_name] = cnts["entail"] / cnts["total"]
            
            per_finding_scores_list.append(per_finding_scores)
            
        return {
            "fcs": fcs_scores,
            "contradiction_rate": contra_rates,
            "per_finding_scores": per_finding_scores_list,
            "details": report_details,
        }

    @torch.no_grad()
    def get_contradiction_scores(
        self,
        finding_labels: torch.Tensor,
        generated_texts: List[str],
        finding_label_names: List[str],
        finding_templates: Dict[str, Dict[str, str]],
        device: torch.device,
        batch_size: int = 64,
        finding_keywords: Optional[Dict[str, List[str]]] = None,
    ) -> torch.Tensor:
        """
        Compute batch contradiction scores for training signal.
        
        Returns mean contradiction probability per sample for use in 
        consistency-weighted loss.
        """
        self.eval()
        B = len(generated_texts)
        scores = torch.zeros(B, device=device)
        
        # Compile all pairs
        pairs = []  # list of (premise, hypothesis, report_idx)
        report_pair_counts = [0] * B
        
        for b in range(B):
            sentences = self._split_sentences(generated_texts[b])
            if not sentences:
                continue
            
            for f_idx, f_name in enumerate(finding_label_names):
                label_val = finding_labels[b, f_idx].item()
                if label_val > 0.5:
                    premise = finding_templates.get("positive", {}).get(f_name, "")
                else:
                    premise = finding_templates.get("negative", {}).get(f_name, "")
                
                if not premise:
                    continue
                
                keywords = finding_keywords.get(f_name, []) if finding_keywords else []
                
                for sentence in sentences:
                    if len(sentence.strip()) < 5:
                        continue
                    
                    # Fast heuristic: if finding is negative (label <= 0.5) and sentence does NOT
                    # contain any keywords for this finding, relationship is neutral (no contradiction).
                    # Skip BERT call for massive CPU speedup.
                    if label_val <= 0.5 and keywords:
                        sent_lower = sentence.lower()
                        if not any(kw in sent_lower for kw in keywords):
                            continue
                    
                    pairs.append((premise, sentence, b))
                    report_pair_counts[b] += 1
            
        if not pairs:
            return scores
            
        # Process in batches
        all_contra_probs = []
        for i in range(0, len(pairs), batch_size):
            sub_pairs = pairs[i : i + batch_size]
            premises = [p[0] for p in sub_pairs]
            hypotheses = [p[1] for p in sub_pairs]
            
            encoding = self.encode_batch(premises, hypotheses)
            encoding = {k: v.to(device) for k, v in encoding.items()}
            
            output = self.forward(**encoding)
            contra_probs = output["probs"][:, CONTRADICTION].cpu().tolist()
            all_contra_probs.extend(contra_probs)
            
        # Accumulate per report
        pair_pointer = 0
        for b in range(B):
            count = report_pair_counts[b]
            if count > 0:
                sub_probs = all_contra_probs[pair_pointer : pair_pointer + count]
                scores[b] = sum(sub_probs) / count
                pair_pointer += count
        
        return scores

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split report text into sentences."""
        import re
        # Split on period, exclamation, question mark (but not abbreviations)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out very short fragments
        return [s.strip() for s in sentences if len(s.strip()) > 3]

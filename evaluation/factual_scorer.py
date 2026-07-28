"""
Factual Consistency Evaluator.

Provides post-hoc offline evaluation of generated report sentences compared
to ground-truth findings using the pretrained FactualConsistencyScorer NLI head.
"""

import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

from models.nli_scorer import FactualConsistencyScorer

logger = logging.getLogger(__name__)


class FactualConsistencyEvaluator:
    """
    Computes corpus-level factual consistency metrics over generated test reports.
    
    Metrics:
        - FCS (Factual Consistency Score): average ratio of entailed sentences.
        - CR (Contradiction Rate): average ratio of contradicted sentences.
        - Pathology breakdown: per-finding accuracy scores.
    """

    def __init__(
        self,
        nli_scorer: FactualConsistencyScorer,
        finding_labels: List[str],
        finding_templates: Dict[str, Dict[str, str]],
        device: torch.device,
    ):
        self.nli_scorer = nli_scorer.to(device)
        self.finding_labels = finding_labels
        self.finding_templates = finding_templates
        self.device = device

    def evaluate_reports(
        self,
        generated_texts: List[str],
        finding_labels_batch: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Evaluate factual consistency across a dataset.
        
        Args:
            generated_texts: List of B generated report text strings.
            finding_labels_batch: [B, 14] ground-truth finding labels.
            
        Returns:
            Dict containing average FCS, CR, and per-finding scores.
        """
        B = len(generated_texts)
        assert B == finding_labels_batch.size(0), "Mismatch between generated reports and labels size!"
        
        # Score reports in a batch
        results = self.nli_scorer.score_reports_batch(
            finding_labels=finding_labels_batch,
            generated_texts=generated_texts,
            finding_label_names=self.finding_labels,
            finding_templates=self.finding_templates,
            device=self.device,
        )
        
        total_fcs = results["fcs"]
        total_cr = results["contradiction_rate"]
        
        # Per finding accumulators
        finding_totals = {name: [] for name in self.finding_labels}
        for r_idx in range(B):
            for f_name, score in results["per_finding_scores"][r_idx].items():
                finding_totals[f_name].append(score)
                
        # Aggregate
        avg_fcs = float(np.mean(total_fcs)) if total_fcs else 0.0
        avg_cr = float(np.mean(total_cr)) if total_cr else 0.0
        
        per_finding_avg = {}
        for f_name, scores in finding_totals.items():
            per_finding_avg[f_name] = float(np.mean(scores)) if scores else 0.0
            
        return {
            "factual_consistency_score": avg_fcs,
            "contradiction_rate": avg_cr,
            "per_finding_breakdown": per_finding_avg,
        }

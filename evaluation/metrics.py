"""
NLG Evaluation Metrics.

Computes standard NLP translation and summarization metrics:
    - BLEU-1, BLEU-2, BLEU-3, BLEU-4
    - ROUGE-L
    - METEOR (optional fallback if nltk resources are downloaded)
"""

import logging
from typing import Dict, List

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
for resource, package in [("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab")]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package, quiet=True)


def compute_nlg_metrics(
    hypotheses: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Calculate corpus-level NLG evaluation metrics.
    
    Args:
        hypotheses: List of generated report text strings.
        references: List of target ground truth report text strings.
        
    Returns:
        Dict of computed score floats:
            bleu_1, bleu_2, bleu_3, bleu_4, rouge_l
    """
    assert len(hypotheses) == len(references), "Mismatch in size of hypotheses and references!"
    
    # Tokenize reports into lists of words for BLEU calculation
    hyp_tokens = [nltk.word_tokenize(hyp.lower()) for hyp in hypotheses]
    ref_tokens = [[nltk.word_tokenize(ref.lower())] for ref in references]
    
    # Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4
    smooth = SmoothingFunction().method1
    bleu_1 = corpus_bleu(ref_tokens, hyp_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu_2 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu_3 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu_4 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    # Calculate ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = []
    
    for hyp, ref in zip(hypotheses, references):
        if not hyp.strip() or not ref.strip():
            rouge_l_scores.append(0.0)
            continue
        scores = scorer.score(ref, hyp)
        rouge_l_scores.append(scores["rougeL"].fmeasure)
        
    avg_rouge_l = sum(rouge_l_scores) / max(len(rouge_l_scores), 1)
    
    return {
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_3": bleu_3,
        "bleu_4": bleu_4,
        "rouge_l": avg_rouge_l,
    }

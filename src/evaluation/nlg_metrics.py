"""
NLG (Natural Language Generation) metrics for Vietnamese VQA evaluation.

This module provides functions for computing various NLG metrics:
- Traditional metrics: BLEU, METEOR, ROUGE-L, CIDEr
- BERTScore with PhoBERT
- SMILE metric for answer evaluation
"""

import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

from .shared_models import SharedBERTScoreModel, SharedSMILEModel
from .text_preprocessing import (
    clean_text,
    segment_vietnamese,
    truncate_sentence,
    preprocess_vietnamese_text,
    normalize_answer,
)


# ============================================================================
# TRADITIONAL NLG METRICS
# ============================================================================

def compute_traditional_metrics(gts: dict, res: dict) -> dict[str, float]:
    """
    Compute BLEU, METEOR, ROUGE, CIDEr scores.
    
    Args:
        gts: Ground truth dict {id: [ref1, ref2, ...]}
        res: Predictions dict {id: [pred]}
        
    Returns:
        Dictionary with metric scores (scaled to 0-100)
    """
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    
    scores = {}
    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    scores[m] = float(s) * 100
            else:
                scores[method] = float(score) * 100
        except Exception:
            if isinstance(method, list):
                scores.update({m: 0.0 for m in method})
            else:
                scores[method] = 0.0
    
    return scores


# ============================================================================
# BERTSCORE
# ============================================================================

# ============================================================================
# BERTSCORE
# ============================================================================

def compute_bertscore_max_ref(hypotheses: list[str], references: list[list[str]], 
                              device: str = "cuda", model_type: str = "bert") -> list[float]:
    """
    Compute BERTScore F1 with max over multiple references.
    
    For each hypothesis, computes BERTScore against all its references
    and returns the maximum F1 score.
    
    Args:
        hypotheses: List of predicted texts
        references: List of reference lists (each sample can have multiple refs)
        device: Device for computation ("cuda" or "cpu")
        model_type: "bert" or "phobert"
        
    Returns:
        List of max F1 scores (scaled to 0-100)
    """
    if not hypotheses:
        return []
    
    bertscore = SharedBERTScoreModel.get_instance(device=device, model_type=model_type)
    max_scores = []
    
    for hyp, refs in zip(hypotheses, references):
        valid_refs = [r for r in refs if r.strip()]
        
        if not hyp.strip() or not valid_refs:
            max_scores.append(0.0)
            continue
        
        # Dynamic batch: repeat hypothesis for each reference
        batch_cands = [hyp] * len(valid_refs)
        batch_refs = valid_refs
        
        bertscore.reset()
        bertscore.update(batch_cands, batch_refs)
        result = bertscore.compute()
        
        f1_scores = result['f1'].cpu().tolist()
        if isinstance(f1_scores, float):
            f1_scores = [f1_scores]
        
        max_scores.append(max(f1_scores) * 100)
    
    return max_scores


# ============================================================================
# COMBINED NLG SCORES
# ============================================================================

def get_nlg_scores(references: list[list[str]], hypotheses: list[str], 
                   device: str = "cuda", max_len: int = 150, model_type: str = "bert") -> dict[str, float]:
    """
    Compute all NLG metrics for Vietnamese text.
    
    Includes preprocessing with Vietnamese word segmentation.
    
    Args:
        references: List of reference lists
        hypotheses: List of predictions
        device: Device for BERTScore computation
        max_len: Maximum words per text (for truncation)
        model_type: "bert" or "phobert" for BERTScore
        
    Returns:
        Dictionary with all metric scores
    """
    # Truncate texts
    hypotheses = [truncate_sentence(h, max_len) for h in hypotheses]
    references = [[truncate_sentence(r, max_len) for r in refs] for refs in references]
    
    # Preprocess Vietnamese text
    hypotheses = [preprocess_vietnamese_text(h) for h in hypotheses]
    references = [[preprocess_vietnamese_text(r) for r in refs] for refs in references]
    
    # Prepare data for traditional metrics
    gts = {i: [clean_text(r) for r in refs] for i, refs in enumerate(references)}
    res = {i: [clean_text(hyp)] for i, hyp in enumerate(hypotheses)}
    
    # Compute traditional metrics
    scores = compute_traditional_metrics(gts, res)
    
    # Compute BERTScore
    max_f1_scores = compute_bertscore_max_ref(hypotheses, references, device, model_type=model_type)
    scores["BERTScore_F1"] = (sum(max_f1_scores) / len(max_f1_scores)) if max_f1_scores else 0.0
    
    return scores


# ============================================================================
# SMILE METRIC
# ============================================================================

def compute_smile_scores(questions: list[str], gt_answers: list[str], 
                         predictions: list[str], 
                         synthetic_answers: list[str] = None,
                         model_type: str = "bert") -> dict[str, float]:
    """
    Compute SMILE scores for answer evaluation.
    
    SMILE (Sentence-level Metrics for Information-Leveraging Evaluation)
    evaluates answer quality by comparing semantic similarity and keyword overlap.
    
    Answers are normalized before evaluation to ensure variants like
    "có", "đúng", "yes", "vâng" are treated equally.
    
    Args:
        questions: List of questions
        gt_answers: List of ground truth answers
        predictions: List of predicted answers
        synthetic_answers: List of pre-generated synthetic full-sentence answers.
                           If None, ground truth answers will be used.
        model_type: "bert" or "phobert"
    
    Returns:
        Dictionary with SMILE metrics (avg, hm)
    """
    if not questions or not gt_answers or not predictions:
        return {"SMILE_avg": 0.0, "SMILE_hm": 0.0}
    
    # Normalize answers (important for yes/no questions)
    gt_answers = [normalize_answer(ans) for ans in gt_answers]
    predictions = [normalize_answer(pred) for pred in predictions]
    
    # Use ground truth answers as fallback if no synthetic answers provided
    if synthetic_answers is None:
        synthetic_answers = gt_answers
    else:
        synthetic_answers = [normalize_answer(ans) for ans in synthetic_answers]
    
    if len(synthetic_answers) != len(questions):
        print(f"Warning: synthetic_answers length ({len(synthetic_answers)}) "
              f"does not match questions length ({len(questions)}). Using GT answers.")
        synthetic_answers = gt_answers
    
    # Prepare data: segment Vietnamese text
    smile_data = []
    
    for i, (q, gt, syn_ans, pred) in enumerate(zip(questions, gt_answers, synthetic_answers, predictions)):
        if not q or not gt or not pred:
            continue
        
        q_seg = segment_vietnamese(clean_text(q))
        gt_seg = segment_vietnamese(clean_text(gt))
        syn_ans_seg = segment_vietnamese(clean_text(syn_ans))
        pred_seg = segment_vietnamese(clean_text(pred))
        
        if q_seg and gt_seg and pred_seg and syn_ans_seg:
            smile_data.append((q_seg, gt_seg, syn_ans_seg, pred_seg))
    
    if not smile_data:
        return {"SMILE_avg": 0.0, "SMILE_hm": 0.0}
    
    # Compute SMILE scores
    smile = SharedSMILEModel.get_instance(model_type=model_type)
    smile_data_array = np.array(smile_data)
    results = smile.generate_scores(smile_data_array)
    
    return {
        "SMILE_avg": float(np.mean(results['avg'])) * 100,
        "SMILE_hm": float(np.mean(results['hm'])) * 100,
    }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "compute_traditional_metrics",
    "compute_bertscore_max_ref",
    "get_nlg_scores",
    "compute_smile_scores",
]

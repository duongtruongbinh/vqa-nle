"""
VQA Evaluation Module.

This package provides evaluation utilities for Vietnamese VQA models:
- shared_models: Singleton models (BERTScore, SMILE, SyntheticAnswerGenerator)
- text_preprocessing: Text cleaning and normalization utilities
- nlg_metrics: NLG metrics computation (BLEU, METEOR, ROUGE, CIDEr, BERTScore, SMILE)
- calculate_scores: Main evaluation script
"""

from .shared_models import (
    SharedBERTScoreModel,
    SharedSMILEModel,
    SharedSyntheticAnswerGenerator,
)

from .text_preprocessing import (
    segment_vietnamese,
    clean_text,
    normalize_answer,
    normalize_explanation,
    truncate_sentence,
    ensure_list,
    preprocess_vietnamese_text,
)

from .nlg_metrics import (
    compute_traditional_metrics,
    compute_bertscore_max_ref,
    get_nlg_scores,
    compute_smile_scores,
)

from .calculate_scores import evaluate_file


__all__ = [
    # Shared models
    "SharedBERTScoreModel",
    "SharedSMILEModel",
    "SharedSyntheticAnswerGenerator",
    # Text preprocessing
    "segment_vietnamese",
    "clean_text",
    "normalize_answer",
    "normalize_explanation",
    "truncate_sentence",
    "ensure_list",
    "preprocess_vietnamese_text",
    # NLG metrics
    "compute_traditional_metrics",
    "compute_bertscore_max_ref",
    "get_nlg_scores",
    "compute_smile_scores",
    # Evaluation
    "evaluate_file",
]

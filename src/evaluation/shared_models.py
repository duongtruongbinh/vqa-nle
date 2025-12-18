"""
Shared singleton models for evaluation metrics.

This module provides singleton pattern implementations for expensive-to-initialize models:
- SharedBERTScoreModel: PhoBERT-based BERTScore for Vietnamese
- SharedSMILEModel: SMILE metric for answer evaluation  
- SharedSyntheticAnswerGenerator: LLM-based synthetic answer generation
"""

import os
import sys

import torch
from tqdm import tqdm
import bert_score

# Add SMILE metric path
SMILE_PATH = '/home/vlai-vqa-nle/minhtq/vqa-nle/smile-metric-qna-eval'
if SMILE_PATH not in sys.path:
    sys.path.append(SMILE_PATH)

# Add synthetic answer generator path
NOTEBOOKS_PATH = '/home/vlai-vqa-nle/minhtq/vqa-nle/notebooks'
if NOTEBOOKS_PATH not in sys.path:
    sys.path.insert(0, NOTEBOOKS_PATH)

from smile.smile import SMILE
from synthetic_answer_generator import (
    load_qwen_text_model,
    generate_synthetic_answer_qwen_text,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE,
)


# ============================================================================
# SHARED BERTSCORE MODEL
# ============================================================================

# ============================================================================
# SHARED BERTSCORE MODEL (using bert-score library)
# ============================================================================

class SharedBERTScoreModel:
    """
    Singleton using original bert-score library (more stable).
    
    Uses PhoBERT-base or BERT-base for BERTScore computation.
    """
    
    _scorers = {}  # Cache BERTScorer objects
    
    # Model mapping
    MODEL_MAPPING = {
        'phobert': 'vinai/phobert-base',
        'bert': 'bert-base-uncased'
    }
    
    @classmethod
    def get_scorer(cls, model_type: str = "bert", device: str = "cuda"):
        """Get or create cached BERTScorer."""
        key = (model_type, device)
        if key not in cls._scorers:
            model_name = cls.MODEL_MAPPING.get(model_type, model_type)
            
            cls._scorers[key] = bert_score.BERTScorer(
                model_type=model_name,
                num_layers=12,
                batch_size=64,
                nthreads=4,
                all_layers=False,
                idf=False,
                device=device,
                lang=None,
                rescale_with_baseline=False
            )
        
        return cls._scorers[key]
    
    @classmethod
    def compute_scores(cls, predictions: list, references: list, 
                       model_type: str = "bert", device: str = "cuda"):
        """
        Compute BERTScore with automatic caching.
        
        Returns:
            Dict with 'precision', 'recall', 'f1' (each as mean value)
        """
        scorer = cls.get_scorer(model_type, device)
        P, R, F1 = scorer.score(predictions, references)
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(), 
            'f1': F1.mean().item()
        }


# ============================================================================
# SHARED SMILE MODEL
# ============================================================================

class SharedSMILEModel:
    """
    Singleton for SMILE metric model.
    
    SMILE (Sentence-level Metrics for Information-Leveraging Evaluation)
    is used for evaluating answer quality in VQA tasks.
    """
    
    _instances = {}
    
    @classmethod
    def get_instance(cls, model_type: str = 'bert'):
        """Get or initialize shared SMILE model."""
        if model_type not in cls._instances:
            cls._instances[model_type] = SMILE(
                emb_model=model_type,
                eval_metrics=['avg', 'hm'],
                assign_bins=False,
                use_exact_matching=True,
                verbose=False
            )
        return cls._instances[model_type]


# ============================================================================
# SHARED SYNTHETIC ANSWER GENERATOR
# ============================================================================

class SharedSyntheticAnswerGenerator:
    """
    Singleton for Synthetic Answer LLM model.
    
    Uses Qwen model to generate full-sentence synthetic answers
    from question-answer pairs for improved SMILE metric evaluation.
    """
    
    _model = None
    _tokenizer = None
    _device = None
    _initialized = False
    
    DEFAULT_MODEL_PATH = "/mnt/dataset1/pretrained_fm/Qwen_Qwen3-4B-Instruct-2507"
    
    @classmethod
    def initialize(cls, model_path: str = None, device: str = "cuda"):
        """Initialize the synthetic answer generator model."""
        if cls._initialized:
            return
        
        if model_path is None:
            model_path = cls.DEFAULT_MODEL_PATH
        
        print(f"Initializing Synthetic Answer Generator with {model_path}...")
        cls._model, cls._tokenizer, cls._device = load_qwen_text_model(
            model_path=model_path,
            device=torch.device(device)
        )
        cls._initialized = True
        print("Synthetic Answer Generator initialized")
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the generator has been initialized."""
        return cls._initialized
    
    @classmethod
    def generate_synthetic_answer(cls, question: str, answer: str, max_new_tokens: int = 128) -> str:
        """Generate a synthetic answer for a single question-answer pair."""
        if not cls._initialized:
            raise RuntimeError("SyntheticAnswerGenerator not initialized. Call initialize() first.")
        
        return generate_synthetic_answer_qwen_text(
            model=cls._model,
            tokenizer=cls._tokenizer,
            question=question,
            answer=answer,
            device=cls._device,
            max_new_tokens=max_new_tokens,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE
        )
    
    @classmethod
    def generate_batch(cls, questions: list[str], answers: list[str], 
                       max_new_tokens: int = 128, show_progress: bool = True) -> list[str]:
        """Generate synthetic answers for a batch of question-answer pairs."""
        if not cls._initialized:
            raise RuntimeError("SyntheticAnswerGenerator not initialized. Call initialize() first.")
        
        synthetic_answers = []
        iterator = zip(questions, answers)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Generating synthetic answers")
        
        for q, a in iterator:
            try:
                syn_ans = cls.generate_synthetic_answer(q, a, max_new_tokens)
            except Exception as e:
                print(f"Warning: Error generating synthetic answer: {e}. Using original answer.")
                syn_ans = a
            synthetic_answers.append(syn_ans)
        
        return synthetic_answers


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "SharedBERTScoreModel",
    "SharedSMILEModel", 
    "SharedSyntheticAnswerGenerator",
]

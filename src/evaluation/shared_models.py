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
from torchmetrics.text import BERTScore

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

class SharedBERTScoreModel:
    """
    Singleton for shared BERTScore model to avoid repeated initialization.
    
    Uses PhoBERT-base for Vietnamese language BERTScore computation.
    Supports both local cache and HuggingFace model loading.
    """
    
    _instance = None
    _device = None
    _model_path = None
    
    HF_MODEL_NAME = "vinai/phobert-base"
    LOCAL_CACHE_DIR = "/mnt/dataset1/pretrained_fm/vinai/phobert-base"
    
    @classmethod
    def get_model_path(cls) -> str:
        """Get model path: use local cache if exists, otherwise use HuggingFace."""
        if os.path.exists(cls.LOCAL_CACHE_DIR):
            config_file = os.path.join(cls.LOCAL_CACHE_DIR, "config.json")
            has_model = (
                os.path.exists(os.path.join(cls.LOCAL_CACHE_DIR, "pytorch_model.bin")) or 
                os.path.exists(os.path.join(cls.LOCAL_CACHE_DIR, "model.safetensors"))
            )
            
            if os.path.exists(config_file) and has_model:
                return cls.LOCAL_CACHE_DIR
        
        os.makedirs(cls.LOCAL_CACHE_DIR, exist_ok=True)
        return cls.HF_MODEL_NAME
    
    @classmethod
    def get_instance(cls, device: str = "cuda") -> BERTScore:
        """Get or initialize shared BERTScore model."""
        if cls._instance is None or cls._device != device:
            cls._device = device
            cls._model_path = cls.get_model_path()
            
            cls._instance = BERTScore(
                model_name_or_path=cls._model_path,
                num_layers=12,
                rescale_with_baseline=False,
                device=device,
                truncation=True,
                max_length=256,
                dist_sync_on_step=False,
                sync_on_compute=False
            )
        
        return cls._instance


# ============================================================================
# SHARED SMILE MODEL
# ============================================================================

class SharedSMILEModel:
    """
    Singleton for SMILE metric model.
    
    SMILE (Sentence-level Metrics for Information-Leveraging Evaluation)
    is used for evaluating answer quality in VQA tasks.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get or initialize shared SMILE model."""
        if cls._instance is None:
            print("😊 Initializing SMILE metric...")
            cls._instance = SMILE(
                emb_model='phobert',
                eval_metrics=['avg', 'hm'],
                assign_bins=False,
                use_exact_matching=True,
                verbose=False
            )
        return cls._instance


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
        
        print(f"🤖 Initializing Synthetic Answer Generator with {model_path}...")
        cls._model, cls._tokenizer, cls._device = load_qwen_text_model(
            model_path=model_path,
            device=torch.device(device)
        )
        cls._initialized = True
        print("✅ Synthetic Answer Generator initialized!")
    
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

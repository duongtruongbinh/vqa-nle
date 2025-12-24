"""
Text preprocessing utilities for Vietnamese VQA evaluation.

This module provides functions for cleaning, normalizing, and preprocessing
Vietnamese text for NLG metrics computation.
"""

import re
import unicodedata

from underthesea import text_normalize, word_tokenize


# ============================================================================
# VIETNAMESE TEXT SEGMENTATION
# ============================================================================

def segment_vietnamese(text: str) -> str:
    """
    Segment Vietnamese text using underthesea word tokenizer.
    
    Args:
        text: Input Vietnamese text
        
    Returns:
        Segmented text with compound words joined by underscores
        
    Example:
        >>> segment_vietnamese("Đây là một ví dụ")
        "Đây là một ví_dụ"
    """
    if not text or not text.strip():
        return ""
    return word_tokenize(text, format="text")


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Remove line breaks, control characters, and normalize whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text with normalized whitespace
    """
    if not text:
        return ""
    
    # Replace special separators and line breaks
    text = text.replace("|||", " ").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    
    # Remove control characters
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    
    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


# ============================================================================
# ANSWER NORMALIZATION
# ============================================================================

def normalize_answer(text: str) -> str:
    """
    Normalize answer for exact matching.
    
    Handles:
    - Text cleaning and lowercasing
    - Boolean answer normalization (yes/no variants)
    - Punctuation removal
    - Word sorting for order-invariant comparison
    
    Args:
        text: Raw answer text
        
    Returns:
        Normalized answer string
    """
    if not text:
        return ""
    
    text = clean_text(text).lower().strip().rstrip(".").replace('"', "").strip()
    
    # Normalize boolean answers
    if text in ["có", "đúng", "vâng", "yes", "true", "correct"]:
        return "có"
    if text in ["không", "sai", "no", "false", "incorrect"]:
        return "không"
    
    # Remove punctuation and normalize whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Sort words for order-invariant comparison
    return " ".join(sorted(text.split()))


def normalize_answer_vqax(text: str) -> str:
    """
    Normalize answer for exact matching.
    
    Handles:
    - Text cleaning and lowercasing
    - Boolean answer normalization (yes/no variants)
    - Punctuation removal
    - Word sorting for order-invariant comparison
    
    Args:
        text: Raw answer text
        
    Returns:
        Normalized answer string
    """
    if not text:
        return ""
    
    text = clean_text(text).lower().strip().rstrip(".").replace('"', "").strip()
    
    # Remove punctuation and normalize whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Sort words for order-invariant comparison
    return " ".join(sorted(text.split()))


def normalize_explanation(text: str) -> str:
    """
    Normalize explanation text.
    
    Args:
        text: Raw explanation text
        
    Returns:
        Normalized explanation string
    """
    text = clean_text(text).strip().rstrip(".").strip()
    
    # Remove common prefixes
    text = text.lower()
    
    return text


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def truncate_sentence(text: str, max_words: int) -> str:
    """
    Truncate sentence to maximum number of words.
    
    Args:
        text: Input text
        max_words: Maximum number of words to keep
        
    Returns:
        Truncated text
    """
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def ensure_list(value) -> list[str]:
    """
    Convert value to list of strings.
    
    Args:
        value: Input value (None, str, list, or other)
        
    Returns:
        List of strings
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def preprocess_vietnamese_text(text: str) -> str:
    """
    Preprocess Vietnamese text using underthesea.
    
    Pipeline:
    1. Text normalization (fix encoding, typos)
    2. Word tokenization (segmentation)
    
    Args:
        text: Raw Vietnamese text
        
    Returns:
        Preprocessed and tokenized text
        
    Example:
        >>> preprocess_vietnamese_text("Ðảm baỏ chất lựơng phòng thí nghịêm")
        "Đảm_bảo chất_lượng phòng thí_nghiệm"
    """
    if not text or not text.strip():
        return ""
    
    # Normalize text (fix encoding issues, typos)
    normalized_text = text_normalize(text)
    
    # Tokenize (segment compound words)
    tokenized_text = word_tokenize(normalized_text, format="text")
    
    return tokenized_text


def sanitize_text_for_bert(text: str) -> str:
    """
    Sanitize text for BERT-based models to prevent CUDA errors.
    
    Removes null bytes, control characters, surrogate pairs, and handles empty strings.
    Uses "." as fallback for empty strings (valid token in all vocabularies).
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text safe for BERT tokenization
    """
    if not text or not text.strip():
        return "."
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove chars outside BMP (emoji, special symbols) which can cause tokenizer errors
    text = ''.join(ch for ch in text if ord(ch) < 65536)
    
    # Remove control characters (except common whitespace)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Cc' or ch in '\n\r\t ')
    
    # Remove surrogate pairs (can cause encoding issues)
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text if text else "."


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "segment_vietnamese",
    "clean_text",
    "normalize_answer",
    "normalize_explanation",
    "truncate_sentence",
    "ensure_list",
    "preprocess_vietnamese_text",
    "sanitize_text_for_bert",
]

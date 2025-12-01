import os
import re
import json

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import unicodedata
import torch
import pandas as pd
from datetime import datetime

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from torchmetrics.text import BERTScore
from underthesea import text_normalize, word_tokenize




# ============================================================================
# SHARED BERTSCORE MODEL
# ============================================================================

class SharedBERTScoreModel:
    """Singleton for shared BERTScore model to avoid repeated initialization."""
    
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
# TEXT PREPROCESSING
# ============================================================================

def clean_text(text: str) -> str:
    """Remove line breaks, control characters, and normalize whitespace."""
    if not text:
        return ""
    
    text = text.replace("|||", " ").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    return re.sub(r"\s+", " ", text).strip()


def normalize_answer(text: str) -> str:
    """Normalize answer for exact matching."""
    if not text:
        return ""
    
    text = clean_text(text).lower().strip().rstrip(".").replace('"', "").strip()
    
    if text in ["có", "đúng", "vâng", "yes", "true", "correct"]:
        return "có"
    if text in ["không", "sai", "no", "false", "incorrect"]:
        return "không"
    
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return " ".join(sorted(text.split()))


def normalize_explanation(text: str) -> str:
    """Normalize explanation text."""
    text = clean_text(text).strip().rstrip(".").strip()
    
    text_lower = text.lower()
    if text_lower.startswith("because "):
        text = text[8:].strip()
    elif text_lower.startswith("vì "):
        text = text[3:].strip()
    
    return text


def truncate_sentence(text: str, max_words: int) -> str:
    """Truncate sentence to maximum number of words."""
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def ensure_list(value) -> list[str]:
    """Convert value to list of strings."""
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
    
    Example:
        Input:  "Ðảm baỏ chất lựơng phòng thí nghịêm"
        Normalized: "Đảm bảo chất lượng phòng thí nghiệm"
        Tokenized: "Đảm_bảo chất_lượng phòng thí_nghiệm"
    """
    if not text or not text.strip():
        return ""
    
    normalized_text = text_normalize(text)
    tokenized_text = word_tokenize(normalized_text, format="text")
    
    return tokenized_text


# ============================================================================
# NLG METRICS
# ============================================================================

def compute_traditional_metrics(gts: dict, res: dict) -> dict[str, float]:
    """Compute BLEU, METEOR, ROUGE, CIDEr scores."""
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


def compute_bertscore_max_ref(hypotheses: list[str], references: list[list[str]], 
                              device: str = "cuda", batch_size: int = 8) -> list[float]:
    """
    Compute BERTScore F1 with max over multiple references.
    Uses batch processing for efficiency.
    """
    if not hypotheses:
        return []
    
    all_candidates = []
    all_references = []
    pair_indices = []
    
    for idx, (hyp, refs) in enumerate(zip(hypotheses, references)):
        valid_refs = [r for r in refs if r.strip()]
        
        if not hyp.strip() or not valid_refs:
            continue
        
        for ref in valid_refs:
            all_candidates.append(hyp)
            all_references.append(ref)
            pair_indices.append(idx)
    
    if not all_candidates:
        return [0.0] * len(hypotheses)
    
    bertscore = SharedBERTScoreModel.get_instance(device=device)
    bertscore.reset()
    
    all_scores = []
    for i in range(0, len(all_candidates), batch_size):
        batch_cands = all_candidates[i:i + batch_size]
        batch_refs = all_references[i:i + batch_size]
        
        bertscore.update(batch_cands, batch_refs)
        batch_result = bertscore.compute()
        
        f1_scores = batch_result['f1'].cpu().tolist()
        if isinstance(f1_scores, float):
            f1_scores = [f1_scores]
        
        all_scores.extend(f1_scores)
        bertscore.reset()
    
    max_scores = [0.0] * len(hypotheses)
    for pair_idx, score in zip(pair_indices, all_scores):
        max_scores[pair_idx] = max(max_scores[pair_idx], score * 100)
    
    return max_scores


def get_nlg_scores(references: list[list[str]], hypotheses: list[str], 
                   device: str = "cuda", max_len: int = 150, batch_size: int = 8) -> dict[str, float]:
    """Compute all NLG metrics."""
    hypotheses = [truncate_sentence(h, max_len) for h in hypotheses]
    references = [[truncate_sentence(r, max_len) for r in refs] for refs in references]
    
    hypotheses = [preprocess_vietnamese_text(h) for h in hypotheses]
    references = [[preprocess_vietnamese_text(r) for r in refs] for refs in references]
    
    gts = {i: [clean_text(r) for r in refs] for i, refs in enumerate(references)}
    res = {i: [clean_text(hyp)] for i, hyp in enumerate(hypotheses)}
    
    scores = compute_traditional_metrics(gts, res)
    
    max_f1_scores = compute_bertscore_max_ref(hypotheses, references, device, batch_size)
    scores["BERTScore_F1"] = (sum(max_f1_scores) / len(max_f1_scores)) if max_f1_scores else 0.0
    
    return scores


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_file(json_path: str, device: str = "cuda", batch_size: int = 8) -> dict:
    """Evaluate a single prediction file - scoring only thinking and answer."""
    data = json.load(open(json_path, "r", encoding="utf-8"))
    data = data[:300]
    total = 0  
    correct = 0
    all_gt_expls, all_pred_thinkings = [], []
    
    for item in data:
        total += 1 
        
        gt_ans = normalize_answer(item["answer"])
        pred_ans = normalize_answer(item.get("predict", ""))
        gt_expls = [normalize_explanation(e) for e in ensure_list(item["explanation"])]
        # Score the "thinking" field instead of "pred_explanation"
        pred_thinking = normalize_explanation(item.get("thinking", ""))
        
        all_gt_expls.append(gt_expls)
        all_pred_thinkings.append(pred_thinking)
        
        if pred_ans == gt_ans:
            correct += 1
    
    results = {
        "accuracy": (correct / total * 100) if total > 0 else 0,
        "total_examples": total,
        "correct_count": correct,
        "thinking_scores": get_nlg_scores(all_gt_expls, all_pred_thinkings, device, batch_size=batch_size),
    }
    
    return results


# ============================================================================
# MAIN
# ============================================================================

FILES_TO_EVALUATE = ['checkpoint-1000-merged.json']

def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions")
    parser.add_argument("--input-dir", type=str, default="/home/vlai-vqa-nle/minhtq/vqa-nle/src/inference/src/inference/results/OTA_grpo/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    
    if FILES_TO_EVALUATE:
        files = [f if f.endswith(".json") else f"{f}.json" for f in FILES_TO_EVALUATE]
    else:
        files = sorted([f for f in os.listdir(args.input_dir) 
                        if f.endswith(".json") and "_score" not in f and "summary" not in f])
    
    print(f"📁 Evaluating {len(files)} file(s)")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows = []
    
    for fname in files:
        fpath = os.path.join(args.input_dir, fname)
        print(f"\n🔎 Evaluating: {fname}")
        
        result = evaluate_file(fpath, device=args.device, batch_size=args.batch_size)
        model_name = os.path.splitext(fname)[0]
        
        all_rows.append({
            "model": model_name,
            "total": result["total_examples"],
            "correct": result["correct_count"],
            "accuracy": round(result["accuracy"], 2),
            **{k: round(v, 2) for k, v in result["thinking_scores"].items()}
        })
        
        print(f"   ✅ {model_name}: Accuracy={result['accuracy']:.2f}%")
    
    df = pd.DataFrame(all_rows)
    first_model = os.path.splitext(files[0])[0]
    csv_path = os.path.join(args.input_dir, f"evaluate_{first_model}_{timestamp}.csv")
    
    df.to_csv(csv_path, index=False, encoding="utf-8")
    
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()

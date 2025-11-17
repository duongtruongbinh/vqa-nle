import os
import re
import json
import argparse
import unicodedata
import torch
import torch

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from torchmetrics.text import BERTScore

import py_vncorenlp
import os
import pandas as pd
from datetime import datetime

# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/absolute/path/to/vncorenlp')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_up_visegmenter():
    vncorenlp_dir = '/home/vlai-vqa-nle/minhtq/vqa-nle/vncorenlp_models'
    if not os.path.exists(vncorenlp_dir):
        os.makedirs(vncorenlp_dir)

    # # Tự động tải model VnCoreNLP nếu chưa có trong thư mục
    # if not os.path.exists(os.path.join(vncorenlp_dir, 'models')):
    #     print("Downloading VnCoreNLP models...")
    #     py_vncorenlp.download_model(save_dir=vncorenlp_dir)
    #     print("Download complete!")

    # Load RDRSegmenter 
    print("Loading VnCoreNLP RDRSegmenter...")
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/home/vlai-vqa-nle/minhtq/vqa-nle/src/inference/vncorenlp_models')
    return rdrsegmenter

rdrsegmenter = set_up_visegmenter()

# rdrsegmenter.segment_text(text) 



# ============================================================================
# SHARED BERTSCORE MODEL
# ============================================================================

class SharedBERTScoreModel:
    """
    Shared BERTScore model để tái sử dụng cho nhiều lần tính toán.
    """
    _shared_bertscore = None
    _device = None
    _model_path = None
    
    @classmethod
    def initialize_bertscore(cls, model_name_or_path="/mnt/dataset1/pretrained_fm/vinai/phobert-base", device="cuda"):
        """
        Khởi tạo shared BERTScore model.
        
        Args:
            model_name_or_path: Đường dẫn đến BERT model.
            device: Device để chạy model (cuda/cpu).
        """
        # Chỉ khởi tạo 1 lần hoặc khi model path/device thay đổi
        if cls._shared_bertscore is None or cls._model_path != model_name_or_path or cls._device != device:
            cls._device = device
            cls._model_path = model_name_or_path
            print(f"   🔧 Initializing shared BERTScore from: {model_name_or_path}")
            print(f"   📍 Device: {cls._device}")
            cls._shared_bertscore = BERTScore(
                model_name_or_path=model_name_or_path,
                num_layers=12,
                rescale_with_baseline=False,
                device=cls._device,
                truncation=True,
                max_length=256,
                dist_sync_on_step=False,
                sync_on_compute=False
            )
            print("   ✅ Shared BERTScore initialized successfully.")
        return cls._shared_bertscore



# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def clean_text(text: str) -> str:
    """Remove line breaks, control characters, and normalize whitespace."""
    if not text:
        return ""
    
    text = text.replace("|||", " ").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    return re.sub(r"\s+", " ", text).strip()


def normalize_answer(text: str) -> str:
    """
    Normalize answer for exact matching.
    """
    if not text:
        return ""
    
    # Clean and lowercase
    """
    Normalize answer for exact matching.
    """
    if not text:
        return ""
    
    # Clean and lowercase
    text = clean_text(text).lower().strip().rstrip(".").replace('"', "").strip()
    
    # Yes/No normalization
    if text in ["có", "đúng", "vâng", "yes", "true", "correct"]:
        return "có"
    if text in ["không", "sai", "no", "false", "incorrect"]:
        return "không"
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Sort words for order-independent matching
    return " ".join(sorted(text.split()))


def fuzzy_match_answer(pred: str, gt: str) -> bool:
    """
    Compare prediction with ground truth using keyword matching.
    Both inputs should already be normalized via normalize_answer().
    
    Args:
        pred: Normalized prediction
        gt: Normalized ground truth
        
    Returns:
        True if match (exact or fuzzy), False otherwise
    """
    # Exact match
    if pred == gt:
        return True
    
    # Keyword matching for short ground truths (≤2 words)
    # Example: gt="đạp xe" matches pred="đạp màu xe xanh"
    if len(gt.split()) <= 2:
        gt_keywords = set(gt.split())
        pred_keywords = set(pred.split())
        
        # Prediction contains all GT keywords and is not too long
        if gt_keywords.issubset(pred_keywords) and len(pred_keywords) <= len(gt_keywords) + 2:
            return True
    
    return False



def normalize_explanation(text: str) -> str:
    """Normalize explanation text."""
    text = clean_text(text).strip().rstrip(".").strip()
    
    # Remove 'because'/'vì' prefix
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


# ============================================================================
# NLG METRICS COMPUTATION
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
        except Exception as e:
            print(f"   ⚠️ Error computing {method}: {e}")
            if isinstance(method, list):
                scores.update({m: 0.0 for m in method})
            else:
                scores[method] = 0.0
    
    return scores



def compute_bertscore_single(candidate: str, reference: str, device: str = "cuda") -> float:
    """
    Tính BERTScore F1 cho một cặp candidate-reference đơn.
    Reset model mỗi lần để đảm bảo độc lập.
    
    Args:
        candidate: Câu dự đoán
        reference: Câu ground truth
        device: cuda hoặc cpu
        
    Returns:
        BERTScore F1 score (0-100)
    """
    if not candidate.strip() or not reference.strip():
        return 0.0
    
    try:
        # Sử dụng shared BERTScore model
        bertscore_metric = SharedBERTScoreModel.initialize_bertscore(
            device=device
        )
        bertscore_metric.reset()
        
        bertscore_metric.update([candidate], [reference])
        f1_score = bertscore_metric.compute()['f1']
        
        # Xử lý cả 0-dim và 1-dim tensor
        if f1_score.dim() == 0:
            return f1_score.item() * 100
        else:
            return f1_score[0].item() * 100
        
    except Exception as e:
        print(f"   ⚠️ Error computing BERTScore for single pair: {e}")
        import traceback
        traceback.print_exc()


def compute_bertscore(candidates: list[str], references: list[str], device: str = "cuda") -> float:
    """
    Compute BERTScore F1 (one-to-one comparison) by looping through each pair.
    Tính theo từng cặp và reset mỗi lần.
    """
    if not candidates or not references:
        return 0.0
    
    if len(candidates) != len(references):
        print(f"   ⚠️ Warning: candidates ({len(candidates)}) and references ({len(references)}) length mismatch")
        return 0.0
    
    scores = []
    for i, (cand, ref) in enumerate(zip(candidates, references)):
        score = compute_bertscore_single(cand, ref, device)
        scores.append(score)
    
    # Trả về trung bình
    return sum(scores) / len(scores) if scores else 0.0


def compute_bertscore(candidates: list[str], references: list[str], device: str = "cuda") -> float:
    """
    Compute BERTScore F1 (one-to-one comparison) by looping through each pair.
    Tính theo từng cặp và reset mỗi lần.
    """
    if not candidates or not references:
        return 0.0
    
    if len(candidates) != len(references):
        print(f"   ⚠️ Warning: candidates ({len(candidates)}) and references ({len(references)}) length mismatch")
        return 0.0
    
    scores = []
    for i, (cand, ref) in enumerate(zip(candidates, references)):
        score = compute_bertscore_single(cand, ref, device)
        scores.append(score)
    
    # Trả về trung bình
    return sum(scores) / len(scores) if scores else 0.0


def compute_bertscore_max_ref(hypotheses: list[str], references: list[list[str]], device: str = "cuda") -> list[float]:
    """
    Compute BERTScore F1 with max over multiple references.
    Tính theo từng cặp hypothesis-reference và lấy max, reset mỗi lần.
    """
    max_f1_scores = []
    """
    Compute BERTScore F1 with max over multiple references.
    Tính theo từng cặp hypothesis-reference và lấy max, reset mỗi lần.
    """
    max_f1_scores = []
    
    for hyp, refs in zip(hypotheses, references):
        valid_refs = [r for r in refs if r.strip()]
        
        
        if not hyp.strip() or not valid_refs:
            max_f1_scores.append(0.0)
            max_f1_scores.append(0.0)
            continue
        
        # Tính BERTScore cho hypothesis với từng reference
        ref_scores = []
        for ref in valid_refs:
            score = compute_bertscore_single(hyp, ref, device)
            ref_scores.append(score)
        
        # Lấy max score
        max_f1_scores.append(max(ref_scores) if ref_scores else 0.0)
        # Tính BERTScore cho hypothesis với từng reference
        ref_scores = []
        for ref in valid_refs:
            score = compute_bertscore_single(hyp, ref, device)
            ref_scores.append(score)
        
        # Lấy max score
        max_f1_scores.append(max(ref_scores) if ref_scores else 0.0)
    
    return max_f1_scores


def segment_text(text: str) -> str:
    """
    Tách từ sử dụng VnCoreNLP (chuẩn cho PhoBERT).
    Đầu vào: "Tôi là sinh viên đại học."
    Đầu ra: "Tôi là sinh_viên đại_học ."
    """
    if not text:
        return ""
    try:
        # rdrsegmenter trả về list các câu, mỗi câu là list các từ
        # Ví dụ: [['Tôi', 'là', 'sinh_viên', 'đại_học', '.']]
        sentences = rdrsegmenter.word_segment(text)
        # Nối lại thành chuỗi chuẩn
        segmented_text = " ".join([" ".join(sentence) for sentence in sentences])
        return segmented_text
    except Exception as e:
        print(f"Error segmenting text: {e}")
        return text


def get_nlg_scores(references: list[list[str]], hypotheses: list[str], device: str = "cuda", max_len: int = 150) -> dict[str, float]:
    """Compute all NLG metrics."""
    # Truncate
    hypotheses = [truncate_sentence(h, max_len) for h in hypotheses]
    references = [[truncate_sentence(r, max_len) for r in refs] for refs in references]

    hypotheses = [segment_text(h) for h in hypotheses]
    references = [[segment_text(r) for r in refs] for refs in references]
    
    # Prepare for traditional metrics
    gts = {i: [clean_text(r) for r in refs] for i, refs in enumerate(references)}
    res = {i: [clean_text(hyp)] for i, hyp in enumerate(hypotheses)}
    
    # Compute traditional metrics
    scores = compute_traditional_metrics(gts, res)
    
    # Compute BERTScore
    print("   ⏳ Computing BERTScore...")
    max_f1_scores = compute_bertscore_max_ref(hypotheses, references, device)
    scores["BERTScore_F1"] = (sum(max_f1_scores) / len(max_f1_scores)) if max_f1_scores else 0.0
    scores["BERTScore_F1"] = (sum(max_f1_scores) / len(max_f1_scores)) if max_f1_scores else 0.0
    print("   ✅ BERTScore complete")
    
    return scores


def evaluate_file(json_path: str, device: str = "cuda") -> dict:
    """Evaluate a single prediction file."""
    data = json.load(open(json_path, "r", encoding="utf-8"))
    results = {
        "accuracy": 0,
        "unfiltered_scores": {},
        "by_answer_type": {}
    }
    
    total = 0  
    correct = 0
    all_gt_expls, all_pred_expls = [], []
    by_type = {}
    
    # Process each example
    for item in data:
        
        total += 1 
        
        gt_ans = normalize_answer(item["answer"])
        pred_ans = normalize_answer(item["predict"])
        gt_expls = [normalize_explanation(e) for e in ensure_list(item["explanation"])]
        pred_expl = normalize_explanation(item["pred_explanation"])
        
        all_gt_expls.append(gt_expls)
        all_pred_expls.append(pred_expl)
        
        ans_type = item["answer_type"]
        if ans_type not in by_type:
            by_type[ans_type] = {"gt_expls": [], "pred_expls": [], "total": 0, "correct": 0}
        
        by_type[ans_type]["gt_expls"].append(gt_expls)
        by_type[ans_type]["pred_expls"].append(pred_expl)
        by_type[ans_type]["total"] += 1
        
        # if fuzzy_match_answer(pred_ans, gt_ans):
        if pred_ans == gt_ans:
            correct += 1
            by_type[ans_type]["correct"] += 1
    
    # Overall scores
    results["accuracy"] = (correct / total * 100) if total > 0 else 0
    results["total_examples"] = total
    results["correct_count"] = correct
    results["unfiltered_scores"] = get_nlg_scores(all_gt_expls, all_pred_expls, device)
    
    # By answer_type
    for ans_type, data_type in by_type.items():
        scores = get_nlg_scores(data_type["gt_expls"], data_type["pred_expls"], device)
        results["by_answer_type"][ans_type] = {
            "accuracy": (data_type["correct"] / data_type["total"] * 100),
            "total_examples": data_type["total"],
            "correct_count": data_type["correct"],
            "unfiltered_scores": scores,
        }
    
    return results

# vintern3b-stage1-BS1-1000steps, vintern3b-stage1-BS2-1000steps-gfpo, vintern-stage1-BS2-1000step.json
FILES_TO_EVALUATE = ['gfpo_bs2.json']

def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions")
    parser.add_argument("--input-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    

    if FILES_TO_EVALUATE:
        files = [f if f.endswith(".json") else f"{f}.json" for f in FILES_TO_EVALUATE]
    else:
        files = sorted([f for f in os.listdir(args.input_dir) 
                        if f.endswith(".json") and "_score" not in f and "summary" not in f])
    
    print(f"📁 Evaluating {len(files)} files")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows = []
    
    for fname in files:
        fpath = os.path.join(args.input_dir, fname)
        print(f"\n🔎 {fname}")
        
        result = evaluate_file(fpath, device=args.device)
        model_name = os.path.splitext(fname)[0]
        
        # Overall row
        all_rows.append({
            "model": model_name,
            "answer_type": "Overall",
            "total": result["total_examples"],
            "correct": result["correct_count"],
            "accuracy": round(result["accuracy"], 2),
            "BLEU-1": round(result["unfiltered_scores"].get("BLEU-1", 0), 2),
            "BLEU-2": round(result["unfiltered_scores"].get("BLEU-2", 0), 2),
            "BLEU-3": round(result["unfiltered_scores"].get("BLEU-3", 0), 2),
            "BLEU-4": round(result["unfiltered_scores"].get("BLEU-4", 0), 2),
            "METEOR": round(result["unfiltered_scores"].get("METEOR", 0), 2),
            "ROUGE_L": round(result["unfiltered_scores"].get("ROUGE_L", 0), 2),
            "CIDEr": round(result["unfiltered_scores"].get("CIDEr", 0), 2),
            "BERTScore_F1": round(result["unfiltered_scores"].get("BERTScore_F1", 0), 2),
        })
        
        # By answer_type rows
        for ans_type, type_data in result["by_answer_type"].items():
            all_rows.append({
                "model": model_name,
                "answer_type": ans_type,
                "total": type_data["total_examples"],
                "correct": type_data["correct_count"],
                "accuracy": round(type_data["accuracy"], 2),
                "BLEU-1": round(type_data["unfiltered_scores"].get("BLEU-1", 0), 2),
                "BLEU-2": round(type_data["unfiltered_scores"].get("BLEU-2", 0), 2),
                "BLEU-3": round(type_data["unfiltered_scores"].get("BLEU-3", 0), 2),
                "BLEU-4": round(type_data["unfiltered_scores"].get("BLEU-4", 0), 2),
                "METEOR": round(type_data["unfiltered_scores"].get("METEOR", 0), 2),
                "ROUGE_L": round(type_data["unfiltered_scores"].get("ROUGE_L", 0), 2),
                "CIDEr": round(type_data["unfiltered_scores"].get("CIDEr", 0), 2),
                "BERTScore_F1": round(type_data["unfiltered_scores"].get("BERTScore_F1", 0), 2),
            })
        
        print(f"   ✅ {model_name}")
    
    # Save results
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.input_dir, f"evaluate_{model_name}_{timestamp}.csv")
    
    df.to_csv(csv_path, index=False, encoding="utf-8")
    
    print(f"\n✅ Saved to {csv_path}")
    print(f"\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()

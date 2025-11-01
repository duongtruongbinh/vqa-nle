import os
import re
import json
import argparse
import unicodedata

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from bert_score import score as bert_score
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# ----------------------------
# Text Cleaning & Normalization
# ----------------------------

def clean_text(text: str) -> str:
    """
    Làm sạch chuỗi text:
    - Thay thế mọi biến thể xuống dòng và '|||' bằng khoảng trắng
    - Loại bỏ ký tự điều khiển (Unicode category == 'Cc')
    - Gom nhiều khoảng trắng liên tiếp thành 1
    """
    if not text:
        return ""
    
    text = (
        text.replace("|||", " ")
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\u2028", " ")
        .replace("\u2029", " ")
    )
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_answer(text: str) -> str:
    """
    Chuẩn hóa câu trả lời:
    - Bước 1: Làm sạch text qua clean_text()
    - Bước 2: Chuyển về lowercase và loại bỏ dấu câu thừa
    - Bước 3: Chuẩn hóa câu trả lời Có/Không
    - Bước 4: Chuẩn hóa từ/cụm từ đồng nghĩa
    - Bước 5: Loại bỏ tiền tố phổ biến
    - Bước 6: Loại bỏ ký tự đặc biệt
    - Bước 7: Chuẩn hóa thứ tự từ (sắp xếp)
    """
    # Bước 1: Làm sạch text
    text = clean_text(text)
    
    # Bước 2: Lowercase và strip, loại bỏ dấu câu
    text = text.lower().strip().rstrip(".").replace('"', "").strip()
    
    # Bước 3: Chuẩn hóa các câu trả lời Có/Không
    if text in ["có", "đúng", "yes", "true", "correct"]:
        return "có"
    if text in ["không", "sai", "no", "false", "incorrect"]:
        return "không"
    
    # Bước 4: Chuẩn hóa từ/cụm từ đồng nghĩa
    synonym_map = {
        "bay diều": "thả diều",
        "diều bay": "thả diều",
        # Thêm các cặp đồng nghĩa khác tại đây
    }
    if text in synonym_map:
        text = synonym_map[text]
    
    # Bước 5: Loại bỏ các tiền tố/hậu tố phổ biến
    prefixes_to_remove = ["con ", "cái ", "chiếc ", "quả ", "hoa ", "màu ", "bên ", "phía "]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    
    # Bước 6: Loại bỏ các ký tự đặc biệt và khoảng trắng thừa
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Bước 7: Chuẩn hóa thứ tự từ
    words = sorted(text.split())
    text = " ".join(words)
    
    return text

def normalize_explanation(text: str) -> str:
    """
    Chuẩn hóa explanation:
    - Làm sạch qua clean_text()
    - Loại bỏ 'because'/'vì' ở đầu câu
    - Loại bỏ dấu chấm cuối
    """
    # Làm sạch text
    text = clean_text(text)
    
    # Strip và loại bỏ dấu chấm cuối
    text = text.strip().rstrip(".").strip()
    
    # Loại bỏ 'because'/'vì' ở đầu câu
    text_lower = text.lower()
    if text_lower.startswith("because "):
        text = text[8:].strip()
    elif text_lower.startswith("vì "):
        text = text[3:].strip()
    
    return text

def normalize_line(text: str) -> str:
    """Chuẩn hóa line cho NLG metrics - chỉ làm sạch xuống dòng và khoảng trắng."""
    text = clean_text(text)
    return text

def ensure_list_explanations(expls) -> list[str]:
    """Convert explanation(s) to list of strings."""
    if expls is None:
        return []
    if isinstance(expls, str):
        return [expls]
    if isinstance(expls, list):
        return [str(x) for x in expls]
    return [str(expls)]

# ----------------------------
# NLG Score Calculation
# ----------------------------
def get_nlg_scores(references: list[list[str]], hypotheses: list[str], device: str = "cuda", max_len: int = 150) -> dict[str, float]:
    
    def truncate_sentence(s: str, length: int) -> str:
        """Truncates a sentence to a maximum number of words."""
        words = s.split()
        if len(words) > length:
            return " ".join(words[:length])
        return s

    truncated_hypotheses = [truncate_sentence(h, max_len) for h in hypotheses]
    truncated_references = [[truncate_sentence(r, max_len) for r in ref_list] for ref_list in references]

    gts = {i: [normalize_line(x) for x in refs] for i, refs in enumerate(truncated_references)}
    res = {i: [normalize_line(hyp)] for i, hyp in enumerate(truncated_hypotheses)}

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    scores: dict[str, float] = {}
    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    scores[m] = float(s) * 100
            else:
                scores[method] = float(score) * 100
        except Exception as e:
            print(f"   ⚠️ Error computing {method}: {e}. Setting score to 0.0")
            if isinstance(method, list):
                for m in method:
                    scores[m] = 0.0
            else:
                scores[method] = 0.0

    print("   ⏳ Computing BERTScore ...")
    
    batched_cands = []
    batched_refs = []
    ref_counts = []

    for hyp, refs in zip(truncated_hypotheses, truncated_references):
        if not hyp.strip() or not any(r.strip() for r in refs):
            ref_counts.append(0)
            continue
        
        valid_refs = [r for r in refs if r.strip()]
        num_refs = len(valid_refs)
        batched_cands.extend([hyp] * num_refs)
        batched_refs.extend(valid_refs)
        ref_counts.append(num_refs)

    max_f1_scores = []
    if batched_cands:
        _, _, all_f1 = bert_score(
            batched_cands, batched_refs, 
            lang="vi", 
            model_type='microsoft/deberta-large-mnli',
            device=device, 
            verbose=False, 
            batch_size=1
        )
        
        current_idx = 0
        for count in ref_counts:
            if count == 0:
                max_f1_scores.append(0.0)
            else:
                f1_group = all_f1[current_idx : current_idx + count]
                max_f1_scores.append(f1_group.max().item())
                current_idx += count
    else:
        max_f1_scores = [0.0] * len(hypotheses)

    if max_f1_scores:
        scores["BERTScore_F1"] = (sum(max_f1_scores) / len(max_f1_scores)) * 100
    else:
        scores["BERTScore_F1"] = 0.0
    
    print("   ✅ BERTScore calculation complete.")
    return scores

# ----------------------------
# Evaluate a Prediction File
# ----------------------------
def evaluate_file(json_path: str, device: str = "cuda") -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_examples = len(data)
    correct_count = 0

    all_gt_expls, all_pred_expls = [], []
    filtered_gt_expls, filtered_pred_expls = [], []
    
    # Dictionary to store metrics by answer_type
    by_answer_type = {}

    for item in data:
        gt_ans = normalize_answer(item.get("answer", ""))
        pred_ans = normalize_answer(item.get("predict", ""))
        
        # Chuẩn hóa explanations qua clean_text
        gt_expls_raw = ensure_list_explanations(item.get("explanation", []))
        gt_expls = [normalize_explanation(expl) for expl in gt_expls_raw]
        pred_expl = normalize_explanation(item.get("pred_explanation", ""))

        all_gt_expls.append(gt_expls)
        all_pred_expls.append(pred_expl)

        # Track by answer_type
        answer_type = item.get("answer_type", "unknown")
        if answer_type not in by_answer_type:
            by_answer_type[answer_type] = {
                "total": 0,
                "correct": 0,
                "all_gt_expls": [],
                "all_pred_expls": [],
                "filtered_gt_expls": [],
                "filtered_pred_expls": []
            }
        
        by_answer_type[answer_type]["total"] += 1
        by_answer_type[answer_type]["all_gt_expls"].append(gt_expls)
        by_answer_type[answer_type]["all_pred_expls"].append(pred_expl)

        if pred_ans == gt_ans:
            correct_count += 1
            filtered_gt_expls.append(gt_expls)
            filtered_pred_expls.append(pred_expl)
            
            by_answer_type[answer_type]["correct"] += 1
            by_answer_type[answer_type]["filtered_gt_expls"].append(gt_expls)
            by_answer_type[answer_type]["filtered_pred_expls"].append(pred_expl)

    accuracy_fraction = (correct_count / total_examples) if total_examples > 0 else 0.0
    
    print("\n--- Starting UNFILTERED evaluation ---")
    unfiltered_scores = get_nlg_scores(all_gt_expls, all_pred_expls, device=device)

    print("\n--- Starting FILTERED evaluation (correct answers only) ---")
    if filtered_pred_expls:
        filtered_scores = get_nlg_scores(filtered_gt_expls, filtered_pred_expls, device=device)
    else:
        print("   ⚠️ No correct answers found. Setting filtered scores to 0.")
        filtered_scores = {k: 0.0 for k in unfiltered_scores.keys()}

    scaled_scores = {k: v * accuracy_fraction for k, v in unfiltered_scores.items()}

    # Calculate metrics for each answer_type
    answer_type_results = {}
    for ans_type, type_data in by_answer_type.items():
        print(f"\n--- Evaluating answer_type: {ans_type} ({type_data['total']} examples) ---")
        
        type_accuracy = (type_data["correct"] / type_data["total"]) if type_data["total"] > 0 else 0.0
        
        # Unfiltered scores for this type
        type_unfiltered_scores = get_nlg_scores(
            type_data["all_gt_expls"], 
            type_data["all_pred_expls"], 
            device=device
        )
        
        # Filtered scores for this type (correct answers only)
        if type_data["filtered_pred_expls"]:
            type_filtered_scores = get_nlg_scores(
                type_data["filtered_gt_expls"], 
                type_data["filtered_pred_expls"], 
                device=device
            )
        else:
            print(f"   ⚠️ No correct answers for {ans_type}. Setting filtered scores to 0.")
            type_filtered_scores = {k: 0.0 for k in type_unfiltered_scores.keys()}
        
        type_scaled_scores = {k: v * type_accuracy for k, v in type_unfiltered_scores.items()}
        
        answer_type_results[ans_type] = {
            "accuracy": type_accuracy * 100,
            "total_examples": type_data["total"],
            "correct_count": type_data["correct"],
            "unfiltered_scores": type_unfiltered_scores,
            "filtered_scores": type_filtered_scores,
            "scaled_scores": type_scaled_scores,
        }

    return {
        "accuracy": accuracy_fraction * 100,
        "task_score": accuracy_fraction * 100,
        "unfiltered_scores": unfiltered_scores,
        "filtered_scores": filtered_scores,
        "scaled_scores": scaled_scores,
        "total_examples": total_examples,
        "correct_count": correct_count,
        "by_answer_type": answer_type_results,
    }

# ----------------------------
# Main Execution
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions (answer + explanation NLG).")
    parser.add_argument("--input-dir", type=str, default="results", help="Directory containing prediction .json files")
    parser.add_argument("--device", type=str, default="cuda", help="Device for BERTScore: cuda or cpu")
    parser.add_argument("--suffix", type=str, default="_score.json", help="Output file suffix for each input")
    args = parser.parse_args()

    in_dir = args.input_dir
    if not os.path.isdir(in_dir):
        print(f"❌ Directory not found: {in_dir}")
        return

    files = sorted([f for f in os.listdir(in_dir) if f.endswith(".json") and "_score" not in f and "summary" not in f])
    if not files:
        print(f"⚠️ No .json files found in {in_dir}")
        return

    print(f"📁 Found {len(files)} files in {in_dir}")
    summary = {}

    for fname in files:
        fpath = os.path.join(in_dir, fname)
        print(f"\n=============================================")
        print(f"🔎 Evaluating: {fname}")
        print(f"=============================================")
        try:
            result = evaluate_file(fpath, device=args.device)
        except Exception as e:
            print(f"   ❌ Critical error processing {fname}: {e}")
            continue

        base = os.path.splitext(fname)[0]
        out_path = os.path.join(in_dir, f"{base}{args.suffix}")
        
        # Round scores for readability
        for score_dict in ["unfiltered_scores", "filtered_scores", "scaled_scores"]:
            if score_dict in result:
                result[score_dict] = {k: round(v, 2) for k, v in result[score_dict].items()}
        result["accuracy"] = round(result["accuracy"], 2)
        
        # Round scores for each answer_type
        if "by_answer_type" in result:
            for ans_type, type_result in result["by_answer_type"].items():
                for score_dict in ["unfiltered_scores", "filtered_scores", "scaled_scores"]:
                    if score_dict in type_result:
                        type_result[score_dict] = {k: round(v, 2) for k, v in type_result[score_dict].items()}
                type_result["accuracy"] = round(type_result["accuracy"], 2)

        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(result, fw, indent=2, ensure_ascii=False)
        print(f"\n   💾 Detailed results saved to: {out_path}")

        summary[base] = {
            "accuracy": result["accuracy"],
            "BLEU-4": result["filtered_scores"].get("BLEU-4", 0.0),
            "METEOR": result["filtered_scores"].get("METEOR", 0.0),
            "ROUGE_L": result["filtered_scores"].get("ROUGE_L", 0.0),
            "CIDEr": result["filtered_scores"].get("CIDEr", 0.0),
            "BERTScore_F1": result["filtered_scores"].get("BERTScore_F1", 0.0),
        }

    summary_path = os.path.join(in_dir, "evaluation_summary.txt")
    
    header = "Model\tBLEU-4\tMETEOR\tROUGE_L\tCIDEr\tBERTScore\tAccuracy\n"
    lines = [header]
    
    for model_name, scores in summary.items():
        line = (
            f"{model_name}\t"
            f"{scores['BLEU-4']:.2f}\t"
            f"{scores['METEOR']:.2f}\t"
            f"{scores['ROUGE_L']:.2f}\t"
            f"{scores['CIDEr']:.2f}\t"
            f"{scores['BERTScore_F1']:.2f}\t"
            f"{scores['accuracy']:.2f}\n"
        )
        lines.append(line)

    with open(summary_path, "w", encoding="utf-8") as fw:
        fw.writelines(lines)

    print("\n\n=============================================")
    print("✅ All evaluations complete. Summary:")
    print("=============================================")
    for line in lines:
        print(line.strip())
        
    print(f"\n📊 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

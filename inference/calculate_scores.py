# file: eval_folder_nlg.py
import os
import re
import json
import argparse
from typing import List, Tuple, Dict

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from bert_score import score as bert_score


# ----------------------------
# Helpers (chuẩn hoá & tiền xử lý)
# ----------------------------
def normalize_line(s: str) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", s).strip()

def norm_answer(s: str) -> str:
    s = (s or "").strip().lower()
    # loại bớt các ký tự nhiễu phổ biến
    s = s.strip().rstrip(".").replace('"', "").strip()
    return s

def clean_pred_expl(s: str) -> str:
    """Bỏ 'because'/'vì' đầu câu (nếu có) + bỏ dấu chấm cuối"""
    t = (s or "").strip().rstrip(".").strip()
    low = t.lower()
    if low.startswith("because"):
        return t[7:].strip()
    if low.startswith("vì"):
        return t[2:].strip()
    return t

def ensure_list_explanations(expls) -> List[str]:
    if expls is None:
        return []
    if isinstance(expls, str):
        return [expls]
    if isinstance(expls, list):
        return [str(x) for x in expls]
    return [str(expls)]


# ----------------------------
# Tính NLG scores
# ----------------------------
def get_nlg_scores(references: List[List[str]], hypotheses: List[str], device: str = "cuda") -> Dict[str, float]:
    """
    references: list các danh sách GT explanations (mỗi phần tử là list[str])
    hypotheses: list các câu giải thích dự đoán (mỗi phần tử là str)
    """
    # Đưa về định dạng pycocoevalcap: id -> [candidates]
    gts = {i: [normalize_line(x) for x in refs] for i, refs in enumerate(references)}
    res = {i: [normalize_line(hyp)] for i, hyp in enumerate(hypotheses)}

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE"),
    ]

    scores: Dict[str, float] = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for m, s in zip(method, score):
                scores[m] = float(s)
        else:
            scores[method] = float(score)

    # BERTScore-F1 trung bình (ngôn ngữ vi)
    # Lấy 1 GT đầu tiên cho mỗi mẫu (giống code gốc)
    single_refs = [refs[0] if refs else "" for refs in references]
    _, _, F1 = bert_score(hypotheses, single_refs, lang="vi", device=device)
    scores["BERTScore_F1"] = float(F1.mean().item())

    return scores


# ----------------------------
# Đánh giá một file prediction
# ----------------------------
def evaluate_file(json_path: str, device: str = "cuda") -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_examples = len(data)
    correct_count = 0

    # Tập tổng (unfiltered)
    all_gt_expls: List[List[str]] = []
    all_pred_expls: List[str] = []

    # Tập đúng answer (filtered)
    filtered_gt_expls: List[List[str]] = []
    filtered_pred_expls: List[str] = []

    for item in data:
        gt_ans = norm_answer(item.get("answer", ""))
        pred_ans = norm_answer(item.get("predict", ""))

        gt_expls = ensure_list_explanations(item.get("explanation", []))
        pred_expl = clean_pred_expl(item.get("pred_explanation", ""))

        # Thêm unfiltered
        all_gt_expls.append(gt_expls)
        all_pred_expls.append(pred_expl)

        # Đếm đúng & thêm filtered
        if pred_ans == gt_ans:
            correct_count += 1
            filtered_gt_expls.append(gt_expls)
            filtered_pred_expls.append(pred_expl)

    accuracy = (correct_count / total_examples) if total_examples > 0 else 0.0
    task_score_final = accuracy  # giữ nguyên như code gốc (task_score/total_examples)

    # NLG cho toàn bộ
    unfiltered_scores = get_nlg_scores(all_gt_expls, all_pred_expls, device=device)

    # NLG cho filtered (nếu có ít nhất 1 ví dụ đúng)
    if filtered_pred_expls:
        filtered_scores = get_nlg_scores(filtered_gt_expls, filtered_pred_expls, device=device)
    else:
        filtered_scores = {k: 0.0 for k in unfiltered_scores.keys()}

    # Scaled = unfiltered * accuracy
    scaled_scores = {k: float(v) * accuracy for k, v in unfiltered_scores.items()}

    return {
        "accuracy": accuracy,
        "task_score": task_score_final,
        "unfiltered_scores": unfiltered_scores,
        "filtered_scores": filtered_scores,
        "scaled_scores": scaled_scores,
        "total_examples": total_examples,
        "correct_count": correct_count,
    }


# ----------------------------
# Chạy trên thư mục
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions (answer + explanation NLG).")
    parser.add_argument("--input-dir", type=str, default="results", help="Thư mục chứa các file .json dự đoán")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị cho BERTScore: cuda hoặc cpu")
    parser.add_argument("--suffix", type=str, default="_score.json", help="Hậu tố file output cho mỗi input")
    args = parser.parse_args()

    in_dir = args.input_dir
    if not os.path.isdir(in_dir):
        print(f"❌ Không tìm thấy thư mục: {in_dir}")
        return

    files = sorted([f for f in os.listdir(in_dir) if f.endswith(".json") and "_score" not in f])
    if not files:
        print(f"⚠️ Không có file .json nào trong {in_dir}")
        return

    print(f"📁 Tìm thấy {len(files)} file trong {in_dir}")
    summary = {}

    for fname in files:
        fpath = os.path.join(in_dir, fname)
        print(f"🔎 Đánh giá: {fname}")

        try:
            result = evaluate_file(fpath, device=args.device)
        except Exception as e:
            print(f"   ❌ Lỗi khi xử lý {fname}: {e}")
            continue

        # Lưu file kết quả cạnh input
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(in_dir, f"{base}{args.suffix}")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(result, fw, indent=2, ensure_ascii=False)
        print(f"   💾 Đã lưu: {out_path}")

        summary[base] = {
            "accuracy": round(result["accuracy"], 4),
            "BLEU-4": round(result["unfiltered_scores"].get("BLEU-4", 0.0), 4),
            "ROUGE_L": round(result["unfiltered_scores"].get("ROUGE_L", 0.0), 4),
            "CIDEr": round(result["unfiltered_scores"].get("CIDEr", 0.0), 4),
            "SPICE": round(result["unfiltered_scores"].get("SPICE", 0.0), 4),
            "BERTScore_F1": round(result["unfiltered_scores"].get("BERTScore_F1", 0.0), 4),
            "total_examples": result["total_examples"],
            "correct_count": result["correct_count"],
        }

    summary_path = os.path.join(in_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fw:
        json.dump(summary, fw, indent=2, ensure_ascii=False)

    print("\n✅ Hoàn tất. Tóm tắt:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"📊 Đã lưu tổng hợp: {summary_path}")


if __name__ == "__main__":
    main()

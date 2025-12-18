import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import re
import argparse
import pandas as pd
from datetime import datetime

from .shared_models import SharedSyntheticAnswerGenerator, SharedBERTScoreModel, SharedSMILEModel
from .text_preprocessing import normalize_answer, normalize_explanation, ensure_list, clean_text
from .nlg_metrics import get_nlg_scores, compute_smile_scores


YES_SET = {"yes", "true", "correct", "có", "đúng", "vâng"}
NO_SET = {"no", "false", "incorrect", "không", "sai"}
FILES_TO_EVALUATE = ['sft_answer_explain.json']


def normalize_unsorted(text):
    if not text:
        return ""
    text = clean_text(text).lower().strip().rstrip(".").replace('"', "").strip()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def check_accuracy(pred_ans, gt_ans, raw_pred, raw_gt):
    if pred_ans == gt_ans:
        return True
    
    if not raw_pred:
        return False
    
    c_pred_raw = clean_text(raw_pred).lower().strip()
    c_pred_nopunct = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', c_pred_raw)).strip()
    c_pred_tokens = set(c_pred_raw.split())
    
    if gt_ans in YES_SET and not c_pred_tokens.isdisjoint(YES_SET):
        return True
    if gt_ans in NO_SET and not c_pred_tokens.isdisjoint(NO_SET):
        return True
    
    gt_unsorted = normalize_unsorted(raw_gt)
    return gt_unsorted and gt_unsorted in c_pred_nopunct


def evaluate_file(json_path: str, device: str = "cuda", use_synthetic_answers: bool = False) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total, correct = 0, 0
    all_gt_expls, all_pred_expls = [], []
    all_questions, all_gt_answers, all_pred_answers = [], [], []
    by_type = {}
    
    for item in data:
        total += 1
        raw_gt, raw_pred = item.get("answer", ""), item.get("predict", "")
        gt_ans, pred_ans = normalize_answer(raw_gt), normalize_answer(raw_pred)
        gt_expls = [normalize_explanation(e) for e in ensure_list(item.get("explanation", []))]
        pred_expl = normalize_explanation(item.get("pred_explanation", ""))
        
        all_gt_expls.append(gt_expls)
        all_pred_expls.append(pred_expl)
        all_questions.append(item.get("question", ""))
        all_gt_answers.append(raw_gt)
        all_pred_answers.append(raw_pred)
        
        ans_type = item.get("answer_type", "other")
        if ans_type not in by_type:
            by_type[ans_type] = {
                "gt_expls": [], "pred_expls": [],
                "questions": [], "gt_answers": [], "pred_answers": [],
                "total": 0, "correct": 0
            }
        
        by_type[ans_type]["gt_expls"].append(gt_expls)
        by_type[ans_type]["pred_expls"].append(pred_expl)
        by_type[ans_type]["questions"].append(item.get("question", ""))
        by_type[ans_type]["gt_answers"].append(raw_gt)
        by_type[ans_type]["pred_answers"].append(raw_pred)
        by_type[ans_type]["total"] += 1
        
        if check_accuracy(pred_ans, gt_ans, raw_pred, raw_gt):
            correct += 1
            by_type[ans_type]["correct"] += 1
    
    all_synthetic_answers = None
    if use_synthetic_answers:
        if not SharedSyntheticAnswerGenerator.is_initialized():
            SharedSyntheticAnswerGenerator.initialize()
        
        all_synthetic_answers = SharedSyntheticAnswerGenerator.generate_batch(
            questions=all_questions, answers=all_gt_answers,
            max_new_tokens=128, show_progress=True
        )
        
        current_idx = 0
        for item in data:
            ans_type = item["answer_type"]
            if "synthetic_answers" not in by_type[ans_type]:
                by_type[ans_type]["synthetic_answers"] = []
            by_type[ans_type]["synthetic_answers"].append(all_synthetic_answers[current_idx])
            current_idx += 1
    
    nlg_scores = get_nlg_scores(all_gt_expls, all_pred_expls, device, model_type='phobert')
    smile_scores = compute_smile_scores(
        all_questions, all_gt_answers, all_pred_answers,
        synthetic_answers=all_synthetic_answers, model_type='phobert'
    )
    
    results = {
        "accuracy": (correct / total * 100) if total > 0 else 0,
        "total_examples": total,
        "correct_count": correct,
        "unfiltered_scores": {**nlg_scores, **smile_scores},
        "by_answer_type": {}
    }
    
    for ans_type, data_type in by_type.items():
        nlg = get_nlg_scores(data_type["gt_expls"], data_type["pred_expls"], device, model_type='phobert')
        smile = compute_smile_scores(
            data_type["questions"], data_type["gt_answers"], data_type["pred_answers"],
            synthetic_answers=data_type.get("synthetic_answers"), model_type='phobert'
        )
        results["by_answer_type"][ans_type] = {
            "accuracy": (data_type["correct"] / data_type["total"] * 100),
            "total_examples": data_type["total"],
            "correct_count": data_type["correct"],
            "unfiltered_scores": {**nlg, **smile},
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions")
    parser.add_argument("--input-dir", type=str, default="src/inference/results/grpo")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-synthetic-answers", action="store_true", default=False)
    parser.add_argument("--syn_ans_model_path", type=str, default=None)
    parser.add_argument("--filenames", nargs="+", default=[])
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()
    
    use_synthetic_answers = not args.no_synthetic_answers
    
    if args.filenames:
        files = [f if f.endswith(".json") else f"{f}.json" for f in args.filenames]
    elif FILES_TO_EVALUATE:
        files = [f if f.endswith(".json") else f"{f}.json" for f in FILES_TO_EVALUATE]
    else:
        files = sorted([f for f in os.listdir(args.input_dir)
                        if f.endswith(".json") and "_score" not in f and "summary" not in f])
    
    SharedBERTScoreModel.get_scorer(model_type='phobert', device=args.device)
    SharedSMILEModel.get_instance(model_type='phobert')
    if use_synthetic_answers:
        SharedSyntheticAnswerGenerator.initialize(model_path=args.syn_ans_model_path, device=args.device)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows = []
    
    for fname in files:
        fpath = os.path.join(args.input_dir, fname)
        result = evaluate_file(fpath, device=args.device, use_synthetic_answers=use_synthetic_answers)
        model_name = os.path.splitext(fname)[0]
        
        all_rows.append({
            "model": model_name, "answer_type": "Overall",
            "total": result["total_examples"], "correct": result["correct_count"],
            "accuracy": round(result["accuracy"], 2),
            **{k: round(v, 2) for k, v in result["unfiltered_scores"].items()}
        })
        
        for ans_type, type_data in result["by_answer_type"].items():
            all_rows.append({
                "model": model_name, "answer_type": ans_type,
                "total": type_data["total_examples"], "correct": type_data["correct_count"],
                "accuracy": round(type_data["accuracy"], 2),
                **{k: round(v, 2) for k, v in type_data["unfiltered_scores"].items()}
            })
    
    df = pd.DataFrame(all_rows)
    
    if args.output_file:
        csv_filename = args.output_file if args.output_file.endswith(".csv") else f"{args.output_file}.csv"
    else:
        first_model = os.path.splitext(files[0])[0]
        csv_filename = f"evaluate_{first_model}_{timestamp}.csv"
    
    csv_path = os.path.join(args.input_dir, csv_filename)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Results saved to: {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

import os
# Set environment variables before torch import
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import json
import argparse
import pandas as pd
from datetime import datetime

from .shared_models import SharedSyntheticAnswerGenerator
from .text_preprocessing import (
    normalize_answer,
    normalize_explanation,
    ensure_list,
)
from .nlg_metrics import get_nlg_scores, compute_smile_scores


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_file(json_path: str, device: str = "cuda", use_synthetic_answers: bool = False) -> dict:
    """
    Evaluate a single prediction file.
    
    Args:
        json_path: Path to JSON file with predictions
        device: Device for model computation
        use_synthetic_answers: Whether to use LLM for synthetic answer generation
        
    Returns:
        Dictionary with evaluation results including accuracy and metric scores
    """
    data = json.load(open(json_path, "r", encoding="utf-8"))
    data = data[:300]  # Limit for testing
    
    total = 0  
    correct = 0
    all_gt_expls, all_pred_expls = [], []
    all_questions, all_gt_answers, all_pred_answers = [], [], []  # For SMILE
    by_type = {}
    
    for item in data:
        total += 1 
        
        gt_ans = normalize_answer(item["answer"])
        pred_ans = normalize_answer(item.get("predict", ""))
        gt_expls = [normalize_explanation(e) for e in ensure_list(item["explanation"])]
        pred_expl = normalize_explanation(item.get("pred_explanation", ""))
        
        all_gt_expls.append(gt_expls)
        all_pred_expls.append(pred_expl)
        
        # Collect raw data for SMILE (before normalization)
        all_questions.append(item.get("question", ""))
        all_gt_answers.append(item.get("answer", ""))
        all_pred_answers.append(item.get("predict", ""))
        
        ans_type = item["answer_type"]
        if ans_type not in by_type:
            by_type[ans_type] = {
                "gt_expls": [], "pred_expls": [], 
                "questions": [], "gt_answers": [], "pred_answers": [],
                "total": 0, "correct": 0
            }
        
        by_type[ans_type]["gt_expls"].append(gt_expls)
        by_type[ans_type]["pred_expls"].append(pred_expl)
        by_type[ans_type]["questions"].append(item.get("question", ""))
        by_type[ans_type]["gt_answers"].append(item.get("answer", ""))
        by_type[ans_type]["pred_answers"].append(item.get("predict", ""))
        by_type[ans_type]["total"] += 1
        
        if pred_ans == gt_ans:
            correct += 1
            by_type[ans_type]["correct"] += 1
    
    # Generate synthetic answers ONCE for all data if requested
    all_synthetic_answers = None
    if use_synthetic_answers:
        if not SharedSyntheticAnswerGenerator.is_initialized():
            SharedSyntheticAnswerGenerator.initialize()
        
        print("📝 Generating synthetic answers for ALL samples...")
        all_synthetic_answers = SharedSyntheticAnswerGenerator.generate_batch(
            questions=all_questions,
            answers=all_gt_answers,
            max_new_tokens=128,
            show_progress=True
        )
        
        # Distribute synthetic answers to subgroups based on indices
        current_idx = 0
        for item in data:
            ans_type = item["answer_type"]
            if "synthetic_answers" not in by_type[ans_type]:
                by_type[ans_type]["synthetic_answers"] = []
            
            by_type[ans_type]["synthetic_answers"].append(all_synthetic_answers[current_idx])
            current_idx += 1

    # Compute NLG scores for explanations
    nlg_scores = get_nlg_scores(all_gt_expls, all_pred_expls, device, model_type='phobert')
    
    # Compute SMILE scores for answers (Overall)
    smile_scores = compute_smile_scores(
        all_questions, all_gt_answers, all_pred_answers, 
        synthetic_answers=all_synthetic_answers,
        model_type='phobert'
    )
    
    results = {
        "accuracy": (correct / total * 100) if total > 0 else 0,
        "total_examples": total,
        "correct_count": correct,
        "unfiltered_scores": {**nlg_scores, **smile_scores},
        "by_answer_type": {}
    }
    
    # Compute per-answer-type scores
    for ans_type, data_type in by_type.items():
        nlg = get_nlg_scores(data_type["gt_expls"], data_type["pred_expls"], device, model_type='phobert')
        
        # Get synthetic answers for this subgroup if available
        subgroup_syn_ans = data_type.get("synthetic_answers", None)
        
        smile = compute_smile_scores(
            data_type["questions"], data_type["gt_answers"], data_type["pred_answers"],
            synthetic_answers=subgroup_syn_ans,
            model_type='phobert'
        )
        
        results["by_answer_type"][ans_type] = {
            "accuracy": (data_type["correct"] / data_type["total"] * 100),
            "total_examples": data_type["total"],
            "correct_count": data_type["correct"],
            "unfiltered_scores": {**nlg, **smile},
        }
    
    return results


# ============================================================================
# MAIN
# ============================================================================

# List of files to evaluate (empty = all JSON files in input-dir)
FILES_TO_EVALUATE = ['4_250_curr_anstype_ver3.json']


def main():
    """Main entry point for VQA evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions")
    parser.add_argument("--input-dir", type=str, default="results",
                        help="Directory containing prediction JSON files")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for computation (cuda or cpu)")
    parser.add_argument("--no-synthetic-answers", action="store_true", default=False,
                        help="Disable LLM synthetic answer generation for SMILE metric (enabled by default)")
    parser.add_argument("--syn-ans-model-path", type=str, default=None,
                        help="Path to LLM model for synthetic answer generation (default: Qwen3-4B-Instruct)")
    parser.add_argument("--filenames", nargs="+", default=[],
                        help="List of specific files to evaluate within input-dir")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Optional output CSV filename (default: evaluate_<model>_<timestamp>.csv)")

    args = parser.parse_args()
    
    # Synthetic answers enabled by default unless --no-synthetic-answers is specified
    use_synthetic_answers = not args.no_synthetic_answers
    
    # Get files to evaluate
    if args.filenames:
        files = [f if f.endswith(".json") else f"{f}.json" for f in args.filenames]
    elif FILES_TO_EVALUATE:
        files = [f if f.endswith(".json") else f"{f}.json" for f in FILES_TO_EVALUATE]
    else:
        files = sorted([f for f in os.listdir(args.input_dir) 
                        if f.endswith(".json") and "_score" not in f and "summary" not in f])
    
    print(f"📁 Evaluating {len(files)} file(s)")
    
    # Pre-initialize synthetic answer generator if needed
    if use_synthetic_answers:
        print("🔧 Pre-initializing Synthetic Answer Generator...")
        SharedSyntheticAnswerGenerator.initialize(
            model_path=args.syn_ans_model_path,
            device=args.device
        )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows = []
    
    for fname in files:
        fpath = os.path.join(args.input_dir, fname)
        print(f"\n🔎 Evaluating: {fname}")
        
        result = evaluate_file(fpath, device=args.device, use_synthetic_answers=use_synthetic_answers)
        model_name = os.path.splitext(fname)[0]
        
        # Add overall results
        all_rows.append({
            "model": model_name,
            "answer_type": "Overall",
            "total": result["total_examples"],
            "correct": result["correct_count"],
            "accuracy": round(result["accuracy"], 2),
            **{k: round(v, 2) for k, v in result["unfiltered_scores"].items()}
        })
        
        # Add per-answer-type results
        for ans_type, type_data in result["by_answer_type"].items():
            all_rows.append({
                "model": model_name,
                "answer_type": ans_type,
                "total": type_data["total_examples"],
                "correct": type_data["correct_count"],
                "accuracy": round(type_data["accuracy"], 2),
                **{k: round(v, 2) for k, v in type_data["unfiltered_scores"].items()}
            })
        
        print(f"   ✅ {model_name}: Accuracy={result['accuracy']:.2f}%")
    
    # Save results to CSV
    df = pd.DataFrame(all_rows)
    
    if args.output_file:
        csv_filename = args.output_file if args.output_file.endswith(".csv") else f"{args.output_file}.csv"
    else:
        first_model = os.path.splitext(files[0])[0]
        csv_filename = f"evaluate_{first_model}_{timestamp}.csv"
    
    csv_path = os.path.join(args.input_dir, csv_filename)
    
    df.to_csv(csv_path, index=False, encoding="utf-8")
    
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()

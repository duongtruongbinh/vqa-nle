import os
# Set environment variables before torch import
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import json
import re
import argparse
import pandas as pd
from datetime import datetime

import numpy as np
from .shared_models import SharedSyntheticAnswerGenerator, SharedSMILEModel
from .text_preprocessing import (
    normalize_answer,
    normalize_answer_vqax,
    normalize_explanation,
    ensure_list,
    clean_text,
)
from .nlg_metrics import get_nlg_scores


# ============================================================================
# EVALUATION
# ============================================================================

def compute_smile_max_ref(questions, gt_answers_list, predictions, synthetic_answers=None):
    """
    Compute SMILE scores with Max-Over-References logic.
    
    This function is implemented locally to avoid modifying nlg_metrics.py
    and to handle list-of-lists for ground truth answers.
    """
    if not questions or not gt_answers_list or not predictions:
        return {"SMILE_avg": 0.0, "SMILE_hm": 0.0}
    
    # Use ground truth answers (first one) as fallback if no synthetic answers provided
    if synthetic_answers is None:
        synthetic_answers = [refs[0] if refs else "" for refs in gt_answers_list]
    else:
         # Normalize synthetic answers
         synthetic_answers = [normalize_answer_vqax(ans) for ans in synthetic_answers]

    if len(synthetic_answers) != len(questions):
        print(f"Warning: Syn Answers ({len(synthetic_answers)}) != Questions ({len(questions)}). Using defaults.")
        synthetic_answers = [refs[0] if refs else "" for refs in gt_answers_list]

    # Normalize predictions
    predictions = [normalize_answer_vqax(p) for p in predictions]
    
    # Prepare flattened data for batch processing
    smile_data = []
    sample_mapping = [] # Maps flattened_index -> original_sample_index
    
    for i, (q, refs, syn_ans, pred) in enumerate(zip(questions, gt_answers_list, synthetic_answers, predictions)):
        if not q or not refs or not pred:
            continue
            
        q_seg = clean_text(q)
        syn_ans_seg = clean_text(syn_ans)
        pred_seg = clean_text(pred)
        
        # Verify
        if not q_seg or not syn_ans_seg or not pred_seg:
             continue
             
        # Create one entry per reference
        for ref in refs:
             ref_norm = normalize_answer_vqax(ref)
             ref_seg = clean_text(ref_norm)
             
             if ref_seg:
                 smile_data.append((q_seg, ref_seg, syn_ans_seg, pred_seg))
                 sample_mapping.append(i)
                 
    if not smile_data:
        return {"SMILE_avg": 0.0, "SMILE_hm": 0.0}
        
    # Get scores from SMILE model
    smile_model = SharedSMILEModel.get_instance(model_type='bert')
    results_flat = smile_model.generate_scores(np.array(smile_data))
    
    # Aggregate: Max over references per sample
    max_avg_scores = {}
    max_hm_scores = {}
    
    avg_flat = results_flat['avg']
    hm_flat = results_flat['hm']
    
    for k, sample_idx in enumerate(sample_mapping):
        s_avg = avg_flat[k]
        s_hm = hm_flat[k]
        
        # Update max for this sample
        if sample_idx not in max_avg_scores:
            max_avg_scores[sample_idx] = s_avg
        else:
             max_avg_scores[sample_idx] = max(max_avg_scores[sample_idx], s_avg)
             
        if sample_idx not in max_hm_scores:
             max_hm_scores[sample_idx] = s_hm
        else:
             max_hm_scores[sample_idx] = max(max_hm_scores[sample_idx], s_hm)
    
    if not max_avg_scores:
        return {"SMILE_avg": 0.0, "SMILE_hm": 0.0}

    # Final result is mean of max scores
    final_avg = np.mean(list(max_avg_scores.values()))
    final_hm = np.mean(list(max_hm_scores.values()))
    
    return {
        "SMILE_avg": float(final_avg) * 100,
        "SMILE_hm": float(final_hm) * 100
    }

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
    all_questions, all_pred_answers = [], []
    all_gt_answers_list = [] # List of lists for SMILE
    all_primary_gt_answers = [] # For synthetic generation
    by_type = {}
    
    for item in data:
        total += 1 
        
        # Get list of valid ground truth answers
        gt_answers_multiref = []
        gt_answers_unsorted = [] # Normalized but NOT SORTED

        def normalize_unsorted(text):
            if not text: return ""
            # Lowercase and clean basic
            text = clean_text(text).lower().strip().rstrip(".").replace('"', "").strip()
            # Remove punctuation (same as normalize_answer_vqax but NO SORT)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        if "answers" in item and isinstance(item["answers"], list):
             for ans_item in item["answers"]:
                 raw_ans = ""
                 if isinstance(ans_item, dict) and "answer" in ans_item:
                     raw_ans = ans_item["answer"]
                 elif isinstance(ans_item, str):
                     raw_ans = ans_item
                 
                 if raw_ans:
                     gt_answers_multiref.append(normalize_answer_vqax(raw_ans))
                     gt_answers_unsorted.append(normalize_unsorted(raw_ans))
        
        # Fallback to single answer if list is empty or missing
        if not gt_answers_multiref:
            raw_ans = item.get("answer", "")
            gt_answers_multiref = [normalize_answer_vqax(raw_ans)]
            gt_answers_unsorted = [normalize_unsorted(raw_ans)]
        
        pred_ans = normalize_answer_vqax(item.get("predict", ""))
        gt_expls = [normalize_explanation(e) for e in ensure_list(item["explanation"])]
        pred_expl = normalize_explanation(item.get("pred_explanation", ""))
        
        all_gt_expls.append(gt_expls)
        all_pred_expls.append(pred_expl)
        
        # Collect raw data for SMILE (before normalization)
        all_questions.append(item.get("question", ""))
        all_pred_answers.append(item.get("predict", ""))
        all_gt_answers_list.append(gt_answers_multiref)
        all_primary_gt_answers.append(item.get("answer", "")) 
        
        ans_type = item.get("answer_type", "other")
        if ans_type not in by_type:
            by_type[ans_type] = {
                "gt_expls": [], "pred_expls": [], 
                "questions": [], "gt_answers_multiref": [], "pred_answers": [],
                "total": 0, "correct": 0
            }
        
        by_type[ans_type]["gt_expls"].append(gt_expls)
        by_type[ans_type]["pred_expls"].append(pred_expl)
        by_type[ans_type]["questions"].append(item.get("question", ""))
        by_type[ans_type]["gt_answers_multiref"].append(gt_answers_multiref)
        by_type[ans_type]["pred_answers"].append(item.get("predict", ""))
        by_type[ans_type]["total"] += 1
        
        if pred_ans in gt_answers_multiref:
            is_correct = True
        else:
            is_correct = False
            # Extended Check: Boolean logic + Substring matching for all types
            if item.get("predict"):
                 c_pred_raw = clean_text(item["predict"]).lower().strip()
                 # Prepare prediction for EXACT substring match (remove punct to match GT format)
                 c_pred_nopunct = re.sub(r'[^\w\s]', '', c_pred_raw)
                 c_pred_nopunct = re.sub(r'\s+', ' ', c_pred_nopunct).strip()
                 
                 c_pred_tokens = set(c_pred_raw.split())
                 
                 YES_SET = {"yes", "true", "correct", "có", "đúng", "vâng"}
                 NO_SET = {"no", "false", "incorrect", "không", "sai"}
                 
                 # 1. Specialized Boolean Logic (using Standard/Sorted GTs)
                 for gt in gt_answers_multiref:
                     if gt in YES_SET:
                         if not c_pred_tokens.isdisjoint(YES_SET):
                             is_correct = True
                             break
                     elif gt in NO_SET:
                         if not c_pred_tokens.isdisjoint(NO_SET):
                             is_correct = True
                             break
                 
                 # 2. General Substring Check (using Unsorted, No-Punct GTs)
                 if not is_correct:
                     for gt_unsorted in gt_answers_unsorted:
                         # We check if the GT (no punct) is inside the Pred (no punct)
                         # This handles "red dog" in "This is a red dog." correctly.
                         if gt_unsorted and gt_unsorted in c_pred_nopunct:
                             is_correct = True
                             break
 
        if is_correct:
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
            answers=all_primary_gt_answers,
            max_new_tokens=128,
            show_progress=True
        )
        
        # Distribute synthetic answers to subgroups based on indices
        current_idx = 0
        for item in data:
            ans_type = item.get("answer_type", "other")
            if "synthetic_answers" not in by_type[ans_type]:
                by_type[ans_type]["synthetic_answers"] = []
            
            by_type[ans_type]["synthetic_answers"].append(all_synthetic_answers[current_idx])
            current_idx += 1
 
    # Compute NLG scores for explanations
    nlg_scores = get_nlg_scores(all_gt_expls, all_pred_expls, device, model_type='bert')
    
    # Compute SMILE scores for answers (Overall) - LOCAL IMPLEMENTATION
    smile_scores = compute_smile_max_ref(
        all_questions, all_gt_answers_list, all_pred_answers, 
        synthetic_answers=all_synthetic_answers
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
        nlg = get_nlg_scores(data_type["gt_expls"], data_type["pred_expls"], device, model_type='bert')
        
        # Get synthetic answers for this subgroup if available
        subgroup_syn_ans = data_type.get("synthetic_answers", None)
        
        smile = compute_smile_max_ref(
            data_type["questions"], data_type["gt_answers_multiref"], data_type["pred_answers"],
            synthetic_answers=subgroup_syn_ans
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
FILES_TO_EVALUATE = ['internvl3-2b.json']


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
    if not df.empty:
      print(f"\n{df.to_string(index=False)}")
    else:
      print("\nDataFrame is empty.")


if __name__ == "__main__":
    main()

import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import argparse
import pandas as pd
from datetime import datetime

from .shared_models import SharedBERTScoreModel, SharedSMILEModel, SharedSyntheticAnswerGenerator
from .text_preprocessing import normalize_answer, normalize_explanation, ensure_list
from .nlg_metrics import get_nlg_scores, compute_smile_scores


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_file(json_path: str, device: str = "cuda", batch_size: int = 8) -> dict:
    """Evaluate predictions - scoring thinking field and answer quality."""
    data = json.load(open(json_path, "r", encoding="utf-8"))
    
    total, correct = 0, 0
    all_gt_expls, all_pred_thinkings = [], []
    all_questions, all_gt_answers, all_pred_answers = [], [], []
    
    for item in data:
        total += 1
        
        gt_ans = normalize_answer(item["answer"])
        pred_ans = normalize_answer(item.get("predict", ""))
        gt_expls = [normalize_explanation(e) for e in ensure_list(item["explanation"])]
        pred_thinking = normalize_explanation(item.get("thinking", ""))
        
        all_gt_expls.append(gt_expls)
        all_pred_thinkings.append(pred_thinking)
        all_questions.append(item.get("question", ""))
        all_gt_answers.append(item.get("answer", ""))
        all_pred_answers.append(item.get("predict", ""))
        
        if pred_ans == gt_ans:
            correct += 1
    
    thinking_scores = get_nlg_scores(all_gt_expls, all_pred_thinkings, device, model_type='phobert')
    
    if not SharedSyntheticAnswerGenerator.is_initialized():
        SharedSyntheticAnswerGenerator.initialize()
    
    synthetic_answers = SharedSyntheticAnswerGenerator.generate_batch(
        all_questions, all_gt_answers, max_new_tokens=128, show_progress=True
    )
    
    smile_scores = compute_smile_scores(
        all_questions, all_gt_answers, all_pred_answers,
        synthetic_answers=synthetic_answers,
        model_type='phobert'
    )
    
    return {
        "accuracy": (correct / total * 100) if total > 0 else 0,
        "total_examples": total,
        "correct_count": correct,
        "thinking_scores": thinking_scores,
        "answer_smile_scores": smile_scores,
    }


# ============================================================================
# MAIN
# ============================================================================

FILES_TO_EVALUATE = ['checkpoint-1000-ver1.json']

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
    print("🔧 Initializing models...")
    SharedSMILEModel.get_instance(model_type='phobert')
    SharedBERTScoreModel.get_scorer(model_type='phobert', device=args.device)
    SharedSyntheticAnswerGenerator.initialize(device=args.device)
    
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
            **{k: round(v, 2) for k, v in result["thinking_scores"].items()},
            **{k: round(v, 2) for k, v in result["answer_smile_scores"].items()}
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

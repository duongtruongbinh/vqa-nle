import os
import re
import json
import argparse
import unicodedata

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from torchmetrics.text import BERTScore

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ============================================================================
# CONFIGURATION
# ============================================================================

# List of files to evaluate. 
# - If empty [], will evaluate ALL .json files in input_dir (or use --files argument)
# - If specified, will only evaluate these files by default
# - Can be overridden by --files argument
FILES_TO_EVALUATE = ['stage1_test_results']

# Example usage:
# FILES_TO_EVALUATE = ["stage3_test_results.json"]
# FILES_TO_EVALUATE = ["model1.json", "model2.json", "model3.json"]


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
    """Normalize answer for exact matching."""
    text = clean_text(text).lower().strip().rstrip(".").replace('"', "").strip()
    
    # Yes/No normalization
    if text in ["có", "đúng", "yes", "true", "correct"]:
        return "có"
    if text in ["không", "sai", "no", "false", "incorrect"]:
        return "không"
    
    # Synonym mapping
    synonym_map = {"bay diều": "thả diều", "diều bay": "thả diều"}
    text = synonym_map.get(text, text)
    
    # Remove prefixes
    prefixes = ["con ", "cái ", "chiếc ", "quả ", "hoa ", "màu ", "bên ", "phía "]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    
    # Remove special characters and sort words
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return " ".join(sorted(text.split()))


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


def compute_bertscore(candidates: list[str], references: list[str], device: str = "cuda") -> float:
    """Compute BERTScore F1 (one-to-one comparison)."""
    if not candidates or not references:
        return 0.0
    
    try:
        bertscore_metric = BERTScore(
            model_name_or_path="/mnt/dataset1/pretrained_fm/vinai/phobert-base",
            num_layers=12,
            rescale_with_baseline=False,
            device=device,
            truncation=True,
            max_length=256
        )
        
        bertscore_metric.update(candidates, references)
        f1_scores = bertscore_metric.compute()['f1']
        
        return (f1_scores.mean().item() if f1_scores.dim() > 0 else f1_scores.item()) * 100
        
    except Exception as e:
        print(f"   ⚠️ Error computing BERTScore: {e}")
        return 0.0


def compute_bertscore_max_ref(hypotheses: list[str], references: list[list[str]], device: str = "cuda") -> list[float]:
    """Compute BERTScore F1 with max over multiple references."""
    batched_cands, batched_refs, ref_counts = [], [], []
    
    for hyp, refs in zip(hypotheses, references):
        valid_refs = [r for r in refs if r.strip()]
        if not hyp.strip() or not valid_refs:
            ref_counts.append(0)
            continue
        
        batched_cands.extend([hyp] * len(valid_refs))
        batched_refs.extend(valid_refs)
        ref_counts.append(len(valid_refs))
    
    max_f1_scores = []
    if batched_cands:
        try:
            bertscore_metric = BERTScore(
                model_name_or_path="/mnt/dataset1/pretrained_fm/vinai/phobert-base",
                num_layers=12,
                rescale_with_baseline=False,
                device=device,
                truncation=True,
                max_length=256
            )
            
            bertscore_metric.update(batched_cands, batched_refs)
            all_f1 = bertscore_metric.compute()['f1']
            all_f1_list = [all_f1.item()] if all_f1.dim() == 0 else all_f1.tolist()
            
            current_idx = 0
            for count in ref_counts:
                if count == 0:
                    max_f1_scores.append(0.0)
                else:
                    max_f1_scores.append(max(all_f1_list[current_idx:current_idx + count]))
                    current_idx += count
                    
        except Exception as e:
            print(f"   ⚠️ Error computing BERTScore: {e}")
            max_f1_scores = [0.0] * len(hypotheses)
    else:
        max_f1_scores = [0.0] * len(hypotheses)
    
    return max_f1_scores


def get_nlg_scores(references: list[list[str]], hypotheses: list[str], device: str = "cuda", max_len: int = 150) -> dict[str, float]:
    """Compute all NLG metrics."""
    # Truncate
    hypotheses = [truncate_sentence(h, max_len) for h in hypotheses]
    references = [[truncate_sentence(r, max_len) for r in refs] for refs in references]
    
    # Prepare for traditional metrics
    gts = {i: [clean_text(r) for r in refs] for i, refs in enumerate(references)}
    res = {i: [clean_text(hyp)] for i, hyp in enumerate(hypotheses)}
    
    # Compute traditional metrics
    scores = compute_traditional_metrics(gts, res)
    
    # Compute BERTScore
    print("   ⏳ Computing BERTScore...")
    max_f1_scores = compute_bertscore_max_ref(hypotheses, references, device)
    scores["BERTScore_F1"] = (sum(max_f1_scores) / len(max_f1_scores) * 100) if max_f1_scores else 0.0
    print("   ✅ BERTScore complete")
    
    return scores


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_file(json_path: str, device: str = "cuda") -> dict:
    """Evaluate a single prediction file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Initialize accumulators
    total = len(data)
    correct = 0
    all_gt_expls, all_pred_expls = [], []
    filtered_gt_expls, filtered_pred_expls = [], []
    all_gt_answers, all_pred_answers = [], []
    by_answer_type = {}
    
    # Process each example
    for item in data:
        gt_ans = normalize_answer(item.get("answer", ""))
        pred_ans = normalize_answer(item.get("predict", ""))
        
        gt_expls = [normalize_explanation(e) for e in ensure_list(item.get("explanation", []))]
        pred_expl = normalize_explanation(item.get("pred_explanation", ""))
        
        all_gt_answers.append(gt_ans)
        all_pred_answers.append(pred_ans)
        all_gt_expls.append(gt_expls)
        all_pred_expls.append(pred_expl)
        
        # Track by answer_type
        ans_type = item.get("answer_type", "unknown")
        if ans_type not in by_answer_type:
            by_answer_type[ans_type] = {
                "total": 0, "correct": 0,
                "all_gt_expls": [], "all_pred_expls": [],
                "filtered_gt_expls": [], "filtered_pred_expls": [],
                "all_gt_answers": [], "all_pred_answers": []
            }
        
        by_answer_type[ans_type]["total"] += 1
        by_answer_type[ans_type]["all_gt_expls"].append(gt_expls)
        by_answer_type[ans_type]["all_pred_expls"].append(pred_expl)
        by_answer_type[ans_type]["all_gt_answers"].append(gt_ans)
        by_answer_type[ans_type]["all_pred_answers"].append(pred_ans)
        
        # Check correctness
        if pred_ans == gt_ans:
            correct += 1
            filtered_gt_expls.append(gt_expls)
            filtered_pred_expls.append(pred_expl)
            by_answer_type[ans_type]["correct"] += 1
            by_answer_type[ans_type]["filtered_gt_expls"].append(gt_expls)
            by_answer_type[ans_type]["filtered_pred_expls"].append(pred_expl)
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    # Compute overall scores
    print("\n--- Computing BERTScore for answers ---")
    answer_bertscore = compute_bertscore(all_pred_answers, all_gt_answers, device)
    
    print("\n--- UNFILTERED evaluation ---")
    unfiltered_scores = get_nlg_scores(all_gt_expls, all_pred_expls, device)
    
    print("\n--- FILTERED evaluation (correct answers only) ---")
    filtered_scores = get_nlg_scores(filtered_gt_expls, filtered_pred_expls, device) if filtered_pred_expls else {k: 0.0 for k in unfiltered_scores}
    
    scaled_scores = {k: v * (correct / total) for k, v in unfiltered_scores.items()}
    
    # Compute scores by answer_type
    answer_type_results = {}
    for ans_type, type_data in by_answer_type.items():
        print(f"\n--- Evaluating answer_type: {ans_type} ({type_data['total']} examples) ---")
        
        type_acc = (type_data["correct"] / type_data["total"]) if type_data["total"] > 0 else 0.0
        type_ans_bert = compute_bertscore(type_data["all_pred_answers"], type_data["all_gt_answers"], device)
        type_unfiltered = get_nlg_scores(type_data["all_gt_expls"], type_data["all_pred_expls"], device)
        type_filtered = get_nlg_scores(type_data["filtered_gt_expls"], type_data["filtered_pred_expls"], device) if type_data["filtered_pred_expls"] else {k: 0.0 for k in type_unfiltered}
        type_scaled = {k: v * type_acc for k, v in type_unfiltered.items()}
        
        answer_type_results[ans_type] = {
            "accuracy": type_acc * 100,
            "answer_bertscore_f1": type_ans_bert,
            "total_examples": type_data["total"],
            "correct_count": type_data["correct"],
            "unfiltered_scores": type_unfiltered,
            "filtered_scores": type_filtered,
            "scaled_scores": type_scaled,
        }
    
    return {
        "accuracy": accuracy,
        "answer_bertscore_f1": answer_bertscore,
        "task_score": accuracy,
        "unfiltered_scores": unfiltered_scores,
        "filtered_scores": filtered_scores,
        "scaled_scores": scaled_scores,
        "total_examples": total,
        "correct_count": correct,
        "by_answer_type": answer_type_results,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions")
    parser.add_argument("--input-dir", type=str, default="results", help="Directory with prediction files")
    parser.add_argument("--device", type=str, default="cuda", help="Device for BERTScore")
    parser.add_argument("--suffix", type=str, default="_score.json", help="Output file suffix")
    parser.add_argument("--files", type=str, nargs="+", default=None, 
                        help="Specific files to evaluate (e.g., model1.json model2.json). If not provided, evaluates all files.")
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"❌ Directory not found: {args.input_dir}")
        return
    
    # Determine which files to process
    # Priority: 1) --files argument, 2) FILES_TO_EVALUATE config, 3) all files
    if args.files:
        # User specified files via argument (highest priority)
        print(f"📋 Using --files argument")
        files = []
        for f in args.files:
            if not f.endswith(".json"):
                f = f"{f}.json"
            if os.path.exists(os.path.join(args.input_dir, f)):
                files.append(f)
            else:
                print(f"⚠️ File not found, skipping: {f}")
        files = sorted(files)
    elif FILES_TO_EVALUATE:
        # User specified files in config (medium priority)
        print(f"📋 Using FILES_TO_EVALUATE config: {FILES_TO_EVALUATE}")
        files = []
        for f in FILES_TO_EVALUATE:
            if not f.endswith(".json"):
                f = f"{f}.json"
            if os.path.exists(os.path.join(args.input_dir, f)):
                files.append(f)
            else:
                print(f"⚠️ File not found, skipping: {f}")
        files = sorted(files)
    else:
        # Default: all JSON files (lowest priority)
        print(f"📋 No specific files specified, evaluating all files")
        files = sorted([f for f in os.listdir(args.input_dir) 
                        if f.endswith(".json") and "_score" not in f and "summary" not in f])
    
    if not files:
        print(f"⚠️ No .json files to evaluate in {args.input_dir}")
        return
    
    print(f"📁 Found {len(files)} file(s) to evaluate")
    
    # Load existing summary if it exists
    summary_path = os.path.join(args.input_dir, "evaluation_summary.txt")
    existing_summary = {}
    
    if os.path.exists(summary_path):
        print(f"📖 Loading existing summary from {summary_path}")
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Skip header line
                for line in lines[1:]:
                    if line.strip():
                        parts = line.strip().split("\t")
                        if len(parts) >= 8:
                            model_name = parts[0]
                            existing_summary[model_name] = {
                                "BLEU-4": float(parts[1]),
                                "METEOR": float(parts[2]),
                                "ROUGE_L": float(parts[3]),
                                "CIDEr": float(parts[4]),
                                "BERTScore_F1": float(parts[5]),
                                "answer_bertscore_f1": float(parts[6]),
                                "accuracy": float(parts[7]),
                            }
                            # Try to load answer_type specific scores if they exist
                            if len(parts) >= 14:
                                existing_summary[model_name].update({
                                    "acc_yes/no": float(parts[8]),
                                    "acc_number": float(parts[9]),
                                    "acc_other": float(parts[10]),
                                    "bert_expl_yes/no": float(parts[11]),
                                    "bert_expl_number": float(parts[12]),
                                    "bert_expl_other": float(parts[13]),
                                })
            print(f"   ✅ Loaded {len(existing_summary)} existing results")
        except Exception as e:
            print(f"   ⚠️ Could not parse existing summary: {e}")
            existing_summary = {}
    
    # Start with existing summary
    summary = existing_summary.copy()
    
    for fname in files:
        fpath = os.path.join(args.input_dir, fname)
        print(f"\n{'='*45}\n🔎 Evaluating: {fname}\n{'='*45}")
        
        try:
            result = evaluate_file(fpath, device=args.device)
        except Exception as e:
            print(f"   ❌ Critical error: {e}")
            continue
        
        # Round and save
        for key in ["unfiltered_scores", "filtered_scores", "scaled_scores"]:
            if key in result:
                result[key] = {k: round(v, 2) for k, v in result[key].items()}
        result["accuracy"] = round(result["accuracy"], 2)
        result["answer_bertscore_f1"] = round(result["answer_bertscore_f1"], 2)
        
        if "by_answer_type" in result:
            for type_result in result["by_answer_type"].values():
                for key in ["unfiltered_scores", "filtered_scores", "scaled_scores"]:
                    if key in type_result:
                        type_result[key] = {k: round(v, 2) for k, v in type_result[key].items()}
                type_result["accuracy"] = round(type_result["accuracy"], 2)
                type_result["answer_bertscore_f1"] = round(type_result["answer_bertscore_f1"], 2)
        
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(args.input_dir, f"{base}{args.suffix}")
        
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(result, fw, indent=2, ensure_ascii=False)
        print(f"\n   💾 Saved to: {out_path}")
        
        # Update summary (overwrite if exists, add if new)
        summary[base] = {
            "accuracy": result["accuracy"],
            "answer_bertscore_f1": result["answer_bertscore_f1"],
            "BLEU-4": result["filtered_scores"].get("BLEU-4", 0.0),
            "METEOR": result["filtered_scores"].get("METEOR", 0.0),
            "ROUGE_L": result["filtered_scores"].get("ROUGE_L", 0.0),
            "CIDEr": result["filtered_scores"].get("CIDEr", 0.0),
            "BERTScore_F1": result["filtered_scores"].get("BERTScore_F1", 0.0),
        }
        
        # Add answer_type specific metrics (computed from by_answer_type in evaluate_file)
        if "by_answer_type" in result:
            for ans_type, type_result in result["by_answer_type"].items():
                # Store accuracy and BERTScore for explanations by answer_type
                summary[base][f"acc_{ans_type}"] = type_result["accuracy"]
                summary[base][f"bert_expl_{ans_type}"] = type_result["filtered_scores"].get("BERTScore_F1", 0.0)
        
        print(f"   🔄 {'Updated' if base in existing_summary else 'Added'} {base} in summary")
    
    # Write summary (with all models: old + new/updated)
    with open(summary_path, "w", encoding="utf-8") as fw:
        # Header with answer_type columns
        fw.write("Model\tBLEU-4\tMETEOR\tROUGE_L\tCIDEr\tBERTScore_Expl\tBERTScore_Ans\tAccuracy\t"
                "Acc_YesNo\tAcc_Number\tAcc_Other\tBERT_YesNo\tBERT_Number\tBERT_Other\n")
        
        # Sort by model name for consistency
        for model in sorted(summary.keys()):
            scores = summary[model]
            
            # Get answer_type specific scores (default to 0 if not available)
            acc_yesno = scores.get("acc_yes/no", 0.0)
            acc_number = scores.get("acc_number", 0.0)
            acc_other = scores.get("acc_other", 0.0)
            bert_yesno = scores.get("bert_expl_yes/no", 0.0)
            bert_number = scores.get("bert_expl_number", 0.0)
            bert_other = scores.get("bert_expl_other", 0.0)
            
            fw.write(f"{model}\t{scores['BLEU-4']:.2f}\t{scores['METEOR']:.2f}\t"
                    f"{scores['ROUGE_L']:.2f}\t{scores['CIDEr']:.2f}\t"
                    f"{scores['BERTScore_F1']:.2f}\t{scores['answer_bertscore_f1']:.2f}\t"
                    f"{scores['accuracy']:.2f}\t"
                    f"{acc_yesno:.2f}\t{acc_number:.2f}\t{acc_other:.2f}\t"
                    f"{bert_yesno:.2f}\t{bert_number:.2f}\t{bert_other:.2f}\n")
    
    print(f"\n\n{'='*45}\n✅ All evaluations complete\n{'='*45}")
    with open(summary_path, "r") as f:
        print(f.read())
    print(f"📊 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

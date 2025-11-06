import string
import re
from pycocoevalcap.rouge.rouge import Rouge
# src/rewards/outcome_rewards.py
from base_rewards import BaseRewardScorer
# def accuracy_reward(completion: str, solution: str, partial_threshold: float = 0.5) -> float:
#     """
#     Reward function cho short-text answers:
#       - 3.0 nếu câu trả lời khớp hoàn toàn với ground truth.
#       - 1.5 nếu số từ trùng khớp >= 50% số từ trong ground truth.
#       - 0.0 nếu không đáp ứng các điều kiện trên.
#     """
#     sol_match = re.search(r"<answer>(.*?)</answer>", solution, flags=re.DOTALL | re.IGNORECASE)
#     ground_truth = sol_match.group(1).strip() if sol_match else solution.strip()

#     # Extract student answer
#     content_match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL | re.IGNORECASE)
#     student_answer = content_match.group(1).strip() if content_match else ""

#     # Normalize
#     def normalize_text(text: str) -> str:
#         return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

#     normalized_gt = normalize_text(ground_truth)
#     normalized_sa = normalize_text(student_answer)

#     # Scoring
#     if not normalized_sa or not normalized_gt:
#         return 0.0
#     if normalized_sa == normalized_gt:
#         return 3.0

#     gt_tokens = [w for w in normalized_gt.split() if w]
#     sa_token_set = set(normalized_sa.split())
#     overlap = sum(1 for w in gt_tokens if w in sa_token_set)
#     ratio = overlap / len(gt_tokens) if gt_tokens else 0.0

#     return 1.5 if ratio >= partial_threshold else 0.0




class AccuracyRewardScorer(BaseRewardScorer):
    """
    Accuracy reward scorer sử dụng hybrid approach: ROUGE-L + BERTScore (PhoBERT)
    Trả về điểm kết hợp (0.0 - 1.0).
    """
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Trọng số cho BERTScore (1-alpha cho ROUGE-L)
                  alpha=0.5 => 50% BERTScore, 50% ROUGE-L
        """
        self.bertscore_metric = self.get_bertscore_metric()
        self.rouge_scorer = Rouge()
        self.alpha = alpha
    
    def calculate_rouge_batch(self, ground_truths: dict, predictions: dict) -> dict:
        """
        Tính ROUGE-L cho batch predictions.
        
        Args:
            ground_truths: {id: ground_truth_string}
            predictions: {id: prediction_string}
            
        Returns:
            {id: rouge_l_score}
        """
        if not predictions:
            return {}
        
        gts = {}
        res = {}
        
        for id_, pred in predictions.items():
            gt = ground_truths.get(id_, "")
            if isinstance(gt, str):
                gt_text = gt.strip()
            else:
                # Nếu là list, lấy phần tử đầu tiên
                gt_text = gt[0].strip() if gt else ""
            
            # ROUGE yêu cầu format đặc biệt
            gts[id_] = [gt_text.lower()] if gt_text else [""]
            res[id_] = [pred.lower().strip()]
        
        try:
            avg_score, individual_scores = self.rouge_scorer.compute_score(gts, res)
            # individual_scores là list theo thứ tự của gts/res keys
            rouge_dict = {id_: score for id_, score in zip(gts.keys(), individual_scores)}
            return rouge_dict
        except Exception as e:
            print(f"Error calculating ROUGE-L: {e}")
            return {id_: 0.0 for id_ in predictions.keys()}
    
    def accuracy_reward(self, completion: str, solution: str) -> float:
        """
        Single sample version - tính hybrid reward cho một sample.
        """
        # Extract answers
        sol_match = re.search(r"<answer>(.*?)</answer>", solution, flags=re.DOTALL | re.IGNORECASE)
        ground_truth = sol_match.group(1).strip() if sol_match else solution.strip()

        content_match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL | re.IGNORECASE)
        student_answer = content_match.group(1).strip() if content_match else ""

        if not student_answer or not ground_truth:
            return 0.0

        # Tính cả hai metrics
        gts_dict = {0: ground_truth}
        preds_dict = {0: student_answer}
        
        bert_score = self.calculate_bertscore_batch(gts_dict, preds_dict).get(0, 0.0)
        rouge_score = self.calculate_rouge_batch(gts_dict, preds_dict).get(0, 0.0)
        
        # Combined reward
        reward = self.alpha * bert_score + (1.0 - self.alpha) * rouge_score
        
        return reward
    
    def accuracy_rewards_batch(self, completions: list[str], solutions: list[str]) -> list[float]:
        """
        Batch version - trả về hybrid reward (ROUGE-L + BERTScore) cho từng sample (0-1).
        """
        if not completions:
            return []
        
        # Extract tất cả answers
        gts_dict = {}
        preds_dict = {}
        
        for i, (completion, solution) in enumerate(zip(completions, solutions)):
            sol_match = re.search(r"<answer>(.*?)</answer>", solution, flags=re.DOTALL | re.IGNORECASE)
            ground_truth = sol_match.group(1).strip() if sol_match else solution.strip()

            content_match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL | re.IGNORECASE)
            student_answer = content_match.group(1).strip() if content_match else ""
            
            gts_dict[i] = ground_truth
            preds_dict[i] = student_answer
        
        # Tính cả ROUGE-L và BERTScore batch
        rouge_scores = self.calculate_rouge_batch(gts_dict, preds_dict)
        bert_scores = self.calculate_bertscore_batch(gts_dict, preds_dict)
        
        # Kết hợp scores với trọng số alpha
        rewards = []
        for i in range(len(completions)):
            rouge_score = rouge_scores.get(i, 0.0)
            bert_score = bert_scores.get(i, 0.0)
            
            # Hybrid reward
            reward = self.alpha * bert_score + (1.0 - self.alpha) * rouge_score
            rewards.append(reward)
            
            print(f"  [Sample {i}] ROUGE-L={rouge_score:.4f}, BERTScore={bert_score:.4f} -> Reward={reward:.4f}")
        
        return rewards
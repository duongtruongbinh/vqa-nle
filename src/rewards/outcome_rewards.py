import string
import re
from typing import Optional, List
from explanation_rewards import ExplanationRewardScorer


def accuracy_reward(completions, solution, image_ids, problems, **kwargs):
    """
    Reward function cho short-text answers:
      - 1.0 nếu câu trả lời của học sinh khớp hoàn toàn với ground truth.
      - 0.5 nếu số từ trùng khớp >= 50% số từ trong ground truth.
      - 0.0 nếu không đáp ứng các điều kiện trên.
    """
    rewards = []
    PARTIAL_THRESHOLD = 0.5

    for comp, sol, image_id, problem in zip(completions, solution, image_ids, problems):
        # Giả sử completion là chuỗi; nếu là list of dict: content = comp[0]["content"]
        content = comp
        reward = 0.0

        # --- Trích xuất câu trả lời từ ground truth và model ---
        sol_match = re.search(r"<answer>(.*?)</answer>",
                              sol, flags=re.DOTALL | re.IGNORECASE)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

        content_match = re.search(
            r"<answer>(.*?)</answer>", content, flags=re.DOTALL | re.IGNORECASE)
        student_answer = content_match.group(
            1).strip() if content_match else ""

        # --- Chuẩn hóa văn bản để so sánh ---
        def normalize_text(text):
            return (
                text.lower()
                    .translate(str.maketrans('', '', string.punctuation))
                    .strip()
            )

        normalized_gt = normalize_text(ground_truth)
        normalized_sa = normalize_text(student_answer)

        # --- Logic chấm điểm ---
        if not normalized_sa or not normalized_gt:
            reward = 0.0
        elif normalized_sa == normalized_gt:
            reward = 1.0
        else:
            gt_tokens = [w for w in normalized_gt.split() if w]
            sa_token_set = set(normalized_sa.split())
            overlap = sum(1 for w in gt_tokens if w in sa_token_set)
            ratio = overlap / len(gt_tokens) if gt_tokens else 0.0

            if ratio >= PARTIAL_THRESHOLD:
                reward = 0.5
            else:
                reward = 0.0

        rewards.append(reward)

    return rewards


def explation_reward(ground_truths: list, predictions: list, image_paths: list, alpha = 0.5):
    scorer = ExplanationRewardScorer(alpha = alpha)
    rewards = scorer.explanation_rewards(
        ground_truths=ground_truths,
        predictions=predictions,
        image_paths=image_paths
    )
    return rewards


import string
import re

def accuracy_reward(completion: str, solution: str, partial_threshold: float = 0.5) -> float:
    """
    Reward function cho short-text answers:
      - 3.0 nếu câu trả lời khớp hoàn toàn với ground truth.
      - 1.5 nếu số từ trùng khớp >= 50% số từ trong ground truth.
      - 0.0 nếu không đáp ứng các điều kiện trên.
    """
    sol_match = re.search(r"<answer>(.*?)</answer>", solution, flags=re.DOTALL | re.IGNORECASE)
    ground_truth = sol_match.group(1).strip() if sol_match else solution.strip()

    # Extract student answer
    content_match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL | re.IGNORECASE)
    student_answer = content_match.group(1).strip() if content_match else ""

    # Normalize
    def normalize_text(text: str) -> str:
        return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

    normalized_gt = normalize_text(ground_truth)
    normalized_sa = normalize_text(student_answer)

    # Scoring
    if not normalized_sa or not normalized_gt:
        return 0.0
    if normalized_sa == normalized_gt:
        return 3.0

    gt_tokens = [w for w in normalized_gt.split() if w]
    sa_token_set = set(normalized_sa.split())
    overlap = sum(1 for w in gt_tokens if w in sa_token_set)
    ratio = overlap / len(gt_tokens) if gt_tokens else 0.0

    return 1.5 if ratio >= partial_threshold else 0.0



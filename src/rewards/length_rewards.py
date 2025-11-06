def length_penalty_answer(pred, truth, ratio=1.3):
    pred_len = len(pred.split())
    truth_len = len(truth.split())
    max_ok = round(truth_len * ratio) # mở rộng biên độ chấp nhận
    
    if pred_len <= max_ok:
        return 1.0
    else:
        excess_ratio = (pred_len - max_ok) / truth_len
        return - min(excess_ratio, 1.0)


def length_penalty_explanation(pred, truth, ratio=1.2, sentence_penalty_weight=0.3):
    """
    Penalty function với kiểm soát số câu:
    - Nếu pred có >= 2 câu (phân tách bởi '.' hoặc ',') -> phạt dựa trên số câu
    - Nếu pred chỉ 1 câu -> kiểm tra độ dài như bình thường
    
    Args:
        pred: Câu prediction
        truth: Câu ground truth
        ratio: Tỷ lệ độ dài tối đa cho phép
        sentence_penalty_weight: Trọng số phạt cho mỗi câu thừa (mặc định 0.3)
    
    Returns:
        float: 1.0 (ok), -0.5 (quá ngắn), hoặc giá trị âm (quá dài/nhiều câu)
    """
    # Tách câu dựa trên dấu '.' hoặc ',' (loại bỏ câu rỗng)
    import re
    pred_sentences = [s.strip() for s in re.split(r'[.,]', pred) if s.strip()]
    truth_sentences = [s.strip() for s in re.split(r'[.,]', truth) if s.strip()]
    
    num_pred_sentences = len(pred_sentences)
    num_truth_sentences = len(truth_sentences)
    
    # Kiểm tra số câu: Nếu pred có >= 2 câu mà truth chỉ 1 câu -> phạt
    if num_truth_sentences == 1 and num_pred_sentences >= 2:
        # Phạt càng nặng khi gen càng nhiều câu
        # Ví dụ: 2 câu -> -0.3, 3 câu -> -0.6, 4 câu -> -0.9, 5+ câu -> -1.0
        excess_sentences = num_pred_sentences - 1
        penalty = -min(excess_sentences * sentence_penalty_weight, 1.0)
        return penalty
    
    # Nếu số câu hợp lệ -> kiểm tra độ dài từ
    pred_len = len(pred.split())
    truth_len = len(truth.split())
    min_ok = max(5, int(truth_len * 0.7))
    max_ok = round(truth_len * ratio)
    
    if min_ok <= pred_len <= max_ok:
        return 1.0
    elif pred_len < min_ok:
        return -0.5
    else:
        excess_ratio = (pred_len - max_ok) / truth_len
        return -min(excess_ratio, 1.0)

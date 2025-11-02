def length_penalty_answer(pred, truth, ratio=1.3):
    pred_len = len(pred.split())
    truth_len = len(truth.split())
    max_ok = round(truth_len * ratio) # mở rộng biên độ chấp nhận
    
    if pred_len <= max_ok:
        return 1.0
    else:
        excess_ratio = (pred_len - max_ok) / truth_len
        return - min(excess_ratio, 1.0)


def length_penalty_explanation(pred, truth, ratio=1.2):
    pred_len = len(pred.split())
    truth_len = len(truth.split())
    min_ok = max(5, int(truth_len * 0.7)) # mở rộng biên độ chấp nhận
    max_ok = round(truth_len * ratio) # mở rộng biên độ chấp nhận
    
    if min_ok <= pred_len <= max_ok:
        return 1.0
    elif pred_len < min_ok:
        return -0.5
    else:
        excess_ratio = (pred_len - max_ok) / truth_len
        return - min(excess_ratio, 1.0)
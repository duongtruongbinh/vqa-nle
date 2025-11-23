import json
import pandas as pd

# Đọc file JSON
with open('/home/vlai-vqa-nle/minhtq/vqa-nle/src/inference/results/grpo/BS2_ver3.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Danh sách để lưu kết quả
results = []

# Lặp qua từng sample
for sample in data:
    predict = sample.get('predict', '')
    pred_explanation = sample.get('pred_explanation', '')
    thinking = sample.get('thinking', '')
    
    # Đếm độ dài
    predict_length = len(predict)
    pred_explanation_length = len(pred_explanation)
    thinking_length = len(thinking)
    
    # Đếm số từ (tách theo khoảng trắng)
    predict_words = len(predict.split())
    pred_explanation_words = len(pred_explanation.split())
    thinking_words = len(thinking.split()) if thinking else 0
    
    results.append({
        'question_id': sample.get('question_id'),
        'answer_type': sample.get('answer_type'),
        'predict': predict,
        'predict_length': predict_length,
        'predict_words': predict_words,
        'pred_explanation': pred_explanation,
        'pred_explanation_length': pred_explanation_length,
        'pred_explanation_words': pred_explanation_words,
        'thinking': thinking,
        'thinking_length': thinking_length,
        'thinking_words': thinking_words
    })

# Tạo DataFrame
df = pd.DataFrame(results)

# Thống kê tổng quan
print("=" * 80)
print("THỐNG KÊ TỔNG QUAN")
print("=" * 80)
print(f"Tổng số samples: {len(df)}")
print()

print("THỐNG KÊ CHO PREDICT:")
print(f"  - Độ dài trung bình (ký tự): {df['predict_length'].mean():.2f}")
print(f"  - Độ dài min: {df['predict_length'].min()}")
print(f"  - Độ dài max: {df['predict_length'].max()}")
print(f"  - Số từ trung bình: {df['predict_words'].mean():.2f}")
print()

print("THỐNG KÊ CHO PRED_EXPLANATION:")
print(f"  - Độ dài trung bình (ký tự): {df['pred_explanation_length'].mean():.2f}")
print(f"  - Độ dài min: {df['pred_explanation_length'].min()}")
print(f"  - Độ dài max: {df['pred_explanation_length'].max()}")
print(f"  - Số từ trung bình: {df['pred_explanation_words'].mean():.2f}")
print()

print("THỐNG KÊ CHO THINKING:")
print(f"  - Độ dài trung bình (ký tự): {df['thinking_length'].mean():.2f}")
print(f"  - Độ dài min: {df['thinking_length'].min()}")
print(f"  - Độ dài max: {df['thinking_length'].max()}")
print(f"  - Số từ trung bình: {df['thinking_words'].mean():.2f}")
print(f"  - Số samples có thinking: {(df['thinking_length'] > 0).sum()}")
print(f"  - Số samples không có thinking: {(df['thinking_length'] == 0).sum()}")
print()

# Thống kê theo answer_type
print("=" * 80)
print("THỐNG KÊ THEO ANSWER_TYPE")
print("=" * 80)
stats_by_type = df.groupby('answer_type').agg({
    'predict_length': ['mean', 'min', 'max'],
    'predict_words': 'mean',
    'pred_explanation_length': ['mean', 'min', 'max'],
    'pred_explanation_words': 'mean',
    'thinking_length': ['mean', 'min', 'max'],
    'thinking_words': 'mean'
}).round(2)
print(stats_by_type)
print()


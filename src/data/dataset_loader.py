import os
import json

SYSTEM_PROMPT_VIVQA_ENHANCED = (
    "<image>Câu hỏi: {question}\n"
    "ĐỊNH DẠNG:\n"
    "<thinking>[Suy luận ngắn gọn]</thinking>\n"
    "<answer>[Câu trả lời]</answer>\n"
    "<explain>[Giải thích 10-15 từ]</explain>"
)


def create_jsonl_for_grpo(split="train", output_file=None):
    """
    Tạo file JSONL theo format của VLM-R1 GRPO
    """
    data_dir = "/mnt/VLAI_data/ViVQA-X"

    if split == 'train':
        data_path = os.path.join(data_dir, 'ViVQA-X_train.json')
        image_dir = 'train2014'  # Đường dẫn tương đối
    elif split == 'val':
        data_path = os.path.join(data_dir, 'ViVQA-X_val.json')
        image_dir = 'val2014'
    else:  # 'test'
        data_path = os.path.join(data_dir, 'ViVQA-X_test.json')
        image_dir = 'val2014'

    # Tải dữ liệu từ file JSON
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Đặt tên file output
    if output_file is None:
        output_dir = 'data/processed'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'ViVQA-X_{split}_grpo.jsonl')

    # Xử lý và ghi vào JSONL
    sample_count = 0
    max_samples = 50
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, item in enumerate(raw_data):
            # Kiểm tra giới hạn số lượng samples
            if max_samples is not None and sample_count >= max_samples:
                break

            image_name = item.get('image_name')
            question = item.get('question')
            answer = item.get('answer')
            explanations = item.get('explanation')

            if not all([image_name, question, answer, explanations, explanations[0]]):
                continue

            explanation = explanations[0]

            # Tạo đường dẫn tương đối (chỉ tên file hoặc thư mục con)
            # VLM-R1 sẽ tự động ghép với image_folders
            relative_image_path = os.path.join(image_dir, image_name)

            # Tạo prompt với format của bạn
            question_with_prompt = SYSTEM_PROMPT_VIVQA_ENHANCED.format(
                question=question)

            # Format solution với thinking tags nếu cần RL training
            # Hoặc đơn giản chỉ là answer + explain
            solution = f"<answer>{answer}</answer><explain>{explanation}</explain>"

            # Tạo entry theo format VLM-R1
            entry = {
                "id": idx + 1,
                "image": image_name,  # CHỈ TÊN FILE, không có đường dẫn đầy đủ
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{question_with_prompt}"
                    },
                    {
                        "from": "gpt",
                        "value": solution
                    }
                ]
            }

            # Ghi một dòng JSONL
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            sample_count += 1

    print(f"✅ Đã tạo file JSONL: {output_file} với {sample_count} samples")
    return output_file


if __name__ == "__main__":
    # Tạo file JSONL cho train và val
    create_jsonl_for_grpo("train")
    create_jsonl_for_grpo("val")
    create_jsonl_for_grpo("test")

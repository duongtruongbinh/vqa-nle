import os
import json


# System prompt - VQA Task
SYSTEM_PROMPT = """<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc.""".strip()


# User content template
USER_CONTENT_TEMPLATE = """
Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong ba giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
<REASONING>[Đưa ra phân tích lập luận chi tiết, từng bước để giải quyết vấn đề.]</REASONING>
<CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>
<EXPLANATION>[Tổng hợp các thông tin từ REASONING và cho ra câu mô tả ngắn gọn các phân tích đặc điểm.] Hình ảnh cho thấy...</EXPLANATION>
    
Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
Câu trả lời:""".strip()


def create_jsonl_for_msswift(split="train", output_file=None, image_base_dir="/mnt/VLAI_data/COCO_Images"):
    """
    Tạo file JSONL theo format của MS-Swift với system prompt và user content (Standard/Stage 2).
    Format: {"messages": [...], "images": [...], "solution": "..."}
    """
    data_dir = "/mnt/VLAI_data/ViVQA-X"

    if split == 'train':
        data_path = os.path.join(data_dir, 'ViVQA-X_train.json')
        image_dir = 'train2014'
    elif split == 'val':
        data_path = os.path.join(data_dir, 'ViVQA-X_val.json')
        image_dir = 'val2014'
    else:
        # Test split usually uses val2014 or test2015 depending on dataset, assuming val2014 here as before
        data_path = os.path.join(data_dir, 'ViVQA-X_test.json')
        image_dir = 'val2014'

    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: File not found: {data_path}")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if output_file is None:
        # Default output directory for standard dataset (all stage 2 now)
        output_dir = f'data/processed/standard'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'ViVQA-X_{split}_msswift.jsonl')

    print(f"Writing to: {output_file}")
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, item in enumerate(raw_data):
            image_name = item.get('image_name')
            image_id = item.get('image_id')
            question = item.get('question')
            answer = item.get('answer')
            explanations = item.get('explanation')

            # Validate required fields
            if not all([image_name, question, answer, explanations]):
                continue
            
            # Extract explanation (handle list or string)
            if isinstance(explanations, list) and len(explanations) > 0:
                explanation = explanations[0]
            elif isinstance(explanations, str):
                explanation = explanations
            else:
                continue

            # Construct absolute image path
            absolute_image_path = os.path.join(image_base_dir, image_dir, image_name)

            # Format user content
            user_content = USER_CONTENT_TEMPLATE.format(question=question)

            # Format assistant response (Solution)
            solution = f"<CONCLUSION>{answer}</CONCLUSION>\n<EXPLANATION>{explanation}</EXPLANATION>"

            # MS-Swift format
            entry = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "images": [absolute_image_path], 
                "solution": solution # Duplicate for clarity/compatibility if needed
            }

            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1

    print(f"Created {output_file} with {count} entries")
    return output_file


if __name__ == "__main__":
    IMAGE_BASE_DIRECTORY = "/mnt/VLAI_data/COCO_Images"
    
    print("--- Generating Standard Dataset ---")
    create_jsonl_for_msswift("train", image_base_dir=IMAGE_BASE_DIRECTORY)
    create_jsonl_for_msswift("val", image_base_dir=IMAGE_BASE_DIRECTORY)
    create_jsonl_for_msswift("test", image_base_dir=IMAGE_BASE_DIRECTORY)
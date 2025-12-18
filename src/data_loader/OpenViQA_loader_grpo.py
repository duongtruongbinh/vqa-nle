import os
import json


# system_instruction_vintern3BR = """<image>Bạn là hệ thống Visual Question Answering (VQA). Nhiệm vụ của bạn là trả lời và giải thích các câu hỏi dựa trên nội dung của hình ảnh được cung cấp.
#     Câu hỏi: {question}
#     Vui lòng trả lời câu hỏi sau dựa trên hình ảnh. Hãy trả lời theo định dạng sau:
#     <answer>Câu trả lời (một từ hoặc cụm từ ngắn)</answer>
#     <think>Giải thích một câu ngắn gọn chứng minh câu trả lời</think>""".strip()

SYSTEM_PROMPTS = """<image>Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc."""
USER_PROMPTS = """
    Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong ba giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
    <REASONING>[Đưa ra phân tích lập luận chi tiết, từng bước để giải quyết vấn đề.]</REASONING>
    <CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>
    <EXPLANATION>[Tổng hợp các thông tin từ REASONING và cho ra câu mô tả ngắn gọn các phân tích đặc điểm.] Hình ảnh cho thấy...</EXPLANATION>
    Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
    Câu trả lời:
"""


def create_jsonl_for_msswift(split="train", output_file=None, data_base_dir="/mnt/VLAI_data/OpenViVQA"):
    split_mapping = {
        'train': ('vlsp2023_train_data.json', 'training-images'),
        'dev': ('vlsp2023_dev_data.json', 'dev-images'),
        'val': ('vlsp2023_dev_data.json', 'dev-images'),
        'test': ('vlsp2023_test_data.json', 'test-images')
    }
    
    if split not in split_mapping:
        print(f"Error: Invalid split '{split}'. Available splits are: {list(split_mapping.keys())}")
        return
    
    data_file, image_dir = split_mapping[split]
    data_path = os.path.join(data_base_dir, data_file)
    
    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: File not found: {data_path}")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    images_dict = raw_data.get('images', {})
    annotations_dict = raw_data.get('annotations', {})
    
    if output_file is None:
        output_dir = f'data/processed/openvivqa'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'OpenViVQA_{split}_msswift.jsonl')
    
    print(f"Writing to: {output_file}")
    count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for ann_id, annotation in annotations_dict.items():
            image_id = str(annotation.get('image_id'))
            question = annotation.get('question')
            answer = annotation.get('answer')
            
            if not all([image_id, question, answer]):
                continue
            
            image_filename = images_dict.get(image_id)
            if not image_filename:
                continue
            
            absolute_image_path = os.path.join(data_base_dir, image_dir, image_filename)
            
            user_content = USER_PROMPTS.format(question=question)
            system_prompt = SYSTEM_PROMPTS
            
            full_response = f"<answer>{answer}</answer>"
            
            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "images": [absolute_image_path], 
                "solution": full_response
            }
            
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1
    
    print(f"Created {output_file} with {count} entries")
    return output_file


if __name__ == "__main__":
    DATA_BASE_DIRECTORY = "/mnt/VLAI_data/OpenViVQA"
    
    print("--- Generating OpenViVQA Dataset for GRPO ---")
    create_jsonl_for_msswift("train", data_base_dir=DATA_BASE_DIRECTORY)
    create_jsonl_for_msswift("dev", data_base_dir=DATA_BASE_DIRECTORY)
    create_jsonl_for_msswift("test", data_base_dir=DATA_BASE_DIRECTORY)

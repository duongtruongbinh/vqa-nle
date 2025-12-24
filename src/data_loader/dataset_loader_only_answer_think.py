import os
import json

SYSTEM_PROMPT = """<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc."""

USER_CONTENT_TEMPLATE = """Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong hai giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
<REASONING>[Đưa ra phân tích lập luận chi tiết, từng bước để giải quyết vấn đề.]</REASONING>
<CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>

Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
Câu trả lời:
"""

def create_jsonl_for_msswift(split="train", output_file=None, image_base_dir="/mnt/VLAI_data/COCO_Images"):
    """
    Tạo file JSONL theo format của MS-Swift với system prompt và user content riêng biệt
    Format: {"messages": [...], "images": [...], "target_answer": "..."}
    """

    data_dir = "/mnt/VLAI_data/ViVQA-X"
    if split == 'train':
        data_path = os.path.join(data_dir, 'ViVQA-X_train.json')
        image_dir = 'train2014'
    elif split == 'val':
        data_path = os.path.join(data_dir, 'ViVQA-X_val.json')
        image_dir = 'val2014'
    else:
        data_path = os.path.join(data_dir, 'ViVQA-X_test.json')
        image_dir = 'val2014'

    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if output_file is None:
        output_dir = 'data/processed/only_think_answer'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'ViVQA-X_{split}_msswift.jsonl')

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, item in enumerate(raw_data):
            image_name = item.get('image_name')
            image_id = item.get('image_id')
            question = item.get('question')
            answer = item.get('answer')
            explanations = item.get('explanation')

            if not all([image_name, question, answer, explanations, explanations[0]]):
                continue

            explanation = explanations[0]

            # Construct absolute image path
            absolute_image_path = os.path.join(image_base_dir, image_dir, image_name)

            # Format user content using the instruction template
            # user_content = system_instruction_vintern3BR.format(question=question)

            # Format câu trả lời đầy đủ (assistant response)
            # Prompt asks for REASONING then CONCLUSION
            full_response = f"<REASONING>{explanation}</REASONING>\n<CONCLUSION>{answer}</CONCLUSION>"

            # MS-Swift format
            system_prompt = SYSTEM_PROMPT
            user_content = USER_CONTENT_TEMPLATE.format(question = question)
            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "images": [absolute_image_path], 
                "solution": full_response
            }

            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Created {output_file}")
    return output_file


if __name__ == "__main__":
    IMAGE_BASE_DIRECTORY = "/mnt/VLAI_data/COCO_Images"
    
    print("--- Generating data for SFT ---")
    create_jsonl_for_msswift("train", image_base_dir=IMAGE_BASE_DIRECTORY)
    create_jsonl_for_msswift("val", image_base_dir=IMAGE_BASE_DIRECTORY)
    create_jsonl_for_msswift("test", image_base_dir=IMAGE_BASE_DIRECTORY)
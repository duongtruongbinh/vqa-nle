import os
import json


SYSTEM_PROMPT = """<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc."""

USER_CONTENT_TEMPLATE = """Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong hai giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
<CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>
<EXPLANATION>[Giải thích một câu ngắn gọn chứng minh câu trả lời.] Hình ảnh cho thấy...</EXPLANATION>

Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
Câu trả lời:
"""


def create_jsonl_for_msswift(split="train", output_file=None, image_base_dir="/mnt/VLAI_data/COCO_Images"):
    """
    Tạo file JSONL theo format của MS-Swift với system, user, và assistant messages.
    Format: {"messages": [system, user, assistant], "images": [...]}
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
        output_dir = 'data/processed/sft'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'ViVQA-X_{split}_msswift.jsonl')

    count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in raw_data:
            image_name = item.get('image_name')
            question = item.get('question')
            answer = item.get('answer')
            explanations = item.get('explanation')

            if not all([image_name, question, answer, explanations]):
                continue
            
            if not explanations or not explanations[0]:
                continue

            explanation = explanations[0]

            # Construct absolute image path
            absolute_image_path = os.path.join(image_base_dir, image_dir, image_name)

            # Format user content
            user_content = USER_CONTENT_TEMPLATE.format(question=question)

            # Format assistant response với CONCLUSION và EXPLANATION tags
            assistant_response = (
                f"<CONCLUSION>\n{answer}\n</CONCLUSION>\n"
                f"<EXPLANATION>\n{explanation}\n</EXPLANATION>"
            )

            # MS-Swift SFT format - CẦN CÓ ASSISTANT MESSAGE
            entry = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_response} 
                ],
                "images": [absolute_image_path]
            }

            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1

    print(f"Created {output_file} with {count} samples")
    return output_file


if __name__ == "__main__":
    IMAGE_BASE_DIRECTORY = "/mnt/VLAI_data/COCO_Images"
    
    print("--- Generating data for SFT ---")
    create_jsonl_for_msswift("train", image_base_dir=IMAGE_BASE_DIRECTORY)
    create_jsonl_for_msswift("val", image_base_dir=IMAGE_BASE_DIRECTORY)
    create_jsonl_for_msswift("test", image_base_dir=IMAGE_BASE_DIRECTORY)
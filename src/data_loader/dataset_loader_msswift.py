import os
import json


# System prompts for internVL
# SYSTEM_PROMPTS = {"stage1": """You are an AI assistant specializing in image analysis. Your task is to provide a concise and accurate description of the provided image.

# Follow these steps precisely:
# - <think> Your step-by-step reasoning process. Analyze the key elements, objects, and setting within the image. </think>
# - <caption> Based on your reasoning, provide a description of the image. </caption>""".strip(),
# "stage2": """You are a Visual Question Answering system. Your task is to answer and explain questions based on the content of the provided image.


# Follow these steps precisely:    
# - <think> Your step-by-step reasoning process. Analyze the image carefully. </think>
# - <answer> Based on your reasoning, provide your Vietnamese answer must be one word or a short phrase. </answer>
# - <explain> Your brief Vietnamese explanation (one sentence that justifies your answer). </explain>""".strip()
# }


# # User content templates
# USER_CONTENT_TEMPLATES = {
#     "stage1": """<image>
# Now, provide a brief description of the image. Let's response in two tag pairs in your response: <think></think>, <caption></caption>.""".strip(),
#     "stage2": """<image>
# Now, answer this question based on the image:
# Question: {question}
# Let's response in three tag pairs in your response: <think></think>, <answer></answer>, <explain></explain>.""".strip()
# }

# System prompts - chứa instructions
SYSTEM_PROMPTS = {
    "stage1": """<image>Bạn là trợ lý AI chuyên về phân tích hình ảnh. Nhiệm vụ của bạn là cung cấp mô tả ngắn gọn và chính xác về hình ảnh được cung cấp.""".strip(),
    
    "stage2": """<image>Bạn là hệ thống Visual Question Answering (VQA). Nhiệm vụ của bạn là trả lời và giải thích các câu hỏi dựa trên nội dung của hình ảnh được cung cấp.""".strip()
}

# User content templates
USER_CONTENT_TEMPLATES = {
    "stage1": """
    Câu hỏi: {question}
    Vui lòng phân tích và mô tả chi tiết hình ảnh. Hãy trả lời theo định dạng sau:
    <CAPTION>Mô tả chi tiết hình ảnh</CAPTION>
    <REASONING>Quá trình suy luận để đi đến kết luận</REASONING>""".strip(),
    
    "stage2": """
    Câu hỏi: {question}
    Vui lòng trả lời câu hỏi sau dựa trên hình ảnh. Hãy trả lời theo định dạng sau:
    <REASONING>Quá trình suy luận chi tiết dẫn đến câu trả lời cuối cùng</REASONING>
    <answer>Câu trả lời (một từ hoặc cụm từ ngắn)</answer>
    <explain>Giải thích một câu ngắn gọn chứng minh câu trả lời</explain>""".strip()
}


def create_jsonl_for_msswift(split="train", stage="stage2", output_file=None, image_base_dir="/mnt/VLAI_data/COCO_Images"):
    """
    Tạo file JSONL theo format của MS-Swift với system prompt và user content riêng biệt
    Format: {"messages": [...], "images": [...], "target_answer": "..."}
    """
    if stage not in SYSTEM_PROMPTS:
        raise ValueError(f"Invalid stage: {stage}. Available stages are: {list(SYSTEM_PROMPTS.keys())}")

    system_prompt = SYSTEM_PROMPTS[stage]
    user_template = USER_CONTENT_TEMPLATES[stage]
    
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
        output_dir = f'data/processed/curriculum/{stage}'
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

            # Format user content
            if stage == "stage1":
                user_content = user_template  
            else:  # stage2
                user_content = user_template.format(question=question)

            # Format câu trả lời đầy đủ (assistant response)
            if stage == "stage1":
                full_response = f"<caption>{image_id}</caption>"
            elif stage == "stage2":
                full_response = f"<answer>{answer}</answer><explain>{explanation}</explain>"

            # MS-Swift format với system, user, assistant riêng biệt + images
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
    
    for stage in ["stage1", "stage2"]:
        print(f"--- Generating data for {stage} ---")
        create_jsonl_for_msswift("train", stage=stage, image_base_dir=IMAGE_BASE_DIRECTORY)
        create_jsonl_for_msswift("val", stage=stage, image_base_dir=IMAGE_BASE_DIRECTORY)
        create_jsonl_for_msswift("test", stage=stage, image_base_dir=IMAGE_BASE_DIRECTORY)
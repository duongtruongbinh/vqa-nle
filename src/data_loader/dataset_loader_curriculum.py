import os
import json



YES_NO_PREFIXES = [
    "none of", "is the", "is this", "is it", "is there", "are these", "are there", "are the",
    "do ", "does ", "can ", "could ", "has ", "have ", "was ", "were ",
    "is he", "is she", "is they"
]

# FACTOID_PREFIXES =[
#     "how many", "what color", "which", "who is", "who are", "where is", "where are", 
#     "what time", "what is", "what are", "what kind", "what type"
# ]

FACTOID_PREFIXES =[
    "which"
]

SYSTEM_PROMPTS = {
    "stage1": """<image>Bạn là trợ lý AI chuyên xác nhận thông tin. Hãy trả lời các câu hỏi Có/Không (Yes/No) dựa trên hình ảnh.""".strip(),
    "stage2": """<image>Bạn là hệ thống Visual Question Answering (VQA). Nhiệm vụ của bạn là suy luận từ các chi tiết trong hình ảnh và chọn đáp án đúng nhất cho câu hỏi.""".strip(),
    "stage3": """<image>Bạn là hệ thống Visual Question Answering (VQA). Nhiệm vụ của bạn là trả lời và giải thích các câu hỏi dựa trên nội dung của hình ảnh được cung cấp.""".strip()
}


USER_CONTENT_TEMPLATES = {
    "stage1": """
    Câu hỏi: {question}
    Hãy trả lời theo định dạng sau:
    <REASONING>Các bước suy luận dựa trên quan sát các chi tiết trong hình ảnh để xác nhận đúng/sai.</REASONING>
    <answer>Câu trả lời (Có/Không).</answer>
    <explain>Câu giải thích ngắn gọn, cô đọng, rút ra từ các điểm chính trong phần suy luận để làm bằng chứng.</explain>""".strip(),
    
    "stage2": """
    Câu hỏi: {question}
    Hãy trả lời theo định dạng sau:
    <REASONING>Các bước suy luận dựa trên quan sát các chi tiết trong hình ảnh để đưa ra đáp án đúng nhất.</REASONING>
    <answer>Câu trả lời ngắn (chọn một đáp án).</answer>
    <explain>Câu giải thích ngắn gọn, cô đọng, rút ra từ các điểm chính trong phần suy luận để làm bằng chứng.</explain>""".strip(),
    
    "stage3": """
    Câu hỏi: {question}
    Hãy trả lời theo định dạng sau:
    <REASONING>Quá trình suy luận chi tiết dựa trên nội dung hình ảnh, dẫn đến câu trả lời cuối cùng.</REASONING>
    <answer>Câu trả lời (một từ hoặc cụm từ ngắn).</answer>
    <explain>Câu giải thích ngắn gọn, cô đọng, tổng hợp các điểm chính trong phần suy luận để làm bằng chứng.</explain>""".strip()
}

STAGE_TYPE_MAPPING = {
    "stage1": "yes_no",
    "stage2": "multiple_choice",
    "stage3": "open_ended"
}

## check question -> keyword


def classify_stage_by_questiontype(item):
    a_type = str(item.get('answer_type', '')).lower().strip()
    q_type = str(item.get('question_type', '')).lower().strip()

    # rule 1: answer_type là 'yes/no'
    if a_type == 'yes/no':
        return 'stage1'
    
    # rule 2: answer_type là 'other', question_type bắt đầu bằng động từ tobe/modal verb
    if a_type == 'other':
        if any(q_type.startswith(prefix) for prefix in YES_NO_PREFIXES):
            return 'stage1'

    if a_type == 'number':
        return 'stage2'

    if any(q_type.startswith(prefix) for prefix in FACTOID_PREFIXES):
        return 'stage2'

    return 'stage3'

def classify_stage_by_question(item):
    a_type = str(item.get('answer_type', '')).lower().strip()
    question = str(item.get('question', '')).lower().strip()

   
    if a_type == 'yes/no':
        return 'stage1'

    
    if ' hay ' in question or ' hoặc ' in question:
        return 'stage2'

   
    return 'stage3'



def create_curriculum_from_metadata(split="train", output_file=None, image_base_dir="/mnt/VLAI_data/COCO_Images"):
    """
    Tạo file JSONL theo format của MS-Swift với system prompt và user content riêng biệt
    Format: {"messages": [...], "images": [...], "target_answer": "..."}
    """
    # if stage not in SYSTEM_PROMPTS:
    #     raise ValueError(f"Invalid stage: {stage}. Available stages are: {list(SYSTEM_PROMPTS.keys())}")

    # system_prompt = SYSTEM_PROMPTS[stage]
    # user_template = USER_CONTENT_TEMPLATES[stage]
    
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


    # if output_file is None:
    #     output_dir = f'data/processed/curriculum'
    #     os.makedirs(output_dir, exist_ok=True)
    #     # output_file = os.path.join(output_dir, f'ViVQA-X_{split}_msswift.jsonl')

    output_base_dir = f'data/processed/curriculum'
    # output_base_dir = f'data/processed/curriculum_vietnamesekey'
    os.makedirs(output_base_dir, exist_ok=True)


    files_map = {}
    stages = ['stage1', 'stage2', 'stage3']
    for stage in stages:
        stage_dir = os.path.join(output_base_dir, stage)
        os.makedirs(stage_dir, exist_ok=True)
        files_map[stage] = open(os.path.join(stage_dir, f'ViVQA-X_{split}_{stage}.jsonl'), 'w', encoding='utf-8')

    count_map = {s: 0 for s in stages}
    

    try:
        for item in raw_data:
            if not all([item.get('image_name'), item.get('question'), item.get('answer'), item.get('explanation')]):
                continue

            image_name = item.get('image_name')
            image_id = item.get('image_id')
            question = item.get('question')
            answer = item.get('answer')
            explanations = item.get('explanation')
            question_type = item.get('question_type')

            explanation = explanations[0] if isinstance(explanations, list) and len(explanations) > 0 else str(explanations)
            absolute_image_path = os.path.join(image_base_dir, image_dir, image_name)

            target_stage = classify_stage_by_question(item)

            system_prompt = SYSTEM_PROMPTS[target_stage]
            user_template = USER_CONTENT_TEMPLATES[target_stage]

            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_template.format(question=question)},
                ],
                "images": [absolute_image_path], 
                "solution": f"<questiontype>{question_type}</questiontype><answer>{answer}</answer><explain>{explanation}</explain>"
            }

            files_map[target_stage].write(json.dumps(entry, ensure_ascii=False) + '\n')
            count_map[target_stage] += 1

    finally:
        for f in files_map.values():
            f.close()

    print(f"Result for {split}:")
    for s in stages:
        print(f"  - {s}: {count_map[s]} samples")

if __name__ == "__main__":
    IMAGE_BASE_DIRECTORY = "/mnt/VLAI_data/COCO_Images"
    
    create_curriculum_from_metadata("train", image_base_dir=IMAGE_BASE_DIRECTORY)
    create_curriculum_from_metadata("val", image_base_dir=IMAGE_BASE_DIRECTORY)
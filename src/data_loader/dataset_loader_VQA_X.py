import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print("os.environ['CUDA_VISIBLE_DEVICES']:", os.environ['CUDA_VISIBLE_DEVICES'])
import json
from collections import Counter

SYSTEM_PROMPT = """<image> You are a helpful visual language assistant, designed for structured reasoning."""
USER_PROMPT = """ 
    When answering image-based questions, you must answer correctly in three stages, each stage following the required format:
    <REASONING>[Provide a detailed, step-by-step analysis and reasoning to solve the problem.]</REASONING>
    <CONCLUSION>[State the final answer as a word or phrase.]</CONCLUSION>
    <EXPLANATION>[Synthesize the information from REASONING and provide a brief description of the analyzed features.] The image shows...</EXPLANATION>
    Please apply this format meticulously to analyze the provided image and answer the question: {question}
    Answer:
    """.strip()

def get_most_common_answer(answers):
        return Counter(answer["answer"] for answer in answers).most_common(1)[0][0]

def create_jsonl_for_grpo(split="train", image_base_dir="/mnt/VLAI_data/COCO_Images", output_file=None):
    """
    Tạo file JSONL theo format của VLM-R1 GRPO
    """
    data_dir = "/mnt/VLAI_data/VQA-X"

    if split == 'train':
        data_path = os.path.join(data_dir, 'vqaX_train.json')
        image_dir = 'train2014'
    elif split == 'val':
        data_path = os.path.join(data_dir, 'vqaX_val.json')
        image_dir = 'val2014'
    else:
        data_path = os.path.join(data_dir, 'vqaX_test.json')
        image_dir = 'val2014'

    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if output_file is None:
        output_dir = 'data/processed_EN'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'VQA-X_{split}_grpo.jsonl')

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for id, item in raw_data.items():
            image_name = item.get('image_name')
            question = item.get('question')
            answer = get_most_common_answer(item.get('answers'))
            explanations = item.get('explanation')

            if not all([image_name, question, answer, explanations, explanations[0]]):
                continue

            explanation = explanations[0]
            absolute_image_path = os.path.join(image_base_dir, image_dir, image_name)

            # format câu hỏi
            # question_with_prompt = prompt.format(question=question)

            # format câu trả lời
            solution = f"<answer>{answer}</answer><explain>{explanation}</explain>"

            system_prompt = SYSTEM_PROMPT
            user_content = USER_PROMPT.format(question = question)
            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "images": [absolute_image_path], 
                "solution": solution
            }
            # print(entry)
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    return output_file


if __name__ == "__main__":
    IMAGE_BASE_DIRECTORY = "/mnt/VLAI_data/COCO_Images"

    create_jsonl_for_grpo("train", image_base_dir=IMAGE_BASE_DIRECTORY)
    create_jsonl_for_grpo("val", image_base_dir=IMAGE_BASE_DIRECTORY)
    create_jsonl_for_grpo("test", image_base_dir=IMAGE_BASE_DIRECTORY)

import os
import json

prompt = """
<image> You are a Visual Question Answering system. Your task is to answer questions based on the content of the provided image. 
You must respond in Vietnamese and your response **must** include all the tags <think> </think>, <answer> </answer>, <explain> </explain>.

Follow these steps precisely:
1. In the <think> tag, provide a step-by-step reasoning process.
2. In the <answer> tag, give one word or one short phrase.
3. In the <explain> tag, provide one brief sentence that justifies your answer.

Now, answer this question based on the image:
Question: {question}
""".strip()


def create_jsonl_for_grpo(split="train", output_file=None):
    """
    Tạo file JSONL theo format của VLM-R1 GRPO
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
        output_dir = 'data/processed'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'ViVQA-X_{split}_grpo.jsonl')

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, item in enumerate(raw_data):
            image_name = item.get('image_name')
            question = item.get('question')
            answer = item.get('answer')
            explanations = item.get('explanation')

            if not all([image_name, question, answer, explanations, explanations[0]]):
                continue

            explanation = explanations[0]
            relative_image_path = os.path.join(image_dir, image_name)

            # format câu hỏi
            question_with_prompt = prompt.format(question=question)

            # format câu trả lời
            solution = f"<answer>{answer}</answer><explain>{explanation}</explain>"

            entry = {
                "id": idx + 1,
                "image": image_name,
                "conversations": [
                    {"from": "human", "value": question_with_prompt},
                    {"from": "gpt", "value": solution}
                ]
            }

            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    return output_file


if __name__ == "__main__":
    create_jsonl_for_grpo("train")
    create_jsonl_for_grpo("val")
    create_jsonl_for_grpo("test")

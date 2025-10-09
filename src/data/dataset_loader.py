import os
import json
from datasets import Dataset, Image

SYSTEM_PROMPT_VIVQA = (
    "Là một trợ lý AI, nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên hình ảnh được cung cấp. "
    "Hãy trình bày câu trả lời theo định dạng sau: đầu tiên là câu trả lời trực tiếp được đặt trong cặp thẻ `<answer> </answer>`, "
    "tiếp theo là phần giải thích chi tiết cho câu trả lời đó, đặt trong cặp thẻ `<explain> </explain>`."
)

def get_dataset(processor, split="train"):
    """
    Tải và chuẩn bị tập dữ liệu ViVQA-X cho split được chỉ định.

    Args:
        processor: Processor của mô hình (ví dụ: AutoProcessor) để định dạng prompt.
        split (str): Split của dữ liệu cần tải ('train', 'val', hoặc 'test').

    Returns:
        datasets.Dataset: Một đối tượng Dataset của Hugging Face đã được xử lý.
    """
    data_dir = "/mnt/VLAI_data/ViVQA-X"
    
    if split == 'train':
        data_path = os.path.join(data_dir, 'ViVQA-X_train.json')
        image_dir = '/mnt/VLAI_data/COCO_Images/train2014'
    elif split == 'val':
        data_path = os.path.join(data_dir, 'ViVQA-X_val.json')
        image_dir = '/mnt/VLAI_data/COCO_Images/val2014'
    else:  # 'test'
        data_path = os.path.join(data_dir, 'ViVQA-X_test.json')
        image_dir = '/mnt/VLAI_data/COCO_Images/val2014'

    # Tải dữ liệu từ file JSON
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Xử lý dữ liệu thô
    processed_data = {
        "image_path": [],
        "prompt": [],
        "solution": []
    }

    for item in raw_data:
        image_name = item.get('image_name')
        question = item.get('question')
        answer = item.get('answer')
        explanations = item.get('explanation')

        if not all([image_name, question, answer, explanations, explanations[0]]):
            continue
        
        explanation = explanations[0]
        # Tạo chuỗi solution theo định dạng trong system prompt
        solution = f"<answer> {answer} </answer><explain> {explanation} </explain>"

        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            continue

        # Tạo prompt với system message và user message
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT_VIVQA},
            {"role": "user", "content": f"<|image|>\n{question}"}
        ]
        
        prompt = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        processed_data["image_path"].append(image_path)
        processed_data["prompt"].append(prompt)
        processed_data["solution"].append(solution)

    # Tạo một đối tượng Dataset của Hugging Face
    dataset = Dataset.from_dict({
        "image": processed_data["image_path"],
        "prompt": processed_data["prompt"],
        "solution": processed_data["solution"],
    })

    # Chuyển đổi cột 'image' để tải hình ảnh khi cần
    dataset = dataset.cast_column("image", Image(decode=True))
    
    return dataset
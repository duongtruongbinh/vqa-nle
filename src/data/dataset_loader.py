import os
import json
from datasets import Dataset, Image

SYSTEM_PROMPT_VIVQA_ENHANCED = (
    "<image>\nBạn là một trợ lý AI chuyên gia, có khả năng phân tích hình ảnh một cách cẩn thận và đa nghi. "
    "Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên hình ảnh được cung cấp. "
    "Trước tiên, hãy thực hiện một chuỗi suy luận chi tiết bên trong cặp thẻ <thinking></thinking>. "
    "Sau khi hoàn tất quá trình suy luận, hãy cung cấp câu trả lời cuối cùng theo đúng định dạng yêu cầu.\n\n"
    "Câu hỏi: {question}\n\n"
    "ĐỊNH DẠNG BẮT BUỘC:\n"
    "<thinking>\n"
    "<SUMMARY>[Tóm tắt ngắn gọn về hình ảnh và yêu cầu của câu hỏi]</SUMMARY>\n"
    "<ANALYSIS>[Phân tích các chi tiết, vật thể, văn bản trong ảnh có liên quan trực tiếp đến câu hỏi. Liệt kê các bằng chứng quan sát được.]</ANALYSIS>\n"
    "<REASONING_STEPS>[Trình bày quá trình lập luận logic từng bước một. Từ các bằng chứng đã phân tích, làm thế nào để đi đến câu trả lời? Giải thích các mối liên hệ.]</REASONING_STEPS>\n"
    "<CONCLUSION>[Đưa ra kết luận cuối cùng từ quá trình lập luận trên.]</CONCLUSION>\n"
    "</thinking>\n"
    "<answer>[Điền câu trả lời trực tiếp và ngắn gọn vào đây]</answer>\n"
    "<explain>[Dựa vào quá trình suy luận trong <thinking>, giải thích cực kỳ ngắn gọn (trong khoảng 10-15 từ)]</explain>"
)


def get_dataset(split="train"):
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

    raw_data = raw_data[:10]

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

        # Tạo prompt trực tiếp bằng cách format string
        prompt = SYSTEM_PROMPT_VIVQA_ENHANCED.format(question=question)

        processed_data["image_path"].append(image_path)
        processed_data["prompt"].append(prompt)
        processed_data["solution"].append(solution)

    # Tạo một đối tượng Dataset của Hugging Face
    dataset = Dataset.from_dict({
        "images": processed_data["image_path"],
        "prompt": processed_data["prompt"],
        "solution": processed_data["solution"],
    })

    dataset = dataset.cast_column("images", Image(decode=True))

    return dataset

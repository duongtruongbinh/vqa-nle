
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
print("os.environ['CUDA_VISIBLE_DEVICES']:", os.environ['CUDA_VISIBLE_DEVICES'])


import json
from underthesea import text_normalize, pos_tag

import cupy
from pycocotools.coco import COCO
from collections import defaultdict

# ====================== CONSTANTS ======================

YES_NO_PREFIXES = [
    "none of", "is the", "is this", "is it", "is there", 
    "are these", "are there", "are the",
    "do ", "does ", "can ", "could ", "has ", "have ", 
    "was ", "were ", "is he", "is she", "is they"
]

FACTOID_PREFIXES = ["which"]

STAGE_TYPE_MAPPING = {
    "stage1": "Có/Không",
    "stage2": "lựa chọn",
    "stage3": "open-ended"
}


# ====================== PROMPTS ======================

SYSTEM_PROMPTS = {
    "stage1": """<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc.""",

    "stage2": """<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc.""",

    "stage3": """<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc."""
}

USER_CONTENT_TEMPLATES = {
    "stage1": """Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong ba giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
    <REASONING>[Đưa ra phân tích lập luận chi tiết, từng bước để giải quyết vấn đề.]</REASONING>
    <CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>
    <EXPLANATION>[Tổng hợp các thông tin từ REASONING và cho ra câu mô tả ngắn gọn các phân tích đặc điểm.] Hình ảnh cho thấy...</EXPLANATION>
    
    Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
    Câu trả lời:
    """,

    "stage2": """Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong ba giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
    <REASONING>[Đưa ra phân tích lập luận chi tiết, từng bước để giải quyết vấn đề.]</REASONING>
    <CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>
    <EXPLANATION>[Tổng hợp các thông tin từ REASONING và cho ra câu mô tả ngắn gọn các phân tích đặc điểm.] Hình ảnh cho thấy...</EXPLANATION>
    
    Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
    Câu trả lời:
    """,

    "stage3": """Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong ba giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
    <REASONING>[Đưa ra phân tích lập luận chi tiết, từng bước để giải quyết vấn đề.]</REASONING>
    <CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>
    <EXPLANATION>[Tổng hợp các thông tin từ REASONING và cho ra câu mô tả ngắn gọn các phân tích đặc điểm.] Hình ảnh cho thấy...</EXPLANATION>
    
    Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
    Câu trả lời:
    """
}

# ====================== HELPER FUNCTIONS ======================

def extract_nouns(text):
    """
    Trích xuất danh từ từ văn bản tiếng Việt.

    Args:
        text (str): Văn bản đầu vào

    Returns:
        list: Danh sách các danh từ
    """
    normalized = text_normalize(text)
    tagged = pos_tag(normalized)
    nouns = [word for word, tag in tagged if tag in ['N', 'Np']]
    return nouns


def extract_objects_from_item(question, explanations):
    """
    Trích xuất tất cả objects từ question và explanations.

    Args:
        question (str): Câu hỏi
        explanations (str/list): Câu giải thích hoặc danh sách giải thích

    Returns:
        list: Danh sách unique objects, giữ nguyên thứ tự xuất hiện
    """
    objects = []

    # Extract từ question
    objects.extend(extract_nouns(question))

    # Extract từ tất cả explanations
    if isinstance(explanations, list):
        for exp in explanations:
            objects.extend(extract_nouns(exp))
    else:
        objects.extend(extract_nouns(str(explanations)))

    # Trả về danh sách unique, giữ nguyên thứ tự
    return list(dict.fromkeys(objects))


def classify_stage_by_questiontype(item):
    """
    Phân loại stage dựa trên question type và answer type.

    Args:
        item (dict): Dictionary chứa question_type và answer_type

    Returns:
        str: 'stage1', 'stage2', hoặc 'stage3'
    """
    a_type = str(item.get('answer_type', '')).lower().strip()
    q_type = str(item.get('question_type', '')).lower().strip()

    # Stage 1: Yes/No questions
    if a_type == 'yes/no':
        return 'stage1'

    if a_type == 'other':
        if any(q_type.startswith(prefix) for prefix in YES_NO_PREFIXES):
            return 'stage1'

    # Stage 2: Number and factoid questions
    if a_type == 'number':
        return 'stage2'

    if any(q_type.startswith(prefix) for prefix in FACTOID_PREFIXES):
        return 'stage2'

    # Stage 3: Open-ended questions (default)
    return 'stage3'

#######
def load_COCO_caption(CAPTION_JSON):

    with open(CAPTION_JSON, "r", encoding="utf-8") as f:
        caption_data = json.load(f)

    image_to_captions = defaultdict(list)
    for ann in caption_data["annotations"]:
        image_to_captions[ann["image_id"]].append(ann["caption"])

    print(f"#images có caption: {len(image_to_captions)}")

    return image_to_captions

# Count number of nouns in a caption
def count_nouns_in_caption(caption, nlp):
    noun_pos = ["PROPN", "NOUN"]

    doc = nlp(caption)
    return sum(1 for token in doc if token.pos_ in noun_pos)

# Get max noun count among all captions for an image
def max_noun_count_for_image(captions, nlp):
    if not captions:
        return 0
    return max(count_nouns_in_caption(c, nlp) for c in captions)


def get_coco_id_from_filename(filename):
    try:
        # Retrieve ID from filename 
        id_part = filename.split('_')[-1].split('.')[0]
        return int(id_part)
    except:
        return None

def classify_stage_by_noun_count(max_noun_count):

    if max_noun_count <= 3:
        return "stage1"
    elif 4 <= max_noun_count <= 5:
        return "stage2"
    else:
        return "stage3"
########

# ====================== MAIN PROCESSING ======================

def create_curriculum_from_metadata(split="train", 
                                    output_file=None, 
                                    object_json_dir="./data/processed/object_lists",
                                    image_base_dir="/mnt/VLAI_data/COCO_Images"):
    """
    Tạo file JSONL theo format của MS-Swift với object extraction.

    Args:
        split (str): 'train', 'val', hoặc 'test'
        output_file: Deprecated, không sử dụng
        image_base_dir (str): Đường dẫn tới thư mục chứa COCO images
a96 `
    Format output: {"messages": [...], "images": [...], "solution": "..."}
    """
    data_dir = "/mnt/VLAI_data/ViVQA-X"
    # Mở files cho từng stage
    stages = ['stage1', 'stage2', 'stage3']

    image_metadata_map = {}
    
    for stage in stages:
        # Format tên file: objects_train_stage1.json
        object_file_name = f"objects_{split}_{stage}.json"
        object_file_path = os.path.join(object_json_dir, object_file_name)
        if os.path.exists(object_file_path):
            with open(object_file_path, 'r', encoding='utf-8') as f:
                stage_data = json.load(f)
                # Gộp dữ liệu vào map tổng
                image_metadata_map.update(stage_data)


    # Xác định đường dẫn data và image
    split_config = {
        'train': ('ViVQA-X_train.json', 'train2014'),
        'val': ('ViVQA-X_val.json', 'val2014'),
        'test': ('ViVQA-X_test.json', 'val2014')
    }

    data_file, image_dir = split_config.get(split, split_config['train'])
    data_path = os.path.join(data_dir, data_file)

    print("Processing split:", split)
    # Đọc data
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Tạo output directory structure
    output_base_dir = 'data/processed/curriculum_reasoning_noun_based'
    os.makedirs(output_base_dir, exist_ok=True)

    
    files_map = {}
    count_map = {stage: 0 for stage in stages}
    stage3_buffer = []


    for stage in stages:
        stage_dir = os.path.join(output_base_dir, stage)
        os.makedirs(stage_dir, exist_ok=True)
        output_path = os.path.join(stage_dir, f'ViVQA-X_{split}_{stage}.jsonl')
        files_map[stage] = open(output_path, 'w', encoding='utf-8')

    try:
        print("Checking data and writing to files...")
        for item in raw_data:
            # Validate dữ liệu bắt buộc
            required_fields = ['image_name', 'question', 'answer', 'explanation']
            if not all(item.get(field) for field in required_fields):
                continue
            
            # Lấy thông tin từ item
            image_name = item['image_name']
            question = item['question']
            answer = item['answer']
            explanations = item['explanation']
            question_type = item.get('question_type', '')
            question_id = item.get('question_id', '')
            ######
            # Retrieve object and noun count từ metadata map
            metadata = image_metadata_map.get(image_name, {"objects": [], "noun_count": 0})

            # Classify stage by noun count 
            max_noun_count = metadata.get("noun_count", 0)

            # Extract objects từ string
            obj_list = metadata.get("objects", [])
            objects_str = ', '.join(obj_list)
            # extracted_objects = extract_objects_from_item(question, explanations)
            # objects_str = ', '.join(extracted_objects) if extracted_objects else ''

            target_stage = classify_stage_by_noun_count(max_noun_count)
            ######

            # Lấy explanation đầu tiên
            explanation = (explanations[0] if isinstance(explanations, list) 
                          and len(explanations) > 0 else str(explanations))

            

            # Đường dẫn ảnh tuyệt đối
            absolute_image_path = os.path.join(image_base_dir, image_dir, image_name)

            # # Phân loại stage
            # target_stage = classify_stage_by_questiontype(item)

            # Tạo entry theo format MS-Swift
            print("Target stage:", target_stage)
            entry = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS[target_stage]},
                    {"role": "user", "content": USER_CONTENT_TEMPLATES[target_stage].format(question=question)}
                ],
                "images": [absolute_image_path],
                "solution": (f"<question_id>{question_id}</question_id>"
                        f"<questiontype>{question_type}</questiontype>"
                        f"<answer>{answer}</answer>"
                        f"<explain>{explanation}</explain>"
                        f"<object_extraction>{objects_str}</object_extraction>"
                        f"<noun_count>{max_noun_count}</noun_count>")
            }

            # Ghi vào file tương ứng
            if target_stage == "stage3":
                stage3_buffer.append((max_noun_count, entry))
                count_map[target_stage] += 1
            else:
                files_map[target_stage].write(json.dumps(entry, ensure_ascii=False) + '\n')
                count_map[target_stage] += 1

            if sum(count_map.values()) % 1000 == 0:
                print(f"Processed {sum(count_map.values())} samples...")

        # Sort and write stage 3 data
        if stage3_buffer:
            print(f"Sorting {len(stage3_buffer)} samples for stage3...")
            stage3_buffer.sort(key=lambda x: x[0])
            for _, entry in stage3_buffer:
                files_map['stage3'].write(json.dumps(entry, ensure_ascii=False) + '\n')

    finally:
        # Đóng tất cả files
        for f in files_map.values():
            f.close()

    # In kết quả thống kê
    print(f"\nResult for {split}:")
    for stage in stages:
        print(f"  - {stage}: {count_map[stage]} samples")
    print(f"  - Total: {sum(count_map.values())} samples\n")


# ====================== MAIN EXECUTION ======================

if __name__ == "__main__":

    print("os.environ['CUDA_VISIBLE_DEVICES']:", os.environ['CUDA_VISIBLE_DEVICES'])
    IMAGE_BASE_DIRECTORY = "/mnt/VLAI_data/COCO_Images"
    OBJECT_JSON_DIR = "./data/processed/object_lists"

    print("="*50)
    print("VQA Curriculum Learning Data Preprocessing")
    print("="*50)

    # create_curriculum_from_metadata("train", image_base_dir=IMAGE_BASE_DIRECTORY,caption_json_path=CAPTION_JSON_TRAIN)
    # create_curriculum_from_metadata("val", image_base_dir=IMAGE_BASE_DIRECTORY,caption_json_path=CAPTION_JSON_VAL)

    create_curriculum_from_metadata(
        split="train", 
        image_base_dir=IMAGE_BASE_DIRECTORY,
        object_json_dir=OBJECT_JSON_DIR
    )


    print("="*50)
    print("Processing completed!")
    print("="*50)
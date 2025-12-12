#!/usr/bin/env python3
"""
Script tổng hợp failed question IDs từ nhiều file JSON và tạo dataset mới (train split).

Usage 1 (nhanh - điền list FAILED_JSON_FILES bên dưới):
    python create_failed_dataset.py

Usage 2 (command line):
    python create_failed_dataset.py --failed_json_files file1.json file2.json ...
"""

import os
import json
import argparse
from typing import List, Set, Dict


# ====================== CONFIG - ĐIỀN VÀO ĐÂY ======================

# Danh sách các file JSON chứa failed question IDs
# Điền vào đây để chạy nhanh, không cần dùng command line
FAILED_JSON_FILES = [
    "/home/vlai-vqa-nle/minhtq/vqa-nle/data/analysis/stage_1/failed_question_ids_stage1.json",
    "/home/vlai-vqa-nle/minhtq/vqa-nle/data/analysis/stage_2/failed_question_ids_stage2.json"
]

IMAGE_BASE_DIR = "/mnt/VLAI_data/COCO_Images"
OUTPUT_DIR = "data/processed/curriculum_reasoning_failed"


# ====================== CONSTANTS ======================

SYSTEM_PROMPT = """<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc."""

USER_CONTENT_TEMPLATE = """Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong ba giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
    <REASONING>[Đưa ra phân tích lập luận chi tiết, từng bước để giải quyết vấn đề.]</REASONING>
    <CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>
    <EXPLANATION>[Tổng hợp các thông tin từ REASONING và cho ra câu mô tả ngắn gọn các phân tích đặc điểm.] Hình ảnh cho thấy...</EXPLANATION>
    
    Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
    Câu trả lời:
    """


# ====================== MAIN FUNCTIONS ======================

def load_failed_question_ids(file_paths: List[str]) -> Set[str]:
    """Load và tổng hợp failed question IDs từ nhiều file JSON."""
    all_failed_ids = set()
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File không tồn tại: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            failed_ids = data.get('failed_question_ids', [])
            all_failed_ids.update(failed_ids)
            print(f"Loaded {len(failed_ids)} IDs from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return all_failed_ids


def load_vivqa_dataset() -> Dict[str, dict]:
    """Load ViVQA-X train dataset và index theo question_id."""
    data_path = "/mnt/VLAI_data/ViVQA-X/ViVQA-X_train.json"
    print(f"Loading ViVQA-X from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    indexed_data = {}
    for item in raw_data:
        qid = str(item.get('question_id', ''))
        if qid:
            indexed_data[qid] = item
    
    print(f"Loaded {len(indexed_data)} items")
    return indexed_data


def filter_dataset_by_question_ids(vivqa_data: Dict[str, dict], failed_ids: Set[str]) -> List[dict]:
    """Lọc dataset theo failed question IDs."""
    filtered_items = []
    not_found = 0
    
    for qid in failed_ids:
        if qid in vivqa_data:
            filtered_items.append(vivqa_data[qid])
        else:
            not_found += 1
    
    if not_found > 0:
        print(f"Warning: {not_found} question IDs không tìm thấy trong dataset")
    
    print(f"Filtered {len(filtered_items)}/{len(failed_ids)} items")
    return filtered_items


def create_ms_swift_entry(item: dict, image_base_dir: str, image_dir: str) -> dict:
    """Tạo entry theo format MS-Swift."""
    explanation = (item['explanation'][0] if isinstance(item['explanation'], list) 
                   and len(item['explanation']) > 0 else str(item['explanation']))
    
    image_path = os.path.join(image_base_dir, image_dir, item['image_name'])
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_CONTENT_TEMPLATE.format(question=item['question'])}
        ],
        "images": [image_path],
        "solution": (f"<question_id>{item.get('question_id', '')}</question_id>"
                     f"<questiontype>{item.get('question_type', '')}</questiontype>"
                     f"<answer>{item['answer']}</answer>"
                     f"<explain>{explanation}</explain>")
    }


def save_failed_dataset(filtered_items: List[dict], image_base_dir: str, output_dir: str):
    """Lưu dataset đã lọc theo format MS-Swift JSONL."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'ViVQA-X_train_failed.jsonl')
    print(f"Saving to: {output_path}")
    
    saved_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in filtered_items:
            if not all(item.get(field) for field in ['image_name', 'question', 'answer', 'explanation']):
                continue
            
            entry = create_ms_swift_entry(item, image_base_dir, 'train2014')
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            saved_count += 1
    
    print(f"Saved {saved_count} entries")


def main():
    parser = argparse.ArgumentParser(description="Tổng hợp failed question IDs và tạo dataset mới (train)")
    
    parser.add_argument('--failed_json_files', nargs='+', 
                        help='Danh sách các file JSON chứa failed_question_ids')
    parser.add_argument('--image_base_dir', type=str, default=IMAGE_BASE_DIR,
                        help=f'Đường dẫn tới thư mục COCO images (default: {IMAGE_BASE_DIR})')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help=f'Thư mục output (default: {OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    # Sử dụng FAILED_JSON_FILES nếu có, nếu không thì dùng command line args
    failed_json_files = args.failed_json_files if args.failed_json_files else FAILED_JSON_FILES
    
    if not failed_json_files:
        print("Error: Cần cung cấp danh sách failed_json_files")
        print("  - Cách 1: Điền vào FAILED_JSON_FILES list trong file")
        print("  - Cách 2: Dùng --failed_json_files trong command line")
        return
    
    print("=" * 60)
    print("TỔNG HỢP FAILED QUESTIONS VÀ TẠO DATASET MỚI")
    print("=" * 60)
    
    # Load failed question IDs
    print(f"\nStep 1: Loading failed question IDs from {len(failed_json_files)} file(s)")
    failed_ids = load_failed_question_ids(failed_json_files)
    print(f"Total unique failed IDs: {len(failed_ids)}")
    
    # Load ViVQA-X train dataset
    print(f"\nStep 2: Loading ViVQA-X train dataset")
    vivqa_data = load_vivqa_dataset()
    
    # Filter dataset
    print(f"\nStep 3: Filtering dataset")
    filtered_items = filter_dataset_by_question_ids(vivqa_data, failed_ids)
    
    # Save dataset
    print(f"\nStep 4: Saving dataset")
    save_failed_dataset(filtered_items, args.image_base_dir, args.output_dir)
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print(f"Failed IDs: {len(failed_ids)} | Saved items: {len(filtered_items)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

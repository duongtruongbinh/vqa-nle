# File: src/data/dataset_collator.py
from PIL import Image


class VinternDataCollator:
    def __init__(self, processor, max_num=6):
        self.processor = processor
        self.max_num = max_num

    def __call__(self, features):
        batch = []

        for f in features:
            item = {
                "prompt": f["prompt"],
                "solution": f.get("solution", None),
            }

            # ✅ Xử lý image đúng cách - KHÔNG preprocess ở đây
            # Chỉ pass PIL Image, để trainer xử lý
            if "images" in f:
                img = f["images"]
                # Vintern expects PIL Image, not preprocessed tensors
                item["images"] = [img] if not isinstance(img, list) else img

            batch.append(item)

        return batch

import torch
import numpy as np
import re
import os
import warnings
from datetime import datetime
from PIL import Image
from torchmetrics.multimodal import CLIPScore
from torchmetrics.text import BERTScore

# Imports này được định nghĩa nhưng không xài - giữ nguyên theo yêu cầu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from transformers import AutoModel, AutoTokenizer
import atexit
from base_rewards import BaseRewardScorer
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.modeling_utils')

class ExplanationRewardScorer(BaseRewardScorer):
    """
    Reward scorer kết hợp BERTScore (semantic similarity) và CLIPScore (image-text alignment).
    Sử dụng PhoBERT cho Vietnamese text evaluation.
    """
    def __init__(self, alpha: float = 0.5, clip_model_name: str = "openai/clip-vit-base-patch16"):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be in [0, 1]")

        self.alpha = alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # CIDEr scorer - định nghĩa nhưng không xài
        self.cider_scorer = Cider()
        print("CIDEr scorer initialized (not used).")

        print(f"Initializing BERTScore (PhoBERT) on device: {self.device}")
        self.bertscore_metric = self.initialize_bertscore()
        print("BERTScore initialized.")

        print(f"Initializing CLIPScore on device: {self.device}")
        self.clip_metric = CLIPScore(model_name_or_path=clip_model_name).to(self.device)
        print("CLIPScore model loaded and ready.")

    def calculate_cider_batch(self, ground_truths: dict, predictions: dict) -> dict:
        res = {img_id: [caption] for img_id, caption in predictions.items()}
        _, individual_scores_array = self.cider_scorer.compute_score(ground_truths, res)
        
        image_ids = list(predictions.keys())
        cider_scores_dict = {img_id: score for img_id, score in zip(image_ids, individual_scores_array)}
        return cider_scores_dict

    def calculate_clip_batch(self, image_paths: dict, predictions: dict) -> dict:
        """
        Tính CLIPScore cho batch predictions.
        
        Args:
            image_paths: {img_id: image_path}
            predictions: {img_id: prediction_string}
            
        Returns:
            {img_id: clip_score}
        """
        image_ids = list(predictions.keys())

        pil_images = []
        pred_captions = []
        valid_img_ids = []
        clip_scores_dict = {img_id: 0.0 for img_id in image_ids}

        for img_id in image_ids:
            image_path = image_paths[img_id]
            pred_caption = str(predictions[img_id]).strip()

            if pred_caption:
                try:
                    image = Image.open(image_path).convert("RGB")
                    pil_images.append(image)
                    pred_captions.append(pred_caption)
                    valid_img_ids.append(img_id)
                except FileNotFoundError:
                    print(f"Warning: Image not found at {image_path}. Skipping.")
                except Exception as e:
                    print(f"Warning: Error loading image {image_path}: {e}. Skipping.")

        if not pil_images:
            return clip_scores_dict

        try:
            processor = self.clip_metric.processor
            model = self.clip_metric.model
            model.eval()

            inputs = processor(
                text=pred_captions,
                images=pil_images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image

            scores_tensor = logits_per_image.diag()

            for img_id, score in zip(valid_img_ids, scores_tensor.tolist()):
                clip_scores_dict[img_id] = score

        except Exception as e:
            print(f"Error during CLIP batch computation: {e}")

        return clip_scores_dict

    def explanation_rewards(self, ground_truths: list[str], predictions: list[str], image_paths: list[str]) -> list[float]:
        """
        Tính combined reward (BERTScore + CLIPScore) cho batch.
        
        Args:
            ground_truths (list[str]): Dạng ["ground_truth 1", "ground_truth 2", ...]
            predictions (list[str]): Dạng ["prediction 1", "prediction 2", ...]
            image_paths (list[str]): Dạng ["path/to/image1.jpg", "path/to/image2.jpg", ...]

        Returns:
            list[float]
        """
        assert len(ground_truths) == len(predictions) == len(image_paths), \
            "Input lists must have the same length."

        if not predictions:
            return []

        # Mỗi sample chỉ có 1 GT string, không phải list
        gts_dict = {i: gt for i, gt in enumerate(ground_truths)}
        preds_dict = {i: pred for i, pred in enumerate(predictions)}
        paths_dict = {i: path for i, path in enumerate(image_paths)}

        bert_scores = self.calculate_bertscore_batch(gts_dict, preds_dict)
        clip_scores = self.calculate_clip_batch(paths_dict, preds_dict)

        final_rewards = []
        for i in range(len(predictions)):
            bert_score = bert_scores.get(i, 0.0)
            clip_score_raw = clip_scores.get(i, 0.0)

            clip_score_normalized = max(0, (clip_score_raw - 15) / (35 - 15))
            reward = self.alpha * bert_score + (1.0 - self.alpha) * clip_score_normalized
            final_rewards.append(reward)
            print(f"  [Sample {i}] BERTScore={bert_score:.4f}, CLIPRaw={clip_score_raw:.2f} (Norm={clip_score_normalized:.4f}) -> Reward={reward:.4f}")

        return final_rewards


# if __name__ == "__main__":
#     ground_truths_list = [
#         ["một chiếc ô tô màu đỏ đậu trên đường phố", "siêu xe thể thao màu đỏ"],
#         ["chú chó đang chơi trên bãi cỏ xanh", "một chú chó lông vàng chạy trong công viên"]
#     ]


#     predictions_list = [
#         "một chiếc xe hơi màu đỏ",
#         "con chó vui vẻ trên đồng cỏ"
#     ]

#     try:
#         Image.new('RGB', (224, 224), color = 'red').save("car_101.jpg")
#         Image.new('RGB', (224, 224), color = 'green').save("dog_102.jpg")
#         image_paths_list = [
#             "car_101.jpg",
#             "dog_102.jpg"
#         ]

#         reward_scorer = ExplanationRewardScorer(alpha=0.5)

#         final_rewards = reward_scorer.explanation_rewards(
#             ground_truths=ground_truths_list,
#             predictions=predictions_list,
#             image_paths=image_paths_list
#         )

#         print("\n--- Final Reward ---")
#         print(final_rewards)

#     except Exception as e:
#         print(f"\nError: {e}")

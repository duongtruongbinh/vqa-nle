import torch
import numpy as np
import re
import os
import warnings
from datetime import datetime
from PIL import Image
from torchmetrics.multimodal import CLIPScore
from torchmetrics.text import BERTScore

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from transformers import AutoModel, AutoTokenizer
import atexit
from base_rewards import BaseRewardScorer
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.modeling_utils')

import py_vncorenlp

# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/absolute/path/to/vncorenlp')

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_up_visegmenter():
    vncorenlp_dir = '/home/vlai-vqa-nle/minhtq/vqa-nle/vncorenlp_models'
    if not os.path.exists(vncorenlp_dir):
        os.makedirs(vncorenlp_dir)

    # # Tự động tải model VnCoreNLP nếu chưa có trong thư mục
    # if not os.path.exists(os.path.join(vncorenlp_dir, 'models')):
    #     print("Downloading VnCoreNLP models...")
    #     py_vncorenlp.download_model(save_dir=vncorenlp_dir)
    #     print("Download complete!")

    # Load RDRSegmenter 
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/home/vlai-vqa-nle/minhtq/vqa-nle/src/inference/vncorenlp_models')
    return rdrsegmenter

rdrsegmenter = set_up_visegmenter()

def segment_text(text: str) -> str:
    if not text:
        return ""
    try:
        sentences = rdrsegmenter.word_segment(text)
        segmented_text = " ".join([" ".join(sentence) for sentence in sentences])
        return segmented_text
    except Exception as e:
        print(f"Error segmenting text: {e}. Text: '{text}'. Returning empty string.")
        return "" 

class ExplanationRewardScorer(BaseRewardScorer):
    """
    Reward scorer kết hợp BERTScore (semantic similarity) và CLIPScore (image-text alignment).
    Sử dụng PhoBERT cho Vietnamese text evaluation.
    """
    def __init__(self, alpha: float = 0.5, clip_model_name: str = "/home/vlai-vqa-nle/.cache/huggingface/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3"):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be in [0, 1]")

        self.alpha = alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cider_scorer = Cider()
        self.bertscore_metric = self.initialize_bertscore()
        self.clip_metric = CLIPScore(model_name_or_path=clip_model_name).to(self.device)

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

    def explanation_rewards(self, ground_truths: list[str], predictions: list[str], 
                        image_paths: list[str], prompt_ids: list[int]) -> list[float]:
        """
        Tính combined reward (BERTScore + CLIPScore) cho batch với normalization per group.
        
        Args:
            ground_truths (list[str]): Dạng ["ground_truth 1", "ground_truth 2", ...]
            predictions (list[str]): Dạng ["prediction 1", "prediction 2", ...]
            image_paths (list[str]): Dạng ["path/to/image1.jpg", "path/to/image2.jpg", ...]
            prompt_ids (list[int]): ID của prompt tương ứng với mỗi completion
                                VD: [0, 0, 0, 1, 1, 1] nghĩa là 3 completions đầu từ prompt 0,
                                    3 completions sau từ prompt 1
        
        Returns:
            list[float]: Rewards cho từng completion
        """
        from collections import defaultdict
        
        assert len(ground_truths) == len(predictions) == len(image_paths) == len(prompt_ids), \
            "Input lists must have the same length."
        
        if not predictions:
            return []
        
        # Chuẩn bị dữ liệu
        gts_dict = {}
        for i, gt in enumerate(ground_truths):
            if isinstance(gt, list):
                # Nếu gt là list, lấy phần tử đầu tiên
                gt_text = gt[0] if gt else ""
            else:
                # Nếu gt là string
                gt_text = gt
            gts_dict[i] = [segment_text(gt_text.strip())]
        preds_dict = {}
        empty_indices = set()
        
        # Check từng prediction sau segment_text
        for i, pred in enumerate(predictions):
            pred_segmented = segment_text(pred.strip())
            if not pred_segmented or not pred_segmented.strip():
                empty_indices.add(i)
            else:
                preds_dict[i] = pred_segmented
        
        paths_dict = {i: path for i, path in enumerate(image_paths)}
        
        # Tính scores cho non-empty predictions
        bert_scores_dict = self.calculate_bertscore_batch(gts_dict, preds_dict)
        clip_scores_raw_dict = self.calculate_clip_batch(paths_dict, preds_dict)
        
        # --- NORMALIZE CLIP SCORES PER GROUP (GRPO) ---
        clip_normalized_dict = {}
        
        # Group samples by prompt_id
        groups = defaultdict(list)
        for i in range(len(predictions)):
            if i not in empty_indices: 
                pid = prompt_ids[i]
                groups[pid].append(i)
        
        # Normalize within each group separately
        for pid, indices in groups.items():
            # Lấy raw CLIP scores của group này
            group_clip_scores = [clip_scores_raw_dict.get(idx, 0.0) for idx in indices]
            group_tensor = torch.tensor(group_clip_scores, device=self.device, dtype=torch.float32)
            
            # Normalize trong group
            if len(group_tensor) > 1:
                g_min = group_tensor.min()
                g_max = group_tensor.max()
                g_range = g_max - g_min
                
                if g_range > 1e-8:
                    # Min-max normalization to [0, 1]
                    normalized = (group_tensor - g_min) / g_range
                else:
                    # Nếu tất cả giá trị giống nhau, assign 0.5
                    normalized = torch.full_like(group_tensor, 0.5)
                
                print(f"   [Group {pid}] Size={len(indices)}, CLIP Min={g_min:.4f}, Max={g_max:.4f}, Range={g_range:.4f}")
            else:
                # Nếu group chỉ có 1 sample, assign 0.5
                normalized = torch.full_like(group_tensor, 0.5)
                print(f"   [Group {pid}] Size=1, assigned CLIPNorm=0.5")
            
            # Map normalized scores back to dict
            for idx, i in enumerate(indices):
                clip_normalized_dict[i] = normalized[idx].item()
        
        # --- BUILD FINAL REWARDS ---
        final_rewards = []
        for i in range(len(predictions)):
            if i in empty_indices:
                # Phạt -1 cho prediction rỗng
                reward = -1.0
                print(f"   [Sample {i}, Group {prompt_ids[i]}] Empty prediction -> Penalty={reward}")
            else:
                # Tính reward bình thường
                bert_score = bert_scores_dict.get(i, 0.0)
                clip_raw = clip_scores_raw_dict.get(i, 0.0)
                clip_norm = clip_normalized_dict.get(i, 0.0)
                
                # Combined reward: alpha * BERT + (1-alpha) * CLIP_normalized
                reward = (self.alpha * bert_score) + ((1.0 - self.alpha) * clip_norm)
                
                print(f"   [Sample {i}, Group {prompt_ids[i]}] BERT={bert_score:.4f}, "
                    f"CLIPRaw={clip_raw:.2f} (CLIPNorm={clip_norm:.4f}) -> Reward={reward:.4f}")
            
            final_rewards.append(reward)
        
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

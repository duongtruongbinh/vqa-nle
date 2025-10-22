import torch
from torchmetrics.multimodal import CLIPScore
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from PIL import Image
import warnings
import numpy as np 
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.modeling_utils')

class ExplanationRewardScorer:
    def __init__(self, alpha: float = 0.5, clip_model_name: str = "openai/clip-vit-base-patch16"):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be in [0, 1]")
            
        self.alpha = alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cider_scorer = Cider()
        print("CIDEr scorer initialized.")

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
        clip_scores = {}
        for img_id, pred_caption in predictions.items():
            image_path = image_paths[img_id]
            try:
                image = Image.open(image_path).convert("RGB")
            except FileNotFoundError:
                print(f"Warning: Image not found at {image_path}. Skipping.")
                clip_scores[img_id] = 0.0
                continue
            
            image_np = np.array(image)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
            
            self.clip_metric.update(image_tensor.to(self.device), [pred_caption])
            score_tensor = self.clip_metric.compute()
            self.clip_metric.reset()
            
            clip_scores[img_id] = score_tensor.item()
            
        return clip_scores

    def explanation_rewards(self, ground_truths: list[list[str]], predictions: list[str], image_paths: list[str]) -> list[float]:
        """
        Args:
            ground_truths (list[list[str]]): Dạng [["caption 1a", ...], ["caption 2a", ...]]
            predictions (list[str]): Dạng ["prediction 1", "prediction 2", ...]
            image_paths (list[str]): Dạng ["path/to/image1.jpg", "path/to/image2.jpg", ...]

        Returns:
            list[float]
        """
        assert len(ground_truths) == len(predictions) == len(image_paths), \
            "Input lists must have the same length."

        gts_dict = {i: gt for i, gt in enumerate(ground_truths)}
        preds_dict = {i: pred for i, pred in enumerate(predictions)}
        paths_dict = {i: path for i, path in enumerate(image_paths)}

        # print("Calculating CIDEr scores...")
        cider_scores = self.calculate_cider_batch(gts_dict, preds_dict)

        # print("Calculating CLIP scores...")
        clip_scores = self.calculate_clip_batch(paths_dict, preds_dict)

        # print("Combining scores to generate final rewards...")
        final_rewards = []
        for i in range(len(predictions)):
            cider_score = cider_scores.get(i, 0.0)
            
            clip_score_raw = clip_scores.get(i, 0.0)
            clip_score_normalized = max(0, (clip_score_raw - 15) / (35 - 15))

            reward = self.alpha * cider_score + (1.0 - self.alpha) * clip_score_normalized
            final_rewards.append(reward) 
            
            # print(f"  Index {i}: CIDEr={cider_score:.2f}, CLIP={clip_score_raw:.2f} -> Reward={reward:.4f}")

        rewards_array = np.array(final_rewards)
        min_reward = np.min(rewards_array)
        reward_range = np.max(rewards_array) - np.min(rewards_array)
        if reward_range == 0:
            final_rewards = [0.0] * len(rewards_array)
        else:
            final_rewards = (rewards_array - min_reward) / reward_range
            final_rewards = final_rewards.tolist()
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

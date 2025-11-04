import string
import re
import unicodedata
from base_rewards import BaseRewardScorer

def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    
    # 1. Chuẩn hóa các câu trả lời Có/Không (giữ nguyên)
    if text in ["có", "đúng", "yes", "true", "correct"]:
        return "có"
    if text in ["không", "sai", "no", "false", "incorrect"]:
        return "không"
        
    # 2. Chuẩn hóa từ/cụm từ đồng nghĩa
    # Ví dụ: "bay diều", "diều bay" đều có thể được hiểu là "thả diều".
    synonym_map = {
        "bay diều": "thả diều",
        "diều bay": "thả diều",
    }
    if text in synonym_map:
        text = synonym_map[text]
        
    # 3. Loại bỏ các tiền tố/hậu tố phổ biến (giữ nguyên)
    prefixes_to_remove = ["con ", "cái ", "chiếc ", "quả ", "hoa ", "màu ", "bên ", "phía "]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):]
            
    # 4. Loại bỏ các ký tự đặc biệt và khoảng trắng thừa (giữ nguyên)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Chuẩn hóa thứ tự từ
    # Ví dụ: "bò sữa" và "sữa bò" sau khi sắp xếp đều trở thành "bò sữa".
    words = sorted(text.split())
    text = " ".join(words)
    
    return text

def clean_text(text: str) -> str:
    """
    Làm sạch chuỗi trước khi gửi tới pycocoevalcap.Meteor
    - Thay thế mọi biến thể xuống dòng và '|||' bằng khoảng trắng
    - Loại bỏ ký tự điều khiển (Unicode category == 'Cc')
    - Gom nhiều khoảng trắng liên tiếp thành 1
    """
    text = (
        text.replace("|||", " ")
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\u2028", " ")
        .replace("\u2029", " ")
    )
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    text = re.sub(r"\s+", " ", text).strip()
    return text

class AccuracyRewardScorer(BaseRewardScorer):
    """
    Accuracy reward scorer sử dụng BERTScore cho semantic matching.
    Trả về điểm BERTScore trực tiếp (0.0 - 1.0).
    """
    
    def __init__(self, model_name_or_path="/mnt/dataset1/pretrained_fm/vinai/phobert-base"):
        """
        Args:
            model_name_or_path: Đường dẫn đến PhoBERT model cho tiếng Việt
        """
        self.model_path = model_name_or_path
        # Khởi tạo shared BERTScore một lần
        self.initialize_bertscore(self.model_path)
    
    def accuracy_rewards_batch(self, completions: list[str], solutions: list[str]) -> list[float]:
        """
        Batch version - trả về BERTScore F1 cho từng sample (0-1).
        Có thể dùng cho 1 sample hoặc nhiều samples.
        """
        if not completions:
            return []
        
        # Extract tất cả answers
        gts_dict = {}
        preds_dict = {}
        
        for i, (completion, solution) in enumerate(zip(completions, solutions)):
            sol_match = re.search(r"<answer>(.*?)</answer>", solution, flags=re.DOTALL | re.IGNORECASE)
            ground_truth = sol_match.group(1).strip() if sol_match else solution.strip()


            content_match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL | re.IGNORECASE)
            student_answer = content_match.group(1).strip() if content_match else ""
            
            gts_dict[i] = ground_truth
            preds_dict[i] = normalize_answer(clean_text(student_answer))
        
        # Tính BERTScore batch
        bert_scores = self.calculate_bertscore_batch(
            gts_dict, 
            preds_dict,
            model_name_or_path=self.model_path
        )
        
        # Trả về scores
        rewards = [bert_scores.get(i, 0.0) for i in range(len(completions))]
        return rewards


import json

class CaptionRewardScorer(BaseRewardScorer):
    """
    Caption reward scorer sử dụng BERTScore cho semantic matching với COCO captions.
    So sánh student caption với tất cả GT captions và lấy max BERTScore.
    Trả về điểm BERTScore trực tiếp (0.0 - 1.0).
    """
    
    def __init__(self, 
                 model_name_or_path="google-bert/bert-base-uncased",
                 coco_train_path="/home/vlai-vqa-nle/minhtq/vqa-nle/data/processed/coco/coco_train2014_with_captions.json",
                 coco_val_path="/home/vlai-vqa-nle/minhtq/vqa-nle/data/processed/coco/coco_val2014_with_captions.json"):
        """
        Args:
            model_name_or_path: Đường dẫn đến BERT model.
            coco_train_path: Đường dẫn đến COCO train captions JSON
            coco_val_path: Đường dẫn đến COCO val captions JSON
        """
        self.model_path = model_name_or_path
        # Khởi tạo shared BERTScore một lần
        self.initialize_bertscore(self.model_path)
        
        # Load COCO captions một lần
        print("Loading COCO captions...")
        self.coco_captions = {}
        
        # Load train captions
        with open(coco_train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            self.coco_captions.update(train_data['captions_map'])
        
        # Load val captions
        with open(coco_val_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
            self.coco_captions.update(val_data['captions_map'])
        
        print(f"Loaded {len(self.coco_captions)} images with captions")
    
    def caption_rewards_batch(self, completions: list[str], solutions: list[str]) -> list[float]:
        """
        Batch version - trả về max BERTScore F1 cho từng sample (0-1).
        So sánh student caption với tất cả GT captions của image.
        """
        if not completions:
            return []
        
        rewards = []
        
        for completion, solution in zip(completions, solutions):
            # Extract image_id từ solution
            sol_match = re.search(r"<CAPTION>(.*?)</CAPTION>", solution, flags=re.DOTALL | re.IGNORECASE)
            image_id_str = sol_match.group(1).strip() if sol_match else solution.strip()
            
            
            # Extract student caption từ completion
            content_match = re.search(r"<CAPTION>(.*?)</CAPTION>", completion, flags=re.DOTALL | re.IGNORECASE)
            student_caption = content_match.group(1).strip() if content_match else ""
            
            if not student_caption:
                rewards.append(0.0)
                continue
            
            # Lấy tất cả GT captions từ COCO
            if image_id_str not in self.coco_captions:
                print(f"Warning: Image ID {image_id_str} not found in COCO captions")
                rewards.append(0.0)
                continue
            
            gt_captions_list = self.coco_captions[image_id_str]  # List of dicts with 'caption' and 'caption_id'
            gt_captions = [item['caption'] for item in gt_captions_list]
            
            # Tính BERTScore với từng GT caption
            gts_dict = {j: gt_cap for j, gt_cap in enumerate(gt_captions)}
            preds_dict = {j: student_caption for j in range(len(gt_captions))}
            
            bert_scores = self.calculate_bertscore_batch(
                gts_dict, 
                preds_dict,
                model_name_or_path=self.model_path
            )
            
            # Lấy max score
            max_score = max(bert_scores.values()) if bert_scores else 0.0
            rewards.append(max_score)
        
        return rewards
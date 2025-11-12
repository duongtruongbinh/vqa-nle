# src/rewards/base_reward.py
import torch
from torchmetrics.text import BERTScore
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class BaseRewardScorer:
    """
    Base class chứa shared BERTScore model để tái sử dụng cho nhiều reward functions.
    """
    _shared_bertscore = None
    _device = None
    _model_path = None
    
    @classmethod
    def initialize_bertscore(cls, model_name_or_path="/mnt/dataset1/pretrained_fm/vinai/phobert-base"):
        """
        Khởi tạo shared BERTScore model với tham số model path tùy chỉnh.
        """
        if cls._shared_bertscore is None or cls._model_path != model_name_or_path:
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model_path = model_name_or_path
            cls._shared_bertscore = BERTScore(
                model_name_or_path=model_name_or_path,
                num_layers=12,
                rescale_with_baseline=False, 
                device=cls._device,
                dist_sync_on_step=False,
                sync_on_compute=False,
                truncation=True,
                max_length=256
            )
        return cls._shared_bertscore
    
    @classmethod
    def calculate_bertscore_batch(cls, ground_truths: dict, predictions: dict, 
                                model_name_or_path="/mnt/dataset1/pretrained_fm/vinai/phobert-base") -> dict:
        """
        Tính BERTScore cho batch predictions bằng cách xử lý từng sample riêng biệt.
        Đảm bảo mỗi sample được tính độc lập, không bị ảnh hưởng bởi samples khác.
        
        Args:
            ground_truths: {id: [gt1, gt2, ...]} hoặc {id: "gt_string"}
            predictions: {id: prediction_string}
            model_name_or_path: Đường dẫn đến BERT model
            
        Returns:
            {id: bertscore_f1}
        """
        ids = list(predictions.keys())
        bert_scores_dict = {id_: 0.0 for id_ in ids}
        
        # Khởi tạo shared metric
        bertscore_metric = cls.initialize_bertscore(model_name_or_path)
        
        # Xử lý từng sample riêng biệt để tránh contamination
        for id_ in ids:
            pred = str(predictions[id_]).strip()
            gt = ground_truths.get(id_, [])
            
            # Chuẩn hóa ground truth
            if isinstance(gt, str):
                gt_text = gt.strip()
            elif isinstance(gt, list) and len(gt) > 0:
                gt_text = str(gt[0]).strip()
            else:
                gt_text = ""
            
            # Skip nếu thiếu data
            if not pred or not gt_text:
                continue
            
            try:
                # QUAN TRỌNG: Reset metric state trước mỗi sample
                bertscore_metric.reset()
                
                # Tính score cho sample này (single item lists)
                bertscore_metric.update([pred], [gt_text])
                score_dict = bertscore_metric.compute()
                
                # Extract F1 score
                f1_score = score_dict['f1']
                if f1_score.dim() == 0:
                    bert_scores_dict[id_] = f1_score.item()
                else:
                    bert_scores_dict[id_] = f1_score[0].item()
                
            except Exception as e:
                # Silently handle errors, keep default 0.0
                pass
        
        return bert_scores_dict
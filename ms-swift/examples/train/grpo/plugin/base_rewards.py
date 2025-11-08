# src/rewards/base_reward.py
import torch
from torchmetrics.text import BERTScore
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
        
        Args:
            model_name_or_path: Đường dẫn đến BERT model.
        """
        # Chỉ khởi tạo 1 lần hoặc khi model path thay đổi
        if cls._shared_bertscore is None or cls._model_path != model_name_or_path:
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model_path = model_name_or_path
            print(f"Initializing shared BERTScore from: {model_name_or_path}")
            print(f"Device: {cls._device}")
            cls._shared_bertscore = BERTScore(
                model_name_or_path=model_name_or_path,
                num_layers=12,
                rescale_with_baseline=False, 
                device=cls._device,
                dist_sync_on_step=False,
                sync_on_compute=False
            )
            print("Shared BERTScore initialized successfully.")
        return cls._shared_bertscore
    
    @classmethod
    def calculate_bertscore_batch(cls, ground_truths: dict, predictions: dict, 
                                   model_name_or_path="/mnt/dataset1/pretrained_fm/vinai/phobert-base") -> dict:
        """
        Tính BERTScore cho batch predictions.
        
        Args:
            ground_truths: {id: [gt1, gt2, ...]}
            predictions: {id: prediction_string}
            model_name_or_path: Đường dẫn đến BERT model (optional)
            
        Returns:
            {id: bertscore_f1}
        """
        ids = list(predictions.keys())
        bert_scores_dict = {id_: 0.0 for id_ in ids}
        valid_preds = []
        valid_gts = []
        valid_ids = []

        for id_ in ids:
            pred = str(predictions[id_]).strip()
            gt = ground_truths.get(id_, [])
            
            # Xử lý khi GT là string hoặc list
            if isinstance(gt, str):
                gt_list = [gt] if gt.strip() else []
            else:
                gt_list = [str(g).strip() for g in gt if str(g).strip()]

            if pred and gt_list:
                valid_preds.append(pred)
                # Pass string thay vì list để tránh bug với single reference
                valid_gts.append(gt_list) 
                valid_ids.append(id_)

        if not valid_preds:
            return bert_scores_dict

        try:
            # Sử dụng shared BERTScore thay vì tạo mới mỗi lần
            bertscore_metric = cls.initialize_bertscore(model_name_or_path)
            
            # Reset metric để clear cache từ lần tính trước
            bertscore_metric.reset()
            
            # Pass string targets thay vì list of lists
            bertscore_metric.update(valid_preds, valid_gts)
            score_dict = bertscore_metric.compute()

            # Xử lý cả 0-dim và 1-dim tensor
            f1_scores = score_dict['f1']
            if f1_scores.dim() == 0:
                bert_scores = [f1_scores.item()]
            else:
                bert_scores = f1_scores.tolist()
            
            for id_, score in zip(valid_ids, bert_scores):
                bert_scores_dict[id_] = score

        except Exception as e:
            print(f"Error during BERTScore batch computation: {e}")
            import traceback
            traceback.print_exc()

        return bert_scores_dict
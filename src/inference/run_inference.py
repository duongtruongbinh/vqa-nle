import os
import json
import argparse
from tqdm import tqdm
from .models.utils import set_seed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MODELS = {
    "internvl": ".models.internvl.InternVLModel",
    "molmo": ".models.molmo.MolmoModel", 
    "qwenvl": ".models.qwenvl.QwenVLModel",
    "videollama": ".models.videollama.VideoLLaMAModel",
    "phi": ".models.phi.PhiModel",
    "ovis": ".models.ovis.OvisModel",
    "minicpm": ".models.minicpm.MiniCPMModel",
    "vintern1b": ".models.vintern1b.Vintern1BModel"
}

import importlib

def get_model_class(model_key: str):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    module_path, class_name = MODELS[model_key].rsplit('.', 1)
    module = importlib.import_module(module_path, package='src.inference')
    return getattr(module, class_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=MODELS.keys())
    parser.add_argument("--image_folder", type=str, default="/mnt/VLAI_data/COCO_Images/val2014")
    parser.add_argument("--data_path", type=str, default="/mnt/VLAI_data/ViVQA-X/ViVQA-X_test.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="src/inference/results/grpo/")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    args = parser.parse_args()

    set_seed(args.seed)
    
    try:
        ModelClass = get_model_class(args.model)
        model = ModelClass()
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # if args.limit:
    #     data = data[:args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    filename = args.output_name if args.output_name else model.model_name
    if not filename.endswith('.json'):
        filename += ".json"
    output_path = os.path.join(args.output_dir, filename)
    
    print(f"Processing {len(data)} samples...")
    
    for item in tqdm(data, desc=f"Inference"):
        img_path = os.path.join(args.image_folder, item['image_name'])
        
        if not os.path.exists(img_path):
            item["predict"] = "ERROR: Image not found"
            item["pred_explanation"] = ""
            continue
        
        try:
            answer, explanation = model.infer_sft_explain_answer(item['question'], img_path)
            
            item["predict"] = answer if answer else "ERROR: Empty answer"
            item["pred_explanation"] = explanation if explanation else ""

        except Exception as e:
            item["predict"] = f"ERROR: {str(e)}"
            item["pred_explanation"] = ""

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
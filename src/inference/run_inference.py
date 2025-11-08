import os
import json
import argparse
from tqdm import tqdm
from models.utils import set_seed
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
MODELS = {
    "internvl": "models.internvl.InternVLModel",
    "molmo": "models.molmo.MolmoModel", 
    "qwenvl": "models.qwenvl.QwenVLModel",
    "videollama": "models.videollama.VideoLLaMAModel",
    "phi": "models.phi.PhiModel",
    "ovis": "models.ovis.OvisModel",
    "minicpm": "models.minicpm.MiniCPMModel",
    "vintern1b": "models.vintern1b.Vintern1BModel"
}


def import_model_class(model_key: str):
    """
    Dynamically import model class only when needed to avoid environment conflicts.
    
    Args:
        model_key: Key from MODELS dict
        
    Returns:
        Model class
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
    
    module_path, class_name = MODELS[model_key].rsplit('.', 1)
    
    try:
        print(f"📦 Importing {class_name} from {module_path}...")
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        return model_class
    except ImportError as e:
        print(f"❌ Failed to import {class_name}: {e}")
        print(f"💡 Make sure the required dependencies for {model_key} are installed")
        raise
    except AttributeError as e:
        print(f"❌ Class {class_name} not found in {module_path}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a selected model")
    parser.add_argument("model", type=str, choices=MODELS.keys(),
                        help=f"Name of the model to run. Choices: {list(MODELS.keys())}")
    parser.add_argument("--image_folder", type=str,
                        default="/mnt/VLAI_data/COCO_Images/val2014", help="Directory of input images.")
    parser.add_argument("--data_path", type=str, default="/mnt/VLAI_data/ViVQA-X/ViVQA-X_test.json",
                        help="Path to the JSON file of sample questions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="src/inference/results/grpo/", help="Directory to save the results.")
    parser.add_argument("--output_name", type=str, default=None, help="Optional: specific name for the output JSON file (e.g., 'my_test_run').")
    args = parser.parse_args()

    set_seed(args.seed)
    
    print(f"🚀 Initializing {args.model} model...")
    try:
        ModelClass = import_model_class(args.model)
        model = ModelClass()
        clean_model_name = model.model_name
        print(f"✅ Successfully loaded {clean_model_name}")
    except Exception as e:
        print(f"❌ Failed to initialize {args.model} model: {e}")
        return 1
    
    print(f"📂 Loading data from {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_name:
        # Nếu người dùng cung cấp tên, sử dụng nó. Thêm .json nếu chưa có.
        name = args.output_name if args.output_name.endswith('.json') else f"{args.output_name}.json"
        output_filename = os.path.join(args.output_dir, name)
    else:
        # Giữ nguyên hành vi mặc định
        output_filename = os.path.join(args.output_dir, f"{clean_model_name}.json")
    
    print(f"📝 Processing {len(data)} samples with {clean_model_name}...")
    print(f"💾 Results will be saved to: {output_filename}")
    
    data_sliced = data[:300]
    for i, item in enumerate(tqdm(data_sliced, desc=f"Running {clean_model_name}")):
        img_path = os.path.join(args.image_folder, item['image_name'])
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            item["predict"] = "ERROR: Image file not found"
            item["pred_explanation"] = ""  # ← THÊM DÒNG NÀY
            continue
        
        try:
            answer, explanation = model.infer(item['question'], img_path)
            
            if answer is None or answer == "":
                print(f"⚠️  Model returned empty answer for {item['image_id']}")
                item["predict"] = "ERROR: Empty answer"
                item["pred_explanation"] = ""
            else:
                item["predict"] = answer
                item["pred_explanation"] = explanation if explanation else ""
                
        except Exception as e:
            print(f"❌ Error processing item {item['image_id']}: {e}")
            import traceback
            traceback.print_exc()  # ← In ra full error để debug
            item["predict"] = f"ERROR: {str(e)}"
            item["pred_explanation"] = ""  # ← THÊM DÒNG NÀY

    print(f"✅ Inference complete!")    
    print(f"💾 Saving final results to {output_filename}")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(data_sliced, f, ensure_ascii=False, indent=2)

    print("🎉 All done!")
    return 0

if __name__ == "__main__":
    exit(main()) 
"""Run inference with GRPO-trained VQA models."""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

# Environment configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from models.utils import set_seed

MODELS = {
    "internvl": "models.internvl.InternVLModel",
    "molmo": "models.molmo.MolmoModel",
    "qwenvl": "models.qwenvl.QwenVLModel",
    "videollama": "models.videollama.VideoLLaMAModel",
    "phi": "models.phi.PhiModel",
    "ovis": "models.ovis.OvisModel",
    "minicpm": "models.minicpm.MiniCPMModel",
    "vintern1b": "models.vintern1b.Vintern1BModel",
}

# Default paths
DEFAULT_IMAGE_FOLDER = "/mnt/VLAI_data/COCO_Images/val2014"
DEFAULT_DATA_PATH = "/mnt/VLAI_data/ViVQA-X/ViVQA-X_test.json"
DEFAULT_OUTPUT_DIR = "src/inference/results/grpo/"


def import_model_class(model_key: str):
    """Dynamically import model class to avoid environment conflicts."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    module_path, class_name = MODELS[model_key].rsplit(".", 1)
    print(f"📦 Importing {class_name} from {module_path}...")

    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def process_sample(model, item: dict, image_folder: Path) -> dict:
    """Process a single sample and return updated item."""
    img_path = image_folder / item["image_name"]

    if not img_path.exists():
        print(f"⚠️  Image not found: {img_path}")
        item["predict"] = "ERROR: Image file not found"
        return item

    think, answer, explanation = model.infer_grpo(item["question"], str(img_path))
    item["thinking"] = think
    item["predict"] = answer
    item["pred_explanation"] = explanation

    print(f"Q: {item['question']}")
    print(f"Thinking: {think}")
    print(f"Predicted: {answer} | GT: {item['answer']}")
    print(f"Explanation: {explanation}")

    return item


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a selected model")
    parser.add_argument("model", choices=MODELS.keys(), help="Model to run")
    parser.add_argument("--image_folder", default=DEFAULT_IMAGE_FOLDER)
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", default=None, help="Custom output filename")
    parser.add_argument("--limit", type=int, default=300, help="Limit number of samples")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    # Load model
    print(f"🚀 Initializing {args.model} model...")
    try:
        model = import_model_class(args.model)()
        print(f"✅ Successfully loaded {model.model_name}")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return 1

    # Load data
    print(f"📂 Loading data from {args.data_path}...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)[: args.limit]

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or model.model_name
    output_file = output_dir / f"{output_name}.json"

    print(f"📝 Processing {len(data)} samples...")
    print(f"💾 Results will be saved to: {output_file}")

    # Process samples
    image_folder = Path(args.image_folder)
    for item in tqdm(data, desc=f"Running {model.model_name}"):
        try:
            process_sample(model, item, image_folder)
        except Exception as e:
            print(f"❌ Error processing {item.get('image_id', 'unknown')}: {e}")
            item["predict"] = f"ERROR: {e}"

    # Save results
    print(f"💾 Saving results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("🎉 All done!")
    return 0


if __name__ == "__main__":
    exit(main())
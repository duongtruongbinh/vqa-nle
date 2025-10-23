# VQA Inference Pipeline

A modular Visual Question Answering (VQA) inference system supporting multiple state-of-the-art vision-language models.

## Quick Start

### 1. Run Inference
```bash
python run_inference.py <model_name> [options]
```

**Available Models:**
- `qwenvl` - Qwen2.5-VL-7B-Instruct  
- `internvl` - InternVL3.5-8B
- `molmo` - Molmo-7B-D-0924
- `videollama` - VideoLLaMA3-7B-Image
- `phi` - Phi-4-multimodal-instruct
- `ovis` - Ovis2.5-9B
- `minicpm` - MiniCPM-o-2-6

**Options:**
- `--image_folder` - Image directory (default: `/mnt/VLAI_data/COCO_Images/val2014`)
- `--data_path` - Questions JSON file (default: `/mnt/VLAI_data/ViVQA-X/ViVQA-X_test.json`)
- `--output_dir` - Results directory (default: `results`)
- `--seed` - Random seed (default: 42)

**Example:**
```bash
python run_inference.py qwenvl --output_dir my_results
```

### 2. Calculate Scores
```bash
python calculate_scores.py [--input-dir results] [--device cuda]
```

Computes NLG metrics (BLEU, ROUGE, CIDEr, SPICE, BERTScore) for explanation evaluation.

## Input Format

Questions JSON should contain:
```json
[
  {
    "image_name": "image.jpg",
    "question": "Câu hỏi của bạn?", 
    "answer": "expected_answer",
    "explanation": ["expected_explanation"]
  }
]
```

## Output Format

Results are saved as JSON with:
```json
[
  {
    "image_name": "image.jpg",
    "question": "Câu hỏi của bạn?",
    "answer": "expected_answer", 
    "predict": "model_answer",
    "pred_explanation": "model_explanation"
  }
]
```

## Dependencies

Core requirements:
- PyTorch
- Transformers 
- PIL/Pillow
- Model-specific packages (see individual model documentation)

For evaluation:
- pycocoevalcap
- bert-score

## Architecture

- `run_inference.py` - Main inference script
- `models/` - Model implementations
  - `base_model.py` - Abstract base class
  - `qwenvl.py`, `internvl.py`, etc. - Model-specific implementations
  - `utils.py` - Shared utilities
- `calculate_scores.py` - NLG evaluation metrics
- `results/` - Output directory

## Adding New Models

1. Create new model file in `models/` inheriting from `VQAModel`
2. Implement `load_model()` and `infer()` methods
3. Add entry to `MODELS` dict in `run_inference.py`
4. Return `tuple[str, str]` (answer, explanation) from `infer()`

See existing models for implementation examples.

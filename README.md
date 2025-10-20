# GRPO VLM Research Project

## Overview

This project is a research environment for fine-tuning Vision Language Models (VLMs) using Group Relative Policy Optimization (GRPO). It provides a structured setup for data processing, model training, evaluation, and inference.

## Project Structure

```
├── configs/            # Experiment configuration files (e.g., hyperparameters)
├── data/               # Datasets (raw and processed)
├── models/             # Saved model checkpoints (local)
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── results/            # Experiment results, logs, and outputs
├── src/                # Reusable source code
│   ├── data/           # Data loading and preprocessing
│   ├── evaluation/     # Model evaluation scripts
│   ├── inference/      # Inference scripts
│   └── training/       # Core training logic
├── tests/              # Unit and integration tests
├── .github/workflows/  # CI/CD workflows
├── Dockerfile          # Docker container definition
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd vqa-nle
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    conda create --name venv python=3.10 -y
    conda activate venv
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Training

### General GRPO Training

To run the GRPO training script, execute the following command from the project's root directory:

```bash
python -m src.training.run_grpo
```

### InternVL-Specific Training

For training with InternVL models, use the specialized script:

```bash
bash VLM-R1/run_scripts/run_grpo_rec_internvl.sh
```

### Running the Training Script

After preparing the data, you can start the training process.

#### 1. Configure the Training Script

Before running, open `VLM-R1/run_scripts/run_grpo_rec_internvl.sh` and verify the following parameters:

-   **`data_paths`**: (Line 7) Should point to your generated `ViVQA-X_train_grpo.jsonl`. The default is `/home/vlai-vqa-nle/minhtq/vqa-nle/data/processed/ViVQA-X_train_grpo.jsonl`.
-   **`image_folders`**: (Line 8) Should point to the directory containing the COCO images (`train2014`, `val2014`). The default is `/mnt/VLAI_data/COCO_Images/train2014`.
-   **`model_path`**: (Line 9) The model identifier from Hugging Face or a local path to the pretrained model you want to fine-tune (e.g., `5CD-AI/Vintern-1B-v3_5`).
-   **`EXP_NAME`**: (Line 14) A unique name for your experiment. This will be used for creating directories for logs and checkpoints.
-   **`CUDA_VISIBLE_DEVICES`**: (Line 26) The index of the GPU to use. Default is `2`. If you want to use a different GPU, change this value (e.g., `0` or `1`).
-   **`--nproc_per_node`**: (Line 27) The number of processes, which should equal the number of GPUs specified in `CUDA_VISIBLE_DEVICES`. For a single GPU, this should be `"1"`.

#### 2. Execute the Script

Navigate to the project's root directory and run the script. The script will automatically change the directory to `VLM-R1/src/open-r1-multimodal` before launching the training.

```bash
cd /home/vlai-vqa-nle/minhtq/vqa-nle/VLM-R1
bash run_scripts/run_grpo_rec_internvl.sh
```

#### 3. Monitor the Training

-   **Logs**: Detailed logs, including debug outputs if `DEBUG_MODE` is `"true"`, will be saved in the `runs/${EXP_NAME}/log/` directory inside the `VLM-R1` folder.
-   **Checkpoints**: Model checkpoints will be saved periodically to `VLM-R1/checkpoints/rl/${EXP_NAME}/`.
-   **W&B**: If not disabled (by uncommenting `export WANDB_DISABLED=true`), training metrics will be reported to Weights & Biases under the run name specified by `${EXP_NAME}`.

## Data Processing

### Custom Data Preparation for InternVL

The project includes custom data processing modules specifically designed for InternVL training:

#### 1. Dataset Loader (`src/data/dataset_loader.py`)

**Purpose**: Converts ViVQA-X dataset to GRPO-compatible JSONL format

**Key Features**:
- **Vietnamese System Prompt**: Enhanced prompt with structured thinking process
- **Format Conversion**: Converts ViVQA-X JSON to VLM-R1 GRPO JSONL format
- **Image Path Handling**: Uses relative paths for flexible image folder configuration

**Usage**:
```python
from src.data.dataset_loader import create_jsonl_for_grpo

# Create training data
create_jsonl_for_grpo("train")

# Create validation data  
create_jsonl_for_grpo("val")

# Create test data
create_jsonl_for_grpo("test")
```

**Output Format**:
```json
{
    "id": 1,
    "image": "COCO_train2014_000000000139.jpg",
    "conversations": [
        {
            "from": "human", 
            "value": "<image>Bạn là một trợ lý AI chuyên gia... Câu hỏi: {question}"
        },
        {
            "from": "gpt",
            "value": "<answer>{answer}</answer><explain>{explanation}</explain>"
        }
    ]
}
```

### Data Setup Instructions

1. **Prepare ViVQA-X Dataset**:
   ```bash
   # Ensure ViVQA-X data is available at:
   /mnt/VLAI_data/ViVQA-X/
   ├── ViVQA-X_train.json
   ├── ViVQA-X_val.json
   └── ViVQA-X_test.json
   ```

2. **Prepare COCO Images**:
   ```bash
   # Ensure COCO images are available at:
   /mnt/VLAI_data/COCO_Images/
   ├── train2014/
   └── val2014/
   ```

3. **Generate GRPO Data**:
   ```bash
   cd /home/vlai-vqa-nle/minhtq/vqa-nle
   python -m src.data.dataset_loader
   ```

## InternVL-Specific Modifications

This project has been specifically adapted to support InternVL models with the following key modifications:

### 1. Module Detection and Routing

**File**: `VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py` (line 942)

Added "vintern" detection to route to `InvernVLModule`:
```python
elif "internvl" in model_name_or_path.lower() or "vintern" in model_name_or_path.lower():
    return InvernVLModule
```

### 2. Environment Configuration

**File**: `VLM-R1/run_scripts/run_grpo_rec_internvl.sh`

- **DS_SKIP_CUDA_CHECK**: Set to `1` (line 1) to skip CUDA compatibility checks.
- **CUDA_VISIBLE_DEVICES**: Set to `2` for 1-GPU training (line 26).
- **Process Configuration**: `torchrun --nproc_per_node="1"` to match GPU count (line 27).

### 3. Training Parameters

**File**: `VLM-R1/run_scripts/run_grpo_rec_internvl.sh`

- **Gradient Checkpointing**: Disabled (`--gradient_checkpointing false`, line 44).
- **Batch Size**: Set to `--per_device_train_batch_size 1` (line 42).
- **PEFT**: Not used with InternVL (no LoRA/PEFT parameters in script).

> **Note on PEFT and InternVL:**
> Using PEFT (LoRA) with InternVL can cause a `TypeError: InternVLChatModel.forward() got an unexpected keyword argument 'inputs_embeds'`. This happens because:
> 1.  PEFT automatically injects the `inputs_embeds` argument when using LoRA with CausalLM models.
> 2.  The `InternVL` model's `forward` method does not accept this argument.
> 3.  The GRPO trainer passes all keyword arguments from PEFT to the model, leading to a crash.
> For this reason, PEFT is disabled for InternVL training.

### 4. DeepSpeed Configuration

**File**: `VLM-R1/src/open-r1-multimodal/local_scripts/zero2.json`

Optimized for InternVL with ZeRO Stage 2 and optimizer offloading:
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

### 5. Model Initialization Fix

To avoid the `self.img_context_token_id is not None` error, the model initialization logic has been updated to correctly handle PEFT models by accessing the underlying base model:

```python
if is_peft_model(model):
    self.vlm_module.post_model_init(model.base_model.model, processing_class)
else:
    self.vlm_module.post_model_init(model, processing_class)
if self.ref_model is not None:
    if is_peft_model(self.ref_model):
        self.vlm_module.post_model_init(self.ref_model.base_model.model, processing_class)
    else:
        self.vlm_module.post_model_init(self.ref_model, processing_class)
```

### 6. Checkpoint Handling Modification

The training script logic was adjusted to ensure that training restarts from the beginning when parameters are changed, rather than always resuming from the latest checkpoint. The `resume_from_checkpoint=True` parameter is now conditional.

```python
# Original logic always resumed if a checkpoint existed
if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
```


    --max_steps 10000 \  # Training sẽ dừng ở điều kiện nào đến trước
    --num_train_epochs 11 \
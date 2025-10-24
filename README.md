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

### Detailed Training Guide

This guide provides comprehensive instructions for running the GRPO training script (`run_grpo_rec_internvl.sh`), including configuration, execution, and customization.

#### 1. Prerequisites

Before running the training script, ensure the following requirements are met:

*   **Hardware**: A CUDA-enabled GPU with at least 24GB of VRAM is recommended for fine-tuning 1B models.
*   **Software**: Ensure all dependencies from `requirements.txt` are installed in your activated conda environment. DeepSpeed is required.
*   **Data**: You must first generate the GRPO-compatible dataset by running the data processing script as described in the "Data Processing" section. Verify that `ViVQA-X_train_grpo.jsonl` exists in the `data/processed/` directory.
*   **Weights & Biases (Optional)**: If you want to log metrics, make sure you are logged into W&B by running `wandb login` in your terminal. To disable it, uncomment `export WANDB_DISABLED=true` (line 26) in the script.

#### 2. Configuration Parameters

The training process is controlled by variables and arguments in `VLM-R1/run_scripts/run_grpo_rec_internvl.sh`. Below is a detailed breakdown of key parameters.

##### **Essential Parameters**

These must be configured correctly for your environment.

| Variable                  | Line | Description                                                                                             |
| ------------------------- | :--: | ------------------------------------------------------------------------------------------------------- |
| `data_paths`              |  8   | **Required**. Path to the `ViVQA-X_train_grpo.jsonl` file.                                               |
| `image_folders`           |  9   | **Required**. Path to the directory containing COCO images (`train2014`).                                 |
| `model_path`              |  10  | The Hugging Face model identifier or a local path to the pretrained model (e.g., `OpenGVLab/InternVL3-1B`). |
| `EXP_NAME`                |  15  | A unique name for your experiment. Logs and checkpoints will be saved under this name.                  |
| `CUDA_VISIBLE_DEVICES`    |  27  | The specific GPU(s) to use (e.g., `0`, `0,1`). Default is `2`.                                            |
| `--nproc_per_node`        |  28  | The number of GPUs to use. This should match the count in `CUDA_VISIBLE_DEVICES`.                       |

> **Important Note on `EXP_NAME`**: Avoid using forward slashes (`/`) in your experiment name (e.g., `my_experiment/run_1`). The system will interpret the slash as a directory separator and create nested folders. This can cause the `resume_from_checkpoint` logic to fail because it may not correctly locate the checkpoint files inside the nested structure. It is recommended to use hyphens (`-`) or underscores (`_`) instead.

##### **Training Hyperparameters**

These parameters control the training loop and performance.

| Argument                        | Line | Description                                                                                             |
| ------------------------------- | :--: | ------------------------------------------------------------------------------------------------------- |
| `--per_device_train_batch_size` |  42  | The number of samples processed per GPU in one forward pass. Adjust based on VRAM.                      |
| `--gradient_accumulation_steps` |  43  | Number of updates steps to accumulate gradients before performing a backward pass. Effective batch size = `nproc_per_node * per_device_train_batch_size * gradient_accumulation_steps`. |
| `--learning_rate`               |  60  | The initial learning rate for the optimizer.                                                            |
| `--max_steps`                   |  46  | The total number of training steps to perform. Overrides `num_train_epochs`.                            |
| `--save_steps`                  |  52  | How often to save a model checkpoint, specified in number of steps.                                     |

##### **GRPO-Specific Parameters**

These parameters are unique to the Group Relative Policy Optimization algorithm.

| Argument              | Line | Description                                                                                    |
| --------------------- | :--: | ---------------------------------------------------------------------------------------------- |
| `--num_generations`   |  53  | The number of candidate responses to generate for each prompt during training.                 |
| `--reward_funcs`      |  55  | A space-separated list of reward functions to use for scoring generations (e.g., `accuracy format`). |
| `--beta`              |  56  | The KL divergence penalty coefficient. Controls how much the policy model can deviate from the reference model. |

##### **Advanced Options**

| Argument                    | Line | Description                                                                                             |
| --------------------------- | :--: | ------------------------------------------------------------------------------------------------------- |
| `--gradient_checkpointing`  |  44  | A memory optimization technique that trades compute for memory. Set to `true` to reduce VRAM usage.     |
| `--freeze_vision_modules`   |  61  | If `true`, the weights of the vision encoder are frozen and not updated during training.                |
| `--push_to_hub`             |  62  | If `true`, automatically pushes the final trained model to the Hugging Face Hub.                        |
| `--hub_model_id`            |  63  | The repository name for the model on the Hugging Face Hub (e.g., `YourUsername/YourModelName`).         |

#### 3. Step-by-Step Execution

1.  **Navigate to the Correct Directory**: The script must be run from the `VLM-R1` directory.
```bash
cd /home/vlai-vqa-nle/minhtq/vqa-nle/VLM-R1
    ```
2.  **Run the Script**:
    ```bash
bash run_scripts/run_grpo_rec_internvl.sh
```
3.  **Verify Execution**: Upon starting, you should see logs indicating the setup of the environment, model loading, and then the start of the training steps.

#### 4. Monitoring and Debugging

*   **Logs**: Training logs are saved to `VLM-R1/runs/${EXP_NAME}/log/`. If `DEBUG_MODE` (line 19) is `"true"`, the log will include detailed rollout information.
*   **Checkpoints**: Model checkpoints are saved in `VLM-R1/checkpoints/rl/${EXP_NAME}/`.
*   **W&B Dashboard**: If enabled, you can monitor training metrics in real-time at `wandb.ai` under the project and run name you configured.

#### 5. Customization Scenarios

##### **Basic Customization**

*   **Adjusting GPU and Batch Size**: For a single GPU setup on GPU `0`, change `CUDA_VISIBLE_DEVICES` to `0` and ensure `--nproc_per_node` is `"1"`. If you encounter CUDA "out of memory" errors, reduce `--per_device_train_batch_size`.
*   **Changing Reward Functions**: Modify the `--reward_funcs` argument (line 55) to add or remove rewards. For example, to add the `explanation` reward, change it to `--reward_funcs accuracy format explanation`.

##### **Advanced Customization**

*   **Adapting for a New VQA Dataset**:
    1.  Modify `src/data/dataset_loader.py` to handle your custom dataset's format.
    2.  Ensure your data loader outputs a `.jsonl` file consistent with the required format (containing `id`, `image`, and `conversations` fields).
    3.  Update the `data_paths` variable in the run script to point to your new dataset file.

*   **Switching to a Different Model**:
    1.  Change the `model_path` variable (line 10) to the Hugging Face identifier or local path of the new model.
    2.  You may need to adjust `--per_device_train_batch_size` and `--learning_rate` depending on the new model's size and architecture.
    3.  Verify that the model is compatible with the existing `InvernVLModule` or implement a new module if needed.

#### 6. Troubleshooting Common Issues

*   **CUDA Out of Memory**: The most common issue.
    *   **Solution**: Decrease `--per_device_train_batch_size`. If that's not enough, enable `--gradient_checkpointing true`. You can also try reducing `--max_completion_length`.
*   **DeepSpeed Errors**: Ensure your DeepSpeed configuration file (`local_scripts/zero2.json`) is correctly formatted and that DeepSpeed was installed properly.
*   **Checkpoint Resumption Behavior**: The training script `grpo_jsonl.py` contains hard-coded logic that automatically resumes from a checkpoint if one is found in the output directory. This **ignores and overrides** the `--resume_from_checkpoint False` flag set in the `run_grpo_rec_internvl.sh` script. To start a fresh training run when parameters are changed, you have three options:
    *   **Option 1 (Workaround, Recommended)**: Change the `EXP_NAME` variable (line 15) in the run script. This creates a new, empty directory for checkpoints and logs, forcing a fresh start.
    *   **Option 2 (Workaround)**: Manually delete the contents of the existing checkpoint directory (`VLM-R1/checkpoints/rl/${EXP_NAME}/`).
    *   **Option 3 (Permanent Fix)**: Modify the logic in `VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py` to respect the script's arguments.

        **Original Code (lines 1216-1219):**
        ```python
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        ```

        **Modified Code:**
        ```python
        # This allows the --resume_from_checkpoint flag to control the behavior.
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        ```

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

### 7. Data Handling Logic (`grpo_jsonl.py`)

-   **File**: `VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py`
-   **Problem**: The original data loading logic was designed to automatically wrap solutions in `<answer>` tags. It first stripped any existing `<answer>` tags from the ground truth and then added them back. This created a logic conflict with pre-formatted data for ViVQA-X, which already contains both `<answer>` and `<explain>` tags, leading to incorrect nested tags (e.g., `<answer>...<explain>...</explain></answer>`).
-   **Modification**: The data processing pipeline has been adjusted to preserve the original tags from the input data. The code that stripped and re-added `<answer>` tags has been changed. The script now directly uses the `solution` string from the JSONL file, assuming it is already correctly formatted with all necessary tags.

    **Original Logic (Before):**
    ```python
    # In the data loading loop
    item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
    
    # In make_conversation_from_jsonl function
    'solution': f"<answer> {example['solution']} </answer>",
    ```

    **Modified Logic (After):**
    ```python
    # In the data loading loop
    item['solution'] = solution_value # Directly use the pre-formatted string

    # In make_conversation_from_jsonl function
    'solution': example['solution'], # Use the solution as-is
    ```

## Customization and Extensibility

This section provides guidance on how to customize the training pipeline, such as adding new reward functions or understanding how data is handled.

### Adding a New Reward Function

To integrate a new reward function into the GRPO training pipeline, follow these steps:

1.  **Implement the Reward Function**:
    -   Open `VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py`.
    -   Define your new Python function. The function must accept `completions`, `solution`, and `**kwargs` as arguments and return a list of floating-point scores. `kwargs` can be used to access additional data like `image_path`.
    -   **Example Signature**:
        ```python
        def my_new_reward(completions, solution, **kwargs):
            scores = []
            # ... your logic here ...
            return scores
        ```

2.  **Normalize the Reward Score**:
    -   It is crucial that your reward function returns scores normalized to a consistent range, typically **[0.0, 1.0]**. This ensures that it combines fairly with other reward functions (`accuracy`, `format`, `explanation`).

3.  **Register the Function**:
    -   In the same file, add your function to the `reward_funcs_registry` dictionary.
        ```python
        reward_funcs_registry = {
            "accuracy": accuracy_reward,
            "format": format_reward,
            "explanation": explanation_reward,
            # Add your new function here
            "my_new_reward": my_new_reward,
        }
        ```

4.  **Activate in Training Script**:
    -   Open your training script (e.g., `VLM-R1/run_scripts/run_grpo_rec_internvl.sh`).
    -   Add the key of your new reward function to the `--reward_funcs` argument.
        ```bash
        --reward_funcs accuracy format explanation my_new_reward \
        ```

### Image Path Handling for Rewards

The data pipeline is optimized to handle images efficiently, especially for reward functions that require image data (e.g., CLIP-based scores).

-   **Path-Based Loading**: Instead of loading images into memory during initial data processing, the script stores file paths in the `image_path` field. This is done in `grpo_jsonl.py` by combining the `image_folders` path from the run script with the image filename from the JSONL data.
-   **Access in Reward Functions**: The list of image paths for the current batch is passed into the `**kwargs` dictionary for all reward functions. A function can access it like this:
    ```python
    def my_reward_with_images(completions, solution, **kwargs):
        if 'image_path' in kwargs:
            image_paths_list = kwargs['image_path']
            # Now you can open and process images using these paths
            for path in image_paths_list:
                # ...
    ```
-   **Singleton Scorer**: For reward models that are computationally expensive to initialize (like CLIP), a global singleton pattern is used (`initialize_explanation_scorer`). This ensures the model is loaded into memory only once per process, preventing repeated initialization on every batch.

## Advanced Customization Guide

This section provides a deeper dive into the specific customizations made for the Vietnamese VQA task, including data preparation, custom reward functions, and VLM module modifications.

### 1. Custom Data Loader with Vietnamese System Prompt

To adapt the training for the ViVQA-X dataset, a custom data loader was created at `src/data/dataset_loader.py`.

**Key Features**:

-   **Vietnamese System Prompt**: A specialized Vietnamese prompt is used to guide the model's response structure, requiring it to generate `<think>`, `<answer>`, and `<explain>` tags. This ensures the model follows a structured reasoning process.

    ```python:4:15:src/data/dataset_loader.py
    prompt = """
    <image> You are a Visual Question Answering system. Your task is to answer questions based on the content of the provided image. 
    You must respond in Vietnamese and your response **must** include all the tags <think> </think>, <answer> </answer>, <explain> </explain>.
    
    Follow these steps precisely:
    1. In the <think> tag, provide a step-by-step reasoning process.
    2. In the <answer> tag, give one word or one short phrase.
    3. In the <explain> tag, provide one brief sentence that justifies your answer.
    
    Now, answer this question based on the image:
    Question: {question}
    """.strip()
    ```

-   **GRPO JSONL Conversion**: The script converts the standard ViVQA-X JSON format into the JSONL format required by the GRPO trainer, injecting the system prompt and structuring the ground truth solution.

    ```python:58:59:src/data/dataset_loader.py
    # format câu trả lời
    solution = f"<answer>{answer}</answer><explain>{explanation}</explain>"
    ```

### 2. Custom Reward Functions

To better evaluate the model's performance on the Vietnamese VQA task, custom reward functions were developed in the `src/rewards/` directory and integrated into `VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py`.

#### 2.1 Explanation Reward (CLIP + CIDEr)

**File**: `src/rewards/explaination_rewards.py`

This reward function evaluates the quality of the generated explanation in the `<explain>` tag by combining two scores:

1.  **CLIP Score**: Measures the semantic similarity between the generated text and the image.
2.  **CIDEr Score**: Measures the consensus between the generated text and the ground truth explanations.

The final score is a weighted average of the two, normalized to a `[0, 1]` range. A singleton pattern (`initialize_explanation_scorer`) is used in `grpo_jsonl.py` to ensure the CLIP model is loaded only once.

#### 2.2 Outcome/Accuracy Reward

**File**: `src/rewards/outcome_rewards.py`

This function provides a more nuanced evaluation of the answer in the `<answer>` tag compared to simple exact matching.

**Scoring Logic**:

-   **3.0 points**: For a perfect, exact match after normalization.
-   **1.5 points**: For a partial match where at least 50% of the ground truth words are present.
-   **0.0 points**: Otherwise.

This logic is integrated into `grpo_jsonl.py` at line `893` as the default accuracy reward.

### 3. VLM Module and Data Handling Modifications

The core training script `grpo_jsonl.py` was modified to support custom models and handle data more efficiently.

#### 3.1 VLM Module Abstraction

A VLM module system (`VLM-R1/src/open-r1-multimodal/src/open_r1/vlm_modules/`) was implemented to easily switch between different model architectures (e.g., Qwen2-VL, InternVL). To add a new model, you would create a new module inheriting from `VLMBaseModule` and implement the required methods. The model is then registered in the `get_vlm_module` function in `grpo_jsonl.py`.

```python:1083:1086:VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py
    elif "internvl" in model_name_or_path.lower() or "vintern" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")
```

#### 3.2 Data Handling: Preserving Tags

The original data loading logic in the GRPO trainer would strip `<answer>` tags and then re-add them. This conflicted with our pre-formatted data which also included `<explain>` tags. The logic was modified to preserve the solution string as-is, ensuring the format is maintained.

-   **Previous Logic**: `item['solution'] = solution_value.replace('<answer>', '')...`
-   **Current Logic** (`line 1142`): `item['solution'] = solution_value`

#### 3.3 Image Path Handling (Lazy Loading)

Instead of loading all images into memory at the start, the data pipeline was modified to only store image file paths. The images are loaded just-in-time during the training step. This significantly reduces memory usage and speeds up initialization. The image paths are then passed to reward functions that require them (like the Explanation Reward).

### 4. Complete Integration Example

Here is a summary of how to use these customizations together:

1.  **Prepare Data**: Use the custom script to generate the GRPO-compatible JSONL file.
    ```python
    from src.data.dataset_loader import create_jsonl_for_grpo
    create_jsonl_for_grpo("train")
    ```

2.  **Integrate Custom Rewards**: In `grpo_jsonl.py`, import and register your custom reward functions.
    ```python
    # Import your custom reward functions
    from src.rewards.explaination_rewards import ExplanationRewardScorer 
    from src.rewards.outcome_rewards import accuracy_reward as custom_accuracy_reward

    # In accuracy_reward function (line 893)
    reward = custom_accuracy_reward(content, sol)

    # In reward_funcs_registry, ensure `explanation` is active
    reward_funcs_registry = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "explanation": explanation_reward, 
    }
    ```

3.  **Configure Training Script**: In `run_grpo_rec_internvl.sh`, add `explanation` to the list of active rewards.
    ```bash
    --reward_funcs accuracy format explanation \
    ```
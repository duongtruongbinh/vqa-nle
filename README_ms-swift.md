# GRPO RUNNING DETAIL STEP-BY-STEP

```bash
cd ms-swift/examples
```

All downloaded LLMs models will be saved at: `/home/vlai-vqa-nle/.cache/modelscope/hub/models/`

### 1. CUSTOM DATASET
Data or prompt can be customized in `examples/custom/dataset_loader.py`
The ouputs are jsonl files, saved at `ms-swift/data_custom`. The dataset directory can be modified, please set `data_dir=your/output_dir` in `dataset_loader.py`.

Format is followed by GRPO format, see https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html


### 2. CUSTOM REWARD FUNCTION
Step 1: Go to `examples/train/grpo/plugin/`, here's the directory contains your reward functions 

Step 2: In the director, we can set up customized function in `plugin.py`:
* Create a class which inherit ORM base class: `YourCustomReward(ORM):`
* Declare call method:
  ```
  YourCustomReward(ORM):
     def __call__(self, completions: List[str], **kwargs) -> List[float]:
        #Your logic here

  orms['my_custom_reward'] = YourCustomReward
  ```
* Inside the function, custom your rewards
  
Step 3: It's required one more decleration after finishing, please add `orms['my_custom_reward'] = YourCustomReward` below you customized class

### 3. Run GRPO
All scripts files are contained in `examples/train/grpo/internal/`, their name start with `grpo_`.

An detail example is bewlow, call your customized reward function after `--reward_funcs`, for model_id, see https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html :
```
MODEL_ID_OR_PATH="OpenGVLab/InternVL3-1B-Instruct" # Model ID
MODEL_TYPE="internvl3" 

... 

swift rlhf \
    --rlhf_type grpo \
    --model_type "$MODEL_TYPE" \
    --model "$MODEL_ID_OR_PATH" \
    --dataset "$TRAIN_DATASET_PATH" \
    --external_plugins "$PLUGIN_PATH" \
    --reward_funcs custom_format_reward custom_accuracy_reward my_custom_reward \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --output_dir "$OUTPUT_DIR" \
    --per_device_eval_batch_size $NUM_GENERATIONS \
    --max_length $MAX_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --save_steps $SAVE_STEPS \
    --max_steps $MAX_STEPS \
    --logging_steps $LOGGING_STEPS \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --num_generations $NUM_GENERATIONS \
    --temperature $TEMPERATURE \
    --top_p 0.9 \
    --top_k 50 \
    --log_completions true \
    --torch_dtype bfloat16 \
    --save_only_model false \
    --save_total_limit 2 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 1 \
    --report_to wandb \
    --beta 0.04 \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --gradient_checkpointing true \
    # --ref_model id_hoặc_path_đến_model_SFT_ban_đầu # Nên chỉ định ref_model
```

for more parameter information, see https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#grpo-arguments

Call the script file in terminal to run, for examples  `bash ms-swift/examples/train/grpo/internal/grpo_internvl3_5_4B_stage1.sh`
### 4. Track training output 
There are two ways:

1. Wandb, please go to project's wandb to see detail

   > **Note on WandB:**
   > WandB might automatically resume from a previous training run (e.g., if it was interrupted or running in the same directory). This can cause the `global_step` to start from an unexpected value instead of 1.
   > 
   > To fix this, you can clear the WandB cache and restart the training:
   > ```bash
   > # Navigate to the training directory
   > cd ms-swift/examples/train/grpo/internal
   > # Remove all cached runs
   > rm -rf wandb/
   > 
   > # Rerun the training script
   > bash ms-swift/examples/train/grpo/internal/grpo_internvl3_5_4B_stage1.sh
   > ```

2. By log files, please go to `examples/training/grpo/output/` to see all checkpoint folder. Note: `Your running version is the lastest checkpoint, please change the name`.

### 5. Merging
Go to `bash examples/training/grpo/internal/merge_lora.sh`, detail is:
```
export CUDA_VISIBLE_DEVICES=0 

MODEL_ID_OR_PATH="OpenGVLab/InternVL3_5-4B-Instruct"  #Set your model_name 
MODEL_TYPE="internvl3" #Set model type
OUTPUT_DIR="/home/vlai-vqa-nle/phatdat/ms-swift/examples/train/grpo/output" #Checkpoint folders, not change
LORA_CHECKPOINT_PATH="/home/vlai-vqa-nle/phatdat/ms-swift/examples/train/grpo/output/vx-xxxxxxx-xxxx/checkpoint-500"  #Copy checkpoint path then pasting here
MERGED_MODEL_OUTPUT_DIR="$OUTPUT_DIR/merged_model_final/your_merged_model_name" #Ouput dir, replace your_merged_model_name by the name you want

swift merge-lora \
    --model_type "$MODEL_TYPE" \
    --model "$MODEL_ID_OR_PATH" \
    --adapters "$LORA_CHECKPOINT_PATH" \
    --output_dir "$MERGED_MODEL_OUTPUT_DIR" \
    --torch_dtype bfloat16 \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_compute_dtype bfloat16

```

### 6. NOTE ON MULTI-STAGE TRAINING

When training with LoRA, `ms-swift` only saves the LoRA adapters, not the entire base model with the adapters in the same checkpoint. Therefore, if you want to perform multi-stage training (e.g., train a second stage on the result of the first stage), you must first merge the LoRA adapters from the previous stage into the base model.

The correct workflow for independent multi-stage training is as follows:

**Step 1: Train Stage 1**

Use the training script as usual. The output checkpoint will only contain the LoRA adapters.

```bash
# Example command for Stage 1 training
swift rlhf \
    --rlhf_type grpo \
    --model OpenGVLab/InternVL3_5-4B-Instruct \
    --dataset "$STAGE1_DATASET" \
    --train_type lora \
    --output_dir "$STAGE1_OUTPUT"
    # ... other parameters
```
> The checkpoint will be saved at `$STAGE1_OUTPUT/vx-xxxxxx-xxxx/checkpoint-xxx/` (containing only LoRA adapters).

**Step 2: Merge Stage 1 LoRA into the Base Model**

Use the `swift export` command with the `--merge_lora` flag to combine the LoRA adapters with the base model. The merged model will be a complete model, ready for Stage 2.

```bash
# Command to merge LoRA into the base model
# If you need quantization, you can specify `--quant_bits 4`.
swift export \
    --model "OpenGVLab/InternVL3_5-4B-Instruct" \
    --ckpt_dir "$STAGE1_OUTPUT/vx-xxxxxx-xxxx/checkpoint-xxx" \
    --merge_lora true \
    --output_dir "$STAGE1_OUTPUT/checkpoint-xxx-merged"
```
> The merged model (base model + LoRA_1) will be saved at: `$STAGE1_OUTPUT/checkpoint-xxx-merged/`.

**Step 3: Train Stage 2 with the Merged Model**

Use the merged model from Step 2 as the base model for Stage 2. This way, the LoRA adapters for Stage 2 (`LoRA_2`) will be trained on a base model that has already integrated the knowledge from Stage 1.

```bash
# Example command for Stage 2 training
swift rlhf \
    --rlhf_type grpo \
    --model "$STAGE1_OUTPUT/checkpoint-xxx-merged" \
    --dataset "$STAGE2_DATASET" \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --output_dir "$STAGE2_OUTPUT" \
    # ... other parameters
```

### 7. Using Reward Weights (`--reward_weights`)

To balance the influence of different reward sources in GRPO, you can use the `--reward_weights` parameter. This allows you to prioritize certain objectives over others.

**How It Works**

You provide a list of numbers that correspond to each of your reward functions and models. The final reward is a weighted sum, meaning a higher weight gives a reward more importance.

**Key Rules:**

1.  **Order is Critical**: The weights are applied in the same order as your rewards are defined: first all functions from `--reward_funcs`, then all models from `--reward_model`.
2.  **Count Must Match**: The number of weights must exactly match the total number of reward functions and models.
3.  **Default is Equal**: If you don't specify `--reward_weights`, all rewards are given an equal weight of `1.0`.

**Practical Example for VQA**

To make the model prioritize `accuracy` twice as much as `format` compliance:

```bash
swift rlhf \
    --rlhf_type grpo \
    --reward_funcs accuracy format \
    --reward_weights 2.0 1.0
```

This simple parameter is very powerful for fine-tuning your model's behavior based on what you find most important.

#!/bin/bash
export HF_ENDPOINT="https://huggingface.co"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_ID_OR_PATH="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/minh-vintern3BR/stage1/merged/checkpoint-1000-merged-ver2"
MODEL_TYPE="internvl3"
TRAIN_DATASET_PATH="/home/vlai-vqa-nle/minhtq/vqa-nle/data/processed/ms-swift/stage2/ViVQA-X_train_msswift.jsonl"
PLUGIN_PATH="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/plugin/plugin.py"
OUTPUT_DIR="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/minh-vintern3BR/stage2"

# Tham số GRPO
MAX_LENGTH=4096
MAX_COMPLETION_LENGTH=1024
BATCH_SIZE_PER_DEVICE=1
NUM_GENERATIONS=8
GRAD_ACCUM_STEPS=8
TEMPERATURE=1
EPOCHS=1
MAX_STEPS=1000
LEARNING_RATE=1e-3

SAVE_STEPS=50
LOGGING_STEPS=1
EVAL_STEPS=1

# ========== GFPO Parameters ==========
ENABLE_GFPO=true

swift rlhf \
    --rlhf_type grpo \
    --model_type "$MODEL_TYPE" \
    --model "$MODEL_ID_OR_PATH" \
    --dataset "$TRAIN_DATASET_PATH" \
    --external_plugins "$PLUGIN_PATH" \
    --reward_funcs custom_accuracy_reward custom_explaination_reward \
    --reward_weights 1 1 \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --freeze_vit True \
    --output_dir "$OUTPUT_DIR" \
    --per_device_eval_batch_size $NUM_GENERATIONS \
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
    --beta 0.04 \
    --log_completions true \
    --torch_dtype bfloat16 \
    --save_only_model false \
    --save_total_limit 2 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --dataset_num_proc 1 \
    --report_to wandb \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --gradient_checkpointing true \
    --enable_gfpo $ENABLE_GFPO \

# --top_k 20 \ # This is not used in the script
echo "Hoàn thành huấn luyện GRPO Stage 2 - Format + Accuracy + Explanation"

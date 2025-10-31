#!/bin/bash
export HF_ENDPOINT="https://huggingface.co"
# --- Cấu hình ---
export CUDA_VISIBLE_DEVICES=2  # Chỉ định GPU muốn sử dụng
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_ID_OR_PATH="OpenGVLab/InternVL3_5-4B-Instruct" # Model ID trên Hugging Face
MODEL_TYPE="internvl3"
TRAIN_DATASET_PATH="/home/vlai-vqa-nle/phatdat/ms-swift/data_custom/ViVQA-X_train_grpo.jsonl"
VAL_DATASET_PATH="/home/vlai-vqa-nle/phatdat/ms-swift/examples/custom/data_custom/ViVQA-X_train_grpo.jsonl" # Tập validation
PLUGIN_PATH="/home/vlai-vqa-nle/phatdat/ms-swift/examples/train/grpo/plugin/plugin.py"
OUTPUT_DIR="/home/vlai-vqa-nle/phatdat/ms-swift/examples/train/grpo/output" # Đường dẫn tương đối

# --- Tham số GRPO & Huấn luyện ---
MAX_LENGTH=1024
MAX_COMPLETION_LENGTH=1024
NUM_GENERATIONS=4
TEMPERATURE=0.9
EPOCHS=1
BATCH_SIZE_PER_DEVICE=1
GRAD_ACCUM_STEPS=4
MAX_STEPS=500
LEARNING_RATE=1e-5

SAVE_STEPS=50
LOGGING_STEPS=5
EVAL_STEPS=5 # **Thêm:** Đặt tần suất đánh giá (thường bằng logging_steps hoặc save_steps)
# --eval_dataset "$VAL_DATASET_PATH" \
swift rlhf \
    --rlhf_type grpo \
    --model_type "$MODEL_TYPE" \
    --model "$MODEL_ID_OR_PATH" \
    --dataset "$TRAIN_DATASET_PATH" \
    --external_plugins "$PLUGIN_PATH" \
    --reward_funcs custom_format_reward custom_accuracy_reward \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit True \
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
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --gradient_checkpointing true \
    # --ref_model id_hoặc_path_đến_model_SFT_ban_đầu # Nên chỉ định ref_model

echo "Hoàn thành huấn luyện GRPO LoRA!"
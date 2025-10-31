export CUDA_VISIBLE_DEVICES=0 

MODEL_ID_OR_PATH="OpenGVLab/InternVL3_5-4B-Instruct" 
MODEL_TYPE="internvl3"
OUTPUT_DIR="/home/vlai-vqa-nle/phatdat/ms-swift/examples/train/grpo/output"
LORA_CHECKPOINT_PATH="/home/vlai-vqa-nle/phatdat/ms-swift/examples/train/grpo/output/v7-20251029-013438/checkpoint-500" 
MERGED_MODEL_OUTPUT_DIR="$OUTPUT_DIR/merged_model_final/InternVL3_5-4B-Instruct-Merged-01"

swift merge-lora \
    --model_type "$MODEL_TYPE" \
    --model "$MODEL_ID_OR_PATH" \
    --adapters "$LORA_CHECKPOINT_PATH" \
    --output_dir "$MERGED_MODEL_OUTPUT_DIR" \
    --torch_dtype bfloat16 \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_compute_dtype bfloat16
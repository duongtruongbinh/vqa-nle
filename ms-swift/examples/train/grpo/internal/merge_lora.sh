export CUDA_VISIBLE_DEVICES=2

OUTPUT="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/minh-vintern3BR/stage2/merged"

swift export \
    --use_hf true \
    --model_type "internvl3" \
    --model "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/minh-vintern3BR/stage1/merged/checkpoint-500-merged" \
    --ckpt_dir "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/minh-vintern3BR/stage2/v0-20251106-013820/checkpoint-500" \
    --merge_lora true \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --output_dir "$OUTPUT/stage1_checkpoint-500-merged"

echo "Hoàn thành merge LoRA với BNB 4-bit quantization"

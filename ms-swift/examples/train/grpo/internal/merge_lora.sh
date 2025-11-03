export CUDA_VISIBLE_DEVICES=1

STAGE2_OUTPUT="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/stage2/merge-model/v2"

swift export \
    --model "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/stage1/merged/checkpoint-500-merged" \
    --ckpt_dir "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/stage2/trained/v9-20251103-110740/checkpoint-500" \
    --merge_lora true \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --output_dir "$STAGE2_OUTPUT/stage1_checkpoint-500-merged"

echo "Hoàn thành merge LoRA với BNB 4-bit quantization"

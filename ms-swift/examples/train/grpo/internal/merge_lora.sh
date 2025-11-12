export CUDA_VISIBLE_DEVICES=2

OUTPUT="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/minh-vintern3BR/stage2/merged"

swift export \
    --use_hf true \
    --model_type "internvl3" \
    --model "5CD-AI/Vintern-3B-R-beta" \
    --ckpt_dir "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/minh-vintern3BR/stage1/v18-20251111-162849/checkpoint-1000" \
    --merge_lora true \
    --output_dir "$OUTPUT/stage1_checkpoint-1000-merged"

echo "Hoàn thành merge LoRA"

export CUDA_VISIBLE_DEVICES=1


swift export \
    --use_hf true \
    --model_type "internvl3" \
    --model "5CD-AI/Vintern-3B-R-beta" \
    --ckpt_dir "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/our/v4-20251218-014814/checkpoint-1000" \
    --merge_lora true \

echo "Hoàn thành merge LoRA"
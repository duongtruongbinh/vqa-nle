export CUDA_VISIBLE_DEVICES=1


swift export \
    --use_hf true \
    --model_type "internvl3" \
    --model "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/curr_nouns/merged/stage2_250_curr_noun_ver3" \
    --ckpt_dir "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/curr_nouns/stage3/v6-20251211-222938/checkpoint-250" \
    --merge_lora true \

echo "Hoàn thành merge LoRA"
export CUDA_VISIBLE_DEVICES=2


swift export \
    --use_hf true \
    --model_type "internvl3" \
    --model "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/curr_nouns/merged/stage3_250_curr_noun_ver3_2" \
    --ckpt_dir "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/sft/output/intern2B_failed_dataset_stage1/v2-20251213-170432/checkpoint-250" \
    --merge_lora true \

echo "Hoàn thành merge LoRA"
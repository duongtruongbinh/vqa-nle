export CUDA_VISIBLE_DEVICES=0


swift export \
    --use_hf true \
    --model_type "internvl3" \
    --model "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/curr_anstype/merged/stage_2_250_curr_anstype_ver2" \
    --ckpt_dir "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/curr_anstype/stage3/v3-20251123-022216/checkpoint-500" \
    --merge_lora true \

echo "Hoàn thành merge LoRA"

export CUDA_VISIBLE_DEVICES=2


swift export \
    --use_hf true \
    --model_type "internvl3" \
    --model "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/curr_anstype/merged/stage2_250_curr_anstype_ver_3" \
    --ckpt_dir "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/curr_anstype/stage3/v1-20251129-014914/checkpoint-500" \
    --merge_lora true \
    --output_dir "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/only_explain_answer/merged" \

echo "Hoàn thành merge LoRA"

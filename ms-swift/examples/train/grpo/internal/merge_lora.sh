export CUDA_VISIBLE_DEVICES=2

STAGE1_OUTPUT="/home/vlai-vqa-nle/phatdat/ms-swift/examples/train/grpo/output/merged_model_stage1"

# If you need quantization, you can specify `--quant_bits 4`.

swift export \
    --model "OpenGVLab/InternVL3_5-4B-Instruct" \
    --ckpt_dir "/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/stage1/trained/checkpoint-500" \
    --merge_lora true \
    --output_dir "$STAGE1_OUTPUT/checkpoint-500-internvl3_5_4B-merged"
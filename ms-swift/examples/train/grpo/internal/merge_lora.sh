export CUDA_VISIBLE_DEVICES=0 

STAGE1_OUTPUT="/home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift/examples/train/grpo/output/stage1"

# If you need quantization, you can specify `--quant_bits 4`.

swift export \
    --model "OpenGVLab/InternVL3_5-4B-Instruct" \
    --ckpt_dir "$STAGE1_OUTPUT/vx-xxxxxx-xxxx/checkpoint-xxx" \
    --merge_lora true \
    --output_dir "$STAGE1_OUTPUT/checkpoint-xxx-merged"
```

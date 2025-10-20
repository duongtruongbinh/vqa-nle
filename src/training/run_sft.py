from trl import SFTTrainer

from transformers import AutoProcessor, AutoModelForImageTextToText
from trl import SFTConfig
from peft import LoraConfig
import torch
from transformers import BitsAndBytesConfig

# Hugging Face model id
model_id = "Qwen/Qwen2-VL-7B-Instruct"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    # attn_implementation="flash_attention_2", # not supported for training
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
processor = AutoProcessor.from_pretrained(model_id)


# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)


args = SFTConfig(
    # directory to save and repository id
    output_dir="qwen2-7b-instruct-amazon-description",
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    # number of steps before performing a backward/update pass
    gradient_accumulation_steps=8,
    # use gradient checkpointing to save memory
    gradient_checkpointing=True,
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=5,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    # use reentrant checkpointing
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="",  # need a dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True}  # important for collator
)
args.remove_unused_columns = False

# Create a data collator to encode text and image pairs


def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(
        example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[
        0] for example in examples]

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs,
                      return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  #
    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652, 151653, 151655]
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(
            processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch


trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collate_fn,
    dataset_text_field="",  # needs dummy value
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)


# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save model
trainer.save_model(args.output_dir)

# free the memory again
del model
del trainer
torch.cuda.empty_cache()

# -*- coding: utf-8 -*-
import torch

from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

# We will create the get_dataset function in the next step inside src/data/dataset_loader.py
from src.data.dataset_loader import get_dataset
from src.rewards.outcome_rewards import format_reward, accuracy_reward


def main():
    model_id = "5CD-AI/Vintern-3B-R-beta"

    # --- Load Model and Processor ---
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # --- LoRA Configuration ---
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Load Dataset ---
    # Load the custom ViVQA-X dataset for the 'train' split
    train_dataset = get_dataset(processor, split="train")

    # --- GRPO Training Configuration ---
    training_args = GRPOConfig(
        output_dir=f"/results/checkpoints/{model_id}-ViVQA-X",
        learning_rate=1e-5,
        remove_unused_columns=False,
        num_train_epochs=1,
        bf16=True,
        per_device_train_batch_size=2,
        max_completion_length=1024,
        num_generations=2,
        max_prompt_length=2048,
        report_to=["tensorboard"],
        logging_steps=10,
        push_to_hub=True,
        save_strategy="steps",
        save_steps=10,
    )

    # --- Initialize and Run Trainer ---
    # IMPORTANT: Ensure your reward functions in 'src/rewards/outcome_rewards.py'
    # can parse the 'solution' field (<answer>...</answer><explain>...</explain>)
    # to calculate rewards correctly.
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting GRPO training...")
    trainer.train()
    print("Training finished.")

    # --- Save Model ---
    print("Saving model...")
    trainer.save_model(training_args.output_dir)
    trainer.push_to_hub(dataset_name="VLAI-AIVN/ViVQA-X")
    print(f"Model saved to {training_args.output_dir} and pushed to hub.")


if __name__ == "__main__":
    main()

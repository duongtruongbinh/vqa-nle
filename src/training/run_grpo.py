# -*- coding: utf-8 -*-
import os
import torch

from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel
from trl import GRPOConfig, GRPOTrainer
import json
# We will create the get_dataset function in the next step inside src/data/dataset_loader.py
from src.data.dataset_loader import get_dataset
from src.data.dataset_collator import VinternDataCollator
from src.trainers.custom_grpo_trainer import VinternGRPOTrainer
from src.rewards.outcome_rewards import format_reward, accuracy_reward
from dotenv import load_dotenv

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    model_id = "5CD-AI/Vintern-3B-R-beta"
    hf_token = os.getenv("HF_TOKEN")

    # --- LoRA Configuration ---
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    # --- Load Dataset ---
    train_dataset = get_dataset(split="train")

    # --- GRPO Training Configuration ---
    training_args = GRPOConfig(
        output_dir=f"results/checkpoints/{model_id}-ViVQA-X",
        learning_rate=1e-5,
        remove_unused_columns=False,
        num_train_epochs=1,
        bf16=True,
        per_device_train_batch_size=2,
        max_completion_length=1024,
        num_generations=2,
        max_prompt_length=2048,
        report_to=["none"],
        logging_steps=10,
        push_to_hub=True,
        save_strategy="steps",
        save_steps=10,
        gradient_checkpointing=False,
        hub_token=hf_token,
    )

    trainer = VinternGRPOTrainer(
        model=model_id,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )

    print("Starting GRPO training...")
    trainer.train()
    print("Training finished.")

    print("Saving model...")
    trainer.save_model(training_args.output_dir)
    trainer.push_to_hub(dataset_name="VLAI-AIVN/ViVQA-X")
    print(f"Model saved to {training_args.output_dir} and pushed to hub.")


if __name__ == "__main__":
    main()

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

import copy
from src.inference.models.utils import load_image

if is_peft_available():
    from peft import PeftConfig, get_peft_model

# if is_wandb_available():
#     import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class VinternProcessor:
    def __init__(self, tokenizer, image_size=448, max_num=12):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_num = max_num
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, text=None, images=None, return_tensors="pt",
                 padding=True, padding_side="left", add_special_tokens=False):
        # Text tokenization
        if text:
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                add_special_tokens=add_special_tokens
            )
            if padding_side == "left":
                text_inputs = self._left_pad(text_inputs)
        else:
            text_inputs = {}

        # Image processing
        if images:
            pixel_values_list = []
            for image in images:
                if isinstance(image, str):
                    pv = load_image(image, input_size=self.image_size,
                                    max_num=self.max_num)
                else:
                    pv = load_image(image, input_size=self.image_size,
                                    max_num=self.max_num)
                pixel_values_list.append(pv)

            # Stack all pixel values
            pixel_values = torch.cat(pixel_values_list, dim=0)
            text_inputs["pixel_values"] = pixel_values

        return text_inputs

    def _left_pad(self, inputs):
        # Convert right padding to left padding
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Flip sequences for left padding
        input_ids = torch.flip(input_ids, dims=[1])
        attention_mask = torch.flip(attention_mask, dims=[1])

        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask
        return inputs

    def batch_decode(self, sequences, skip_special_tokens=True):
        return self.tokenizer.batch_decode(sequences,
                                           skip_special_tokens=skip_special_tokens)


class VinternGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        image_size: int = 448,
        max_num_images: int = 12,
    ):
        # Model loading
        if args is None:
            model_name = model if isinstance(
                model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        model_init_kwargs = args.model_init_kwargs or {}

        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype", torch.bfloat16)

            # Load Vintern model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=False,
                **model_init_kwargs
            )
        else:
            model_id = model.config._name_or_path

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=False,
            )
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class for Vintern
        if processing_class is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=False
            )
            processing_class = VinternProcessor(
                tokenizer,
                image_size=image_size,
                max_num=max_num_images
            )
            pad_token_id = tokenizer.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values):
        # Vintern forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True
        )

        logits = outputs.logits  # (B, L, V)
        logits = logits[:, :-1, :]  # Exclude last logit
        input_ids = input_ids[:, 1:]  # Exclude first token

        # Compute per-token log probabilities
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(
                log_probs, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)

        return torch.stack(per_token_logps)

    def _log_metrics(self, completion_mask, rewards, std_grouped_rewards, per_token_kl):
        """Log training metrics"""
        completion_length = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)
        ).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item()
        )

        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(
                std_grouped_rewards).mean().item()
        )

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) /
                   completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )

    def _compute_rewards(self, prompts, completions, images, inputs):
        """Compute rewards từ reward functions"""
        device = self.accelerator.device
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device)

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                # Reward model case
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c}
                                for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)[
                        "text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]

                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True,
                    padding_side="right", add_special_tokens=False
                )
                reward_inputs = {k: v.to(device)
                                 for k, v in reward_inputs.items()}

                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(
                        **reward_inputs).logits[:, 0]
            else:
                # Custom reward function case
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in [
                    "prompt", "image"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend(
                            [example[key]] * self.num_generations)

                # Không cần pass images vào đây nếu reward function không dùng
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    **reward_kwargs
                )
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        return rewards_per_func.sum(dim=1)

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError(
                "VinternGRPOTrainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]
        images = [x["image"] for x in inputs]

        # Apply chat template nếu cần
        prompts_text = [
            maybe_apply_chat_template(
                example, self.processing_class.tokenizer)["prompt"]
            for example in inputs
        ]

        # Process prompts với images
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]

        # Truncate if needed
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            all_completions = []

            # Prepare inputs for generation
            gen_inputs = {
                "input_ids": prompt_ids,
                "attention_mask": prompt_mask,
                "pixel_values": pixel_values,
            }

            for i in range(self.num_generations):
                temp_generation_config = copy.deepcopy(self.generation_config)
                temp_generation_config.num_return_sequences = 1

                completion = unwrapped_model.generate(
                    **gen_inputs,
                    generation_config=temp_generation_config
                )
                all_completions.append(completion)

            # Pad and stack completions
            max_length = max(c.size(1) for c in all_completions)
            padded_completions = []

            for completion in all_completions:
                if completion.size(1) < max_length:
                    padding = torch.full(
                        (completion.size(0), max_length - completion.size(1)),
                        self.processing_class.tokenizer.pad_token_id,
                        dtype=completion.dtype,
                        device=completion.device
                    )
                    padded_completion = torch.cat([completion, padding], dim=1)
                else:
                    padded_completion = completion
                padded_completions.append(padded_completion)

            prompt_completion_ids = torch.cat(padded_completions, dim=0)

        # Extract completions
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Replicate pixel_values for num_generations
        pixel_values = pixel_values.repeat_interleave(
            self.num_generations, dim=0)

        # Mask after EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(
            1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[
            is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(
            1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate masks
        prompt_mask = prompt_mask.repeat_interleave(
            self.num_generations, dim=0)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Compute log probabilities
        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, attention_mask, pixel_values
        )
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, pixel_values
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, pixel_values
                    )
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Compute KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - \
            (ref_per_token_logps - per_token_logps) - 1

        # Decode completions
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # Compute rewards
        prompts = [prompt for prompt in prompts for _ in range(
            self.num_generations)]
        images = [img for img in images for _ in range(self.num_generations)]

        rewards = self._compute_rewards(prompts, completions, images)

        # Group-wise normalization
        mean_grouped_rewards = rewards.view(-1,
                                            self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations)

        advantages = (rewards - mean_grouped_rewards) / \
            (std_grouped_rewards + 1e-4)

        # Compute loss
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * \
            advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        loss = ((per_token_loss * completion_mask).sum(dim=1) /
                completion_mask.sum(dim=1)).mean()

        # Log metrics
        self._log_metrics(completion_mask, rewards,
                          std_grouped_rewards, per_token_kl)

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key,
                   val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        # citation = textwrap.dedent(
        #     """\
        #     @article{zhihong2024deepseekmath,
        #         title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
        #         author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
        #         year         = 2024,
        #         eprint       = {arXiv:2402.03300},
        #     """
        # )

        # model_card = generate_model_card(
        #     base_model=base_model,
        #     model_name=model_name,
        #     hub_model_id=self.hub_model_id,
        #     dataset_name=dataset_name,
        #     tags=tags,
        #     wandb_url=wandb.run.get_url() if is_wandb_available(
        #     ) and wandb.run is not None else None,
        #     comet_url=get_comet_experiment_url(),
        #     trainer_name="GRPO",
        #     trainer_citation=citation,
        #     paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        #     paper_id="2402.03300",
        # )

        # model_card.save(os.path.join(self.args.output_dir, "README.md"))

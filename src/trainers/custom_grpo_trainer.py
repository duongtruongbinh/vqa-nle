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
import json
import os
from src.data.image_preprocessing import load_image
import math
if is_peft_available():
    from peft import PeftConfig, get_peft_model

# if is_wandb_available():
#     import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class VinternProcessor:
    """
    Đóng vai 'processing_class' cho Vintern (tokenizer + tiền xử lý ảnh + build prompt).
    Trả về: {input_ids, attention_mask, pixel_values, image_flags}
    """

    def __init__(self, model, tokenizer, image_size=448, max_num_tiles=12, use_thumbnail=True):
        self.model = model
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.use_thumbnail = use_thumbnail

        # thiết lập pad/eos cho Trainer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # id của <IMG_CONTEXT> để model biết chỗ cắm features
        img_ctx_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = img_ctx_id
        if self.model.img_context_token_id is None:
            raise ValueError(
                "Vintern: không tìm thấy <IMG_CONTEXT> trong tokenizer.")

        # số token ảnh mỗi tile (thường 256)
        if not hasattr(self.model, "num_image_token"):
            raise ValueError(
                "Vintern: thiếu thuộc tính model.num_image_token.")
        self.num_image_token = int(self.model.num_image_token)

    def _encode_one(self, text: str, image) -> dict:
        pixel_values = load_image(image, self.image_size, self.max_num_tiles)
        num_patches = pixel_values.shape[0]
        num_img_tokens = self.num_image_token * num_patches

        # 2) build image span + prompt
        image_span = "<img>" + "<IMG_CONTEXT>" * num_img_tokens + "</img>"
        prompt = f"<|im_start|>user\n{image_span}\n{text}<|im_end|>\n<|im_start|>assistant\n"

        enc = self.tokenizer(prompt, return_tensors="pt",
                             padding=False, add_special_tokens=False)
        image_flags = torch.ones((num_patches, 1), dtype=torch.long)

        return {
            "input_ids": enc["input_ids"][0],           # (L,)
            "attention_mask": enc["attention_mask"][0],  # (L,)
            "pixel_values": pixel_values,               # (T, 3, H, W)
            "image_flags": image_flags,                 # (T, 1)
        }

    def __call__(self, text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        """
        Batching đơn giản: xử lý từng mẫu rồi pad text theo 'left'.
        """
        assert padding_side == "left", "Vintern cần left padding cho causal LM."
        batch = [self._encode_one(t, im) for t, im in zip(text, images)]

        # pad input_ids/attention_mask (left)
        maxL = max(x["input_ids"].size(0) for x in batch)
        input_ids, attn_masks = [], []
        for x in batch:
            pad_len = maxL - x["input_ids"].size(0)
            if pad_len > 0:
                pad_ids = torch.full(
                    (pad_len,), self.pad_token_id, dtype=torch.long)
                pad_ms = torch.zeros((pad_len,), dtype=torch.long)
                input_ids.append(torch.cat([pad_ids, x["input_ids"]], dim=0))
                attn_masks.append(
                    torch.cat([pad_ms,  x["attention_mask"]], dim=0))
            else:
                input_ids.append(x["input_ids"])
                attn_masks.append(x["attention_mask"])

        # concat ảnh theo batch: Vintern forward kỳ vọng (sum_T, 3, H, W) + image_flags (sum_T, 1)
        pixel_values = torch.cat([x["pixel_values"] for x in batch], dim=0)
        image_flags = torch.cat([x["image_flags"] for x in batch], dim=0)

        out = {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attn_masks, dim=0),
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }
        return out

    def save_pretrained(self, save_directory: str):
        """
        Save tokenizer plus the few processor-specific fields so Trainer
        can checkpoint/push without errors.
        """
        os.makedirs(save_directory, exist_ok=True)

        # 1) save underlying tokenizer (standard HF way)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(save_directory)

        # 2) save minimal processor config
        cfg = {
            "_processor_class": "VinternProcessor",
            "image_size": self.image_size,
            "max_num_tiles": self.max_num_tiles,
            "use_thumbnail": self.use_thumbnail,
            # optional: record special token id for sanity
            "img_context_token_id": int(getattr(self.model, "img_context_token_id", -1)),
            "num_image_token": int(getattr(self, "num_image_token", 256)),
        }
        with open(os.path.join(save_directory, "vintern_processor_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str, model, **kwargs):
        """
        Optional loader so you (or someone else) can restore the processor
        from a checkpoint/hub repo later.
        """
        tok = AutoTokenizer.from_pretrained(
            save_directory, trust_remote_code=True, use_fast=False
        )
        cfg_path = os.path.join(
            save_directory, "vintern_processor_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        else:
            cfg = {}

        return cls(
            model=model,
            tokenizer=tok,
            image_size=cfg.get("image_size", kwargs.get("image_size", 448)),
            max_num_tiles=cfg.get(
                "max_num_tiles", kwargs.get("max_num_tiles", 12)),
            use_thumbnail=cfg.get(
                "use_thumbnail", kwargs.get("use_thumbnail", True)),
        )


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
                model=model,
                tokenizer=tokenizer,
                image_size=image_size,
                max_num_tiles=max_num_images,
                use_thumbnail=False,
            )

        _img_tid = self._set_img_ctx_token(model, processing_class.tokenizer)
        if self.ref_model is not None:
            self._set_img_ctx_token(self.ref_model, processing_class.tokenizer)

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
            do_sample=getattr(args, "do_sample", True),
            temperature=getattr(args, "temperature", 0.7),
            top_p=getattr(args, "top_p", 0.9),
            top_k=getattr(args, "top_k", None),
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

    @staticmethod
    def _set_img_ctx_token(model_like, tokenizer):
        tid = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        # đặt lên wrapper
        setattr(model_like, "img_context_token_id", tid)
        # đặt lên base peft nếu có
        base = getattr(model_like, "base_model", None)
        if base is not None:
            setattr(base, "img_context_token_id", tid)
            # một số wrapper có .model
            inner = getattr(base, "model", None)
            if inner is not None:
                setattr(inner, "img_context_token_id", tid)
        return tid

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_flags, **_):
        # Bóc các wrapper (Accelerate/PEFT) để lấy core model không chèn inputs_embeds
        core = self.accelerator.unwrap_model(model)
        # PeftModel -> .base_model (PeftModelForCausalLM) -> .model (InternVLChatForCausalLM/InternVLChatModel)
        base = getattr(core, "base_model", None)
        if base is not None:
            core_inner = getattr(base, "model", None)
            if core_inner is not None:
                core = core_inner

        # Gọi trực tiếp forward của InternVLChat*, chỉ với input_ids (không inputs_embeds)
        out = core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_flags=image_flags,
            use_cache=False,          # an toàn cho tính log-prob
            return_dict=True,
        )
        logits = out.logits  # (B, L, V)
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        log_probs = logits.log_softmax(dim=-1)
        token_log_prob = torch.gather(
            log_probs, dim=2, index=input_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_prob

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
            raise ValueError("GRPOTrainer does not support return_outputs")

        device = self.accelerator.device

        # Lấy text (question/prompt) & image
        if "question" in inputs[0]:
            texts = [ex["question"] for ex in inputs]
        else:
            texts = [ex["prompt"] for ex in inputs]
        images = [ex["image"] for ex in inputs]
        image_ids = [ex.get("image_id", None) for ex in inputs]
        problems = [ex.get("problem", None) for ex in inputs]

        # Encode bằng processing_class (build prompt + tiles)
        enc = self.processing_class(text=texts, images=images, return_tensors="pt",
                                    padding=True, padding_side="left", add_special_tokens=False)
        device = self.accelerator.device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        pixel_values = enc["pixel_values"].to(
            device, dtype=self.model.dtype)  # bf16/float16 tùy model
        image_flags = enc["image_flags"].to(device)

        if self.max_prompt_length is not None:
            img_tid = getattr(self.model, "img_context_token_id", None)
            if img_tid is not None:
                B, L = input_ids.size()
                new_ids, new_mask = [], []
                for b in range(B):
                    ids = input_ids[b]
                    ms = attention_mask[b]
                    positions = (ids == img_tid).nonzero(
                        as_tuple=False).flatten()
                    if L <= self.max_prompt_length or positions.numel() == 0:
                        new_ids.append(ids[-self.max_prompt_length:])
                        new_mask.append(ms[-self.max_prompt_length:])
                        continue
                    img_start = int(positions[0])
                    img_end = int(positions[-1]) + 1
                    img_len = img_end - img_start
                    if img_len > self.max_prompt_length:
                        # too many image tokens: rely on fewer tiles
                        keep_start = img_end - self.max_prompt_length
                        keep_end = img_end
                    else:
                        # prefer to keep the tail, but never cut the image block
                        keep_start = max(0, L - self.max_prompt_length)
                        if keep_start > img_start:
                            keep_start = img_start
                        keep_end = keep_start + self.max_prompt_length
                    new_ids.append(ids[keep_start:keep_end])
                    new_mask.append(ms[keep_start:keep_end])
                input_ids = torch.stack(new_ids, dim=0)
                attention_mask = torch.stack(new_mask, dim=0)
            else:
                input_ids = input_ids[:, -self.max_prompt_length:]
                attention_mask = attention_mask[:, -self.max_prompt_length:]

        # === Generate completions ===
        unwrapped = self.accelerator.unwrap_model(model)

        num_generations = self.generation_config.num_return_sequences
        temp_gc = copy.deepcopy(self.generation_config)
        temp_gc.num_return_sequences = 1

        # ép sinh tối thiểu 1 token + cấu hình dừng và cache
        temp_gc.use_cache = True
        setattr(temp_gc, "min_new_tokens", 1)
        temp_gc.pad_token_id = self.processing_class.pad_token_id
        eos_id = self.processing_class.tokenizer.convert_tokens_to_ids(
            "<|im_end|>")
        if eos_id is None:
            eos_id = self.processing_class.tokenizer.eos_token_id
        temp_gc.eos_token_id = eos_id

        all_completions = []
        for _ in range(num_generations):
            gen_ids = unwrapped.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=temp_gc,
                pixel_values=pixel_values,
            )
            all_completions.append(gen_ids)

        # pad & stack completions
        max_len = max(x.size(1) for x in all_completions)
        padded = [torch.cat([g, torch.full((g.size(0), max_len-g.size(1)),
                                           self.processing_class.pad_token_id,
                                           dtype=g.dtype, device=g.device)], dim=1) if g.size(1) < max_len else g
                  for g in all_completions]
        prompt_completion_ids = torch.cat(padded, dim=0)

        prompt_len = input_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_len]
        completion_ids = prompt_completion_ids[:, prompt_len:]
        prompt_mask = attention_mask.repeat_interleave(num_generations, dim=0)

        # Mask theo EOS (loại EOS khỏi loss)
        if completion_ids.size(1) == 0:
            # Không có token sinh mới → mask rỗng, tránh argmax lỗi
            completion_mask = torch.zeros(
                (completion_ids.size(0), 0), dtype=torch.int, device=device
            )
        else:
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(
                1), dtype=torch.long, device=device)
            has_eos = is_eos.any(dim=1)
            eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
            seq_idx = torch.arange(is_eos.size(
                1), device=device).expand_as(is_eos)
            completion_mask = (seq_idx < eos_idx.unsqueeze(1)).int()

        attn_mask_full = torch.cat([prompt_mask, completion_mask], dim=1)

        pv = pixel_values.repeat_interleave(num_generations, dim=0)
        img_flags = image_flags.repeat_interleave(num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, attn_mask_full,
            pixel_values=pv, image_flags=img_flags
        )[:, prompt_len-1:]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attn_mask_full,
                    pixel_values=pv, image_flags=img_flags
                )[:, prompt_len-1:]
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attn_mask_full,
                        pixel_values=pv, image_flags=img_flags
                    )[:, prompt_len-1:]

        delta = (ref_token_logps - per_token_logps).clamp(-20, 20)
        per_token_kl = torch.exp(delta) - delta - 1

        # Decode completions cho reward
        completions = self.processing_class.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True)
        prompts_rep = [t for t in texts for _ in range(num_generations)]

        # Tính reward (giữ nguyên code reward_funcs hiện có của bạn)
        rewards_per_func = torch.zeros(
            len(prompts_rep), len(self.reward_funcs), device=device)
        for i, (rf, rproc) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(rf, PreTrainedModel):
                # nối text+comp (non-chat)
                texts_for_rm = [p + c for p,
                                c in zip(prompts_rep, completions)]
                rm_inputs = rproc(texts_for_rm, return_tensors="pt", padding=True,
                                  padding_side="right", add_special_tokens=False)
                rm_inputs = super()._prepare_inputs(rm_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = rf(**rm_inputs).logits[:, 0]
            else:
                # custom function
                reward_kwargs = {k: [] for k in inputs[0].keys() if k not in [
                    "prompt", "completion", "image"]}
                for k in reward_kwargs:
                    for ex in inputs:
                        reward_kwargs[k].extend([ex[k]] * num_generations)
                out_r = rf(prompts=prompts_rep, completions=completions, image_ids=image_ids * num_generations,
                           problems=problems * num_generations, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(
                    out_r, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)
        mean_group = rewards.view(-1, num_generations).mean(dim=1)
        std_group = rewards.view(-1, num_generations).std(dim=1)

        mean_group = mean_group.repeat_interleave(num_generations, dim=0)
        std_group = std_group.repeat_interleave(num_generations, dim=0)

        # safeguard std=0 + clip advantage (tùy chọn)
        std_eps = 1e-6
        zero_std = (std_group < std_eps)
        eff_std = torch.where(zero_std, torch.ones_like(std_group), std_group)
        advantages = (rewards - mean_group) / (eff_std + 1e-8)
        advantages = torch.where(
            zero_std, torch.zeros_like(advantages), advantages)
        advantages = advantages.clamp(-getattr(self.args, "adv_clip", 5.0),
                                      getattr(self.args, "adv_clip", 5.0))

        # beta schedule (tuỳ chọn)
        beta = self.beta
        if hasattr(self, "state") and getattr(self.state, "max_steps", None):
            prog = self.state.global_step / max(1, self.state.max_steps)
            bmin = getattr(self.args, "beta_min", beta)
            bmax = getattr(self.args, "beta_max", beta)
            beta = float(bmin + 0.5*(bmax - bmin) *
                         (1 - math.cos(prog * 3.14159)))

        ratio = torch.exp(per_token_logps - per_token_logps.detach())
        per_token_loss = - \
            (ratio * advantages.unsqueeze(1) - beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) /
                (completion_mask.sum(dim=1).clamp(min=1))).mean()

        # Log metrics
        self._log_metrics(completion_mask, rewards, std_group, per_token_kl)

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

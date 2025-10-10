# File: src/trainers/custom_grpo_trainer.py
import torch
from PIL import Image
from trl import GRPOTrainer
from typing import List, Optional
from torch.utils.data import DataLoader


class VinternGRPOTrainer(GRPOTrainer):
    """Custom GRPO Trainer for Vintern VLM"""

    def __init__(self, *args, data_collator=None, **kwargs):
        self._custom_data_collator = data_collator
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        """Override để inject custom data collator"""
        if self._custom_data_collator is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self._custom_data_collator,
                num_workers=self.args.dataloader_num_workers if hasattr(
                    self.args, 'dataloader_num_workers') else 0,
                pin_memory=self.args.dataloader_pin_memory if hasattr(
                    self.args, 'dataloader_pin_memory') else True,
                shuffle=True,
            )
        return super().get_train_dataloader()

    def _generate_and_score_completions(self, batch):
        """Override để xử lý images từ dataset đúng cách"""

        # batch là list of dicts
        if not isinstance(batch, list):
            return super()._generate_and_score_completions(batch)

        print(f"DEBUG - Batch type: {type(batch)}")
        print(f"DEBUG - Batch length: {len(batch)}")
        if batch:
            print(f"DEBUG - First item keys: {batch[0].keys()}")
            if "images" in batch[0]:
                print(f"DEBUG - First images type: {type(batch[0]['images'])}")

        # Process images cho từng item
        for item in batch:
            if "images" in item and item["images"] is not None:
                images = item["images"]

                if not isinstance(images, list):
                    images = [images]

                processed_tensors = []
                for img in images:
                    if isinstance(img, Image.Image):
                        from src.data.image_preprocessing import load_image
                        pixel_values = load_image(img, max_num=6)
                        processed_tensors.append(pixel_values)
                    elif isinstance(img, torch.Tensor):
                        processed_tensors.append(img)

                item["images"] = processed_tensors

        return super()._generate_and_score_completions(batch)

    def _generate_single_turn(self, prompts: List[str], images: Optional[List] = None):
        """Override để xử lý images đúng cách với Vintern model"""

        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        processed_prompts = [
            p.replace('<image>', IMG_CONTEXT_TOKEN) for p in prompts]

        prompt_inputs = self.processing_class(
            processed_prompts,
            padding=True,
            truncation=True,
            max_length=self.args.max_prompt_length,
            return_tensors="pt",
        ).to(self.accelerator.device)

        forward_kwargs = {}
        if images is not None and len(images) > 0:
            # ✅ THAY ĐỔI QUAN TRỌNG: Xử lý từng sample riêng biệt
            # Thay vì flatten tất cả, ta sẽ batch chúng đúng cách

            batch_pixel_values = []
            for img_list in images:
                if isinstance(img_list, list) and len(img_list) > 0:
                    # Lấy first item nếu là list
                    img = img_list[0] if isinstance(
                        img_list[0], (Image.Image, torch.Tensor)) else img_list

                    # Process image to tiles
                    if isinstance(img, Image.Image):
                        from src.data.image_preprocessing import load_image
                        # Shape: [num_tiles, 3, 448, 448]
                        pixel_values = load_image(img, max_num=6)
                        batch_pixel_values.append(pixel_values)
                    elif isinstance(img, torch.Tensor):
                        batch_pixel_values.append(img)

            if batch_pixel_values:
                # Stack thành batch: [batch_size, num_tiles, 3, 448, 448]
                # Nhưng model.generate() của Vintern expect flatten format
                # Nên ta cần concatenate và track num_patches

                num_patches_list = [pv.shape[0] for pv in batch_pixel_values]
                pixel_values = torch.cat(batch_pixel_values, dim=0).to(
                    self.accelerator.device, dtype=self.model.dtype
                )

                forward_kwargs["pixel_values"] = pixel_values
                forward_kwargs["num_patches"] = num_patches_list

                print(f"DEBUG - pixel_values shape: {pixel_values.shape}")
                print(f"DEBUG - num_patches: {num_patches_list}")

        # ✅ QUAN TRỌNG: Gọi model.generate với PEFT wrapper đúng cách
        with torch.no_grad():
            try:
                # Vintern's generate expects different format
                # We need to call the base model's generate, not through PEFT
                if hasattr(self.model, 'generate'):
                    generation_output = self.model.generate(
                        **prompt_inputs,
                        **forward_kwargs,
                        max_new_tokens=self.args.max_completion_length,
                        do_sample=True,
                        temperature=0.7,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                else:
                    raise AttributeError("Model does not have generate method")

            except Exception as e:
                print(f"ERROR during generation: {e}")
                print(f"Prompt input IDs: {prompt_inputs['input_ids'].shape}")
                if "pixel_values" in forward_kwargs:
                    print(
                        f"Pixel values: {forward_kwargs['pixel_values'].shape}")
                    print(
                        f"Num patches: {forward_kwargs.get('num_patches', 'None')}")
                raise

        prompt_ids = prompt_inputs["input_ids"]
        completion_ids = generation_output.sequences[:, prompt_ids.shape[1]:]
        logprobs = None

        return prompt_ids, completion_ids, logprobs, forward_kwargs

import torch
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms import v2 as Tv2
from PIL import Image

from .base_model import VQAModel
from .utils import get_system_prompt, parse_output, get_grpo_system_prompt, parse_output_grpo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ABSOLUTE_PROJECT_PATH = '/home/vlai-vqa-nle/phatdat/finetune_vintern/'
# MODEL_PATH = "/mnt/dataset1/pretrained_fm/Vintern-1B-v3_5"
#Fine-tuned model:
MODEL_PATH  = ABSOLUTE_PROJECT_PATH + "work_dirs/internvl_chat_v2_0/Vintern_1B_v3_5_finetune_lora_vivqa-x_merge"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size: int):
    return Tv2.Compose([
        Tv2.Lambda(lambda img: img.convert('RGB') if hasattr(img, "mode") and img.mode != 'RGB' else img),
        Tv2.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        Tv2.ToImage(),
        Tv2.ToDtype(torch.float32, scale=True),
        Tv2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images



def get_image(image_path):
  """Opens and returns an image from a given path using Pillow.

  Args:
    image_path: The path to the image file.

  Returns:
    A PIL Image object, or None if the file is not found.
  """
  try:
    img = Image.open(image_path)
    return img
  except FileNotFoundError:
    print(f"Error: Image not found at {image_path}")
    return None
  except Exception as e:
    print(f"An error occurred: {e}")
    return None
  

class Vintern1BModel(VQAModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = "/mnt/dataset1/pretrained_fm/Vintern-1B-v3_5"
        self.image_size = 448
        self.transform = build_transform(self.image_size)
        self._set_clean_model_name()
        self.load_model()

    def load_model(self):
        self.model = AutoModel.from_pretrained(MODEL_PATH,
                            torch_dtype=torch.bfloat16,
                            use_flash_attn=False,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)   
    
    def _load_image(self, image_file, input_size=448, max_num=12):
        #check image_file is a path or image
        if isinstance(image_file, str):
            image = Image.open(image_file).convert('RGB')
        else:
            image = image_file
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def infer(self, question: str, image_path: str) -> tuple[str, str]:
        pixel_values = self._load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        
        generation_config = dict(
            max_new_tokens= 100,
            pad_token_id = self.tokenizer.eos_token_id
        )

        system_instruction = get_system_prompt()
        user_content = f"Câu hỏi: {question}"
        prompt = f"{system_instruction}\n" + user_content

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config=generation_config
            )
        return parse_output(response) 

    def infer_grpo(self, question: str, image_path: str) -> tuple[str, str]:
        pixel_values = self._load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        
        generation_config = dict(
            max_new_tokens= 256,
            pad_token_id = self.tokenizer.eos_token_id
        )

        system_instruction = get_grpo_system_prompt()
        user_content = f"""Now, answer this question based on the image: 
        Question: {question}. 
        Let's response in three tag pairs in your response: <think></think>, <answer></answer>, <explain></explain>."""
        prompt = f"{system_instruction}\n" + user_content

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config=generation_config
            )
        return parse_output_grpo(response) 
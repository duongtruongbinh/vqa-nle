#!/usr/bin/env python3
"""
Generate REASONING cho ViVQA-X dataset sử dụng CoCoA (Gemini + UQLM Semantic Negentropy).

Usage:
    python -m src.data_loader.generate_semneg_reasoning --split train
    python -m src.data_loader.generate_semneg_reasoning --split train --failed_only
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import base64
import asyncio
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from tqdm import tqdm
import torch

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from uqlm import BlackBoxUQ


# ==================== CONFIG ====================

# Model settings (đọc từ environment hoặc dùng default)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-pro"


# CoCoA settings
TEMPERATURE = 1.0
NUM_RESPONSES = 3
SLEEP_DELAY_SECONDS = 1.0  

# Paths
VIVQA_PATH = "/mnt/VLAI_data/ViVQA-X/ViVQA-X_train.json"
COCO_DIR = "/mnt/VLAI_data/COCO_Images/train2014/"
OUTPUT_DIR = "data/processed/cocoa_reasoning/"
FAILED_IDS_FILES = [
    "data/analysis/stage_1/failed_question_ids_stage1.json",
    "data/analysis/stage_2/failed_question_ids_stage2.json",
    "data/analysis/stage_3/failed_question_ids_stage3.json"
]

# Prompt template
PROMPT = """Tôi có một hình ảnh, một câu hỏi và câu trả lời đúng chuẩn. Tôi cần bạn tuân thủ nghiêm ngặt định dạng với một phần cụ thể: REASONING. Điều quan trọng là bạn phải tuân thủ chính xác cấu trúc này và REASONING của bạn phải hỗ trợ logic cho câu trả lời đúng được cung cấp một cách chính xác.
Để giải thích rõ hơn: Trong phần REASONING, hãy phác thảo một quá trình suy nghĩ từng bước dựa trên hình ảnh dẫn đến câu trả lời đúng. Khác với định dạng trước, không được đưa ra Tóm tắt, Chú thích hoặc Kết luận. Việc tập trung hoàn toàn vào quá trình suy luận là bắt buộc.
Đây là cách định dạng cần có:
<REASONING>
[Cung cấp một chuỗi suy nghĩ, giải thích logic về vấn đề bằng tiếng Việt. Phần này cần phác thảo REASONING từng bước phù hợp với các chi tiết trực quan của hình ảnh và kết luận logic với câu trả lời chuẩn được cung cấp.]
</REASONING>
(Đừng quên thẻ <REASONING>!)
Hãy áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh, câu hỏi và câu trả lời chuẩn đã cho, đảm bảo rằng REASONING hoàn toàn khớp với REASONING chuẩn. 

Dữ liệu đầu vào:
- Hình ảnh: [Hình ảnh đầu vào]
- Câu hỏi: {question}
- Câu trả lời đúng: {answer}
"""


# ==================== DATA CLASS ====================

@dataclass
class ReasoningResult:
    question_id: str
    reasoning: str
    score: float
    error: Optional[str] = None


# ==================== GENERATOR ====================

# Global singleton instance
_GENERATOR_INSTANCE = None

class CoCoAReasoningGenerator:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI API key is required!")
        
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=TEMPERATURE,
            google_api_key=GEMINI_API_KEY
        )
        
        self.scorer = BlackBoxUQ(
            llm=self.llm,
            scorers=["semantic_negentropy"],
            use_best=True
        )
    
    async def generate_single(self, item: Dict, img_path: str) -> ReasoningResult:
        try:
            with open(img_path, "rb") as f:
                b64_string = base64.b64encode(f.read()).decode("utf-8")
            
            message_content = [
                {"type": "text", "text": PROMPT.format(
                    question=item['question'],
                    answer=item['answer']
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64_string}"
                }}
            ]
            
            # Wrap in no_grad to save VRAM during NLI model forward pass
            with torch.no_grad():
                results = await self.scorer.generate_and_score(
                    prompts=[[HumanMessage(content=message_content)]],
                    num_responses=NUM_RESPONSES
                )
            
            df = results.to_df()
            return ReasoningResult(
                question_id=str(item['question_id']),
                reasoning=df['response'].values[0],
                score=float(df['semantic_negentropy'].values[0])
            )
        except Exception as e:
            return ReasoningResult(
                question_id=str(item.get('question_id', 'unknown')),
                reasoning="",
                score=0.0,
                error=str(e)
            )
    
    def clear_cache(self):
        """Manually clear internal caches of UQLM components to prevent memory leaks."""
        try:
            # Access SemanticEntropy scorer
            if hasattr(self.scorer, "scorer_objects"):
                sem_entropy = self.scorer.scorer_objects.get("semantic_negentropy")
                if sem_entropy:
                    # Clear SemanticClusterer cache (Critical: Contains Base64 prompt keys)
                    if hasattr(sem_entropy, "clusterer") and hasattr(sem_entropy.clusterer, "nli_scores"):
                        for key in sem_entropy.clusterer.nli_scores:
                            if isinstance(sem_entropy.clusterer.nli_scores[key], dict):
                                sem_entropy.clusterer.nli_scores[key].clear()
                    
                    # Clear NLI probability cache
                    if hasattr(sem_entropy, "nli") and hasattr(sem_entropy.nli, "probabilities"):
                        sem_entropy.nli.probabilities.clear()
                        
        except Exception as e:
            print(f"Warning: Cache cleanup failed: {e}")

    async def generate_batch(self, items: List[Dict], img_dir: str) -> List[ReasoningResult]:
        results = []
        for item in tqdm(items, desc="Generating REASONING"):
            img_path = os.path.join(img_dir, item['image_name'])
            result = await self.generate_single(item, img_path)
            results.append(result)
            if result.error:
                tqdm.write(f"Error for {result.question_id}: {result.error}")
            
            # Critical: Clear UQLM caches after every generation to free up RAM/VRAM
            self.clear_cache()
            
            # Sleep to avoid rate limit
            await asyncio.sleep(SLEEP_DELAY_SECONDS)
        return results
    
    def cleanup(self):
        """Explicitly cleanup resources."""
        print("Cleaning up resources...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("CUDA cache cleared")


def get_generator() -> CoCoAReasoningGenerator:
    """Get or create singleton generator instance."""
    global _GENERATOR_INSTANCE
    
    if _GENERATOR_INSTANCE is None:
        print(f"Initializing generator (model: {GEMINI_MODEL})...")
        _GENERATOR_INSTANCE = CoCoAReasoningGenerator()
    else:
        print("Reusing existing generator instance")
    
    return _GENERATOR_INSTANCE


# ==================== UTILITIES ====================

def load_existing_ids() -> Set[str]:
    """Load question_ids that have already been processed."""
    output_path = os.path.join(OUTPUT_DIR, "cocoa_reasoning.json")
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            existing_ids = set(existing_data.keys())
            print(f"Found {len(existing_ids)} already-processed question_ids")
            return existing_ids
        except json.JSONDecodeError:
            print("Existing file is empty or invalid, starting fresh")
            return set()
    return set()


def load_failed_ids() -> Set[str]:
    failed_ids = set()
    for file_path in FAILED_IDS_FILES:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                failed_ids.update(data.get('failed_question_ids', []))
            print(f"Loaded {len(data.get('failed_question_ids', []))} IDs from {os.path.basename(file_path)}")
    return failed_ids


def load_vivqa_data(failed_ids: Set[str], existing_ids: Set[str]) -> List[Dict]:
    print(f"Loading ViVQA-X from: {VIVQA_PATH}")
    
    with open(VIVQA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter to failed IDs only
    data = [item for item in data if str(item.get('question_id')) in failed_ids]
    print(f"Filtered to {len(data)} items (failed IDs only)")
    
    # Skip already-processed items
    original_count = len(data)
    data = [item for item in data if str(item.get('question_id')) not in existing_ids]
    skipped_count = original_count - len(data)
    if skipped_count > 0:
        print(f"Skipping {skipped_count} already-processed items")
    print(f"{len(data)} items remaining to process")
    
    return data


def save_results(results: List[ReasoningResult]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load existing data if available
    mapping_path = os.path.join(OUTPUT_DIR, "cocoa_reasoning.json")
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                existing_mapping = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read existing file ({e}), starting fresh")
            existing_mapping = {}
    else:
        existing_mapping = {}
    
    # Merge new results with existing
    new_mapping = {r.question_id: r.reasoning for r in results if not r.error}
    existing_mapping.update(new_mapping)
    
    # Save merged data
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(existing_mapping, f, ensure_ascii=False, indent=2)
    
    # Stats
    num_success = len([r for r in results if not r.error])
    num_failed = len([r for r in results if r.error])
    total_saved = len(existing_mapping)
    
    print(f"\n{'='*60}")
    print(f"Saved {num_success}/{len(results)} reasonings")
    print(f"Output: {mapping_path}")
    print(f"Total in file: {total_saved} question_ids")
    if num_failed > 0:
        print(f"Failed: {num_failed} items")
    print(f"{'='*60}")


# ==================== MAIN ====================

async def main():
    print("\n" + "="*60)
    print("CoCoA REASONING GENERATOR")
    print("="*60)
    
    # Load failed IDs and existing processed IDs
    failed_ids = load_failed_ids()
    existing_ids = load_existing_ids()
    data = load_vivqa_data(failed_ids=failed_ids, existing_ids=existing_ids)
    
    if len(data) == 0:
        print("No data to process!")
        return
    
    # Get or create generator (singleton)
    generator = get_generator()
    
    print(f"Generating for {len(data)} items...\n")
    results = await generator.generate_batch(data, COCO_DIR)
    
    # Save
    save_results(results)
    
    print("\nDONE!")


if __name__ == "__main__":
    asyncio.run(main())

import torch
import re


def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility across libraries.

    Args:
        seed (int): The seed to use.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_clean_model_name(model_path_or_name: str) -> str:
    """
    Extracts clean model name from HuggingFace model path/name.
    
    Examples:
        Tower-Babel_Babel-9B-Chat -> Babel-9B-Chat
        DAMO-NLP-SG_VideoLLaMA3-7B-Image -> VideoLLaMA3-7B-Image
        OpenGVLab_InternVL3-8B -> InternVL3-8B
        Qwen_Qwen2.5-VL-7B-Instruct -> Qwen2.5-VL-7B-Instruct
        allenai_Molmo-7B-D-0924 -> Molmo-7B-D-0924
    
    Args:
        model_path_or_name (str): Full model path or HuggingFace model name
        
    Returns:
        str: Clean model name without organization prefix
    """
    model_name = model_path_or_name.split('/')[-1]
    
    if '_' in model_name:
        parts = model_name.split('_', 1)
        if len(parts) > 1:
            return parts[1]  # Return everything after first underscore
    
    return model_name


def extract_clean_filename(filename: str) -> str:
    """
    Extracts clean model name from prediction filename.
    
    Examples:
        DAMO-NLP-SG_VideoLLaMA3-7B-Image_normal.json -> VideoLLaMA3-7B-Image_normal
        OpenGVLab_InternVL3-8B_normal.json -> InternVL3-8B_normal
        VideoLLaMA3-7B-Image_normal.json -> VideoLLaMA3-7B-Image_normal
        
    Args:
        filename (str): Prediction filename
        
    Returns:
        str: Clean filename with organization prefix removed
    """
    base_name = filename.replace('.json', '')

    if '_' in base_name:
        parts = base_name.split('_')
        return '_'.join(parts[1:]) if len(parts) > 2 else base_name
    
    return base_name


def get_system_prompt() -> str:
    """
    Generates the system prompt for VQA models.

    Returns:
        str: The formatted system prompt.
    """
    system_instruction = (
        "Answer the following question based solely on the image content "
        "with a single term in Vietnamese, followed by a brief Vietnamese explanation.\n"
        "The explanation must be one short sentence only and point to the visual evidence that justifies the answer.\n"
        "Use this output format:\n"
        "Trả lời: <Your concise answer>\n"
        "Giải thích: <Your brief explanation>"
    )
    return system_instruction


def get_grpo_system_prompt():
    system_instruction = f"""
    You are a Visual Question Answering system. Your task is to answer and explain questions based on the content of the provided image.

    Follow these steps precisely:    
    - <think> Your step-by-step reasoning process. Analyze the image carefully. </think>
    - <answer> Based on your reasoning, provide your Vietnamese answer must be one word or a short phrase. </answer>
    - <explain> Your brief Vietnamese explanation (one sentence that justifies your answer). </explain>""".strip()
    
    return system_instruction

def parse_output(response: str) -> tuple[str, str]:
    """
    Extracts (answer, explanation) from model output.
    - Case-insensitive for 'Answer:' and 'Explanation:'.
    - Robust when labels are missing.
    - If no labels: use first two non-empty lines -> line1=answer, line2=explanation.
    """
    text = (response or "").strip()
    answer, explanation = "", ""

    m_ans = re.search(r'(?is)\btrả\s*lời\s*:\s*(.*?)(?:\n\s*\bgiải\s*thích\s*:|$)', text, re.I)
    m_exp = re.search(r'(?is)\bgiải\s*thích\s*:\s*(.*)$', text, re.I)

    if m_ans:
        answer = m_ans.group(1).strip()

    if m_exp:
        explanation = m_exp.group(1).strip()

    # Fallback: if missing either, use first two non-empty lines
    if not answer or (not explanation and 'giải thích' not in text.lower()):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not answer and lines:
            answer = lines[0]
        if not explanation and len(lines) >= 2:
            explanation = lines[1]
            
    if explanation.strip() == "":
        print("="*50)
        print(response)
        print("="*50)
    return answer, explanation

def parse_output_grpo(response: str) -> tuple[str, str, str]:
    """
    Trích xuất (think, answer, explanation) từ output của model
    dựa trên format thẻ <think>, <answer>, và <explain>.
    - Sử dụng re.DOTALL để xử lý nội dung đa dòng bên trong thẻ.
    - Robust khi thẻ bị thiếu hoặc không có nội dung.
    """
    text = (response or "").strip()
    think, answer, explanation = "", "", ""

    # 1. Trích xuất nội dung thẻ <think>
    m_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m_think:
        think = m_think.group(1).strip()

    # 2. Trích xuất nội dung thẻ <answer>
    m_ans = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m_ans:
        answer = m_ans.group(1).strip()

    # 3. Trích xuất nội dung thẻ <explain>
    m_exp = re.search(r"<explain>(.*?)</explain>", text, re.DOTALL)
    if m_exp:
        explanation = m_exp.group(1).strip()

    # 4. Fallback: (Ít quan trọng hơn khi dùng thẻ, nhưng vẫn giữ lại)
    #    Nếu không tìm thấy bất kỳ thẻ nào
    if not think and not answer and not explanation:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        non_tag_lines = [ln for ln in lines if not ln.startswith('<')]
        
        if len(non_tag_lines) >= 1:
            # Không thể biết dòng đầu là think hay answer,
            # nhưng ta gán nó cho answer theo logic cũ.
            answer = non_tag_lines[0]
        if len(non_tag_lines) >= 2:
            explanation = non_tag_lines[1]
        # Không có fallback rõ ràng cho 'think'

    # 5. Debug (Giữ lại từ hàm gốc của bạn)
    if explanation.strip() == "" and answer.strip() != "":
        print("="*50)
        print(f"DEBUG (grpo): Không tìm thấy 'explanation' hoặc 'explanation' rỗng.\nResponse:\n{response}")
        print("="*50)
        
    return think, answer, explanation

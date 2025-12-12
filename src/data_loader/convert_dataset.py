import json
import os

input_file = "/home/vlai-vqa-nle/minhtq/vqa-nle/data/processed/sft/ViVQA-X_train_msswift.jsonl"
output_file = "/home/vlai-vqa-nle/minhtq/vqa-nle/data/processed/sft/ViVQA-X_train_msswift_fixed.jsonl"

print(f"Converting {input_file} to {output_file}...")

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)
        messages = data.get('messages', [])
        solution = data.get('solution')
        
        if solution:
            # Check if assistant message already exists (just in case)
            has_assistant = any(m.get('role') == 'assistant' for m in messages)
            if not has_assistant:
                messages.append({
                    "role": "assistant",
                    "content": solution
                })
                data['messages'] = messages
        
        # Write the modified data
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')

print("Conversion complete.")

#!/bin/bash

# --- Bước 1: Cài "Core" (Nặng nhất & Quan trọng nhất) trước ---
echo ">>> Installing PyTorch 2.6.0 (CUDA 12.4)..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# --- Bước 2: Cài Flash Attention (Phụ thuộc vào Torch đã cài ở B1) ---
echo ">>> Installing Flash Attention..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# --- Bước 3: Cài các thư viện phụ trợ từ requirements.txt ---
echo ">>> Installing dependencies..."
# Lưu ý: Trong requirements.txt nhớ để 'transformers==4.46.0' (hoặc bản nào tương thích Torch 2.6)
pip install -r requirements.txt

# --- Bước 4: Cài ms-swift (Editable Mode) ---
echo ">>> Installing ms-swift in Editable Mode..."
cd /home/vlai-vqa-nle/minhtq/vqa-nle/ms-swift
# Dùng --no-build-isolation để nó nhận diện ngay Torch/FlashAttn đã cài
pip install -e . --no-build-isolation

echo ">>> Installation COMPLETED!"
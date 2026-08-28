#!/usr/bin/env bash
# ==============================================================================
# Diffusion Policy 1-Click Training Script
# Task: Astribot Making Coffee (Real Robot 3-Camera Setup)
# Optimized for: NVIDIA RTX 6000 / H100 / Blackwell GPUs
# ==============================================================================

set -e

# Change directory to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🚀 Khởi động Training Diffusion Policy (Astribot Coffee)"
echo "📂 Thư mục dự án: $SCRIPT_DIR"
echo "============================================================"

# Tự động kích hoạt môi trường conda nếu có
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda info --envs | grep -q "robodiff"; then
        echo "🔄 Kích hoạt Conda environment: robodiff"
        conda activate robodiff
    fi
fi

# Thiết lập GPU mặc định (cuda:0 nếu chưa chỉ định)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

echo "⚙️  GPU Device: $CUDA_VISIBLE_DEVICES"
echo "⚙️  Config: train_diffusion_unet_real_image_workspace"
echo "⚙️  Task: astribot_making_coffee_image"
echo "📊 WandB Project: astribot_making_coffee"
echo "============================================================"

# Chạy training (cho phép truyền thêm tham số override qua CLI)
python train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    "$@"

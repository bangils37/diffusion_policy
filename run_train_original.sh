#!/usr/bin/env bash
# ==============================================================================
# Diffusion Policy Baseline Training Script (Original Configuration)
# Task: Astribot Making Coffee (Real Robot 3-Camera Setup)
# Settings: Batch Size 64, LR 1e-4, 8 Workers, 1000 Epochs
# ==============================================================================

set -e

# Change directory to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🚀 Khởi động Training Diffusion Policy (Original Baseline)"
echo "📂 Thư mục dự án: $SCRIPT_DIR"
echo "============================================================"

# Tự động kích hoạt môi trường ảo (venv hoặc conda)
if [ -f "$SCRIPT_DIR/env/bin/activate" ]; then
    echo "🔄 Kích hoạt Python Virtualenv: $SCRIPT_DIR/env"
    source "$SCRIPT_DIR/env/bin/activate"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda info --envs | grep -q "robodiff"; then
        echo "🔄 Kích hoạt Conda environment: robodiff"
        conda activate robodiff
    fi
fi

# Tự động chọn GPU đang có VRAM trống nhiều nhất (nếu chưa truyền CUDA_VISIBLE_DEVICES)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    BEST_GPU=$(python -c "
import torch
best_gpu = 0
max_free = -1
for i in range(torch.cuda.device_count()):
    try:
        free, _ = torch.cuda.mem_get_info(i)
        if free > max_free:
            max_free = free
            best_gpu = i
    except Exception:
        pass
print(best_gpu)
" 2>/dev/null || echo "0")
    export CUDA_VISIBLE_DEVICES="$BEST_GPU"
    echo "🎯 Tự động chọn GPU có VRAM trống nhiều nhất: GPU $CUDA_VISIBLE_DEVICES"
else
    echo "⚙️  GPU Device (thủ công): $CUDA_VISIBLE_DEVICES"
fi

# Thiết lập các siêu tham số nguyên bản (Baseline Paper)
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1.0e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
echo "⚙️  Batch Size: $BATCH_SIZE"
echo "⚙️  Num Workers: $NUM_WORKERS"
echo "⚙️  Learning Rate: $LR"
echo "⚙️  Total Epochs: $NUM_EPOCHS"
echo "⚙️  Checkpoint Every: $CHECKPOINT_EVERY epochs"
echo "⚙️  Config: train_diffusion_unet_real_image_workspace"
echo "⚙️  Task: astribot_making_coffee_image"
echo "📊 WandB Project: astribot_making_coffee"
echo "============================================================"

# Chạy training với cấu hình nguyên bản
python train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    dataloader.batch_size="$BATCH_SIZE" \
    val_dataloader.batch_size="$BATCH_SIZE" \
    dataloader.num_workers="$NUM_WORKERS" \
    val_dataloader.num_workers="$NUM_WORKERS" \
    optimizer.lr="$LR" \
    training.num_epochs="$NUM_EPOCHS" \
    training.checkpoint_every="$CHECKPOINT_EVERY" \
    "$@"

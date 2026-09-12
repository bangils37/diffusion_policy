#!/usr/bin/env bash
# ==============================================================================
# DIFFUSION POLICY TRANSFORMER MEM: Short-Video Memory Vision Training (OpenPI Architecture)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/env/bin/activate" ]; then
    source "$SCRIPT_DIR/env/bin/activate"
fi

# 1. Tham số cấu hình
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG_NAME="train_diffusion_transformer_mem_real_image_workspace"
TASK_NAME="astribot_making_coffee_mem_image"
DATASET_PATH="${DATASET_PATH:-/home/anhnb9/Documents/datasets/astri_making_coffee_v21.zarr}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2.0e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-300}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
MEM_NUM_FRAMES="${MEM_NUM_FRAMES:-12}"
MEM_FRAME_STRIDE="${MEM_FRAME_STRIDE:-15}"
WANDB_PROJECT="${WANDB_PROJECT:-astribot_making_coffee}"
WANDB_MODE="${WANDB_MODE:-online}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "============================================================"
echo "🚀 KHỞI CHẠY DIFFUSION POLICY TRANSFORMER MEM (OPENPI MEM ENCODER)"
echo "🎯 GPU: $CUDA_VISIBLE_DEVICES"
echo "⚙️  Config: $CONFIG_NAME"
echo "⚙️  Task: $TASK_NAME"
echo "🎞️  Memory Frames: $MEM_NUM_FRAMES (Stride: $MEM_FRAME_STRIDE ticks)"
echo "⚙️  Batch Size: $BATCH_SIZE | Workers: $NUM_WORKERS | LR: $LEARNING_RATE"
echo "⚙️  Total Epochs: $NUM_EPOCHS | Checkpoint Every: $CHECKPOINT_EVERY"
echo "📊 WandB: $WANDB_PROJECT (mode: $WANDB_MODE)"
echo "📁 Dataset: $DATASET_PATH"
echo "============================================================"

PYTHON_BIN="python"
if [ -f "$SCRIPT_DIR/env/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/env/bin/python"
fi

"$PYTHON_BIN" train.py \
    --config-name="$CONFIG_NAME" \
    task="$TASK_NAME" \
    task.dataset.dataset_path="$DATASET_PATH" \
    mem_num_frames="$MEM_NUM_FRAMES" \
    mem_frame_stride="$MEM_FRAME_STRIDE" \
    dataloader.batch_size="$BATCH_SIZE" \
    val_dataloader.batch_size="$BATCH_SIZE" \
    dataloader.num_workers="$NUM_WORKERS" \
    val_dataloader.num_workers="$NUM_WORKERS" \
    optimizer.lr="$LEARNING_RATE" \
    training.lr_warmup_steps=1000 \
    training.num_epochs="$NUM_EPOCHS" \
    training.checkpoint_every="$CHECKPOINT_EVERY" \
    logging.project="$WANDB_PROJECT" \
    logging.mode="$WANDB_MODE" \
    "$@"

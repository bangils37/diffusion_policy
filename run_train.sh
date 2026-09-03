#!/usr/bin/env bash
# ==============================================================================
# 🚀 DIFFUSION POLICY 1-CLICK ALL-IN-ONE TRAINING RUNNER
# ==============================================================================
# Hướng dẫn:
# 1. Chỉnh sửa cấu hình & đường dẫn dataset trực tiếp ở mục "USER CONFIGURATION" bên dưới.
# 2. Chạy lệnh: ./run_train.sh
# ==============================================================================

set -e

# ==============================================================================
# 🎯 USER CONFIGURATION (Chỉnh sửa các tham số ở đây)
# ==============================================================================

# 1. Tên Task & Kiến trúc Mô hình (Model Architecture)
TASK_NAME="astribot_making_coffee_image"                  # File config trong diffusion_policy/config/task/
MODEL_TYPE="transformer"                                  # "transformer" (Transformer-Base) | "unet" (CNN/UNet Baseline)

# Tự động gán config tương ứng theo MODEL_TYPE (hoặc ghi đè trực tiếp tên file config)
if [ "$MODEL_TYPE" = "transformer" ]; then
    CONFIG_NAME="train_diffusion_transformer_real_image_workspace"
else
    CONFIG_NAME="train_diffusion_unet_real_image_workspace"
fi

# 2. Đường dẫn Dataset (.zarr hoặc .zarr.zip)
DATASET_PATH="/home/anhnb9/Documents/datasets/astri_making_coffee_v21.zarr"

# 3. Siêu tham số Huấn luyện (Training Hyperparameters)
BATCH_SIZE=256              # Kích thước batch (256 cho RTX 6000/H100, 128 cho card tầm trung, 64 chuẩn paper)
NUM_WORKERS=16              # Số luồng CPU load & giải nén dữ liệu (8 - 16)
LEARNING_RATE="1.0e-4"      # Tốc độ học (1.0e-4 cho Transformer, 2.0e-4 cho UNet batch 256)
NUM_EPOCHS=150              # Tổng số epoch cần train
CHECKPOINT_EVERY=10         # Tần suất lưu checkpoint (mỗi N epoch)
LR_WARMUP_STEPS=1000        # Số bước warmup learning rate (Transformer rất cần warmup)

# 4. Cấu hình Transformer-Base (Chỉ áp dụng khi MODEL_TYPE="transformer")
TRANSFORMER_LAYERS=12       # Số layer Transformer (12 cho bản base/large)
TRANSFORMER_HEADS=8         # Số attention heads
TRANSFORMER_EMB_DIM=512     # Chiều embedding (512 cho bản base/large)

# 5. Phần cứng (Hardware) & Weights & Biases (WandB)
GPU_ID=""                   # Để trống "" để tự động chọn GPU nhiều VRAM trống nhất, hoặc chỉ định "0", "1", "3"...
WANDB_PROJECT="astribot_making_coffee"  # Tên project trên WandB
WANDB_MODE="online"         # "online" (đẩy lên cloud) | "offline" (lưu máy cục bộ) | "disabled" (tắt WandB)

# ==============================================================================
# ⚙️ TỰ ĐỘNG KHỞI TẠO & CHẠY TIẾN TRÌNH (Không cần sửa phần bên dưới)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🤖 KHỞI ĐỘNG DIFFUSION POLICY TRAINING"
echo "📂 Thư mục: $SCRIPT_DIR"
echo "============================================================"

# 1. Kiểm tra và kích hoạt môi trường
if [ ! -d "$SCRIPT_DIR/env" ] && ! conda info --envs 2>/dev/null | grep -q "robodiff"; then
    echo "⚠️ Chưa tìm thấy môi trường ảo! Đang tự động chạy setup.sh..."
    bash "$SCRIPT_DIR/setup.sh"
fi

if [ -f "$SCRIPT_DIR/env/bin/activate" ]; then
    echo "🔄 Kích hoạt Virtualenv: $SCRIPT_DIR/env"
    source "$SCRIPT_DIR/env/bin/activate"
elif command -v conda &> /dev/null && conda info --envs | grep -q "robodiff"; then
    eval "$(conda shell.bash hook)"
    echo "🔄 Kích hoạt Conda environment: robodiff"
    conda activate robodiff
fi

# 2. Tự động chọn GPU phù hợp
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "⚙️  GPU Device (thủ công): GPU $CUDA_VISIBLE_DEVICES"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "⚙️  GPU Device (môi trường): GPU $CUDA_VISIBLE_DEVICES"
else
    # Tự động tìm GPU còn nhiều VRAM trống nhất
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
fi

# 3. Kiểm tra đường dẫn Dataset & Tùy biến tham số theo mô hình
EXTRA_ARGS=()
if [ -n "$DATASET_PATH" ]; then
    if [ -e "$DATASET_PATH" ]; then
        echo "📁 Dataset: $DATASET_PATH (Đã tìm thấy)"
        EXTRA_ARGS+=("task.dataset.dataset_path=$DATASET_PATH")
    else
        echo "⚠️ CẢNH BÁO: Không tìm thấy file dataset tại: $DATASET_PATH"
        echo "   Sẽ sử dụng đường dẫn mặc định trong task config."
    fi
fi

# Thiết lập tham số optimizer và architecture phù hợp từng model
if [[ "$CONFIG_NAME" == *"transformer"* ]]; then
    EXTRA_ARGS+=("optimizer.learning_rate=$LEARNING_RATE")
    EXTRA_ARGS+=("policy.n_layer=$TRANSFORMER_LAYERS")
    EXTRA_ARGS+=("policy.n_head=$TRANSFORMER_HEADS")
    EXTRA_ARGS+=("policy.n_emb=$TRANSFORMER_EMB_DIM")
else
    EXTRA_ARGS+=("optimizer.lr=$LEARNING_RATE")
fi

# 4. Thiết lập biến môi trường tăng tốc
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

echo "⚙️  Kiến trúc: $MODEL_TYPE"
echo "⚙️  Task: $TASK_NAME"
echo "⚙️  Config Workspace: $CONFIG_NAME"
echo "⚙️  Batch Size: $BATCH_SIZE"
echo "⚙️  Num Workers: $NUM_WORKERS"
echo "⚙️  Learning Rate: $LEARNING_RATE"
echo "⚙️  Total Epochs: $NUM_EPOCHS"
echo "⚙️  Checkpoint Every: $CHECKPOINT_EVERY epochs"
if [[ "$CONFIG_NAME" == *"transformer"* ]]; then
    echo "⚙️  Transformer: layers=$TRANSFORMER_LAYERS, heads=$TRANSFORMER_HEADS, emb=$TRANSFORMER_EMB_DIM"
fi
echo "📊 WandB Project: $WANDB_PROJECT (mode: $WANDB_MODE)"
echo "============================================================"

# 5. Khởi chạy Training
python train.py \
    --config-name="$CONFIG_NAME" \
    task="$TASK_NAME" \
    dataloader.batch_size="$BATCH_SIZE" \
    val_dataloader.batch_size="$BATCH_SIZE" \
    dataloader.num_workers="$NUM_WORKERS" \
    val_dataloader.num_workers="$NUM_WORKERS" \
    training.lr_warmup_steps="$LR_WARMUP_STEPS" \
    training.num_epochs="$NUM_EPOCHS" \
    training.checkpoint_every="$CHECKPOINT_EVERY" \
    logging.project="$WANDB_PROJECT" \
    logging.mode="$WANDB_MODE" \
    "${EXTRA_ARGS[@]}" \
    "$@"

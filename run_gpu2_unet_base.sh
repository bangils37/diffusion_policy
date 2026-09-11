#!/usr/bin/env bash
# ==============================================================================
# GPU 2: Diffusion U-Net - BASELINE (Resuming with Batch Size: 256)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/env/bin/activate" ]; then
    source "$SCRIPT_DIR/env/bin/activate"
fi

export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OUTPUT_DIR="data/outputs/2026.08.28/17.55.17_train_diffusion_unet_real_image_astribot_making_coffee_image"
DATASET_PATH="/home/anhnb9/Documents/datasets/astri_making_coffee_v21.zarr"
BATCH_SIZE=256
NUM_WORKERS=6
LR=2.0e-4
NUM_EPOCHS=300
CHECKPOINT_EVERY=10
WANDB_PROJECT="astribot_making_coffee"
WANDB_ID="mhv1glal"

echo "============================================================"
echo "🚀 [GPU 2] TIẾP TỤC TRAINING U-NET BASE (BS 256)"
echo "📂 Output: $OUTPUT_DIR"
echo "⚙️  Batch Size: $BATCH_SIZE | Num Workers: $NUM_WORKERS | LR: $LR"
echo "⚙️  Architecture: down_dims=[512, 1024, 2048]"
echo "📊 WandB: $WANDB_PROJECT (Run ID: $WANDB_ID)"
echo "============================================================"

python train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    hydra.run.dir="$OUTPUT_DIR" \
    task="astribot_making_coffee_image" \
    task.dataset.dataset_path="$DATASET_PATH" \
    dataloader.batch_size="$BATCH_SIZE" \
    val_dataloader.batch_size="$BATCH_SIZE" \
    dataloader.num_workers="$NUM_WORKERS" \
    val_dataloader.num_workers="$NUM_WORKERS" \
    optimizer.lr="$LR" \
    training.lr_warmup_steps=1000 \
    training.num_epochs="$NUM_EPOCHS" \
    training.checkpoint_every="$CHECKPOINT_EVERY" \
    logging.project="$WANDB_PROJECT" \
    logging.mode="online" \
    logging.id="$WANDB_ID" \
    logging.resume="allow" \
    "$@"

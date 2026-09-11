#!/usr/bin/env bash
# ==============================================================================
# GPU 3: Diffusion U-Net - LONG HORIZON & RESNET34 (Batch Size: 256, Seed: 3407)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/env/bin/activate" ]; then
    source "$SCRIPT_DIR/env/bin/activate"
fi

export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OUTPUT_DIR="data/outputs/experiments/train_diffusion_unet_resnet34_horizon32_seed3407"
DATASET_PATH="/home/anhnb9/Documents/datasets/astri_making_coffee_v21.zarr"
BATCH_SIZE=256
NUM_WORKERS=6
LR=2.0e-4
SEED=3407
NUM_EPOCHS=300
CHECKPOINT_EVERY=10
WANDB_PROJECT="astribot_making_coffee"
WANDB_NAME="unet_resnet34_horizon32_bs256_seed3407"

echo "============================================================"
echo "🚀 [GPU 3] KHỞI CHẠY U-NET RESNET34 LONG HORIZON (Seed: $SEED)"
echo "📂 Output: $OUTPUT_DIR"
echo "⚙️  Batch Size: $BATCH_SIZE | Num Workers: $NUM_WORKERS | LR: $LR | Seed: $SEED"
echo "⚙️  Architecture: resnet34, horizon=32, n_action_steps=16"
echo "📊 WandB: $WANDB_PROJECT ($WANDB_NAME)"
echo "============================================================"

python train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    hydra.run.dir="$OUTPUT_DIR" \
    task="astribot_making_coffee_image" \
    task.dataset.dataset_path="$DATASET_PATH" \
    horizon=32 \
    n_action_steps=16 \
    policy.obs_encoder.rgb_model.name=resnet34 \
    dataloader.batch_size="$BATCH_SIZE" \
    val_dataloader.batch_size="$BATCH_SIZE" \
    dataloader.num_workers="$NUM_WORKERS" \
    val_dataloader.num_workers="$NUM_WORKERS" \
    optimizer.lr="$LR" \
    training.seed="$SEED" \
    training.lr_warmup_steps=1000 \
    training.num_epochs="$NUM_EPOCHS" \
    training.checkpoint_every="$CHECKPOINT_EVERY" \
    logging.project="$WANDB_PROJECT" \
    logging.mode="online" \
    logging.name="$WANDB_NAME" \
    "$@"

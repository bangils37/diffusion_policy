#!/usr/bin/env bash
# ==============================================================================
# GPU 1: Diffusion Transformer - BASELINE (Resuming with Batch Size: 256)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/env/bin/activate" ]; then
    source "$SCRIPT_DIR/env/bin/activate"
fi

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OUTPUT_DIR="data/outputs/2026.09.03/10.26.32_train_diffusion_transformer_real_image_astribot_making_coffee_image"
DATASET_PATH="/home/anhnb9/Documents/datasets/astri_making_coffee_v21.zarr"
BATCH_SIZE=256
NUM_WORKERS=6
LR=1.0e-4
NUM_EPOCHS=150
CHECKPOINT_EVERY=10
WANDB_PROJECT="astribot_making_coffee"
WANDB_ID="f5qol23j"

echo "============================================================"
echo "🚀 [GPU 1] TIẾP TỤC TRAINING TRANSFORMER BASE (BS 256)"
echo "📂 Output: $OUTPUT_DIR"
echo "⚙️  Batch Size: $BATCH_SIZE | Num Workers: $NUM_WORKERS | LR: $LR"
echo "⚙️  Architecture: n_layer=12, n_head=8, n_emb=512"
echo "📊 WandB: $WANDB_PROJECT (Run ID: $WANDB_ID)"
echo "============================================================"

python train.py \
    --config-name=train_diffusion_transformer_real_image_workspace \
    hydra.run.dir="$OUTPUT_DIR" \
    task="astribot_making_coffee_image" \
    task.dataset.dataset_path="$DATASET_PATH" \
    dataloader.batch_size="$BATCH_SIZE" \
    val_dataloader.batch_size="$BATCH_SIZE" \
    dataloader.num_workers="$NUM_WORKERS" \
    val_dataloader.num_workers="$NUM_WORKERS" \
    optimizer.learning_rate="$LR" \
    policy.n_layer=12 \
    policy.n_head=8 \
    policy.n_emb=512 \
    training.lr_warmup_steps=1000 \
    training.num_epochs="$NUM_EPOCHS" \
    training.checkpoint_every="$CHECKPOINT_EVERY" \
    logging.project="$WANDB_PROJECT" \
    logging.mode="online" \
    logging.id="$WANDB_ID" \
    logging.resume="allow" \
    "$@"

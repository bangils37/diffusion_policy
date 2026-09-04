#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/env/bin/activate" ]; then
    source "$SCRIPT_DIR/env/bin/activate"
elif command -v conda &> /dev/null && conda info --envs | grep -q "robodiff"; then
    eval "$(conda shell.bash hook)"
    conda activate robodiff
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="data/outputs/2026.09.03/10.26.32_train_diffusion_transformer_real_image_astribot_making_coffee_image"
DATASET_PATH="/home/anhnb9/Documents/datasets/astri_making_coffee_v21.zarr"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LR="${LR:-1.0e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-150}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
WANDB_PROJECT="astribot_making_coffee"
WANDB_ID="f5qol23j"

echo "============================================================"
echo "🚀 TIẾP TỤC TRAINING TRANSFORMER-BASE (Resuming from Epoch 40)"
echo "📂 Thư mục output: $OUTPUT_DIR"
echo "🎯 GPU: GPU $CUDA_VISIBLE_DEVICES"
echo "⚙️  Batch Size: $BATCH_SIZE"
echo "⚙️  Num Workers: $NUM_WORKERS"
echo "⚙️  Learning Rate: $LR"
echo "⚙️  Target Epochs: $NUM_EPOCHS"
echo "⚙️  Checkpoint Every: $CHECKPOINT_EVERY epochs"
echo "📊 WandB: $WANDB_PROJECT (run: $WANDB_ID)"
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

#!/bin/bash
# Script to sync offline/active runs to Weights & Biases cloud
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WANDB_BIN="$ROOT_DIR/env/bin/wandb"
PROJECT="astribot_making_coffee"
ENTITY="nguyenbanganh30-vnu"

echo "=== Syncing Transformer Run to W&B ==="
TRANS_DIR="$ROOT_DIR/data/outputs/2026.09.03/10.26.32_train_diffusion_transformer_real_image_astribot_making_coffee_image/wandb"
if [ -d "$TRANS_DIR" ]; then
    TRANS_WANDB=$(ls -td "$TRANS_DIR"/run-*/run-*.wandb 2>/dev/null | head -n 1)
    if [ -n "$TRANS_WANDB" ]; then
        echo "Found: $TRANS_WANDB"
        "$WANDB_BIN" sync --legacy --include-online -p "$PROJECT" -e "$ENTITY" "$TRANS_WANDB" || true
    fi
fi

echo "=== Syncing CNN / UNet Run to W&B ==="
CNN_DIR="$ROOT_DIR/data/outputs/2026.08.28/17.55.17_train_diffusion_unet_real_image_astribot_making_coffee_image/wandb"
if [ -d "$CNN_DIR" ]; then
    CNN_WANDB=$(ls -td "$CNN_DIR"/run-*/run-*.wandb 2>/dev/null | head -n 1)
    if [ -n "$CNN_WANDB" ]; then
        echo "Found: $CNN_WANDB"
        "$WANDB_BIN" sync --legacy --include-online -p "$PROJECT" -e "$ENTITY" "$CNN_WANDB" || true
    fi
fi

echo "=== Sync complete! ==="

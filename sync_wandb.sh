#!/bin/bash
# Script to sync offline/active runs to Weights & Biases cloud
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WANDB_BIN="$ROOT_DIR/env/bin/wandb"

echo "=== Syncing Transformer Run to W&B ==="
TRANS_WANDB=$(find "$ROOT_DIR/data/outputs/2026.09.03/10.26.32_train_diffusion_transformer_real_image_astribot_making_coffee_image/wandb" -name "run-*.wandb" 2>/dev/null | head -n 1)
if [ -n "$TRANS_WANDB" ]; then
    "$WANDB_BIN" sync --no-skip-online "$TRANS_WANDB" || true
fi

echo "=== Syncing CNN / UNet Run to W&B ==="
CNN_WANDB=$(find "$ROOT_DIR/data/outputs/2026.08.28/17.55.17_train_diffusion_unet_real_image_astribot_making_coffee_image/wandb" -name "run-mhv1glal.wandb" 2>/dev/null | head -n 1)
if [ -n "$CNN_WANDB" ]; then
    "$WANDB_BIN" sync --no-skip-online "$CNN_WANDB" || true
fi

echo "=== Sync complete! ==="

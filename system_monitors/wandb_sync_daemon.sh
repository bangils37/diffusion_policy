#!/usr/bin/env bash
# ==============================================================================
# W&B Auto-Sync Daemon
# ==============================================================================
# Tự động quét và đồng bộ các file log .wandb cục bộ lên Weights & Biases cloud
# Sử dụng cờ --legacy để loại bỏ hoàn toàn lỗi EOF (transactionlog error).
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/home/anhnb9/Documents/diffusion_policy"
WANDB_BIN="$ROOT_DIR/env/bin/wandb"
LOG_FILE="$SCRIPT_DIR/wandb_sync_daemon.log"

PROJECT="${WANDB_PROJECT:-astribot_making_coffee}"
ENTITY="${WANDB_ENTITY:-nguyenbanganh30-vnu}"
INTERVAL="${SYNC_INTERVAL:-300}"  # 300 giây (5 phút)

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log "🚀 W&B Auto-Sync Daemon khởi động (Project: $PROJECT, Entity: $ENTITY, Interval: ${INTERVAL}s)"

if [ ! -x "$WANDB_BIN" ]; then
    log "❌ Không tìm thấy wandb binary tại: $WANDB_BIN"
    exit 1
fi

while true; do
    log "🔄 Bắt đầu chu kỳ đồng bộ..."

    # 1. Đồng bộ Transformer run mới nhất
    TRANS_DIR="$ROOT_DIR/data/outputs/2026.09.03/10.26.32_train_diffusion_transformer_real_image_astribot_making_coffee_image/wandb"
    if [ -d "$TRANS_DIR" ]; then
        TRANS_WANDB=$(ls -td "$TRANS_DIR"/run-*/run-*.wandb 2>/dev/null | head -n 1)
        if [ -n "$TRANS_WANDB" ]; then
            log "📤 Đồng bộ Transformer: $TRANS_WANDB"
            "$WANDB_BIN" sync --legacy --include-online -p "$PROJECT" -e "$ENTITY" "$TRANS_WANDB" >> "$LOG_FILE" 2>&1 || true
        fi
    fi

    # 2. Đồng bộ CNN / UNet run mới nhất
    CNN_DIR="$ROOT_DIR/data/outputs/2026.08.28/17.55.17_train_diffusion_unet_real_image_astribot_making_coffee_image/wandb"
    if [ -d "$CNN_DIR" ]; then
        CNN_WANDB=$(ls -td "$CNN_DIR"/run-*/run-*.wandb 2>/dev/null | head -n 1)
        if [ -n "$CNN_WANDB" ]; then
            log "📤 Đồng bộ CNN/UNet: $CNN_WANDB"
            "$WANDB_BIN" sync --legacy --include-online -p "$PROJECT" -e "$ENTITY" "$CNN_WANDB" >> "$LOG_FILE" 2>&1 || true
        fi
    fi

    log "✅ Chu kỳ đồng bộ hoàn tất. Nghỉ ${INTERVAL}s trước chu kỳ kế tiếp..."
    sleep "$INTERVAL"
done

#!/usr/bin/env bash
# ==============================================================================
# CÔNG CỤ: W&B Auto-Sync Daemon (Tiến Trình Tự Động Đồng Bộ Weights & Biases)
# Tác giả: AI Assistant / AnhNB9
# Mục đích:
#     Tự động quét định kỳ các thư mục log của các mô hình AI đang huấn luyện
#     (Diffusion Transformer & CNN/UNet) và đẩy các metrics, biểu đồ loss, learning rate
#     lên hệ thống đám mây Weights & Biases (W&B).
#
# Bối cảnh kỹ thuật quan trọng:
#     1. Tránh lỗi 'transactionlog unexpected EOF':
#        Phiên bản wandb (0.29.x) có cơ chế đọc transaction log mới rất dễ văng lỗi EOF
#        khi file log đang được ghi dở bởi tiến trình training khác. Script này bắt buộc
#        phải sử dụng cờ:
#            wandb sync --legacy --include-online ...
#        để kích hoạt chế độ đồng bộ tương thích ngược cổ điển an toàn 100%.
#     2. Đồng bộ các run offline hoặc đứt kết nối mạng:
#        Khi server mất mạng tạm thời hoặc training bị kill, các điểm dữ liệu chưa đồng bộ
#        sẽ được daemon này phát hiện và đẩy bù đầy đủ lên dashboard.
#
# Biến môi trường có thể cấu hình (Environment Variables):
#     - WANDB_PROJECT   : Tên dự án W&B (Mặc định: 'astribot_making_coffee')
#     - WANDB_ENTITY    : Tài khoản hoặc Workspace W&B (Mặc định: 'nguyenbanganh30-vnu')
#     - SYNC_INTERVAL   : Khoảng thời gian nghỉ giữa 2 lần đồng bộ tính bằng giây (Mặc định: 300 = 5 phút)
#
# File nhật ký:
#     - 'wandb_sync_daemon.log': Ghi nhận mốc thời gian và kết quả chi tiết từng lần đẩy dữ liệu.
#
# Hướng dẫn sử dụng:
#     1. Chạy độc lập:
#        ./wandb_sync_daemon.sh
#     2. Chạy với chu kỳ 1 phút (60s):
#        SYNC_INTERVAL=60 ./wandb_sync_daemon.sh
#     3. Chạy nền bằng Nohup:
#        nohup ./wandb_sync_daemon.sh > /dev/null 2>&1 &
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

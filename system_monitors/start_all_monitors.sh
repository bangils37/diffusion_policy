#!/usr/bin/env bash
# ==============================================================================
# 🚀 Start All AI Server System Monitors in Tmux
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TMUX_NAME="sys_monitors"

echo "============================================================"
echo "🛡️  KHỞI ĐỘNG HỆ THỐNG GIÁM SÁT TOÀN DIỆN (sys_monitors)"
echo "📂 Thư mục: $SCRIPT_DIR"
echo "============================================================"

# Hủy session cũ nếu đang tồn tại
tmux kill-session -t "$TMUX_NAME" 2>/dev/null || true

# 1. Tạo session mới và chạy GPU Watcher
echo "1️⃣  Khởi động GPU Watcher (Window 0)..."
tmux new-session -d -s "$TMUX_NAME" -n "gpu_watcher" \
    "python3 $SCRIPT_DIR/gpu_watcher.py --min-vram-gb 45 --check-interval 10"

# 2. Tạo window chạy Process Guard
echo "2️⃣  Khởi động Process Guard Sentinel (Window 1)..."
tmux new-window -t "$TMUX_NAME" -n "process_guard" \
    "python3 $SCRIPT_DIR/process_guard.py --patterns train.py diffusion --tmux-sessions train_transformer train_cnn --interval 5"

# 3. Tạo window chạy W&B Auto-Sync Daemon
echo "3️⃣  Khởi động W&B Auto-Sync Daemon (Window 2)..."
tmux new-window -t "$TMUX_NAME" -n "wandb_sync" \
    "bash $SCRIPT_DIR/wandb_sync_daemon.sh"

# 4. Chạy kiểm tra reboot 1 lần
echo "4️⃣  Kiểm tra trạng thái Boot & Uptime..."
python3 "$SCRIPT_DIR/reboot_detector.py"

echo "============================================================"
echo "✅ TẤT CẢ CÁC ỨNG DỤNG GIÁM SÁT ĐÃ ĐƯỢC KÍCH HOẠT NỀN TRONG TMUX!"
echo "👉 Xem dashboard trực tiếp: ./status.sh"
echo "👉 Vào xem chi tiết các màn hình giám sát: tmux attach -t $TMUX_NAME"
echo "   (Chuyển giữa các window: Ctrl+B rồi bấm 0, 1, 2. Thoát ra: Ctrl+B rồi bấm D)"
echo "============================================================"

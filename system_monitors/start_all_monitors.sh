#!/usr/bin/env bash
# ==============================================================================
# CÔNG CỤ: Start All System Monitors (Khởi Động Trọn Bộ Giám Sát Trong Tmux)
# Tác giả: AI Assistant / AnhNB9
# Mục đích:
#     Khởi chạy toàn bộ hệ sinh thái ứng dụng giám sát hệ thống (Watchdog Suite)
#     bên trong một session Tmux độc lập mang tên 'sys_monitors'.
#
# Cấu trúc các cửa sổ (Windows) được khởi tạo trong Tmux:
#     - Window 0 [gpu_watcher]   : Quét liên tục VRAM 4 card GPU mỗi 10 giây, sẵn sàng
#                                  phát hiện GPU trống và kích hoạt job.
#     - Window 1 [process_guard] : Trực chờ các PID huấn luyện mô hình (train.py, diffusion),
#                                  tự động điều tra ai đã kill tiến trình qua /var/log/auth.log.
#     - Window 2 [wandb_sync]    : Chạy tiến trình đồng bộ ngầm định kỳ dữ liệu lên W&B
#                                  bằng giao thức an toàn không lỗi EOF.
#
# Lợi ích:
#     - Giữ cho toàn bộ các script giám sát chạy liên tục 24/7 ngay cả khi đóng terminal SSH.
#     - Cho phép quản trị viên attach vào xem live log của từng công cụ bất cứ lúc nào.
#
# Cách sử dụng:
#     ./start_all_monitors.sh
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

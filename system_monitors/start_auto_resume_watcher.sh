#!/usr/bin/env bash
# ==============================================================================
# CÔNG CỤ: Start Auto Resume Watcher in Tmux
# Tác giả: AI Assistant / AnhNB9
# Mục đích:
#     Khởi chạy daemon giám sát và tự động kích hoạt resume training Diffusion Policy
#     (U-Net >= 30GB VRAM, Transformer >= 60GB VRAM) ngầm 24/7 trong phiên Tmux.
#
# Cách sử dụng:
#     ./start_auto_resume_watcher.sh
#
# Sau khi chạy:
#     - Xem live log giám sát: tmux attach -t auto_resume_watcher
#     - Thoát ra màn hình ngoài: Ctrl+B rồi bấm D
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TMUX_NAME="auto_resume_watcher"

echo "============================================================"
echo "🛡️  KHỞI ĐỘNG AUTO RESUME WATCHER (Session Tmux: $TMUX_NAME)"
echo "⚙️  Quy tắc phân bổ: U-Net >= 30GB VRAM | Transformer >= 60GB VRAM"
echo "============================================================"

# Hủy session cũ nếu đang tồn tại
tmux kill-session -t "$TMUX_NAME" 2>/dev/null || true

# Tạo session mới chạy auto_resume_watcher.py
tmux new-session -d -s "$TMUX_NAME" \
    "python3 $SCRIPT_DIR/auto_resume_watcher.py --transformer-vram 60 --unet-vram 30 --interval 10"

echo "✅ Auto Resume Watcher đã được khởi chạy ngầm thành công trong Tmux!"
echo "👉 Xem trực tiếp quá trình canh GPU: tmux attach -t $TMUX_NAME"
echo "👉 Nhật ký theo dõi: tail -f $SCRIPT_DIR/auto_resume_watcher.log"
echo "============================================================"

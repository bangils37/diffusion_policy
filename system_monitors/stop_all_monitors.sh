#!/usr/bin/env bash
# ==============================================================================
# CÔNG CỤ: Stop All System Monitors (Dừng Toàn Bộ Hệ Thống Giám Sát Tmux)
# Tác giả: AI Assistant / AnhNB9
# Mục đích:
#     Dừng an toàn toàn bộ phiên làm việc Tmux 'sys_monitors', qua đó chấm dứt hoạt động
#     của tất cả các ứng dụng giám sát chạy ngầm (GPU Watcher, Process Guard, W&B Sync).
#
# Cách sử dụng:
#     ./stop_all_monitors.sh
# ==============================================================================

TMUX_NAME="sys_monitors"
echo "🛑 Đang yêu cầu dừng session Tmux: $TMUX_NAME..."
if tmux kill-session -t "$TMUX_NAME" 2>/dev/null; then
    echo "✅ Đã dừng toàn bộ các ứng dụng giám sát thành công!"
else
    echo "ℹ️ Phiên làm việc Tmux '$TMUX_NAME' hiện không hoạt động (đã dừng trước đó)."
fi

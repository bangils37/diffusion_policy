#!/usr/bin/env bash
# Dừng toàn bộ các ứng dụng giám sát
TMUX_NAME="sys_monitors"
echo "Đang dừng session tmux $TMUX_NAME..."
tmux kill-session -t "$TMUX_NAME" 2>/dev/null && echo "✅ Đã dừng thành công!" || echo "ℹ️ Session $TMUX_NAME không tồn tại."

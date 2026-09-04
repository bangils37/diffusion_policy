#!/usr/bin/env bash
# ==============================================================================
# CÔNG CỤ: Stop Auto Resume Watcher
# Tác giả: AI Assistant / AnhNB9
# Mục đích: Dừng phiên Tmux auto_resume_watcher an toàn.
# ==============================================================================

TMUX_NAME="auto_resume_watcher"

echo "🛑 Đang dừng session Tmux: $TMUX_NAME..."
if tmux kill-session -t "$TMUX_NAME" 2>/dev/null; then
    echo "✅ Đã dừng Auto Resume Watcher thành công!"
else
    echo "ℹ️ Phiên làm việc Tmux '$TMUX_NAME' hiện không hoạt động."
fi

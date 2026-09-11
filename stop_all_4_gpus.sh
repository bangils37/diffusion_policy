#!/usr/bin/env bash
# ==============================================================================
# CÔNG CỤ: Dừng Toàn Bộ 4 Tiến Trình Training Trên 4 GPU
# ==============================================================================
echo "🛑 Đang dừng toàn bộ các tiến trình training Diffusion Policy..."
pkill -f "python train.py" || true
sleep 2
echo "✅ Đã dừng toàn bộ tiến trình. Kiểm tra lại:"
ps aux | grep "python train.py" | grep -v grep || echo "Không còn tiến trình nào đang chạy."

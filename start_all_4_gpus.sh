#!/usr/bin/env bash
# ==============================================================================
# CÔNG CỤ: Khởi Chạy Song Song 4 Mô Hình Trên 4 GPU (Đồng Bộ Batch Size: 256)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs
mkdir -p data/outputs/experiments

echo "🛑 Đang dọn dẹp các tiến trình training cũ..."
pkill -15 -f "python train.py" 2>/dev/null || true
sleep 2
pkill -9 -f "python train.py" 2>/dev/null || true
sleep 1

echo ""
echo "🚀 ĐANG KHỞI ĐỘNG ĐỒNG THỜI CẢ 4 GPU VỚI BATCH SIZE = 256 (Seed: 3407 cho GPU 0 & 3)..."
echo "-------------------------------------------------------------------------------------"

echo "🟢 [GPU 0] Transformer Large (L16-H16-D768, BS=256, Seed=3407)..."
nohup bash run_gpu0_transformer_large.sh > logs/gpu0_transformer_large.log 2>&1 &
echo "   -> PID: $! | Log: logs/gpu0_transformer_large.log"

sleep 3

echo "🔵 [GPU 1] Transformer Base (Resumed L12-H8-D512, BS=256)..."
nohup bash run_gpu1_transformer_base.sh > logs/gpu1_transformer_base.log 2>&1 &
echo "   -> PID: $! | Log: logs/gpu1_transformer_base.log"

sleep 3

echo "🟡 [GPU 2] U-Net Base (Resumed DownDims[512..2048], BS=256)..."
nohup bash run_gpu2_unet_base.sh > logs/gpu2_unet_base.log 2>&1 &
echo "   -> PID: $! | Log: logs/gpu2_unet_base.log"

sleep 3

echo "🟣 [GPU 3] U-Net Long-Horizon (ResNet34, Horizon=32, BS=256, Seed=3407)..."
nohup bash run_gpu3_unet_long_horizon.sh > logs/gpu3_unet_long_horizon.log 2>&1 &
echo "   -> PID: $! | Log: logs/gpu3_unet_long_horizon.log"

echo "-------------------------------------------------------------------------------------"
echo "🎉 TẤT CẢ 4 GPU ĐÃ ĐƯỢC KÍCH HOẠT THÀNH CÔNG VỚI BATCH SIZE = 256!"
echo "💡 Dùng lệnh: bash status_4_gpus.sh để theo dõi trạng thái thời gian thực."

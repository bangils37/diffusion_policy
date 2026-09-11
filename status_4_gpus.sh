#!/usr/bin/env bash
# ==============================================================================
# CÔNG CỤ: Theo Dõi Trạng Thái 4 GPU Training Dashboard
# ==============================================================================
echo "=========================================================================================="
echo "📊 TRẠNG THÁI 4 GPU TRAINING (NVIDIA RTX PRO 6000 Blackwell - 96 GB/GPU - Batch Size: 256)"
echo "=========================================================================================="

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=, read -r idx name mem_used mem_total util temp; do
        printf "🎯 GPU %-2s | VRAM: %6s / %6s MiB (%4.1f%%) | GPU Util: %3s%% | Temp: %2s°C\n" "$idx" "$mem_used" "$mem_total" "$(echo "scale=1; $mem_used * 100 / $mem_total" | bc -l 2>/dev/null || echo 0)" "$util" "$temp"
    done
fi

echo "------------------------------------------------------------------------------------------"
echo "📜 TIẾN ĐỘ TRAINING TỪ LOGS CỦA TỪNG GPU:"
echo "------------------------------------------------------------------------------------------"

echo "[GPU 0 - Transformer Large (Seed 3407)]:"
if [ -f logs/gpu0_transformer_large.log ]; then
    tail -n 2 logs/gpu0_transformer_large.log | sed 's/^/   /'
else
    echo "   (Chưa có log)"
fi

echo "[GPU 1 - Transformer Base]:"
if [ -f logs/gpu1_transformer_base.log ]; then
    tail -n 2 logs/gpu1_transformer_base.log | sed 's/^/   /'
else
    echo "   (Chưa có log)"
fi

echo "[GPU 2 - U-Net Base]:"
if [ -f logs/gpu2_unet_base.log ]; then
    tail -n 2 logs/gpu2_unet_base.log | sed 's/^/   /'
else
    echo "   (Chưa có log)"
fi

echo "[GPU 3 - U-Net Long-Horizon ResNet34 (Seed 3407)]:"
if [ -f logs/gpu3_unet_long_horizon.log ]; then
    tail -n 2 logs/gpu3_unet_long_horizon.log | sed 's/^/   /'
else
    echo "   (Chưa có log)"
fi

echo "=========================================================================================="

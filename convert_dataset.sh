#!/usr/bin/env bash
# ==============================================================================
# 🔄 1-Click LeRobot to Diffusion Policy Dataset Converter
# Tự động chuyển đổi dataset LeRobot sang Zarr và cập nhật đường dẫn vào run_train.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# 🎯 CẤU HÌNH ĐẦU VÀO (Có thể chỉnh sửa hoặc truyền qua tham số dòng lệnh)
# ==============================================================================
# Ví dụ: /path/to/lerobot_dataset hoặc lerobot/pusht
INPUT_DATASET="${1:-/home/anhnb9/Documents/datasets/lerobot_dataset}"

# Đường dẫn Zarr output mong muốn (để trống "" để tự động đặt tên theo dataset)
OUTPUT_ZARR="${2:-}"

# Kích thước ảnh resize (để trống nếu muốn giữ nguyên độ phân giải gốc của camera)
RESIZE_HEIGHT="" # ví dụ: 240
RESIZE_WIDTH=""  # ví dụ: 320

# ==============================================================================
# ⚙️ TIẾN HÀNH CHUYỂN ĐỔI
# ==============================================================================

echo "============================================================"
echo "🔄 BẮT ĐẦU CHUYỂN ĐỔI DATASET TỪ LEROBOT SANG DIFFUSION POLICY"
echo "📥 Nguồn: $INPUT_DATASET"
echo "============================================================"

# 1. Kích hoạt môi trường ảo
if [ -f "$SCRIPT_DIR/env/bin/activate" ]; then
    source "$SCRIPT_DIR/env/bin/activate"
elif command -v conda &> /dev/null && conda info --envs | grep -q "robodiff"; then
    eval "$(conda shell.bash hook)"
    conda activate robodiff
fi

# Đảm bảo các thư viện đọc parquet & video có sẵn
python -c "import pandas, pyarrow, cv2" 2>/dev/null || {
    echo "📦 Đang cài đặt bổ sung pandas, pyarrow..."
    pip install pandas pyarrow
}

# 2. Xử lý đường dẫn Output
if [ -z "$OUTPUT_ZARR" ]; then
    DATASET_NAME="$(basename "$INPUT_DATASET" | tr '.' '_')"
    OUTPUT_ZARR="$SCRIPT_DIR/data/${DATASET_NAME}.zarr"
fi

RESIZE_ARGS=()
if [ -n "$RESIZE_HEIGHT" ] && [ -n "$RESIZE_WIDTH" ]; then
    RESIZE_ARGS+=(--img_height "$RESIZE_HEIGHT" --img_width "$RESIZE_WIDTH")
fi

# 3. Thực thi chuyển đổi
python "$SCRIPT_DIR/convert_lerobot_to_zarr.py" \
    "$INPUT_DATASET" \
    -o "$OUTPUT_ZARR" \
    "${RESIZE_ARGS[@]}"

OUTPUT_ZARR_ABS="$(cd "$(dirname "$OUTPUT_ZARR")" && pwd)/$(basename "$OUTPUT_ZARR")"

# 4. Tự động ghi đường dẫn vào run_train.sh
if [ -f "$SCRIPT_DIR/run_train.sh" ]; then
    echo "📝 Đang tự động cập nhật DATASET_PATH vào run_train.sh..."
    
    # Dùng sed để thay thế giá trị DATASET_PATH trong run_train.sh
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|^DATASET_PATH=.*|DATASET_PATH=\"$OUTPUT_ZARR_ABS\"|g" "$SCRIPT_DIR/run_train.sh"
    else
        sed -i "s|^DATASET_PATH=.*|DATASET_PATH=\"$OUTPUT_ZARR_ABS\"|g" "$SCRIPT_DIR/run_train.sh"
    fi

    echo "✅ Đã ghi thành công: DATASET_PATH=\"$OUTPUT_ZARR_ABS\" vào run_train.sh!"
fi

echo "============================================================"
echo "🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH CHUYỂN ĐỔI!"
echo "👉 Bây giờ bạn chỉ cần chạy lệnh sau để train:"
echo "   ./run_train.sh"
echo "============================================================"

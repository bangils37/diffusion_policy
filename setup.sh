#!/usr/bin/env bash
# ==============================================================================
# 🛠️ 1-Click Environment Setup Script for Diffusion Policy
# Compatible with: Ubuntu/Linux, NVIDIA RTX (30/40/6000), A100, H100, Blackwell
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🚀 BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG CHO DIFFUSION POLICY"
echo "📂 Thư mục: $SCRIPT_DIR"
echo "============================================================"

# 1. Kiểm tra Python 3
PYTHON_BIN=""
for cmd in python3.10 python3.9 python3.11 python3; do
    if command -v "$cmd" &> /dev/null; then
        PY_VER=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        PY_MAJOR=$("$cmd" -c 'import sys; print(sys.version_info.major)')
        PY_MINOR=$("$cmd" -c 'import sys; print(sys.version_info.minor)')
        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 9 ] && [ "$PY_MINOR" -le 11 ]; then
            PYTHON_BIN="$cmd"
            echo "✅ Tìm thấy Python phù hợp: $PYTHON_BIN (Phiên bản $PY_VER)"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "⚠️ Không tìm thấy Python 3.9 - 3.11 chuyên dụng, dùng python3 mặc định..."
    PYTHON_BIN="python3"
fi

# 2. Khởi tạo Virtual Environment (./env)
VENV_DIR="$SCRIPT_DIR/env"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Đang tạo virtualenv tại: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "ℹ️  Môi trường virtualenv đã tồn tại tại: $VENV_DIR"
fi

# Kích hoạt venv
source "$VENV_DIR/bin/activate"

# 3. Nâng cấp Pip & Build Tools
echo "🔄 Nâng cấp pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# 4. Cài đặt PyTorch với CUDA hỗ trợ Hopper & Blackwell
echo "🔥 Đang cài đặt PyTorch (CUDA)..."
if ! python -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
    # Cài đặt PyTorch tương thích CUDA 12.x / 13.x
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || pip install torch torchvision
fi

# In thông tin PyTorch & CUDA
python -c "import torch; print(f'✅ PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device Count: {torch.cuda.device_count()}')"

# 5. Cài đặt các thư viện phụ thuộc từ requirements.txt
echo "📚 Đang cài đặt các thư viện cần thiết từ requirements.txt..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# 6. Cài đặt repo này ở chế độ Editable (-e)
echo "📦 Cài đặt diffusion_policy package (pip install -e .)..."
pip install -e .

echo "============================================================"
echo "🎉 CÀI ĐẶT MÔI TRƯỜNG THÀNH CÔNG 100%!"
echo "👉 Bây giờ bạn chỉ cần cấu hình và chạy: ./run_train.sh"
echo "============================================================"

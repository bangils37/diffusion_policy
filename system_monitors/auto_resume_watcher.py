#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CÔNG CỤ: Auto Resume Watcher & Dynamic GPU Allocator
         (Giám Sát VRAM Thông Minh & Tự Động Khôi Phục Huấn Luyện AI)
Tác giả: AI Assistant / AnhNB9
Mục đích:
    Trực chờ 24/7 trên máy chủ GPU (ai-server-1). Tự động theo dõi mức chiếm dụng VRAM
    của 4 card NVIDIA RTX PRO 6000 (96GB VRAM mỗi card).
    Ngay khi phát hiện có card GPU được giải phóng và đạt ngưỡng VRAM yêu cầu:
    1. Ngưỡng >= 60 GB VRAM: Kích hoạt khôi phục (resume) tiến trình
       Diffusion Policy Transformer ('train_transformer' qua resume_train_transformer.sh).
    2. Ngưỡng >= 30 GB VRAM: Kích hoạt khôi phục (resume) tiến trình
       Diffusion Policy U-Net ('train_unet' qua resume_train_unet.sh).

Nguyên lý và Thuật toán điều phối (Dynamic GPU Scheduling):
    - Định kỳ (mỗi 10-15 giây) quét thông số phần cứng qua 'nvidia-smi'.
    - Kiểm tra trạng thái sống/chết của 2 session Tmux:
        * 'train_transformer': Huấn luyện Transformer-base
        * 'train_unet': Huấn luyện U-Net / CNN-base
    - Nếu tác vụ nào ĐÃ ĐANG CHẠY: Bỏ qua, tránh khởi chạy trùng lặp.
    - Nếu tác vụ CHƯA CHẠY:
        * Sắp xếp các GPU theo dung lượng VRAM trống giảm dần.
        * Ưu tiên cấp phát GPU cho Transformer nếu có GPU rảnh >= 60 GB.
        * Tiếp tục cấp phát GPU còn lại cho U-Net nếu có GPU rảnh >= 30 GB.
        * Mỗi GPU chỉ cấp phát cho tối đa 1 tác vụ tại một thời điểm.
    - Khi kích hoạt:
        * Tự động truyền biến môi trường 'GPU_ID=<id>' tương ứng vào script.
        * Chạy trong session Tmux nền riêng biệt để đảm bảo ổn định 24/7.
        * Gửi thông báo Desktop (notify-send) và ghi nhật ký vào 'auto_resume_watcher.log'.

Các tham số dòng lệnh (CLI Arguments):
    --transformer-vram : VRAM tối thiểu (GB) cho Transformer (Mặc định: 60.0 GB).
    --unet-vram        : VRAM tối thiểu (GB) cho U-Net (Mặc định: 30.0 GB).
    --interval         : Chu kỳ kiểm tra (giây) (Mặc định: 10 giây).
    --log-file         : Đường dẫn file nhật ký (Mặc định: 'auto_resume_watcher.log').
    --exit-when-done   : Tự động kết thúc script Watcher sau khi ĐÃ KHỞI CHẠY THÀNH CÔNG các task.
                         (Mặc định là False: Watcher tiếp tục chạy ngầm để bảo vệ, tự động
                         resume lại nếu sau này task bị kill hoặc crash giữa chừng).
    --once             : Chỉ quét 1 lần duy nhất rồi thoát (Dry-run kiểm tra).

Cơ chế an toàn (Process Safety):
    - Khi các tác vụ đã chạy (RUNNING): Watcher TUYỆT ĐỐI KHÔNG tự kill hay can thiệp vào task.
    - Watcher chỉ đóng vai trò người quan sát (Observer/Dispatcher), đảm bảo tiến trình huấn
      luyện được thực thi độc lập trong session Tmux an toàn.

Ví dụ sử dụng:
    1. Chạy trực tiếp theo dõi liên tục và tự phục hồi 24/7 (Khuyên dùng):
       python3 auto_resume_watcher.py

    2. Tự thoát sau khi đã kích hoạt thành công cả 2 task:
       python3 auto_resume_watcher.py --exit-when-done

    3. Chạy kiểm tra nhanh hiện trạng GPU và tác vụ:
       python3 auto_resume_watcher.py --once
================================================================================
"""

import os
import sys
import time
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Thiết lập đường dẫn thư mục
BASE_DIR = Path("/home/anhnb9/Documents/diffusion_policy")
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_LOG_FILE = SCRIPT_DIR / "auto_resume_watcher.log"

# Đường dẫn các script resume training chuẩn
TRANSFORMER_SCRIPT = BASE_DIR / "resume_train_transformer.sh"
UNET_SCRIPT = BASE_DIR / "resume_train_unet.sh"

# ANSI Colors cho hiển thị Terminal sinh động
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_CYAN = "\033[96m"
C_MAGENTA = "\033[95m"


def log_msg(msg: str, log_file: Path):
    """Ghi log ra Terminal và đồng thời append vào file log."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_msg = msg
    for code in [C_RESET, C_BOLD, C_RED, C_GREEN, C_YELLOW, C_BLUE, C_CYAN, C_MAGENTA]:
        clean_msg = clean_msg.replace(code, "")
    print(f"[{ts}] {msg}")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {clean_msg}\n")
    except Exception:
        pass


def send_notification(title: str, message: str):
    """Gửi thông báo màn hình Desktop qua notify-send (nếu có môi trường GUI)."""
    try:
        subprocess.run(
            ["notify-send", "-u", "critical", "-t", "8000", title, message],
            capture_output=True,
            timeout=2,
        )
    except Exception:
        pass


def get_gpu_status() -> List[Dict]:
    """
    Truy vấn thông số VRAM và tải của từng GPU thông qua nvidia-smi.

    Returns:
        List[Dict]: Danh sách thông tin từng GPU bao gồm:
            - 'index': ID card GPU (0, 1, 2, 3)
            - 'name': Tên GPU (ví dụ: RTX PRO 6000)
            - 'free_gb': VRAM còn trống (GB)
            - 'used_gb': VRAM đang dùng (GB)
            - 'total_gb': Tổng dung lượng VRAM (GB)
            - 'util': Tỷ lệ tải GPU (%)
            - 'temp': Nhiệt độ (°C)
    """
    gpus = []
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=8)
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                free_mb = float(parts[2])
                used_mb = float(parts[3])
                total_mb = float(parts[4])
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "free_gb": free_mb / 1024.0,
                    "used_gb": used_mb / 1024.0,
                    "total_gb": total_mb / 1024.0,
                    "util": int(parts[5]),
                    "temp": int(parts[6]),
                })
    except subprocess.TimeoutExpired:
        print("⚠️ nvidia-smi bị timeout (>8s), có thể do driver GPU đang bận xử lý.")
    except Exception as e:
        print(f"Lỗi khi đọc nvidia-smi: {e}")
    return gpus


def is_tmux_session_running(session_name: str) -> bool:
    """Kiểm tra xem một phiên Tmux có đang tồn tại hay không."""
    try:
        res = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True,
            timeout=3,
        )
        return res.returncode == 0
    except Exception:
        return False


def is_training_process_running(pattern: str) -> bool:
    """Kiểm tra xem có tiến trình huấn luyện tương ứng đang chạy không."""
    try:
        res = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return res.returncode == 0 and bool(res.stdout.strip())
    except Exception:
        return False


def start_training_in_tmux(session_name: str, script_path: Path, gpu_id: int, log_file: Path) -> bool:
    """
    Kích hoạt tiến trình huấn luyện trong một phiên Tmux mới với GPU được chỉ định.

    Args:
        session_name: Tên session Tmux (ví dụ: 'train_transformer' hoặc 'train_unet')
        script_path: Đường dẫn tới script bash resume tương ứng
        gpu_id: ID card GPU được phân bổ (0, 1, 2, 3)
        log_file: File lưu nhật ký

    Returns:
        bool: True nếu khởi chạy thành công, False nếu thất bại
    """
    if not script_path.exists():
        log_msg(f"{C_RED}❌ Không tìm thấy script: {script_path}{C_RESET}", log_file)
        return False

    # Lệnh khởi chạy với GPU_ID được export
    run_cmd = f"cd {BASE_DIR} && GPU_ID={gpu_id} bash {script_path}"
    log_msg(
        f"{C_GREEN}{C_BOLD}🚀 ĐANG KÍCH HOẠT [{session_name}] TRÊN GPU {gpu_id}...{C_RESET}",
        log_file,
    )
    log_msg(f"   Lệnh thực thi: {C_CYAN}{run_cmd}{C_RESET}", log_file)

    try:
        # Hủy session cũ nếu có
        subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)
        time.sleep(1)

        # Tạo session tmux mới và chạy lệnh
        cmd = ["tmux", "new-session", "-d", "-s", session_name, run_cmd]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)

        notify_title = f"🚀 Khởi Chạy Huấn Luyện Thành Công!"
        notify_body = f"Tác vụ: {session_name}\nGPU: #{gpu_id}\nLệnh: {script_path.name}"
        send_notification(notify_title, notify_body)

        log_msg(
            f"{C_GREEN}✅ Đã khởi chạy thành công trong Tmux session '{session_name}'! (Xem: tmux attach -t {session_name}){C_RESET}",
            log_file,
        )
        return True
    except Exception as e:
        log_msg(f"{C_RED}❌ Lỗi khi khởi tạo Tmux session '{session_name}': {e}{C_RESET}", log_file)
        return False


def run_cycle(
    transformer_vram_min: float,
    unet_vram_min: float,
    log_file: Path,
) -> Tuple[bool, bool]:
    """
    Thực hiện 1 chu kỳ kiểm tra tài nguyên GPU và điều phối tác vụ.

    Returns:
        Tuple[bool, bool]: (transformer_is_running, unet_is_running)
    """
    # 1. Kiểm tra trạng thái hiện tại của 2 tác vụ
    trans_running = is_tmux_session_running("train_transformer") or is_training_process_running("train_diffusion_transformer")
    unet_running = is_tmux_session_running("train_unet") or is_training_process_running("train_diffusion_unet")

    # 2. Lấy danh sách GPU
    gpus = get_gpu_status()
    if not gpus:
        log_msg(f"{C_RED}⚠️ Không thể kết nối tới GPU qua nvidia-smi!{C_RESET}", log_file)
        return trans_running, unet_running

    # In trạng thái nhanh
    status_parts = []
    for g in gpus:
        color = C_RED if g["free_gb"] < 15 else (C_YELLOW if g["free_gb"] < 45 else C_GREEN)
        status_parts.append(f"GPU{g['index']}: {color}{g['free_gb']:.1f}/{g['total_gb']:.0f}GB Free{C_RESET}")
    print(f"📊 [{' | '.join(status_parts)}]")

    # Hiển thị trạng thái tác vụ
    trans_str = f"{C_GREEN}RUNNING{C_RESET}" if trans_running else f"{C_RED}STOPPED{C_RESET}"
    unet_str = f"{C_GREEN}RUNNING{C_RESET}" if unet_running else f"{C_RED}STOPPED{C_RESET}"
    print(f"🤖 Trạng thái tác vụ: [DP-Transformer: {trans_str} (cần >= {transformer_vram_min}GB)] | [DP-UNet: {unet_str} (cần >= {unet_vram_min}GB)]")

    if trans_running and unet_running:
        # Cả 2 đều đang chạy ổn định
        return trans_running, unet_running

    # 3. Phân bổ thông minh
    # Sắp xếp các GPU theo dung lượng VRAM trống giảm dần
    sorted_gpus = sorted(gpus, key=lambda x: x["free_gb"], reverse=True)
    allocated_gpus = set()

    # --- Bước A: Ưu tiên phân bổ cho Transformer nếu chưa chạy ---
    if not trans_running:
        for g in sorted_gpus:
            if g["index"] not in allocated_gpus and g["free_gb"] >= transformer_vram_min:
                log_msg(
                    f"{C_MAGENTA}{C_BOLD}🎯 PHÁT HIỆN GPU {g['index']} ĐẠT CHUẨN CHO TRANSFORMER ({g['free_gb']:.1f} GB >= {transformer_vram_min} GB)!{C_RESET}",
                    log_file,
                )
                success = start_training_in_tmux(
                    session_name="train_transformer",
                    script_path=TRANSFORMER_SCRIPT,
                    gpu_id=g["index"],
                    log_file=log_file,
                )
                if success:
                    allocated_gpus.add(g["index"])
                    trans_running = True
                    # Chờ 15s để process kịp phân bổ VRAM
                    time.sleep(15)
                break

    # --- Bước B: Phân bổ cho U-Net nếu chưa chạy ---
    if not unet_running:
        for g in sorted_gpus:
            if g["index"] not in allocated_gpus and g["free_gb"] >= unet_vram_min:
                log_msg(
                    f"{C_CYAN}{C_BOLD}🎯 PHÁT HIỆN GPU {g['index']} ĐẠT CHUẨN CHO U-NET ({g['free_gb']:.1f} GB >= {unet_vram_min} GB)!{C_RESET}",
                    log_file,
                )
                success = start_training_in_tmux(
                    session_name="train_unet",
                    script_path=UNET_SCRIPT,
                    gpu_id=g["index"],
                    log_file=log_file,
                )
                if success:
                    allocated_gpus.add(g["index"])
                    unet_running = True
                    # Chờ 15s để process kịp phân bổ VRAM
                    time.sleep(15)
                break

    # Nếu vẫn chưa có đủ GPU
    if not trans_running or not unet_running:
        waiting = []
        if not trans_running:
            waiting.append(f"Transformer (>={transformer_vram_min}GB)")
        if not unet_running:
            waiting.append(f"U-Net (>={unet_vram_min}GB)")
        print(f"⏳ Vẫn đang chờ GPU trống cho: {', '.join(waiting)}...")

    return trans_running, unet_running


def main():
    parser = argparse.ArgumentParser(
        description="Auto Resume Watcher: Tự động canh GPU trống VRAM và resume training Diffusion Policy",
    )
    parser.add_argument(
        "--transformer-vram",
        type=float,
        default=50.0,
        help="Dung lượng VRAM trống tối thiểu (GB) cho Transformer (mặc định: 50.0 GB)",
    )
    parser.add_argument(
        "--unet-vram",
        type=float,
        default=50.0,
        help="Dung lượng VRAM trống tối thiểu (GB) cho U-Net (mặc định: 50.0 GB)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Chu kỳ quét VRAM tính bằng giây (mặc định: 10 giây)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(DEFAULT_LOG_FILE),
        help=f"Đường dẫn lưu nhật ký hoạt động (mặc định: {DEFAULT_LOG_FILE.name})",
    )
    parser.add_argument(
        "--exit-when-done",
        action="store_true",
        help="Tự động thoát Watcher sau khi cả 2 tác vụ đều đã được kích hoạt thành công",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Chỉ kiểm tra 1 lần duy nhất rồi thoát",
    )

    args = parser.parse_args()
    log_path = Path(args.log_file)

    print(f"{C_BOLD}{C_CYAN}=============================================================================={C_RESET}")
    print(f"{C_BOLD}{C_CYAN}🛡️  AUTO RESUME WATCHER & DYNAMIC GPU ALLOCATOR ĐANG CHẠY{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}=============================================================================={C_RESET}")
    print(f"⚙️  Ngưỡng VRAM Transformer : >= {args.transformer_vram} GB -> Session: train_transformer")
    print(f"⚙️  Ngưỡng VRAM U-Net       : >= {args.unet_vram} GB -> Session: train_unet")
    print(f"⚙️  Chu kỳ quét            : {args.interval} giây")
    print(f"⚙️  Chế độ kết thúc        : {'Tự thoát khi cả 2 task chạy' if args.exit_when_done else 'Chạy thường trực bảo vệ 24/7'}")
    print(f"📂 Nhật ký lưu tại         : {log_path}")
    print(f"{C_BOLD}{C_CYAN}------------------------------------------------------------------------------{C_RESET}")

    if args.once:
        run_cycle(args.transformer_vram, args.unet_vram, log_path)
        return

    log_msg(f"🚀 Auto Resume Watcher bắt đầu vòng lặp giám sát (chu kỳ {args.interval}s)...", log_path)

    try:
        while True:
            trans_ok, unet_ok = run_cycle(args.transformer_vram, args.unet_vram, log_path)
            if args.exit_when_done and trans_ok and unet_ok:
                log_msg(
                    f"{C_GREEN}{C_BOLD}🎉 CẢ 2 TÁC VỤ (TRANSFORMER & U-NET) ĐÃ ĐƯỢC KÍCH HOẠT THÀNH CÔNG!{C_RESET}",
                    log_path,
                )
                log_msg("ℹ️ Đã hoàn thành sứ mệnh điều phối. Watcher tự động thoát (exit 0) theo cờ --exit-when-done.", log_path)
                sys.exit(0)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log_msg("🛑 Đã nhận tín hiệu dừng từ người dùng. Tạm biệt!", log_path)
        sys.exit(0)


if __name__ == "__main__":
    main()

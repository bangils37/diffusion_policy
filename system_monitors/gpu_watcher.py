#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CÔNG CỤ: GPU Watcher & Automated Dispatcher (Giám Sát VRAM & Tự Động Kích Hoạt)
Tác giả: AI Assistant / AnhNB9
Mục đích:
    Tự động theo dõi mức độ chiếm dụng VRAM của tất cả các GPU NVIDIA theo chu kỳ.
    Khi phát hiện có GPU đạt ngưỡng dung lượng VRAM trống theo yêu cầu (ví dụ >= 45 GB):
    1. Phát cảnh báo trực quan trên terminal (màu sắc nổi bật) và gửi thông báo Desktop (notify-send).
    2. Tự động kích hoạt câu lệnh hoặc script huấn luyện (ví dụ: resume training) tương ứng với GPU_ID vừa được giải phóng.

Nguyên lý hoạt động:
    - Sử dụng lệnh 'nvidia-smi --query-gpu=...' để lấy thông tin chính xác từng miligiây.
    - Quét VRAM Free, VRAM Used, VRAM Total, GPU Utilization (%) và Nhiệt độ (°C).
    - Hỗ trợ chạy liên tục (Daemon mode) hoặc kiểm tra nhanh 1 lần (Check-once mode).
    - Hỗ trợ tự động tạo session Tmux để chạy nền an toàn.

Các tham số dòng lệnh (CLI Arguments):
    --min-vram-gb     : Dung lượng VRAM trống tối thiểu (GB) cần để kích hoạt (Mặc định: 45.0 GB).
    --check-interval  : Chu kỳ quét lại trạng thái GPU (tính bằng giây) (Mặc định: 10 giây).
    --trigger-cmd     : Lệnh Bash tự động chạy khi tìm thấy GPU thỏa mãn điều kiện.
    --tmux-session    : Tên session Tmux sẽ tạo để chạy lệnh trigger (ví dụ: 'train_transformer').
    --target-gpus     : Giới hạn chỉ canh các GPU nhất định (ví dụ: '1,3'), để trống nếu canh tất cả.
    --log-file        : Đường dẫn lưu file log hoạt động (Mặc định: 'gpu_watcher.log').
    --once            : Chỉ kiểm tra 1 lần rồi thoát (Exit code 0 nếu có GPU đủ VRAM, 1 nếu không).

Ví dụ sử dụng thực tế:
    1. Chỉ theo dõi và in trạng thái ra màn hình mỗi 10 giây:
       python3 gpu_watcher.py --min-vram-gb 45

    2. Canh GPU 1 và GPU 3, khi có GPU trống >= 45GB thì TỰ ĐỘNG resume Transformer vào Tmux:
       python3 gpu_watcher.py --min-vram-gb 45 --target-gpus "1,3" \\
           --trigger-cmd "bash /home/anhnb9/Documents/diffusion_policy/resume_train_transformer.sh" \\
           --tmux-session "train_transformer"

    3. Kiểm tra nhanh xem hiện tại có GPU nào trống >= 50GB không:
       python3 gpu_watcher.py --min-vram-gb 50 --once
================================================================================
"""

import os
import sys
import time
import argparse
import subprocess
import datetime
from typing import List, Dict, Optional

# ANSI Colors phục vụ hiển thị trực quan
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_gpu_info() -> List[Dict]:
    """
    Truy vấn thông tin VRAM, tải và nhiệt độ của tất cả GPU qua nvidia-smi.
    Trả về danh sách dict chứa thông tin từng GPU.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        gpus = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_mb": float(parts[2]),
                    "free_mb": float(parts[3]),
                    "used_mb": float(parts[4]),
                    "util_percent": float(parts[5]),
                    "temp_c": float(parts[6]),
                    "free_gb": float(parts[3]) / 1024.0,
                    "total_gb": float(parts[2]) / 1024.0,
                    "used_gb": float(parts[4]) / 1024.0,
                })
        return gpus
    except Exception as e:
        print(f"{RED}[LỖI] Không thể truy vấn lệnh nvidia-smi: {e}{RESET}")
        return []


def send_notification(title: str, message: str):
    """Gửi thông báo dạng Desktop Notification nếu môi trường có biến DISPLAY."""
    if os.environ.get("DISPLAY"):
        try:
            subprocess.run(["notify-send", "-u", "critical", title, message], check=False)
        except Exception:
            pass


def log_event(log_file: Optional[str], text: str):
    """Ghi nhận sự kiện kèm timestamp ra màn hình console và file log."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {text}"
    print(line)
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="GPU Watcher: Tự động canh GPU trống VRAM và kích hoạt tiến trình huấn luyện",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ví dụ: python3 gpu_watcher.py --min-vram-gb 45 --trigger-cmd 'bash resume_train_transformer.sh'"
    )
    parser.add_argument("--min-vram-gb", type=float, default=45.0,
                        help="Dung lượng VRAM trống tối thiểu (GB) để kích hoạt (mặc định: 45 GB)")
    parser.add_argument("--check-interval", type=int, default=10,
                        help="Chu kỳ kiểm tra (giây) (mặc định: 10 giây)")
    parser.add_argument("--trigger-cmd", type=str, default="",
                        help="Lệnh bash tự động chạy khi có GPU đủ VRAM (ví dụ: 'bash resume_train_transformer.sh')")
    parser.add_argument("--tmux-session", type=str, default="",
                        help="Tên session Tmux để chạy trigger-cmd (ví dụ: 'train_transformer')")
    parser.add_argument("--target-gpus", type=str, default="",
                        help="Danh sách GPU muốn canh (ví dụ: '1,3' hoặc để trống để canh tất cả)")
    parser.add_argument("--log-file", type=str, default="gpu_watcher.log",
                        help="Tên file lưu nhật ký hoạt động (mặc định: gpu_watcher.log)")
    parser.add_argument("--once", action="store_true",
                        help="Chỉ kiểm tra 1 lần duy nhất rồi thoát (mã trả về 0 nếu có GPU đạt chuẩn, 1 nếu không)")

    args = parser.parse_args()

    target_gpus = [int(x.strip()) for x in args.target_gpus.split(",")] if args.target_gpus else []

    print(f"{BOLD}{CYAN}=============================================================================={RESET}")
    print(f"{BOLD}{CYAN}👁️  GPU WATCHER & RESOURCE SENTINEL (GIÁM SÁT VRAM VÀ ĐIỀU PHỐI TỰ ĐỘNG){RESET}")
    print(f"🎯 Ngưỡng VRAM trống yêu cầu : {BOLD}{GREEN}>= {args.min_vram_gb} GB{RESET}")
    print(f"⏱️  Chu kỳ quét               : {args.check_interval} giây")
    if target_gpus:
        print(f"🎯 Các GPU được chỉ định    : GPU {target_gpus}")
    else:
        print(f"🎯 Phạm vi theo dõi          : Tất cả các GPU hiện có trên hệ thống")
    if args.trigger_cmd:
        print(f"🚀 Lệnh tự động kích hoạt    : {YELLOW}{args.trigger_cmd}{RESET}")
        if args.tmux_session:
            print(f"🪟 Session Tmux tạo mới      : {args.tmux_session}")
    print(f"📝 File log ghi nhận         : {args.log_file}")
    print(f"{BOLD}{CYAN}=============================================================================={RESET}")

    iteration = 0
    while True:
        iteration += 1
        gpus = get_gpu_info()
        available_gpus = []

        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        status_line = []

        for gpu in gpus:
            idx = gpu["index"]
            if target_gpus and idx not in target_gpus:
                continue

            free_gb = gpu["free_gb"]
            total_gb = gpu["total_gb"]
            util = gpu["util_percent"]
            temp = gpu["temp_c"]

            is_ok = free_gb >= args.min_vram_gb
            color = GREEN if is_ok else RED

            status_line.append(f"GPU {idx}: {color}{free_gb:.1f}/{total_gb:.0f}GB trống ({util:.0f}%, {temp:.0f}°C){RESET}")

            if is_ok:
                available_gpus.append(gpu)

        print(f"\r[{now_str}] Quét lần #{iteration}: " + " | ".join(status_line), end="", flush=True)

        if available_gpus:
            # Chọn GPU có dung lượng VRAM trống lớn nhất
            best_gpu = max(available_gpus, key=lambda x: x["free_gb"])
            best_id = best_gpu["index"]
            best_free = best_gpu["free_gb"]

            print("\n")
            log_event(args.log_file, f"{GREEN}🎉 PHÁT HIỆN GPU ĐỦ ĐIỀU KIỆN: GPU {best_id} hiện có {best_free:.2f} GB VRAM trống (>= ngưỡng {args.min_vram_gb} GB){RESET}")
            send_notification(
                "GPU Watcher Thông Báo",
                f"GPU {best_id} hiện có {best_free:.1f} GB VRAM trống! Sẵn sàng kích hoạt tiến trình."
            )

            # Thực thi lệnh kích hoạt tự động nếu được thiết lập
            if args.trigger_cmd:
                log_event(args.log_file, f"🚀 Đang tiến hành kích hoạt lệnh trên GPU {best_id}...")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(best_id)
                env["GPU_ID"] = str(best_id)

                if args.tmux_session:
                    tmux_cmd = f"tmux new-session -d -s {args.tmux_session} 'GPU_ID={best_id} CUDA_VISIBLE_DEVICES={best_id} {args.trigger_cmd}'"
                    log_event(args.log_file, f"Lệnh Tmux thực thi: {tmux_cmd}")
                    subprocess.run(tmux_cmd, shell=True)
                else:
                    log_event(args.log_file, f"Lệnh thực thi trực tiếp: GPU_ID={best_id} {args.trigger_cmd}")
                    subprocess.Popen(f"GPU_ID={best_id} CUDA_VISIBLE_DEVICES={best_id} {args.trigger_cmd}", shell=True, env=env)

                log_event(args.log_file, "✅ Đã kích hoạt tiến trình huấn luyện thành công. Kết thúc phiên trực của GPU Watcher.")
                break

            if args.once:
                sys.exit(0)

        if args.once and not available_gpus:
            print("\n")
            sys.exit(1)

        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()

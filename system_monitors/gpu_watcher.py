#!/usr/bin/env python3
"""
GPU Watcher & Resource Sentinel
-------------------------------
Theo dõi mức chiếm dụng VRAM của các GPU theo thời gian thực.
Khi phát hiện GPU có VRAM trống đạt ngưỡng yêu cầu:
1. Báo động qua console / desktop notification.
2. Tự động kích hoạt lệnh/script training tương ứng với GPU_ID vừa được giải phóng.

Sử dụng:
    python3 gpu_watcher.py --min-vram-gb 45 --check-interval 10
    python3 gpu_watcher.py --min-vram-gb 45 --trigger-cmd "bash resume_train_transformer.sh"
"""

import os
import sys
import time
import argparse
import subprocess
import datetime
from typing import List, Dict, Optional

# ANSI Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_gpu_info() -> List[Dict]:
    """Truy vấn thông tin VRAM và tiến trình của tất cả GPU qua nvidia-smi."""
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
        print(f"{RED}[ERROR] Không thể truy vấn nvidia-smi: {e}{RESET}")
        return []


def get_gpu_processes(gpu_index: int) -> List[Dict]:
    """Lấy danh sách các tiến trình đang chiếm GPU."""
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        # fallback simple ps check
        return []
    except Exception:
        return []


def send_notification(title: str, message: str):
    """Gửi notification ra desktop nếu có DISPLAY."""
    if os.environ.get("DISPLAY"):
        try:
            subprocess.run(["notify-send", "-u", "critical", title, message], check=False)
        except FileNotFoundError:
            pass


def log_event(log_file: Optional[str], text: str):
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
    parser = argparse.ArgumentParser(description="GPU Watcher & Automated Dispatcher")
    parser.add_argument("--min-vram-gb", type=float, default=45.0,
                        help="Dung lượng VRAM trống tối thiểu (GB) để kích hoạt (mặc định: 45 GB)")
    parser.add_argument("--check-interval", type=int, default=10,
                        help="Khoảng thời gian kiểm tra (giây) (mặc định: 10s)")
    parser.add_argument("--trigger-cmd", type=str, default="",
                        help="Lệnh bash tự động thực thi khi có GPU đủ VRAM (ví dụ: 'bash resume_train_transformer.sh')")
    parser.add_argument("--tmux-session", type=str, default="",
                        help="Tên session tmux để chạy trigger-cmd (ví dụ: 'train_transformer')")
    parser.add_argument("--target-gpus", type=str, default="",
                        help="Danh sách GPU muốn canh (ví dụ: '1,3' hoặc để trống để canh tất cả)")
    parser.add_argument("--log-file", type=str, default="gpu_watcher.log",
                        help="File lưu log hoạt động")
    parser.add_argument("--once", action="store_true",
                        help="Chỉ kiểm tra 1 lần rồi thoát (exit 0 nếu có GPU, 1 nếu không)")

    args = parser.parse_args()

    target_gpus = [int(x.strip()) for x in args.target_gpus.split(",")] if args.target_gpus else []

    print(f"{BOLD}{CYAN}============================================================{RESET}")
    print(f"{BOLD}{CYAN}👁️  GPU WATCHER & RESOURCE SENTINEL ACTIVATED{RESET}")
    print(f"🎯 Ngưỡng VRAM trống yêu cầu: {BOLD}{GREEN}>= {args.min_vram_gb} GB{RESET}")
    print(f"⏱️  Chu kỳ quét: {args.check_interval}s")
    if target_gpus:
        print(f"🎯 Chỉ theo dõi GPU: {target_gpus}")
    if args.trigger_cmd:
        print(f"🚀 Lệnh tự động kích hoạt: {YELLOW}{args.trigger_cmd}{RESET}")
        if args.tmux_session:
            print(f"🪟 Tmux session: {args.tmux_session}")
    print(f"📝 Log file: {args.log_file}")
    print(f"{BOLD}{CYAN}============================================================{RESET}")

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

            status_line.append(f"GPU {idx}: {color}{free_gb:.1f}/{total_gb:.0f}GB free ({util:.0f}%, {temp:.0f}°C){RESET}")

            if is_ok:
                available_gpus.append(gpu)

        print(f"\r[{now_str}] #{iteration}: " + " | ".join(status_line), end="", flush=True)

        if available_gpus:
            best_gpu = max(available_gpus, key=lambda x: x["free_gb"])
            best_id = best_gpu["index"]
            best_free = best_gpu["free_gb"]

            print("\n")
            log_event(args.log_file, f"{GREEN}🎉 PHÁT HIỆN GPU TRỐNG: GPU {best_id} có {best_free:.2f} GB VRAM trống (>= {args.min_vram_gb} GB){RESET}")
            send_notification(
                "GPU Watcher Alert",
                f"GPU {best_id} hiện có {best_free:.1f} GB VRAM trống! Sẵn sàng kích hoạt training."
            )

            # Nếu có trigger command
            if args.trigger_cmd:
                log_event(args.log_file, f"🚀 Đang khởi động lệnh kích hoạt trên GPU {best_id}...")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(best_id)
                env["GPU_ID"] = str(best_id)

                if args.tmux_session:
                    tmux_cmd = f"tmux new-session -d -s {args.tmux_session} 'GPU_ID={best_id} CUDA_VISIBLE_DEVICES={best_id} {args.trigger_cmd}'"
                    log_event(args.log_file, f"Lệnh tmux: {tmux_cmd}")
                    subprocess.run(tmux_cmd, shell=True)
                else:
                    log_event(args.log_file, f"Thực thi trực tiếp: GPU_ID={best_id} {args.trigger_cmd}")
                    subprocess.Popen(f"GPU_ID={best_id} CUDA_VISIBLE_DEVICES={best_id} {args.trigger_cmd}", shell=True, env=env)

                log_event(args.log_file, "✅ Đã kích hoạt tiến trình thành công. Dừng GPU Watcher.")
                break

            if args.once:
                sys.exit(0)

        if args.once and not available_gpus:
            print("\n")
            sys.exit(1)

        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()

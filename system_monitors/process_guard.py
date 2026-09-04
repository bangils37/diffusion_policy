#!/usr/bin/env python3
"""
Process Guard & Forensic Sentinel
---------------------------------
Giám sát các tiến trình huấn luyện AI quan trọng (Diffusion Policy, VLA...).
Khi một tiến trình biến mất hoặc bị ngắt đột ngột:
1. Ghi nhận thời điểm kết thúc.
2. Tự động quét `/var/log/auth.log` qua sudo để xác định chính xác AI đã kill tiến trình (user, command, TTY).
3. Ghi báo cáo điều tra vào file log và gửi cảnh báo.

Sử dụng:
    python3 process_guard.py --patterns "train.py" "res_ssl"
    python3 process_guard.py --tmux-sessions "train_transformer" "train_cnn"
"""

import os
import sys
import time
import argparse
import subprocess
import datetime
import re
from typing import Dict, List, Optional

# ANSI Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_active_processes(patterns: List[str]) -> Dict[int, Dict]:
    """Tìm tất cả tiến trình khớp với patterns."""
    procs = {}
    try:
        res = subprocess.run(["ps", "-eo", "pid,user,lstart,cmd"], capture_output=True, text=True, check=True)
        for line in res.stdout.strip().splitlines()[1:]:
            parts = line.strip().split(maxsplit=6)
            if len(parts) >= 7:
                pid = int(parts[0])
                user = parts[1]
                cmd = parts[6]

                # Bỏ qua chính script này và lệnh grep
                if "process_guard.py" in cmd or "grep " in cmd:
                    continue

                for pat in patterns:
                    if pat.lower() in cmd.lower():
                        procs[pid] = {
                            "pid": pid,
                            "user": user,
                            "cmd": cmd,
                            "first_seen": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        break
    except Exception as e:
        print(f"{RED}[ERROR] Không thể lấy danh sách tiến trình: {e}{RESET}")
    return procs


def get_active_tmux_sessions() -> List[str]:
    """Lấy danh sách các session tmux hiện tại."""
    try:
        res = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
        if res.returncode == 0:
            sessions = []
            for line in res.stdout.strip().splitlines():
                if ":" in line:
                    sessions.append(line.split(":")[0].strip())
            return sessions
    except Exception:
        pass
    return []


def investigate_kill_in_auth_log(pid: int, event_time: datetime.datetime) -> Optional[str]:
    """Tra cứu trong /var/log/auth.log xem có ai vừa chạy sudo kill <pid> không."""
    try:
        # Tìm lệnh kill liên quan đến PID
        cmd = ["sudo", "-n", "grep", "-E", f"COMMAND=.*kill.*{pid}|COMMAND=.*reboot", "/var/log/auth.log"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            lines = res.stdout.strip().splitlines()
            return lines[-1]
    except Exception:
        pass
    return None


def send_alert(title: str, message: str):
    """Gửi cảnh báo qua notify-send nếu có GUI."""
    if os.environ.get("DISPLAY"):
        try:
            subprocess.run(["notify-send", "-u", "critical", title, message], check=False)
        except Exception:
            pass


def log_entry(log_file: str, message: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {message}"
    print(entry)
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Process Guard & Forensic Sentinel")
    parser.add_argument("--patterns", nargs="+", default=["train.py", "diffusion"],
                        help="Danh sách từ khóa tiến trình cần giám sát (ví dụ: 'train.py')")
    parser.add_argument("--tmux-sessions", nargs="+", default=["train_transformer", "train_cnn"],
                        help="Tên các session tmux cần theo dõi (ví dụ: 'train_transformer')")
    parser.add_argument("--interval", type=int, default=5,
                        help="Chu kỳ quét tiến trình (giây) (mặc định: 5s)")
    parser.add_argument("--log-file", type=str, default="process_guard.log",
                        help="File lưu log sự kiện giám sát")
    parser.add_argument("--auto-restart-cmd", type=str, default="",
                        help="Lệnh tự động phục hồi nếu tiến trình bị kill")

    args = parser.parse_args()

    print(f"{BOLD}{CYAN}============================================================{RESET}")
    print(f"{BOLD}{CYAN}🛡️  PROCESS GUARD & SENTINEL ACTIVATED{RESET}")
    print(f"🎯 Patterns giám sát: {YELLOW}{args.patterns}{RESET}")
    print(f"🎯 Tmux sessions giám sát: {YELLOW}{args.tmux_sessions}{RESET}")
    print(f"⏱️  Chu kỳ kiểm tra: {args.interval}s")
    print(f"📝 Log file: {args.log_file}")
    print(f"{BOLD}{CYAN}============================================================{RESET}")

    known_procs = get_active_processes(args.patterns)
    known_tmux = set(get_active_tmux_sessions())

    log_entry(args.log_file, f"Bắt đầu giám sát {len(known_procs)} tiến trình và {len(known_tmux)} session tmux.")
    for pid, info in known_procs.items():
        log_entry(args.log_file, f"  -> [PID {pid}] (User: {info['user']}): {info['cmd'][:80]}...")

    while True:
        time.sleep(args.interval)
        current_procs = get_active_processes(args.patterns)
        current_tmux = set(get_active_tmux_sessions())

        # 1. Kiểm tra các tiến trình bị biến mất
        dead_pids = set(known_procs.keys()) - set(current_procs.keys())
        for pid in dead_pids:
            proc_info = known_procs[pid]
            now = datetime.datetime.now()
            log_entry(args.log_file, f"{RED}{BOLD}🚨 CẢNH BÁO: Tiến trình [PID {pid}] đã bị DỪNG!{RESET}")
            log_entry(args.log_file, f"   Chi tiết lệnh: {proc_info['cmd'][:120]}")

            # Điều tra nguyên nhân
            investigation = investigate_kill_in_auth_log(pid, now)
            if investigation:
                log_entry(args.log_file, f"{YELLOW}🔍 KẾT QUẢ ĐIỀU TRA: Phát hiện lệnh can thiệp trong auth.log:{RESET}")
                log_entry(args.log_file, f"   👉 {investigation}")
                send_alert(
                    "PROCESS KILLED BY USER",
                    f"Tiến trình PID {pid} đã bị can thiệp!\nLog: {investigation}"
                )
            else:
                log_entry(args.log_file, "ℹ️ Không tìm thấy lệnh sudo kill cụ thể. Có thể do tiến trình tự hoàn tất, bị crash hoặc nhận SIGTERM từ session.")
                send_alert(
                    "Process Stopped",
                    f"Tiến trình PID {pid} ({proc_info['user']}) đã dừng."
                )

            del known_procs[pid]

        # 2. Kiểm tra các session tmux bị tắt
        dead_tmux = known_tmux - current_tmux
        for s in dead_tmux:
            if s in args.tmux_sessions:
                log_entry(args.log_file, f"{RED}⚠️ Session tmux '{s}' đã biến mất / bị tắt!{RESET}")

        # 3. Cập nhật các tiến trình mới xuất hiện
        new_pids = set(current_procs.keys()) - set(known_procs.keys())
        for pid in new_pids:
            known_procs[pid] = current_procs[pid]
            log_entry(args.log_file, f"{GREEN}➕ Phát hiện tiến trình mới [PID {pid}]: {current_procs[pid]['cmd'][:70]}...{RESET}")

        known_tmux = current_tmux


if __name__ == "__main__":
    main()

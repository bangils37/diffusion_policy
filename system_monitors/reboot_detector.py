#!/usr/bin/env python3
"""
Reboot Detector & System Uptime Sentinel
----------------------------------------
Tự động phát hiện khi máy chủ bị khởi động lại (reboot).
Tra cứu auth.log và wtmp để tìm ra chính xác thời điểm và người thực thi lệnh reboot.

Sử dụng:
    python3 reboot_detector.py
    python3 reboot_detector.py --daemon --check-interval 60
"""

import os
import sys
import json
import time
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import Optional, Dict

STATE_FILE = Path(__file__).parent / ".boot_state.json"
LOG_FILE = Path(__file__).parent / "reboot_history.log"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_boot_time() -> str:
    """Lấy thời điểm khởi động gần nhất của hệ thống."""
    try:
        res = subprocess.run(["who", "-b"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        try:
            with open("/proc/uptime", "r") as f:
                uptime_sec = float(f.read().split()[0])
            boot_dt = datetime.datetime.now() - datetime.timedelta(seconds=uptime_sec)
            return boot_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "unknown"


def find_reboot_actor() -> Optional[str]:
    """Tìm ai đã chạy lệnh reboot gần nhất trong auth.log."""
    try:
        cmd = ["sudo", "-n", "grep", "-E", "COMMAND=.*reboot|systemd-logind.*reboot", "/var/log/auth.log"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            lines = res.stdout.strip().splitlines()
            return lines[-1]
    except Exception:
        pass
    return None


def get_reboot_history(n=5) -> str:
    """Lấy lịch sử các lần reboot gần nhất."""
    try:
        res = subprocess.run(["last", "reboot", f"-n{n}"], capture_output=True, text=True)
        return res.stdout.strip()
    except Exception:
        return "Không thể lấy lịch sử reboot."


def log_event(text: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {text}"
    print(entry)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass


def check_and_record():
    current_boot = get_boot_time()
    last_boot = None

    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                last_boot = data.get("last_boot")
        except Exception:
            pass

    if last_boot is None:
        # Lần đầu tiên chạy
        log_event(f"ℹ️ Khởi tạo ghi nhận boot time lần đầu: {current_boot}")
        with open(STATE_FILE, "w") as f:
            json.dump({"last_boot": current_boot, "updated_at": str(datetime.datetime.now())}, f)
        return False
    elif current_boot != last_boot:
        # PHÁT HIỆN REBOOT MỚI!
        actor = find_reboot_actor()
        log_event(f"{RED}{BOLD}🚨 PHÁT HIỆN MÁY CHỦ VỪA BỊ REBOOT!{RESET}")
        log_event(f"   Thời điểm boot mới: {current_boot} (cũ: {last_boot})")
        if actor:
            log_event(f"   👉 Nhật ký lệnh gây reboot: {YELLOW}{actor}{RESET}")
        else:
            log_event("   ℹ️ Không tìm thấy lệnh sudo reboot trong auth.log (có thể reboot cứng / mất điện / kernel panic).")

        # Cập nhật state
        with open(STATE_FILE, "w") as f:
            json.dump({"last_boot": current_boot, "updated_at": str(datetime.datetime.now()), "actor": actor}, f)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Reboot Detector & System Uptime Sentinel")
    parser.add_argument("--daemon", action="store_true", help="Chạy ở chế độ daemon theo dõi liên tục")
    parser.add_argument("--check-interval", type=int, default=60, help="Chu kỳ kiểm tra (giây) khi chạy daemon")
    parser.add_argument("--history", action="store_true", help="Xem lịch sử các lần reboot gần nhất")

    args = parser.parse_args()

    if args.history:
        print(f"{BOLD}{CYAN}=== LỊCH SỬ REBOOT CỦA SERVER ==={RESET}")
        print(get_reboot_history(10))
        actor = find_reboot_actor()
        if actor:
            print(f"\n{YELLOW}Lệnh reboot gần nhất trong auth.log:{RESET}\n{actor}")
        return

    if args.daemon:
        print(f"{BOLD}{CYAN}🔄 Reboot Detector Daemon đang chạy (chu kỳ {args.check_interval}s)...{RESET}")
        check_and_record()
        while True:
            time.sleep(args.check_interval)
            check_and_record()
    else:
        # Chạy 1 lần
        rebooted = check_and_record()
        if not rebooted:
            print(f"{GREEN}✅ Hệ thống đang hoạt động ổn định. Uptime boot: {get_boot_time()}{RESET}")


if __name__ == "__main__":
    main()

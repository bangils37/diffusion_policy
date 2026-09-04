#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CÔNG CỤ: Reboot Detector & System Uptime Sentinel (Phát Hiện Máy Chủ Khởi Động Lại & Điều Tra Nguyên Nhân)
Tác giả: AI Assistant / AnhNB9
Mục đích:
    Tự động ghi nhận và phát hiện sự kiện máy chủ (ai-server-1) bị reboot / khởi động lại.
    Khi máy chủ vừa khởi động lại hoặc kiểm tra định kỳ:
    1. So sánh mốc thời gian boot hiện tại với mốc thời gian lưu trong cache (.boot_state.json).
    2. Nếu phát hiện thời điểm boot mới (tức server vừa bị reboot):
       - Tự động quét /var/log/auth.log qua sudo để tìm xem ai (User/UID) đã gõ lệnh 'sudo reboot'
         hoặc 'shutdown' và thời điểm gõ lệnh.
       - Phân biệt giữa reboot chủ động bằng lệnh sudo và reboot do mất điện/kernel panic.
       - Ghi sự kiện chi tiết vào file lịch sử 'reboot_history.log' để tra cứu hậu kiểm.
    3. Hỗ trợ xem nhanh lịch sử 10 lần reboot gần nhất cùng nguyên nhân thông qua cờ --history.

Nguyên lý hoạt động:
    - Sử dụng lệnh 'who -b' hoặc đọc trực tiếp '/proc/uptime' để tính toán thời điểm boot chính xác.
    - Lưu trạng thái boot lần gần nhất vào file ẩn '.boot_state.json'.
    - Lọc nhật ký /var/log/auth.log tìm các mẫu log: 'COMMAND=.*reboot', 'COMMAND=.*shutdown',
      hoặc log từ 'systemd-logind'.

Các tham số dòng lệnh (CLI Arguments):
    --daemon          : Chạy ở chế độ daemon theo dõi liên tục định kỳ trong nền.
    --check-interval  : Chu kỳ kiểm tra (giây) khi chạy daemon (Mặc định: 60 giây).
    --history         : Hiển thị danh sách lịch sử các lần reboot gần nhất kèm log lệnh gây reboot.

Ví dụ sử dụng thực tế:
    1. Kiểm tra nhanh trạng thái boot hiện tại (chạy 1 lần):
       python3 reboot_detector.py

    2. Xem lịch sử các lần reboot trước đây và người thực thi:
       python3 reboot_detector.py --history

    3. Chạy thường trực dưới dạng daemon kiểm tra mỗi 30 giây:
       python3 reboot_detector.py --daemon --check-interval 30
================================================================================
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
    """
    Lấy thời điểm khởi động gần nhất (boot time) của hệ điều hành.

    Returns:
        str: Chuỗi thời gian boot (ví dụ: '2026-09-04 17:44') hoặc 'unknown' nếu gặp lỗi.
    """
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
    """
    Tìm kiếm thông tin người đã chạy lệnh reboot gần nhất trong nhật ký bảo mật /var/log/auth.log.

    Sử dụng lệnh 'sudo -n grep ...' để trích xuất dòng lệnh chứa 'COMMAND=.*reboot' hoặc 'systemd-logind'.

    Returns:
        Optional[str]: Dòng log chi tiết chứa user, UID, TTY và lệnh reboot, hoặc None nếu không tìm thấy.
    """
    try:
        cmd = ["sudo", "-n", "grep", "-E", "COMMAND=.*reboot|systemd-logind.*reboot", "/var/log/auth.log"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            lines = res.stdout.strip().splitlines()
            return lines[-1]
    except Exception:
        pass
    return None


def get_reboot_history(n: int = 5) -> str:
    """
    Trích xuất lịch sử n lần khởi động lại gần nhất của hệ thống thông qua lệnh 'last reboot'.

    Args:
        n (int): Số lượng bản ghi reboot cần lấy (mặc định: 5).

    Returns:
        str: Chuỗi văn bản định dạng bảng lịch sử reboot từ wtmp.
    """
    try:
        res = subprocess.run(["last", "reboot", f"-n{n}"], capture_output=True, text=True)
        return res.stdout.strip()
    except Exception:
        return "Không thể lấy lịch sử reboot."


def log_event(text: str):
    """
    Ghi nhật ký sự kiện kèm dấu thời gian ra màn hình Console và lưu vào file reboot_history.log.

    Args:
        text (str): Nội dung thông điệp sự kiện cần ghi nhận.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {text}"
    print(entry)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass


def check_and_record() -> bool:
    """
    Kiểm tra trạng thái boot hiện tại so với bản ghi trước đó để phát hiện máy chủ vừa khởi động lại.

    Quy trình:
    1. Lấy mốc thời gian boot hiện tại của máy chủ.
    2. Đọc mốc boot trước đó từ file cache .boot_state.json.
    3. Nếu chưa có cache: Khởi tạo cache ban đầu.
    4. Nếu mốc boot thay đổi (phát hiện reboot mới):
       - Gọi hàm find_reboot_actor() tra cứu /var/log/auth.log.
       - Xuất cảnh báo đỏ trên console và ghi chi tiết vào reboot_history.log.
       - Cập nhật mốc boot mới vào .boot_state.json.

    Returns:
        bool: True nếu phát hiện vừa có sự kiện reboot mới, False nếu hệ thống vẫn chạy bình thường.
    """
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
    """
    Hàm thực thi chính của công cụ Reboot Detector.
    Xử lý các đối số dòng lệnh (--daemon, --history, --check-interval) và điều phối luồng thực thi.
    """
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

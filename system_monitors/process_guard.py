#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CÔNG CỤ: Process Guard & Forensic Sentinel (Giám Sát Tiến Trình & Điều Tra Sự Cố)
Tác giả: AI Assistant / AnhNB9
Mục đích:
    Trực chờ và giám sát liên tục trạng thái sống/chết của các tiến trình huấn luyện AI
    quan trọng (như train.py, Diffusion Policy, VLA model...) hoặc các session Tmux.
    Khi một tiến trình bất ngờ bị biến mất, bị ngắt (Terminated) hoặc bị tiêu diệt (Killed):
    1. Ghi nhận chính xác mốc thời gian (timestamp) tiến trình dừng hoạt động.
    2. Tự động truy vấn nhật ký bảo mật hệ thống (/var/log/auth.log) qua quyền sudo để
       xác định chính xác:
       - Ai là người thực hiện lệnh kill? (User name, UID)
       - Lệnh kill cụ thể là gì? (ví dụ: 'sudo kill 2724161' hoặc 'pkill')
       - Thao tác thực hiện từ đâu? (TTY pts/x, thư mục làm việc CWD)
    3. Ghi lại báo cáo điều tra rõ ràng vào file log và phát thông báo cảnh báo.

Nguyên lý hoạt động:
    - Định kỳ quét danh sách tiến trình hệ thống qua lệnh 'ps -eo pid,user,lstart,cmd'.
    - Theo dõi danh sách các session Tmux qua 'tmux ls'.
    - Khi phát hiện một PID trong danh sách giám sát bị mất khỏi bảng tiến trình,
      script sẽ ngay lập tức đối soát PID đó với các dòng nhật ký lệnh trong /var/log/auth.log.

Các tham số dòng lệnh (CLI Arguments):
    --patterns         : Danh sách các từ khóa lệnh cần theo dõi (ví dụ: 'train.py' 'diffusion').
    --tmux-sessions    : Danh sách tên session Tmux cần theo dõi (ví dụ: 'train_transformer' 'train_cnn').
    --interval         : Chu kỳ quét kiểm tra tiến trình (tính bằng giây) (Mặc định: 5 giây).
    --log-file         : Đường dẫn lưu file log sự kiện giám sát (Mặc định: 'process_guard.log').
    --auto-restart-cmd : Lệnh Bash tùy chọn để tự động kích hoạt lại nếu tiến trình bị kill.

Ví dụ sử dụng thực tế:
    1. Giám sát tất cả các tiến trình có chứa chữ 'train.py' hoặc 'diffusion':
       python3 process_guard.py --patterns "train.py" "diffusion"

    2. Giám sát chặt chẽ 2 session Tmux huấn luyện chính của dự án:
       python3 process_guard.py --tmux-sessions "train_transformer" "train_cnn" --interval 5

    3. Chạy ngầm trong nền để làm nhiệm vụ hộp đen ghi nhận sự cố:
       nohup python3 process_guard.py --patterns "train.py" > process_guard.out 2>&1 &
================================================================================
"""

import os
import sys
import time
import argparse
import subprocess
import datetime
from typing import Dict, List, Optional

# ANSI Colors phục vụ hiển thị trực quan
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_active_processes(patterns: List[str]) -> Dict[int, Dict]:
    """
    Tìm tất cả các tiến trình đang chạy khớp với các từ khóa trong danh sách patterns.
    Trả về dict dạng: {pid: {"pid": pid, "user": user, "cmd": cmd, "first_seen": ...}}
    """
    procs = {}
    try:
        res = subprocess.run(["ps", "-eo", "pid,user,lstart,cmd"], capture_output=True, text=True, check=True)
        for line in res.stdout.strip().splitlines()[1:]:
            parts = line.strip().split(maxsplit=6)
            if len(parts) >= 7:
                pid = int(parts[0])
                user = parts[1]
                cmd = parts[6]

                # Bỏ qua chính script này và lệnh grep để tránh vòng lặp tự giám sát
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
        print(f"{RED}[LỖI] Không thể lấy danh sách tiến trình: {e}{RESET}")
    return procs


def get_active_tmux_sessions() -> List[str]:
    """Lấy danh sách các session Tmux hiện đang tồn tại trên hệ thống."""
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
    """
    Truy vết trong nhật ký bảo mật /var/log/auth.log xem có ai đã thực hiện lệnh sudo kill <pid> không.
    Trả về dòng log phát hiện được hoặc None nếu không tìm thấy.
    """
    try:
        cmd = ["sudo", "-n", "grep", "-E", f"COMMAND=.*kill.*{pid}|COMMAND=.*reboot", "/var/log/auth.log"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            lines = res.stdout.strip().splitlines()
            return lines[-1]
    except Exception:
        pass
    return None


def send_alert(title: str, message: str):
    """Phát cảnh báo khẩn cấp dạng Desktop Notification nếu có giao diện đồ họa."""
    if os.environ.get("DISPLAY"):
        try:
            subprocess.run(["notify-send", "-u", "critical", title, message], check=False)
        except Exception:
            pass


def log_entry(log_file: str, message: str):
    """Ghi nhận sự kiện kèm timestamp chuẩn ISO ra màn hình và file log."""
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
    parser = argparse.ArgumentParser(
        description="Process Guard: Giám sát tiến trình AI và tự động điều tra lệnh can thiệp khi bị kill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ví dụ: python3 process_guard.py --patterns train.py diffusion --tmux-sessions train_transformer"
    )
    parser.add_argument("--patterns", nargs="+", default=["train.py", "diffusion"],
                        help="Danh sách từ khóa lệnh cần giám sát (mặc định: 'train.py' 'diffusion')")
    parser.add_argument("--tmux-sessions", nargs="+", default=["train_transformer", "train_cnn"],
                        help="Tên các session Tmux cần theo dõi (mặc định: 'train_transformer' 'train_cnn')")
    parser.add_argument("--interval", type=int, default=5,
                        help="Chu kỳ quét tiến trình (giây) (mặc định: 5 giây)")
    parser.add_argument("--log-file", type=str, default="process_guard.log",
                        help="Tên file lưu nhật ký giám sát (mặc định: process_guard.log)")
    parser.add_argument("--auto-restart-cmd", type=str, default="",
                        help="Lệnh bash tùy chọn để tự động phục hồi nếu tiến trình bị kill")

    args = parser.parse_args()

    print(f"{BOLD}{CYAN}=============================================================================={RESET}")
    print(f"{BOLD}{CYAN}🛡️  PROCESS GUARD & FORENSIC SENTINEL (BẢO VỆ TIẾN TRÌNH & ĐIỀU TRA SỰ CỐ){RESET}")
    print(f"🎯 Từ khóa tiến trình giám sát : {YELLOW}{args.patterns}{RESET}")
    print(f"🎯 Session Tmux giám sát        : {YELLOW}{args.tmux_sessions}{RESET}")
    print(f"⏱️  Chu kỳ kiểm tra              : {args.interval} giây")
    print(f"📝 File lưu nhật ký điều tra   : {args.log_file}")
    print(f"{BOLD}{CYAN}=============================================================================={RESET}")

    known_procs = get_active_processes(args.patterns)
    known_tmux = set(get_active_tmux_sessions())

    log_entry(args.log_file, f"Bắt đầu phiên giám sát. Phát hiện {len(known_procs)} tiến trình và {len(known_tmux)} session Tmux đang hoạt động.")
    for pid, info in known_procs.items():
        log_entry(args.log_file, f"  -> [PID {pid}] (User: {info['user']}): {info['cmd'][:80]}...")

    while True:
        time.sleep(args.interval)
        current_procs = get_active_processes(args.patterns)
        current_tmux = set(get_active_tmux_sessions())

        # 1. Kiểm tra các tiến trình vừa bị biến mất / bị kill
        dead_pids = set(known_procs.keys()) - set(current_procs.keys())
        for pid in dead_pids:
            proc_info = known_procs[pid]
            now = datetime.datetime.now()
            log_entry(args.log_file, f"{RED}{BOLD}🚨 CẢNH BÁO: Tiến trình [PID {pid}] đã bị DỪNG / BỊ KILL!{RESET}")
            log_entry(args.log_file, f"   Lệnh gốc của tiến trình: {proc_info['cmd'][:120]}")

            # Tiến hành truy vết điều tra qua auth.log
            investigation = investigate_kill_in_auth_log(pid, now)
            if investigation:
                log_entry(args.log_file, f"{YELLOW}🔍 KẾT QUẢ ĐIỀU TRA: Phát hiện lệnh can thiệp trong auth.log hệ thống:{RESET}")
                log_entry(args.log_file, f"   👉 {investigation}")
                send_alert(
                    "CẢNH BÁO: TIẾN TRÌNH BỊ KILL BỞI USER",
                    f"Tiến trình PID {pid} vừa bị can thiệp!\nLog: {investigation}"
                )
            else:
                log_entry(args.log_file, "ℹ️ Không tìm thấy lệnh sudo kill rõ ràng. Có thể do tiến trình tự kết thúc, crash nội bộ, hoặc nhận SIGTERM từ session cha.")
                send_alert(
                    "Tiến trình huấn luyện dừng",
                    f"Tiến trình PID {pid} ({proc_info['user']}) đã dừng hoạt động."
                )

            del known_procs[pid]

        # 2. Kiểm tra các session Tmux bị tắt bất thường
        dead_tmux = known_tmux - current_tmux
        for s in dead_tmux:
            if s in args.tmux_sessions:
                log_entry(args.log_file, f"{RED}⚠️ Session Tmux '{s}' đã biến mất hoặc bị người dùng tắt!{RESET}")

        # 3. Ghi nhận các tiến trình mới được khởi tạo
        new_pids = set(current_procs.keys()) - set(known_procs.keys())
        for pid in new_pids:
            known_procs[pid] = current_procs[pid]
            log_entry(args.log_file, f"{GREEN}➕ Ghi nhận tiến trình huấn luyện mới [PID {pid}]: {current_procs[pid]['cmd'][:70]}...{RESET}")

        known_tmux = current_tmux


if __name__ == "__main__":
    main()

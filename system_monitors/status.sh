#!/usr/bin/env bash
# ==============================================================================
# 📊 System Monitors Status & Health Dashboard
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_RED="\033[91m"
C_GREEN="\033[92m"
C_YELLOW="\033[93m"
C_BLUE="\033[94m"
C_CYAN="\033[96m"

clear
echo -e "${C_BOLD}${C_CYAN}==============================================================================${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}🛡️  AI SERVER SYSTEM & TRAINING HEALTH DASHBOARD${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}==============================================================================${C_RESET}"

# 1. Server Uptime & Hostname
HOSTNAME=$(hostname)
UPTIME=$(uptime -p 2>/dev/null || uptime)
BOOT_TIME=$(who -b 2>/dev/null | awk '{print $3, $4}')
echo -e "🖥️  ${C_BOLD}Máy chủ:${C_RESET} ${C_YELLOW}$HOSTNAME${C_RESET} (ai-server-1) | ${C_BOLD}Boot:${C_RESET} $BOOT_TIME | ${C_BOLD}Uptime:${C_RESET} $UPTIME"
echo ""

# 2. GPU Status
echo -e "${C_BOLD}${C_BLUE}--- 🎮 GPU VRAM & TIẾN TRÌNH CHIẾM DỤNG ---${C_RESET}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits | while IFS=',' read -r idx name used free total util temp; do
        idx=$(echo "$idx" | tr -d ' ')
        name=$(echo "$name" | xargs)
        used=$(echo "$used" | tr -d ' ')
        free=$(echo "$free" | tr -d ' ')
        total=$(echo "$total" | tr -d ' ')
        util=$(echo "$util" | tr -d ' ')
        temp=$(echo "$temp" | tr -d ' ')

        free_gb=$(awk "BEGIN {printf \"%.1f\", $free/1024}")
        total_gb=$(awk "BEGIN {printf \"%.0f\", $total/1024}")

        color=$C_RED
        if (( free > 40000 )); then
            color=$C_GREEN
        elif (( free > 15000 )); then
            color=$C_YELLOW
        fi

        echo -e "  GPU $idx (${name}): ${color}${C_BOLD}${free_gb}/${total_gb} GB Free${C_RESET} (Load: ${util}%, ${temp}°C)"
    done
else
    echo "  nvidia-smi không khả dụng."
fi
echo ""

# 3. Active AI Training Tmux Sessions
echo -e "${C_BOLD}${C_BLUE}--- 🤖 TIẾN TRÌNH TRAINING ĐANG CHẠY (TMUX) ---${C_RESET}"
if command -v tmux &> /dev/null; then
    SESSIONS=$(tmux ls 2>/dev/null)
    if [ -n "$SESSIONS" ]; then
        echo "$SESSIONS" | while read -r line; do
            s_name=$(echo "$line" | cut -d':' -f1)
            echo -e "  ${C_GREEN}● [Active Session]${C_RESET} ${C_BOLD}$s_name${C_RESET} ($line)"
        done
    else
        echo -e "  ${C_YELLOW}Không có tmux session training nào đang chạy.${C_RESET}"
    fi
fi
echo ""

# 4. Monitoring Daemons Status
echo -e "${C_BOLD}${C_BLUE}--- 👁️  TRẠNG THÁI CÁC ỨNG DỤNG GIÁM SÁT ---${C_RESET}"
check_daemon() {
    local name="$1"
    local pattern="$2"
    local pid
    pid=$(pgrep -f "$pattern" 2>/dev/null | head -n 1)
    if [ -n "$pid" ]; then
        echo -e "  $name: ${C_GREEN}● RUNNING${C_RESET} (PID: $pid)"
    else
        echo -e "  $name: ${C_RED}○ STOPPED${C_RESET}"
    fi
}

check_daemon "GPU Watcher      " "gpu_watcher.py"
check_daemon "Process Guard    " "process_guard.py"
check_daemon "W&B Auto-Sync    " "wandb_sync_daemon.sh"
check_daemon "Reboot Detector  " "reboot_detector.py"
echo ""

# 5. Recent Alerts / Logs
echo -e "${C_BOLD}${C_BLUE}--- 🚨 NHẬT KÝ CẢNH BÁO GẦN NHẤT ---${C_RESET}"
if [ -f "$SCRIPT_DIR/process_guard.log" ]; then
    echo -e "${C_BOLD}[process_guard.log]${C_RESET}"
    tail -n 3 "$SCRIPT_DIR/process_guard.log" | sed 's/^/  /'
else
    echo "  Chưa có sự kiện process_guard."
fi

if [ -f "$SCRIPT_DIR/reboot_history.log" ]; then
    echo -e "${C_BOLD}[reboot_history.log]${C_RESET}"
    tail -n 2 "$SCRIPT_DIR/reboot_history.log" | sed 's/^/  /'
fi

echo -e "${C_BOLD}${C_CYAN}==============================================================================${C_RESET}"

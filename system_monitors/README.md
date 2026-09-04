# 🛡️ AI Server System Monitors & Watchdogs Suite

Bộ công cụ trực chờ, giám sát tài nguyên phần cứng (GPU/RAM), tiến trình huấn luyện AI (Process Supervisor), tự động đồng bộ Weights & Biases và điều tra nguyên nhân sự cố hệ thống (Reboot/Kill Sentinel).

Thư mục chính:
* Workspace: `/home/anhnb9/Documents/diffusion_policy/system_monitors`
* Thư mục hệ thống: `/home/anhnb9/Documents/system_monitors`

---

## 🚀 Cách sử dụng nhanh

### 1. Bật toàn bộ hệ thống giám sát chạy ngầm 24/7:
```bash
cd /home/anhnb9/Documents/system_monitors
./start_all_monitors.sh
```
*Hệ thống sẽ tạo 1 session tmux mang tên `sys_monitors` quản lý tất cả các daemon.*

### 2. Xem Dashboard trạng thái tức thì (1 lệnh):
```bash
./status.sh
```
*Hiển thị trực quan: Uptime, VRAM 4 GPU, tiến trình training đang chạy, trạng thái các daemon và cảnh báo gần nhất.*

### 3. Xem màn hình các daemon đang chạy:
```bash
tmux attach -t sys_monitors
```
* Điều hướng: Bấm `Ctrl + B` rồi bấm phím số:
  * `0`: Màn hình **GPU Watcher**
  * `1`: Màn hình **Process Guard**
  * `2`: Màn hình **W&B Auto-Sync**
* Rời khỏi màn hình mà không làm dừng: Bấm `Ctrl + B` rồi bấm `D`.

### 4. Dừng toàn bộ các ứng dụng giám sát:
```bash
./stop_all_monitors.sh
```

---

## 🛠️ Chi tiết các ứng dụng chuyên biệt

### 1. `gpu_watcher.py` (Canh GPU trống & Tự động bật Training)
* **Tính năng**: Quét VRAM của cả 4 GPU theo chu kỳ. Khi phát hiện GPU có VRAM trống đạt ngưỡng (ví dụ: $\ge 45\text{ GB}$), nó sẽ lập tức báo động và có thể **tự động kích hoạt lệnh training**:
* **Ví dụ**:
  ```bash
  # Canh khi nào có GPU trống >= 45 GB VRAM:
  python3 gpu_watcher.py --min-vram-gb 45 --check-interval 10

  # Canh và TỰ ĐỘNG bật script training ngay khi có GPU trống:
  python3 gpu_watcher.py --min-vram-gb 45 \
      --trigger-cmd "bash /home/anhnb9/Documents/diffusion_policy/resume_train_transformer.sh" \
      --tmux-session "train_transformer"
  ```

---

### 2. `process_guard.py` (Giám sát Tiến trình & Điều tra Forensics)
* **Tính năng**: Giám sát các PID huấn luyện AI (`train.py`, `diffusion`).
* Khi phát hiện một tiến trình bị ngắt (`Terminated` / `Killed`), script sẽ **tự động đọc `/var/log/auth.log` qua sudo** để tìm ra:
  * Ai là người đã chạy lệnh kill?
  * Lệnh kill chính xác là gì?
  * Thời điểm và TTY thực thi.
* **Ví dụ**:
  ```bash
  python3 process_guard.py --patterns train.py diffusion --interval 5
  ```

---

### 3. `wandb_sync_daemon.sh` (Tự động đồng bộ W&B định kỳ)
* **Tính năng**: Định kỳ mỗi 5 phút quét qua các thư mục log `.wandb` và thực hiện đồng bộ an toàn bằng cờ `--legacy` (loại bỏ hoàn toàn lỗi `transactionlog: unexpected EOF`).
* Đảm bảo dashboard W&B luôn có dữ liệu mới nhất ngay cả khi tiến trình training bị chập chờn mạng.

---

### 4. `reboot_detector.py` (Phát hiện & Lưu vết Server Reboot)
* **Tính năng**: Phát hiện máy chủ bị khởi động lại đột ngột.
* Tra cứu và ghi lại ai là người đã thực hiện lệnh `sudo reboot` và thời điểm diễn ra sự cố.
* Xem lịch sử reboot:
  ```bash
  python3 reboot_detector.py --history
  ```

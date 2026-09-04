# 🛡️ AI Server System Monitors & Watchdogs Suite

Bộ công cụ trực chờ, giám sát tài nguyên phần cứng, bảo vệ tiến trình huấn luyện AI (Process Supervisor), tự động đồng bộ số liệu Weights & Biases (W&B) và điều tra nguyên nhân sự cố hệ thống (Reboot / Kill Sentinel) trên máy chủ AI (`ai-server-1` / `VMO-SENTK1-D3`).

---

## 📂 Vị trí thư mục

Bộ công cụ được triển khai song song tại:
* **Thư mục ứng dụng hệ thống**: `/home/anhnb9/Documents/system_monitors/`
* **Thư mục workspace dự án**: `/home/anhnb9/Documents/diffusion_policy/system_monitors/`

---

## 📋 Danh mục và chức năng chi tiết của từng Script

| STT | Tên Script | Ngôn ngữ | Chức năng chính |
| :--- | :--- | :--- | :--- |
| 1 | [`auto_resume_watcher.py`](file:///home/anhnb9/Documents/diffusion_policy/system_monitors/auto_resume_watcher.py) | Python 3 | **Trực chờ VRAM động 2 tác vụ**: Tự động kích hoạt resume DP U-Net khi rảnh $\ge 30\text{ GB}$ và DP Transformer khi rảnh $\ge 60\text{ GB}$. |
| 2 | [`start_auto_resume_watcher.sh`](file:///home/anhnb9/Documents/diffusion_policy/system_monitors/start_auto_resume_watcher.sh) | Bash | Kích hoạt `auto_resume_watcher.py` chạy ngầm 24/7 trong phiên Tmux `auto_resume_watcher`. |
| 3 | [`stop_auto_resume_watcher.sh`](file:///home/anhnb9/Documents/diffusion_policy/system_monitors/stop_auto_resume_watcher.sh) | Bash | Dừng phiên Tmux `auto_resume_watcher` an toàn. |
| 4 | [`gpu_watcher.py`](file:///home/anhnb9/Documents/diffusion_policy/system_monitors/gpu_watcher.py) | Python 3 | Giám sát mức chiếm dụng VRAM của 4 card GPU RTX PRO 6000, phát hiện GPU rảnh và tự động kích hoạt script training khi đủ VRAM. |
| 5 | [`process_guard.py`](file:///home/anhnb9/Documents/diffusion_policy/system_monitors/process_guard.py) | Python 3 | Hộp đen giám sát tiến trình AI; khi tiến trình bị tắt/kill, tự động truy vấn `/var/log/auth.log` qua sudo để xác định danh tính người đã kill. |
| 6 | [`reboot_detector.py`](file:///home/anhnb9/Documents/diffusion_policy/system_monitors/reboot_detector.py) | Python 3 | Phát hiện máy chủ bị khởi động lại đột ngột; tra cứu nhật ký hệ thống để tìm ra ai đã thực thi lệnh reboot/shutdown. |
| 7 | [`wandb_sync_daemon.sh`](file:///home/anhnb9/Documents/diffusion_policy/system_monitors/wandb_sync_daemon.sh) | Bash | Daemon chạy ngầm đồng bộ định kỳ log offline lên Weights & Biases bằng cờ `--legacy` an toàn, loại bỏ triệt để lỗi EOF. |
| 8 | [`status.sh`](file:///home/anhnb9/Documents/diffusion_policy/system_monitors/status.sh) | Bash | Dashboard dòng lệnh hiển thị tổng quan tức thì: Uptime, VRAM 4 GPU, phiên Tmux huấn luyện, trạng thái các daemon và cảnh báo mới nhất. |
| 9 | [`start_all_monitors.sh`](file:///home/anhnb9/Documents/diffusion_policy/system_monitors/start_all_monitors.sh) | Bash | Script 1 chạm khởi chạy trọn bộ công cụ giám sát vào trong 1 phiên Tmux độc lập (`sys_monitors`) để hoạt động ngầm 24/7. |
| 10 | [`stop_all_monitors.sh`](file:///home/anhnb9/Documents/diffusion_policy/system_monitors/stop_all_monitors.sh) | Bash | Dừng an toàn phiên Tmux `sys_monitors`, giải phóng toàn bộ các tiến trình giám sát chạy ngầm. |

---

## 🔍 Hướng dẫn chi tiết từng công cụ

### 1. `auto_resume_watcher.py` — Giám Sát VRAM Động & Tự Động Khôi Phục Cả 2 Mô Hình (U-Net & Transformer)

* **Mục đích**: Chạy thường trực 24/7 trên máy chủ. Tự động kiểm tra liên tục VRAM của cả 4 GPU NVIDIA RTX PRO 6000 (96 GB mỗi card). Ngay khi có GPU rảnh đạt ngưỡng VRAM yêu cầu, nó sẽ **lập tức thực hiện luồng resume training** với code mới nhất và checkpoint mới nhất:
  * **Diffusion Policy Transformer**: Yêu cầu VRAM trống $\ge 60\text{ GB}$. Kích hoạt session Tmux `train_transformer` chạy [`resume_train_transformer.sh`](file:///home/anhnb9/Documents/diffusion_policy/resume_train_transformer.sh).
  * **Diffusion Policy U-Net (CNN)**: Yêu cầu VRAM trống $\ge 30\text{ GB}$. Kích hoạt session Tmux `train_unet` chạy [`resume_train_unet.sh`](file:///home/anhnb9/Documents/diffusion_policy/resume_train_unet.sh).
* **Thuật toán điều phối thông minh**:
  * Tự động kiểm tra trạng thái sống/chết của từng tác vụ (`is_tmux_session_running` và `is_training_process_running`).
  * Nếu cả 2 tác vụ đều đang chạy: Bỏ qua, tránh spawn trùng lặp.
  * Khi có GPU trống $\ge 60\text{ GB}$: Ưu tiên cấp phát GPU đó cho Transformer (nếu Transformer chưa chạy).
  * Khi có GPU trống $\ge 30\text{ GB}$: Cấp phát GPU đó cho U-Net (nếu U-Net chưa chạy).
  * Tự động truyền biến `GPU_ID` tương ứng (ví dụ: `GPU_ID=0`, `GPU_ID=1`...) để script train chạy đúng card GPU rảnh mà không gây xung đột tài nguyên.
  * Nếu một tiến trình đang train bị kill / crash bất ngờ, watcher sẽ nhận diện và **tự động resume lại ngay khi có GPU trống**.
* **Các tham số dòng lệnh (CLI Options)**:
  * `--transformer-vram`: Ngưỡng VRAM trống tối thiểu (GB) cho Transformer (Mặc định: `60.0`).
  * `--unet-vram`: Ngưỡng VRAM trống tối thiểu (GB) cho U-Net (Mặc định: `30.0`).
  * `--interval`: Chu kỳ quét VRAM tính bằng giây (Mặc định: `10`).
  * `--log-file`: Đường dẫn lưu nhật ký hoạt động (Mặc định: `auto_resume_watcher.log`).
  * `--once`: Chỉ kiểm tra 1 lần rồi thoát (Dùng để test trạng thái).
* **Khởi chạy ngầm nhanh bằng 1 lệnh Bash**:
  ```bash
  cd /home/anhnb9/Documents/system_monitors
  ./start_auto_resume_watcher.sh
  ```
* **Lệnh xem trực tiếp & dừng**:
  ```bash
  # Xem live log quá trình canh GPU và kích hoạt:
  tmux attach -t auto_resume_watcher

  # Dừng watcher:
  ./stop_auto_resume_watcher.sh
  ```

---

### 2. `gpu_watcher.py` — Giám Sát VRAM Đơn Lẻ & Tự Động Kích Hoạt Tùy Biến

* **Mục đích**: Canh chừng mức độ chiếm dụng VRAM của 4 GPU NVIDIA RTX PRO 6000 (96 GB mỗi card). Khi phát hiện có GPU đạt dung lượng VRAM trống theo yêu cầu (mặc định $\ge 45\text{ GB}$):
  1. Đổi màu terminal cảnh báo nổi bật và phát thông báo Desktop (`notify-send`).
  2. Tự động chạy lệnh hoặc script huấn luyện tương ứng với GPU rảnh đó (chạy trong session Tmux độc lập).
* **Các tham số dòng lệnh (CLI Options)**:
  * `--min-vram-gb`: Dung lượng VRAM trống tối thiểu (GB) cần để kích hoạt (Mặc định: `45.0`).
  * `--check-interval`: Chu kỳ kiểm tra lại trạng thái GPU tính bằng giây (Mặc định: `10`).
  * `--trigger-cmd`: Lệnh Bash tự động chạy khi phát hiện GPU thỏa điều kiện.
  * `--tmux-session`: Tên session Tmux sẽ tạo để chạy lệnh trigger (ví dụ: `train_transformer`).
  * `--target-gpus`: Giới hạn chỉ canh một số GPU cụ thể (ví dụ: `1,3`), để trống nếu canh toàn bộ.
  * `--log-file`: Tên file ghi nhật ký hoạt động (Mặc định: `gpu_watcher.log`).
  * `--once`: Chỉ quét 1 lần rồi thoát (Exit code 0 nếu có GPU đủ VRAM, 1 nếu không).
* **Ví dụ sử dụng**:
  ```bash
  # Canh khi nào có GPU trống >= 45 GB VRAM (chu kỳ quét 10s):
  python3 gpu_watcher.py --min-vram-gb 45 --check-interval 10

  # Canh và TỰ ĐỘNG bật script training Transformer khi có GPU trống:
  python3 gpu_watcher.py --min-vram-gb 45 \
      --trigger-cmd "bash /home/anhnb9/Documents/diffusion_policy/resume_train_transformer.sh" \
      --tmux-session "train_transformer"

  # Kiểm tra nhanh 1 lần xem hiện tại có GPU nào trống >= 50 GB không:
  python3 gpu_watcher.py --min-vram-gb 50 --once
  ```

---

### 2. `process_guard.py` — Giám Sát Tiến Trình & Hộp Đen Điều Tra Sự Cố (Forensics)

* **Mục đích**: Trực chờ và giám sát liên tục tình trạng của các tiến trình AI quan trọng (`train.py`, `diffusion`) hoặc các session Tmux huấn luyện.
* **Nguyên lý phát hiện thủ phạm**:
  * Định kỳ kiểm tra bảng tiến trình qua `ps` và `tmux ls`.
  * Khi phát hiện một PID bị mất tích bất ngờ (bị Terminated hoặc SIGKILL):
    * Lập tức thực thi `sudo -n grep ... /var/log/auth.log` để trích xuất nhật ký bảo mật hệ thống.
    * Xác định chính xác: **User nào (UID nào)** đã gõ lệnh `sudo kill <PID>`, thời điểm gõ lệnh và TTY thực thi.
    * Ghi toàn bộ kết luận điều tra vào `process_guard.log`.
* **Các tham số dòng lệnh (CLI Options)**:
  * `--patterns`: Danh sách từ khóa lệnh cần theo dõi (ví dụ: `train.py` `diffusion`).
  * `--tmux-sessions`: Danh sách tên các session Tmux cần giám sát (ví dụ: `train_transformer` `train_cnn`).
  * `--interval`: Chu kỳ quét kiểm tra tính bằng giây (Mặc định: `5`).
  * `--log-file`: Tên file ghi nhật ký sự kiện (Mặc định: `process_guard.log`).
  * `--auto-restart-cmd`: Lệnh Bash tùy chọn để tự khởi động lại nếu tiến trình bị kill.
* **Ví dụ sử dụng**:
  ```bash
  # Giám sát các tiến trình train.py và diffusion:
  python3 process_guard.py --patterns "train.py" "diffusion" --interval 5

  # Giám sát chặt chẽ 2 session Tmux huấn luyện của dự án:
  python3 process_guard.py --tmux-sessions "train_transformer" "train_cnn"
  ```

---

### 3. `reboot_detector.py` — Phát Hiện & Ghi Nhận Lịch Sử Máy Chủ Reboot

* **Mục đích**: Tự động phát hiện khi máy chủ vừa khởi động lại (sau sự cố hoặc sau bảo trì).
* **Nguyên lý hoạt động**:
  * So sánh mốc thời gian boot hiện tại (`who -b` hoặc `/proc/uptime`) với bản ghi cache trong `.boot_state.json`.
  * Khi phát hiện mốc boot mới:
    * Tra cứu `/var/log/auth.log` qua sudo để tìm xem ai đã gõ lệnh `sudo reboot` hoặc `shutdown`.
    * Phân biệt rõ sự cố reboot chủ động do người dùng gõ lệnh hay do mất điện / kernel panic.
    * Ghi nhận lịch sử chi tiết vào `reboot_history.log`.
* **Các tham số dòng lệnh (CLI Options)**:
  * `--daemon`: Chạy ở chế độ daemon theo dõi liên tục định kỳ trong nền.
  * `--check-interval`: Chu kỳ kiểm tra tính bằng giây (Mặc định: `60`).
  * `--history`: Hiển thị danh sách 10 lần reboot gần nhất cùng dòng log lệnh gây reboot.
* **Ví dụ sử dụng**:
  ```bash
  # Kiểm tra nhanh trạng thái boot hiện tại (1 lần):
  python3 reboot_detector.py

  # Xem lịch sử các lần reboot trước đây và người thực thi:
  python3 reboot_detector.py --history

  # Chạy thường trực dưới dạng daemon kiểm tra mỗi 30 giây:
  python3 reboot_detector.py --daemon --check-interval 30
  ```

---

### 4. `wandb_sync_daemon.sh` — Tự Động Đồng Bộ Dữ Liệu Lên Weights & Biases

* **Mục đích**: Quét định kỳ các thư mục log huấn luyện của Transformer và CNN/UNet trong `data/outputs/` và đồng bộ lên W&B cloud.
* **Đặc tính kỹ thuật quan trọng**:
  * Sử dụng cờ: `wandb sync --legacy --include-online -p <project> -e <entity> <path>`
  * **Giải quyết dứt điểm lỗi**: `transactionlog: error reading record: unexpected EOF` vốn xảy ra thường xuyên trên wandb bản 0.29.x khi đồng bộ các file log đang mở.
  * Tự động bù đắp dữ liệu khi mạng bị ngắt quãng hoặc sau khi tiến trình training bị dừng đột ngột.
* **Các biến môi trường cấu hình**:
  * `WANDB_PROJECT`: Tên dự án W&B (Mặc định: `astribot_making_coffee`).
  * `WANDB_ENTITY`: Tên Entity / Workspace W&B (Mặc định: `nguyenbanganh30-vnu`).
  * `SYNC_INTERVAL`: Chu kỳ giữa 2 lần đồng bộ tính bằng giây (Mặc định: `300` = 5 phút).
* **File nhật ký**: `wandb_sync_daemon.log`.
* **Ví dụ sử dụng**:
  ```bash
  # Chạy với cấu hình mặc định (đồng bộ mỗi 5 phút):
  ./wandb_sync_daemon.sh

  # Chạy với chu kỳ 1 phút (60 giây):
  SYNC_INTERVAL=60 ./wandb_sync_daemon.sh
  ```

---

### 5. `status.sh` — Bảng Điều Khiển Sức Khỏe Máy Chủ & Huấn Luyện (Dashboard)

* **Mục đích**: Cung cấp bức tranh toàn cảnh về sức khỏe máy chủ chỉ trong 1 lệnh duy nhất.
* **Các thành phần trên giao diện Terminal Dashboard**:
  1. **Thông tin máy chủ**: Hostname, thời điểm boot gần nhất, thời gian uptime.
  2. **Tài nguyên 4 GPU RTX PRO 6000**: VRAM Free / Total (GB), Tải GPU (%), Nhiệt độ (°C) kèm màu cảnh báo (Xanh: $>40\text{ GB}$ rảnh, Vàng: $15-40\text{ GB}$, Đỏ: $<15\text{ GB}$).
  3. **Tiến trình AI Training**: Liệt kê các session Tmux đang hoạt động.
  4. **Trạng thái các Daemon giám sát**: Báo rõ từng công cụ đang `RUNNING` hay `STOPPED`.
  5. **Nhật ký cảnh báo gần nhất**: Trích xuất ngay các sự kiện kill process hoặc reboot mới nhất.
* **Ví dụ sử dụng**:
  ```bash
  # Xem trạng thái tức thời:
  ./status.sh

  # Chế độ Live Dashboard tự làm mới mỗi 3 giây:
  watch -n 3 -c ./status.sh
  ```

---

### 6. `start_all_monitors.sh` & `stop_all_monitors.sh` — Điều Phối Toàn Bộ Hệ Thống

* **`start_all_monitors.sh`**:
  * Tự động tạo phiên Tmux `sys_monitors` gồm 3 cửa sổ:
    * `Window 0` (`gpu_watcher`): Chạy `gpu_watcher.py` canh VRAM mỗi 10s.
    * `Window 1` (`process_guard`): Chạy `process_guard.py` canh tiến trình mỗi 5s.
    * `Window 2` (`wandb_sync`): Chạy `wandb_sync_daemon.sh` đồng bộ W&B mỗi 5 phút.
  * Tự động chạy 1 lần kiểm tra `reboot_detector.py`.
* **`stop_all_monitors.sh`**:
  * Dừng toàn bộ phiên Tmux `sys_monitors`, chấm dứt tất cả các ứng dụng giám sát nền.
* **Ví dụ sử dụng**:
  ```bash
  # Bật toàn bộ hệ thống giám sát:
  ./start_all_monitors.sh

  # Dừng toàn bộ hệ thống giám sát:
  ./stop_all_monitors.sh
  ```

---

## 🎮 Hướng dẫn thao tác với Tmux Session `sys_monitors`

Khi hệ thống giám sát đã được bật bằng `./start_all_monitors.sh`, bạn có thể truy cập vào xem trực tiếp màn hình hoạt động của từng công cụ:

```bash
# Đính kèm vào phiên giám sát:
tmux attach -t sys_monitors
```

* **Chuyển đổi giữa các cửa sổ giám sát**:
  * Bấm tổ hợp phím `Ctrl + B`, sau đó bấm phím số:
    * `0`: Xem màn hình **GPU Watcher**
    * `1`: Xem màn hình **Process Guard**
    * `2`: Xem màn hình **W&B Auto-Sync**
* **Thoát ra ngoài màn hình chính mà không làm dừng ứng dụng**:
  * Bấm `Ctrl + B`, sau đó bấm phím `D` (Detach).

---

## 📝 Danh mục các file Logs và Cache trạng thái

| Tên File | Chức năng lưu trữ |
| :--- | :--- |
| `gpu_watcher.log` | Lịch sử quét VRAM, các lần phát hiện GPU rảnh và sự kiện trigger training. |
| `process_guard.log` | Lịch sử giám sát tiến trình AI, báo cáo chi tiết kẻ đã thực hiện lệnh kill. |
| `reboot_history.log` | Nhật ký các lần server khởi động lại kèm user và dòng lệnh `sudo reboot`. |
| `wandb_sync_daemon.log` | Nhật ký chi tiết từng lần đồng bộ file `.wandb` lên W&B cloud. |
| `.boot_state.json` | File ẩn lưu mốc thời gian boot gần nhất phục vụ so sánh phát hiện reboot mới. |

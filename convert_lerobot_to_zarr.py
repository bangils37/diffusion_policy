#!/usr/bin/env bash
"""":
# Runner script wrapper if executed directly
exec python3 "$0" "$@"
"""
import os
import sys
import json
import argparse
import pathlib
import numpy as np
import zarr
import cv2
from tqdm import tqdm
from typing import Dict, List, Optional, Any

# Ensure diffusion_policy is in python path
ROOT_DIR = str(pathlib.Path(__file__).parent.absolute())
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()


def decode_video_to_frames(video_path: str, target_size: Optional[tuple] = None) -> np.ndarray:
    """Decode an MP4 video file into an RGB numpy array (T, H, W, C)."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames = []
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size is not None:
                # target_size = (H, W) -> cv2.resize expects (W, H)
                frame = cv2.resize(frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")

    return np.stack(frames, axis=0)


def sanitize_key_name(key: str) -> str:
    """Convert LeRobot keys like 'observation.images.cam_head' to 'cam_head'."""
    if key.startswith("observation.images."):
        return key.replace("observation.images.", "")
    elif key.startswith("observation.image."):
        return key.replace("observation.image.", "")
    elif key == "observation.state" or key == "observation.robot_state":
        return "state"
    elif key.startswith("observation."):
        return key.replace("observation.", "")
    return key


def convert_lerobot_dataset(
    dataset_path_or_id: str,
    output_zarr_path: str,
    target_img_size: Optional[tuple] = None,
    key_mapping: Optional[Dict[str, str]] = None,
) -> str:
    """
    Convert a LeRobot format dataset (local folder or HF repo) into Diffusion Policy Zarr ReplayBuffer format.
    """
    dataset_dir = dataset_path_or_id

    # 1. If it's a Hugging Face repo ID, download it using huggingface_hub
    if not os.path.exists(dataset_dir):
        print(f"📥 '{dataset_dir}' không phải đường dẫn local, thử tải từ Hugging Face Hub...")
        try:
            from huggingface_hub import snapshot_download
            dataset_dir = snapshot_download(repo_id=dataset_path_or_id, repo_type="dataset")
            print(f"✅ Đã tải dataset về: {dataset_dir}")
        except Exception as e:
            raise FileNotFoundError(f"Không tìm thấy dataset local hoặc repo HF: {dataset_path_or_id}. Lỗi: {e}")

    meta_dir = os.path.join(dataset_dir, "meta")
    info_json_path = os.path.join(meta_dir, "info.json")
    episodes_jsonl_path = os.path.join(meta_dir, "episodes.jsonl")
    data_dir = os.path.join(dataset_dir, "data")
    videos_dir = os.path.join(dataset_dir, "videos")

    # 2. Read dataset info
    info = {}
    if os.path.exists(info_json_path):
        with open(info_json_path, "r") as f:
            info = json.load(f)
        print(f"📋 Dataset Info: FPS={info.get('fps')}, Total Episodes={info.get('total_episodes')}, Total Frames={info.get('total_frames')}")

    # 3. Read Parquet tabular data (State, Action, Episode Indices)
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Vui lòng cài đặt pandas và pyarrow: pip install pandas pyarrow")

    parquet_files = sorted(pathlib.Path(dataset_dir).glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Không tìm thấy file .parquet nào trong thư mục: {dataset_dir}")

    print(f"🔍 Đang đọc {len(parquet_files)} file parquet...")
    df_list = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(df_list, ignore_index=True)

    # Detect episode column
    if "episode_index" in df.columns:
        ep_col = "episode_index"
    elif "episode" in df.columns:
        ep_col = "episode"
    else:
        ep_col = None

    if ep_col is not None:
        episode_groups = [group for _, group in df.groupby(ep_col)]
    else:
        # Fallback if no episode index
        episode_groups = [df]

    total_episodes = len(episode_groups)
    print(f"📊 Tìm thấy tổng cộng {total_episodes} episodes trong dataset.")

    # 4. Prepare Output ReplayBuffer
    output_zarr_path = os.path.expanduser(output_zarr_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_zarr_path)), exist_ok=True)
    if os.path.exists(output_zarr_path):
        import shutil
        print(f"⚠️ Thư mục output {output_zarr_path} đã tồn tại, tiến hành ghi đè...")
        shutil.rmtree(output_zarr_path)

    replay_buffer = ReplayBuffer.create_from_path(output_zarr_path, mode="a")

    # 5. Process each episode
    for ep_idx, ep_df in enumerate(tqdm(episode_groups, desc="Converting Episodes")):
        episode_data: Dict[str, np.ndarray] = {}

        # A. Extract Action
        if "action" in ep_df.columns:
            actions = np.array(ep_df["action"].tolist(), dtype=np.float32)
            episode_data["action"] = actions

        # B. Extract State / Observation State
        state_col = None
        for col in ["observation.state", "observation.robot_state", "state"]:
            if col in ep_df.columns:
                state_col = col
                break
        if state_col is not None:
            states = np.array(ep_df[state_col].tolist(), dtype=np.float32)
            episode_data["state"] = states

        # C. Extract Camera Videos / Images
        # Check videos directory: videos/observation.images.<cam>/episode_*.mp4 or videos/<cam>/episode_*.mp4
        if os.path.exists(videos_dir):
            for cam_folder in sorted(os.listdir(videos_dir)):
                cam_path = os.path.join(videos_dir, cam_folder)
                if not os.path.isdir(cam_path):
                    continue

                cam_key = sanitize_key_name(cam_folder)
                if key_mapping and cam_key in key_mapping:
                    cam_key = key_mapping[cam_key]

                # Find matching video file for this episode (e.g. episode_000000.mp4, episode_0.mp4)
                possible_names = [
                    f"episode_{ep_idx:06d}.mp4",
                    f"episode_{ep_idx:05d}.mp4",
                    f"episode_{ep_idx}.mp4",
                    f"chunk-000/episode_{ep_idx:06d}.mp4",
                ]
                video_file = None
                for name in possible_names:
                    cand = os.path.join(cam_path, name)
                    if os.path.exists(cand):
                        video_file = cand
                        break

                if video_file is None:
                    # Search recursively in cam_path
                    search_res = list(pathlib.Path(cam_path).glob(f"*episode_{ep_idx:06d}*.mp4")) or \
                                 list(pathlib.Path(cam_path).glob(f"*episode_{ep_idx}*.mp4"))
                    if search_res:
                        video_file = str(search_res[0])

                if video_file is not None:
                    frames = decode_video_to_frames(video_file, target_size=target_img_size)
                    episode_data[cam_key] = frames

        # D. Check any raw image columns in parquet
        for col in ep_df.columns:
            if col.startswith("observation.images.") or col.startswith("observation.image."):
                cam_key = sanitize_key_name(col)
                if key_mapping and cam_key in key_mapping:
                    cam_key = key_mapping[cam_key]

                if cam_key not in episode_data:
                    # Extract list of image arrays / bytes
                    val_sample = ep_df[col].iloc[0]
                    if isinstance(val_sample, (list, np.ndarray)):
                        img_arr = np.array(ep_df[col].tolist(), dtype=np.uint8)
                        episode_data[cam_key] = img_arr

        # E. Add episode to ReplayBuffer
        replay_buffer.add_episode(episode_data, compressors="disk")

    print("\n" + "=" * 60)
    print(f"🎉 CHUYỂN ĐỔI DATASET THÀNH CÔNG!")
    print(f"📂 Zarr Output: {output_zarr_path}")
    print(f"📊 Tổng số episodes: {replay_buffer.n_episodes}")
    print(f"📊 Tổng số steps/frames: {replay_buffer.n_steps}")
    print("📋 Các trường dữ liệu & Shape:")
    for key, val in replay_buffer.data.items():
        print(f"   - {key}: shape={val.shape}, dtype={val.dtype}, chunks={val.chunks}")
    print("=" * 60)

    return output_zarr_path


def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi LeRobot Dataset sang định dạng Zarr của Diffusion Policy.")
    parser.add_argument("input_path", type=str, help="Đường dẫn thư mục LeRobot dataset local hoặc HuggingFace Repo ID (ví dụ: lerobot/pusht)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Đường dẫn file/thư mục output .zarr (Mặc định: data/converted_lerobot_dataset.zarr)")
    parser.add_argument("--img_height", type=int, default=None, help="Resize chiều cao ảnh (tùy chọn)")
    parser.add_argument("--img_width", type=int, default=None, help="Resize chiều rộng ảnh (tùy chọn)")
    args = parser.parse_args()

    input_path = args.input_path
    if args.output is None:
        basename = os.path.basename(os.path.normpath(input_path)).replace(".", "_")
        output_zarr = os.path.join(ROOT_DIR, "data", f"{basename}.zarr")
    else:
        output_zarr = args.output

    target_size = None
    if args.img_height is not None and args.img_width is not None:
        target_size = (args.img_height, args.img_width)

    convert_lerobot_dataset(
        dataset_path_or_id=input_path,
        output_zarr_path=output_zarr,
        target_img_size=target_size,
    )


if __name__ == "__main__":
    main()

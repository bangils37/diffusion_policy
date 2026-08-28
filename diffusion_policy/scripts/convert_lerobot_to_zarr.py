"""
High-performance script to convert LeRobot dataset (v2.1 parquet + mp4) into Diffusion Policy Zarr ReplayBuffer format.
Uses preallocated Zarr datasets and multi-processing for parallel decoding and Jpeg2k encoding.
"""
import os
import sys
import pathlib
import json
import click
import numpy as np
import pyarrow.parquet as pq
import cv2
import zarr
import numcodecs
import multiprocessing
import concurrent.futures
from tqdm import tqdm

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
register_codecs()


def process_episode_video(video_path: str, target_size: tuple = None) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_size is not None:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"Failed to read frames from {video_path}")
    return np.array(frames, dtype=np.uint8)


def write_episode_worker(args):
    episode_idx, start_idx, end_idx, dataset_dir, zarr_dir, cam_names, target_size = args
    dataset_path = pathlib.Path(dataset_dir)
    zarr_path = pathlib.Path(zarr_dir)
    
    # 1. Read parquet
    parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    states = np.stack(df['observation.state'].values).astype(np.float32)
    actions = np.stack(df['action'].values).astype(np.float32)
    n_frames = len(df)
    
    # 2. Open zarr array slice
    root = zarr.open(str(zarr_path), mode='a')
    data_group = root['data']
    
    data_group['state'][start_idx:end_idx] = states
    data_group['action'][start_idx:end_idx] = actions
    
    # 3. Read & write videos
    for cam in cam_names:
        video_path = dataset_path / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{episode_idx:06d}.mp4"
        frames = process_episode_video(str(video_path), target_size=target_size)
        if len(frames) != n_frames:
            if len(frames) < n_frames:
                pad = np.repeat(frames[-1:], n_frames - len(frames), axis=0)
                frames = np.concatenate([frames, pad], axis=0)
            else:
                frames = frames[:n_frames]
        data_group[cam][start_idx:end_idx] = frames
        
    return episode_idx, n_frames


@click.command()
@click.option('--input', '-i', default='/home/anhnb9/Documents/datasets/astri_making_coffee_v21', help='Path to LeRobot dataset')
@click.option('--output', '-o', default='/home/anhnb9/Documents/datasets/astri_making_coffee_v21.zarr', help='Output Zarr path')
@click.option('--resolution', '-r', default='320x240', help='Target resolution as WxH (e.g. 320x240 or 640x480 or none)')
@click.option('--num_workers', '-n', default=max(1, multiprocessing.cpu_count() - 2), help='Number of parallel workers')
@click.option('--max_episodes', '-m', default=None, type=int, help='Limit number of episodes (for testing)')
def main(input, output, resolution, num_workers, max_episodes):
    input_path = pathlib.Path(os.path.expanduser(input))
    output_path = pathlib.Path(os.path.expanduser(output))
    
    episodes_file = input_path / "meta" / "episodes.jsonl"
    assert episodes_file.exists(), f"Cannot find {episodes_file}"
    
    episodes_meta = []
    with open(episodes_file, "r") as f:
        for line in f:
            if line.strip():
                episodes_meta.append(json.loads(line))
                
    if max_episodes is not None:
        episodes_meta = episodes_meta[:max_episodes]
        
    total_episodes = len(episodes_meta)
    
    # Calculate episode slice offsets
    episode_lengths = [ep['length'] for ep in episodes_meta]
    episode_ends = np.cumsum(episode_lengths).astype(np.int64)
    episode_starts = np.insert(episode_ends[:-1], 0, 0)
    total_frames = int(episode_ends[-1])
    
    cam_names = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
    
    target_size = None
    if resolution.lower() != 'none':
        w, h = map(int, resolution.split('x'))
        target_size = (w, h)
        img_h, img_w = h, w
    else:
        img_h, img_w = 480, 640
        
    print(f"=== Converting LeRobot Dataset to Zarr ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total episodes: {total_episodes}, Total frames: {total_frames}")
    print(f"Resolution: {img_w}x{img_h}, Workers: {num_workers}")
    
    # Pre-allocate Zarr Store
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)
    
    image_compressor = Jpeg2k(level=50)
    lowdim_compressor = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE)
    
    print("Pre-allocating arrays...")
    data_group.zeros('state', shape=(total_frames, 16), chunks=(1000, 16), dtype=np.float32, compressor=lowdim_compressor)
    data_group.zeros('action', shape=(total_frames, 16), chunks=(1000, 16), dtype=np.float32, compressor=lowdim_compressor)
    
    for cam in cam_names:
        data_group.zeros(cam, shape=(total_frames, img_h, img_w, 3), chunks=(1, img_h, img_w, 3), dtype=np.uint8, compressor=image_compressor)
        
    meta_group.array('episode_ends', episode_ends, compressor=None, overwrite=True)
    
    tasks = []
    for i, ep in enumerate(episodes_meta):
        ep_idx = ep['episode_index']
        s_idx = int(episode_starts[i])
        e_idx = int(episode_ends[i])
        tasks.append((ep_idx, s_idx, e_idx, str(input_path), str(output_path), cam_names, target_size))
        
    cv2.setNumThreads(1)
    
    print("Converting episodes in parallel...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(write_episode_worker, t): t[0] for t in tasks}
        with tqdm(total=total_episodes, desc="Encoding & Writing to Zarr") as pbar:
            for future in concurrent.futures.as_completed(futures):
                ep_idx, n_frames = future.result()
                pbar.update(1)
                
    # Consolidate metadata
    zarr.consolidate_metadata(store)
    print(f"\nDone! Successfully created {output_path} ({total_episodes} episodes, {total_frames} frames).")
    print(root.tree())


if __name__ == '__main__':
    main()

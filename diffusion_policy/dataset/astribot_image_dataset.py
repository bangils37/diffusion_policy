from typing import Dict, List, Optional
import torch
import numpy as np
import zarr
import os
import copy
import threading
import time
from threadpoolctl import threadpool_limits
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
    array_to_stats
)
register_codecs()


class AstribotImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=16,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_latency_steps=0,
            seed=42,
            val_ratio=0.05,
            max_train_episodes=None,
            use_cache=True,
            max_cache_size_gb=50.0,
            background_preload=True,
            preload_delay_sec=1.0,
            preload_sleep_sec=0.01,
        ):
        assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist!"
        
        # Load ReplayBuffer
        self.cache_store = None
        if dataset_path.endswith('.zip'):
            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(src_store=zip_store, store=zarr.MemoryStore())
        elif use_cache and max_cache_size_gb > 0:
            store = zarr.DirectoryStore(os.path.expanduser(dataset_path))
            self.cache_store = zarr.LRUStoreCache(store, max_size=int(max_cache_size_gb * (1024**3)))
            root = zarr.group(store=self.cache_store)
            replay_buffer = ReplayBuffer.create_from_group(root)
        else:
            replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode='r')
            
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type_name = attr.get('type', 'low_dim')
            if type_name == 'rgb':
                rgb_keys.append(key)
            elif type_name == 'low_dim':
                lowdim_keys.append(key)
                
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Start background preloading thread to pre-warm RAM cache gradually
        if background_preload and self.cache_store is not None:
            self._start_background_preload(
                root=replay_buffer.root,
                cache_store=self.cache_store,
                max_bytes=int(max_cache_size_gb * (1024**3)),
                delay_sec=preload_delay_sec,
                sleep_sec=preload_sleep_sec
            )

    def _start_background_preload(self, root, cache_store, max_bytes, delay_sec=1.0, sleep_sec=0.01):
        def _preloader():
            time.sleep(delay_sec)
            try:
                # Preload low-dim keys (small, fast)
                for key in self.lowdim_keys + ['action']:
                    if key in root['data']:
                        _ = root['data'][key][:]

                # Preload RGB images chunk-by-chunk
                for key in self.rgb_keys:
                    if key not in root['data']:
                        continue
                    arr = root['data'][key]
                    chunk_len = arr.chunks[0] if hasattr(arr, 'chunks') and arr.chunks is not None else 100
                    total_len = arr.shape[0]

                    for start_idx in range(0, total_len, chunk_len):
                        cur_size = getattr(cache_store, '_current_size', 0)
                        if cur_size >= max_bytes:
                            # Reached 50GB limit, stop preloading
                            return
                        # Touch 1 element in chunk to trigger LRU cache population
                        _ = arr[start_idx:start_idx+1]
                        if sleep_sec > 0:
                            time.sleep(sleep_sec)
            except Exception:
                pass

        thread = threading.Thread(target=_preloader, daemon=True, name="AstribotDatasetPreloader")
        thread.start()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # Action normalizer (16 DoF)
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        
        # State normalizer (16 DoF)
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        
        # RGB image normalizer
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
            
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'][:])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with threadpool_limits(1):
            data = self.sampler.sample_sequence(idx)

            T_slice = slice(self.n_obs_steps)
            obs_dict = dict()
            
            for key in self.rgb_keys:
                # Shape: (T_obs, H, W, C) -> (T_obs, C, H, W) normalized to [0, 1]
                obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
                del data[key]
                
            for key in self.lowdim_keys:
                obs_dict[key] = data[key][T_slice].astype(np.float32)
                del data[key]
            
            action = data['action'].astype(np.float32)
            if self.n_latency_steps > 0:
                action = action[self.n_latency_steps:]

            torch_data = {
                'obs': dict_apply(obs_dict, torch.from_numpy),
                'action': torch.from_numpy(action)
            }
            return torch_data

from typing import Dict, List, Optional
import torch
import numpy as np
import zarr
import os
import copy
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
            mem_num_frames: Optional[int]=None,
            mem_frame_stride: int=15,
        ):
        assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist!"
        
        # Load ReplayBuffer
        if dataset_path.endswith('.zip'):
            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(src_store=zip_store, store=zarr.MemoryStore())
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
        if mem_num_frames is not None:
            # When using temporal memory, RGB keys are sampled directly via strided temporal indexing
            for key in rgb_keys:
                key_first_k[key] = 0
            if n_obs_steps is not None:
                for key in lowdim_keys:
                    key_first_k[key] = n_obs_steps
        elif n_obs_steps is not None:
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
        self.mem_num_frames = mem_num_frames
        self.mem_frame_stride = mem_frame_stride
        self.episode_ends = replay_buffer.episode_ends[:]

        if mem_num_frames is not None:
            # Compute past offsets ending at 0: [-(num_frames - 1)*stride, ..., -stride, 0]
            self.mem_offsets = np.array([
                -(mem_num_frames - 1 - i) * mem_frame_stride for i in range(mem_num_frames)
            ], dtype=np.int64)
        else:
            self.mem_offsets = None

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
            obs_dict = dict()

            if self.mem_num_frames is not None:
                # Sequence Memory Video Sampling
                buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
                    = self.sampler.indices[idx]
                
                # Anchor step: latest observation step
                anchor_offset = (self.n_obs_steps - 1) if (self.n_obs_steps is not None and self.n_obs_steps > 0) else 0
                anchor_idx = buffer_start_idx + sample_start_idx + anchor_offset

                # Find episode boundary for clamping
                ep_idx = np.searchsorted(self.episode_ends, anchor_idx, side='right')
                ep_start = 0 if ep_idx == 0 else self.episode_ends[ep_idx - 1]
                ep_end = self.episode_ends[ep_idx]

                # Strided sampling into past, clamped at episode boundaries
                mem_indices = np.clip(anchor_idx + self.mem_offsets, ep_start, ep_end - 1)
                min_i, max_i = mem_indices[0], mem_indices[-1]
                rel_indices = mem_indices - min_i

                for key in self.rgb_keys:
                    # Single contiguous slice from Zarr, indexed in memory via numpy
                    chunk = self.replay_buffer[key][min_i : max_i + 1]
                    sampled_frames = chunk[rel_indices] # (T_mem, H, W, C)
                    obs_dict[key] = np.moveaxis(sampled_frames, -1, 1).astype(np.float32) / 255.0
                    if key in data:
                        del data[key]
            else:
                T_slice = slice(self.n_obs_steps)
                for key in self.rgb_keys:
                    # Shape: (T_obs, H, W, C) -> (T_obs, C, H, W) normalized to [0, 1]
                    obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
                    del data[key]
                
            T_slice = slice(self.n_obs_steps)
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

# Alias for explicit config referencing
AstribotMemImageDataset = AstribotImageDataset


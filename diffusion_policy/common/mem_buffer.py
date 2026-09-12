from typing import Dict, List, Optional, Union
from collections import deque
import numpy as np
import torch


class MemVideoBuffer:
    """
    Rolling observation history buffer for online inference with short-video memory.
    Maintains chronological frame streams per camera and generates strided history clips
    compatible with MemShortVideoEncoder and MemMultiImageObsEncoder.
    
    Equivalent to OpenPI deploy buffer (custom_motion2_mem/deploy_ros2.py).
    """
    def __init__(
            self, 
            camera_names: List[str], 
            num_frames: int = 12, 
            frame_stride: int = 15
        ):
        self.camera_names = list(camera_names)
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.depth = 1 + (self.num_frames - 1) * self.frame_stride

        self._buffers: Dict[str, deque] = {
            cam: deque(maxlen=self.depth) for cam in self.camera_names
        }

    def reset(self):
        """Clears all history buffers, e.g. at the start of a new episode or task."""
        for cam in self.camera_names:
            self._buffers[cam].clear()

    def push(self, cam_name: str, frame: Union[np.ndarray, torch.Tensor]):
        """
        Pushes a new observation frame for the given camera.
        Frame shape can be (H, W, C) or (C, H, W).
        """
        if cam_name not in self._buffers:
            raise KeyError(f"Unknown camera: {cam_name}. Configured cameras: {self.camera_names}")
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        self._buffers[cam_name].append(frame)

    def is_ready(self, cam_name: Optional[str] = None) -> bool:
        """Returns True if the buffer has received at least 1 frame."""
        if cam_name is not None:
            return len(self._buffers[cam_name]) > 0
        return all(len(self._buffers[cam]) > 0 for cam in self.camera_names)

    def is_full(self, cam_name: Optional[str] = None) -> bool:
        """Returns True if the buffer has filled its full historical depth."""
        if cam_name is not None:
            return len(self._buffers[cam_name]) >= self.depth
        return all(len(self._buffers[cam]) >= self.depth for cam in self.camera_names)

    def clip(self, cam_name: str) -> np.ndarray:
        """
        Samples num_frames spaced frame_stride apart ending at the newest frame.
        Before the history is full, early slots clamp to the oldest available frame.
        Returns:
            np.ndarray of shape (num_frames, ...) matching frame shape.
        """
        buf = self._buffers[cam_name]
        n_pushed = len(buf)
        if n_pushed == 0:
            raise RuntimeError(f"Buffer for {cam_name} is empty. Push at least 1 frame before clip.")

        # Compute relative indices ending at the newest frame (index n_pushed - 1)
        # Clamped at 0 (oldest available frame)
        indices = [
            max(0, n_pushed - 1 - (self.num_frames - 1 - i) * self.frame_stride)
            for i in range(self.num_frames)
        ]
        
        frames_list = list(buf)
        sampled = [frames_list[idx] for idx in indices]
        return np.stack(sampled, axis=0)

    def get_obs_dict(
            self, 
            low_dim_obs: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
        ) -> Dict[str, torch.Tensor]:
        """
        Builds a batched observation dictionary ready for policy.predict_action(obs_dict).
        Images are returned with shape (1, num_frames, C, H, W) normalized to [0, 1].
        """
        obs_dict: Dict[str, torch.Tensor] = {}

        for cam in self.camera_names:
            clip = self.clip(cam) # (T_mem, ...)
            # Ensure shape is (T_mem, C, H, W)
            if clip.ndim == 4 and clip.shape[-1] in (1, 3):
                # (T, H, W, C) -> (T, C, H, W)
                clip = np.moveaxis(clip, -1, 1)

            tensor_clip = torch.from_numpy(clip).to(dtype=dtype)
            if tensor_clip.max() > 1.0:
                tensor_clip = tensor_clip / 255.0

            # Add batch dimension (B=1, T_mem, C, H, W)
            obs_dict[cam] = tensor_clip.unsqueeze(0)

        if low_dim_obs is not None:
            for k, v in low_dim_obs.items():
                if not isinstance(v, torch.Tensor):
                    v = torch.from_numpy(np.array(v)).to(dtype=dtype)
                if v.ndim == 1:
                    v = v.unsqueeze(0) # (1, D)
                obs_dict[k] = v

        if device is not None:
            obs_dict = {k: v.to(device) for k, v in obs_dict.items()}

        return obs_dict

import os
import tempfile
import numpy as np
import torch

from diffusion_policy.model.vision.mem_video import (
    MemVideoConfig, 
    MemShortVideoEncoder, 
    sinusoidal_time_embedding
)
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.mem_obs_encoder import MemMultiImageObsEncoder


def test_sinusoidal_embedding():
    positions = torch.tensor([-45.0, -30.0, -15.0, 0.0])
    emb_dim = 64
    emb = sinusoidal_time_embedding(positions, emb_dim)
    assert emb.shape == (4, 64)
    # The last position (0.0) must be all zeros as in OpenPI
    assert torch.all(emb[-1] == 0.0)
    # Earlier positions must be non-zero
    assert torch.any(emb[0] != 0.0)


def test_temporal_block_causality():
    cfg = MemVideoConfig(hidden_dim=64, num_frames=8, frame_stride=15, num_heads=2, num_layers=2)
    model = MemShortVideoEncoder(cfg)
    model.eval()

    x1 = torch.randn(1, 8, 4, 64)
    x2 = x1.clone()
    # Modify only the last frame t=7
    x2[:, 7, :, :] = torch.randn(1, 4, 64)

    with torch.no_grad():
        block = model.blocks[0]
        b1 = block(x1)
        b2 = block(x2)
        # Differences for t=0..6 must be 0 (strictly causal)
        diff_past = (b1[:, :7] - b2[:, :7]).abs().max().item()
        diff_future = (b1[:, 7:] - b2[:, 7:]).abs().max().item()
        assert diff_past < 1e-6, f"TemporalBlock is NOT causal! diff={diff_past}"
        assert diff_future > 1e-3, "Modified future frame should change future output"


def test_mem_short_video_encoder_shapes_and_grads():
    cfg = MemVideoConfig(hidden_dim=128, num_frames=12, frame_stride=15, num_heads=4, num_layers=2)
    model = MemShortVideoEncoder(cfg)

    # 4D tensor: [B=2, T=12, P=9, D=128]
    x4 = torch.randn(2, 12, 9, 128, requires_grad=True)
    out4 = model(x4)
    assert out4.shape == (2, 9, 128)
    out4.sum().backward()
    assert x4.grad is not None

    # 3D tensor: [B=2, T=12, D=128]
    x3 = torch.randn(2, 12, 128, requires_grad=True)
    out3 = model(x3)
    assert out3.shape == (2, 128)
    out3.sum().backward()
    assert x3.grad is not None


def test_mem_multi_image_obs_encoder():
    shape_meta = {
        'obs': {
            'cam_head': {'shape': [3, 240, 320], 'type': 'rgb'},
            'cam_left_wrist': {'shape': [3, 240, 320], 'type': 'rgb'},
            'cam_right_wrist': {'shape': [3, 240, 320], 'type': 'rgb'},
            'state': {'shape': [16], 'type': 'low_dim'}
        },
        'action': {'shape': [16]}
    }

    rgb_model = get_resnet('resnet18')
    encoder = MemMultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        resize_shape=[240, 320],
        crop_shape=[216, 288],
        random_crop=True,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=True,
        mem_num_frames=12,
        mem_frame_stride=15,
        mem_temporal_num_heads=8,
        mem_temporal_num_layers=2
    )

    out_shape = encoder.output_shape()
    # 3 cameras * 512 + 16 state dim = 1552
    assert out_shape == (1552,)

    B, T = 2, 12
    obs_dict = {
        'cam_head': torch.randn(B, T, 3, 240, 320),
        'cam_left_wrist': torch.randn(B, T, 3, 240, 320),
        'cam_right_wrist': torch.randn(B, T, 3, 240, 320),
        'state': torch.randn(B, T, 16)
    }
    features = encoder(obs_dict)
    assert features.shape == (B, 1552)

    loss = features.sum()
    loss.backward()
    print("MemMultiImageObsEncoder forward & backward pass passed!")


def test_mem_sampling_offsets_and_clamping():
    mem_num_frames = 12
    mem_frame_stride = 15
    offsets = np.array([
        -(mem_num_frames - 1 - i) * mem_frame_stride for i in range(mem_num_frames)
    ], dtype=np.int64)

    # 1. Normal index well within episode
    anchor_idx = 300
    ep_start = 0
    ep_end = 500
    mem_indices = np.clip(anchor_idx + offsets, ep_start, ep_end - 1)
    assert len(mem_indices) == 12
    assert mem_indices[-1] == 300
    assert mem_indices[0] == 300 - 165
    assert np.all(np.diff(mem_indices) == 15)

    # 2. Near start of episode: oldest frames clamp to ep_start
    anchor_idx = 40
    mem_indices = np.clip(anchor_idx + offsets, ep_start, ep_end - 1)
    assert mem_indices[0] == 0
    assert mem_indices[-1] == 40
    # Must be non-decreasing
    assert np.all(np.diff(mem_indices) >= 0)


if __name__ == '__main__':
    test_sinusoidal_embedding()
    test_temporal_block_causality()
    test_mem_short_video_encoder_shapes_and_grads()
    test_mem_multi_image_obs_encoder()
    test_mem_sampling_offsets_and_clamping()
    print("ALL TESTS PASSED SUCCESSFULLY!")

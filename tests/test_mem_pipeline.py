import os
import tempfile
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.vision.mem_video import (
    MemVideoConfig, 
    MemShortVideoEncoder, 
    sinusoidal_time_embedding
)
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.mem_obs_encoder import MemMultiImageObsEncoder
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.policy.diffusion_unet_mem_image_policy import DiffusionUnetMemImagePolicy
from diffusion_policy.policy.diffusion_transformer_mem_image_policy import DiffusionTransformerMemImagePolicy
from diffusion_policy.common.mem_buffer import MemVideoBuffer


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


def test_mem_video_buffer():
    cams = ['cam_head', 'cam_wrist']
    buf = MemVideoBuffer(cams, num_frames=12, frame_stride=15)
    assert buf.depth == 166
    assert not buf.is_ready()

    # 1. Push 1 frame: cold buffer
    frame = np.ones((240, 320, 3), dtype=np.uint8) * 42
    buf.push('cam_head', frame)
    buf.push('cam_wrist', frame)
    assert buf.is_ready()
    assert not buf.is_full()

    clip_head = buf.clip('cam_head')
    assert clip_head.shape == (12, 240, 320, 3)
    assert np.all(clip_head == 42)

    # 2. Push up to full depth
    for i in range(1, 200):
        buf.push('cam_head', np.ones((240, 320, 3), dtype=np.uint8) * (i % 256))
        buf.push('cam_wrist', np.ones((240, 320, 3), dtype=np.uint8) * (i % 256))

    assert buf.is_full()
    clip_head = buf.clip('cam_head')
    assert clip_head.shape == (12, 240, 320, 3)
    assert clip_head[-1, 0, 0, 0] == 199 % 256

    obs_dict = buf.get_obs_dict(low_dim_obs={'state': np.zeros(16, dtype=np.float32)})
    assert obs_dict['cam_head'].shape == (1, 12, 3, 240, 320)
    assert obs_dict['state'].shape == (1, 16)

    # 3. Reset
    buf.reset()
    assert not buf.is_ready()


def _get_dummy_normalizer(shape_meta):
    normalizer = LinearNormalizer()
    normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
    for k, v in shape_meta['obs'].items():
        normalizer[k] = SingleFieldLinearNormalizer.create_identity()
    return normalizer


def test_diffusion_unet_mem_policy():
    shape_meta = {
        'obs': {
            'cam_head': {'shape': [3, 240, 320], 'type': 'rgb'},
            'state': {'shape': [16], 'type': 'low_dim'}
        },
        'action': {'shape': [16]}
    }
    rgb_model = get_resnet('resnet18')
    obs_encoder = MemMultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        resize_shape=[240, 320],
        crop_shape=[216, 288],
        random_crop=False,
        use_group_norm=True,
        mem_num_frames=12,
        mem_frame_stride=15,
        mem_temporal_num_heads=4,
        mem_temporal_num_layers=2
    )

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon'
    )

    policy = DiffusionUnetMemImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        obs_encoder=obs_encoder,
        horizon=16,
        n_action_steps=8,
        n_obs_steps=1,
        num_inference_steps=4,
        down_dims=[128, 256]
    )
    policy.set_normalizer(_get_dummy_normalizer(shape_meta))

    B, T_mem = 2, 12
    batch = {
        'obs': {
            'cam_head': torch.randn(B, T_mem, 3, 240, 320),
            'state': torch.randn(B, T_mem, 16)
        },
        'action': torch.randn(B, 16, 16)
    }

    # 1. Training loss & backward pass
    loss = policy.compute_loss(batch)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    loss.backward()

    # 2. Inference: predict_action
    policy.eval()
    with torch.no_grad():
        result = policy.predict_action(batch['obs'])
        assert 'action' in result and 'action_pred' in result
        assert result['action'].shape == (B, 8, 16)
        assert result['action_pred'].shape == (B, 16, 16)


def test_diffusion_transformer_mem_policy():
    shape_meta = {
        'obs': {
            'cam_head': {'shape': [3, 240, 320], 'type': 'rgb'},
            'state': {'shape': [16], 'type': 'low_dim'}
        },
        'action': {'shape': [16]}
    }
    rgb_model = get_resnet('resnet18')
    obs_encoder = MemMultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        resize_shape=[240, 320],
        crop_shape=[216, 288],
        random_crop=False,
        use_group_norm=True,
        mem_num_frames=12,
        mem_frame_stride=15,
        mem_temporal_num_heads=4,
        mem_temporal_num_layers=2
    )

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon'
    )

    policy = DiffusionTransformerMemImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        obs_encoder=obs_encoder,
        horizon=16,
        n_action_steps=8,
        n_obs_steps=1,
        num_inference_steps=4,
        n_layer=4,
        n_head=4,
        n_emb=128
    )
    policy.set_normalizer(_get_dummy_normalizer(shape_meta))

    B, T_mem = 2, 12
    batch = {
        'obs': {
            'cam_head': torch.randn(B, T_mem, 3, 240, 320),
            'state': torch.randn(B, T_mem, 16)
        },
        'action': torch.randn(B, 16, 16)
    }

    # 1. Training loss & backward pass
    loss = policy.compute_loss(batch)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    loss.backward()

    # 2. Inference: predict_action
    policy.eval()
    with torch.no_grad():
        result = policy.predict_action(batch['obs'])
        assert 'action' in result and 'action_pred' in result
        assert result['action'].shape == (B, 8, 16)
        assert result['action_pred'].shape == (B, 16, 16)


def test_overfit_single_batch():
    """Sanity check: verifying loss strictly drops on a single fixed batch for UNet and Transformer."""
    shape_meta = {
        'obs': {
            'cam_head': {'shape': [3, 96, 96], 'type': 'rgb'},
            'state': {'shape': [4], 'type': 'low_dim'}
        },
        'action': {'shape': [4]}
    }
    
    # Test UNet MEM overfit
    rgb_model_u = get_resnet('resnet18')
    obs_encoder_u = MemMultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model_u,
        resize_shape=[96, 96],
        crop_shape=[88, 88],
        random_crop=False,
        use_group_norm=True,
        mem_num_frames=4,
        mem_frame_stride=5,
        mem_temporal_num_heads=2,
        mem_temporal_num_layers=1
    )
    scheduler_u = DDIMScheduler(
        num_train_timesteps=20,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    policy_u = DiffusionUnetMemImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler_u,
        obs_encoder=obs_encoder_u,
        horizon=8,
        n_action_steps=4,
        n_obs_steps=1,
        num_inference_steps=4,
        down_dims=[64, 128]
    )
    policy_u.set_normalizer(_get_dummy_normalizer(shape_meta))
    opt_u = policy_u.get_optimizer(lr=1e-3)

    B, T_mem = 2, 4
    batch = {
        'obs': {
            'cam_head': torch.randn(B, T_mem, 3, 96, 96),
            'state': torch.randn(B, T_mem, 4)
        },
        'action': torch.randn(B, 8, 4)
    }

    # For deterministic overfit convergence verification, use fixed seed before compute_loss
    initial_loss = None
    final_loss = None
    for step in range(25):
        torch.manual_seed(100)
        opt_u.zero_grad()
        loss = policy_u.compute_loss(batch)
        loss.backward()
        opt_u.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    print(f"UNet MEM single batch loss: step 0 = {initial_loss:.4f}, step 25 = {final_loss:.4f}")
    assert final_loss < initial_loss * 0.7, f"UNet MEM loss should decrease on fixed batch ({final_loss} vs {initial_loss})"

    # Test Transformer MEM overfit
    rgb_model_t = get_resnet('resnet18')
    obs_encoder_t = MemMultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model_t,
        resize_shape=[96, 96],
        crop_shape=[88, 88],
        random_crop=False,
        use_group_norm=True,
        mem_num_frames=4,
        mem_frame_stride=5,
        mem_temporal_num_heads=2,
        mem_temporal_num_layers=1
    )
    scheduler_t = DDIMScheduler(
        num_train_timesteps=20,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    policy_t = DiffusionTransformerMemImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler_t,
        obs_encoder=obs_encoder_t,
        horizon=8,
        n_action_steps=4,
        n_obs_steps=1,
        num_inference_steps=4,
        n_layer=2,
        n_head=2,
        n_emb=64
    )
    policy_t.set_normalizer(_get_dummy_normalizer(shape_meta))
    opt_t = policy_t.get_optimizer(learning_rate=1e-3)

    t_initial_loss = None
    t_final_loss = None
    for step in range(25):
        torch.manual_seed(100)
        opt_t.zero_grad()
        loss = policy_t.compute_loss(batch)
        loss.backward()
        opt_t.step()
        if step == 0:
            t_initial_loss = loss.item()
        t_final_loss = loss.item()

    print(f"Transformer MEM single batch loss: step 0 = {t_initial_loss:.4f}, step 25 = {t_final_loss:.4f}")
    assert t_final_loss < t_initial_loss * 0.7, f"Transformer MEM loss should decrease on fixed batch ({t_final_loss} vs {t_initial_loss})"


if __name__ == '__main__':
    print("Running test_sinusoidal_embedding...")
    test_sinusoidal_embedding()
    print("Running test_temporal_block_causality...")
    test_temporal_block_causality()
    print("Running test_mem_short_video_encoder_shapes_and_grads...")
    test_mem_short_video_encoder_shapes_and_grads()
    print("Running test_mem_multi_image_obs_encoder...")
    test_mem_multi_image_obs_encoder()
    print("Running test_mem_sampling_offsets_and_clamping...")
    test_mem_sampling_offsets_and_clamping()
    print("Running test_mem_video_buffer...")
    test_mem_video_buffer()
    print("Running test_diffusion_unet_mem_policy...")
    test_diffusion_unet_mem_policy()
    print("Running test_diffusion_transformer_mem_policy...")
    test_diffusion_transformer_mem_policy()
    print("Running test_overfit_single_batch...")
    test_overfit_single_batch()
    print("\n✅ ALL TESTS PASSED SUCCESSFULLY!")

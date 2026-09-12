from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.mem_obs_encoder import MemMultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class DiffusionUnetMemImagePolicy(BaseImagePolicy):
    """
    Diffusion Policy with Memory Video Vision Encoder (diffusion-policy mem).
    Conditions action generation on multi-camera observations enriched with causal
    short-video temporal memory across past frames.
    """
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: Union[DDPMScheduler, DDIMScheduler],
            obs_encoder: MemMultiImageObsEncoder,
            horizon: int, 
            n_action_steps: int, 
            n_obs_steps: int = 1,
            num_inference_steps: Optional[int] = None,
            obs_as_global_cond: bool = True,
            diffusion_step_embed_dim: int = 128,
            down_dims = (256, 512, 1024),
            kernel_size: int = 5,
            n_groups: int = 8,
            cond_predict_scale: bool = True,
            **kwargs):
        super().__init__()

        # Parse action dimension
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # Observation feature dimension from MemMultiImageObsEncoder
        obs_feature_dim = obs_encoder.output_shape()[0]

        # Setup diffusion model conditioning
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference ============
    def conditional_sample(self, 
            condition_data: torch.Tensor, 
            condition_mask: torch.Tensor,
            local_cond=None, 
            global_cond=None,
            generator=None,
            **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )

        # Set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. Apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. Predict model output
            model_output = model(
                trajectory, t, 
                local_cond=local_cond, 
                global_cond=global_cond
            )

            # 3. Compute previous step: x_{t-1} -> x_t
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
            ).prev_sample

        # Finally ensure condition holds
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include observation keys in shape_meta
        result: must include "action" and "action_pred" keys
        """
        assert 'past_action' not in obs_dict
        # Normalize observations
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        # Pass multi-frame memory observations to MemMultiImageObsEncoder
        global_cond = self.obs_encoder(nobs) # shape: (B, obs_feature_dim)

        # Empty data for action generation
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # Run conditional sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=None, 
            global_cond=global_cond,
            **self.kwargs
        )
        
        # Unnormalize predicted action
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Extract n_action_steps
        action = action_pred[:, :self.n_action_steps]
        
        return {
            'action': action,
            'action_pred': action_pred
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.compute_loss(batch)

    # ========= training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            lr: float = 2.0e-4, 
            weight_decay: float = 1.0e-6,
            obs_encoder_lr: Optional[float] = None,
            obs_encoder_weight_decay: Optional[float] = None,
            betas: tuple = (0.95, 0.999),
            eps: float = 1.0e-8,
            **kwargs
        ) -> torch.optim.Optimizer:
        if obs_encoder_lr is None:
            obs_encoder_lr = lr
        if obs_encoder_weight_decay is None:
            obs_encoder_weight_decay = weight_decay

        param_groups = [
            {"params": self.model.parameters(), "lr": lr, "weight_decay": weight_decay},
            {"params": self.obs_encoder.parameters(), "lr": obs_encoder_lr, "weight_decay": obs_encoder_weight_decay},
        ]
        return torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]

        # Extract vision memory global condition
        global_cond = self.obs_encoder(nobs) # (B, obs_feature_dim)
        trajectory = nactions

        # Sample noise to add to the actions
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random timesteps for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory, timesteps, 
            local_cond=None, 
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='mean')
        return loss

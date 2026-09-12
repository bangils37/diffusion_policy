from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.mem_obs_encoder import MemMultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class DiffusionTransformerMemImagePolicy(BaseImagePolicy):
    """
    Diffusion Policy with Transformer architecture and Memory Video Vision Encoder
    (diffusion-policy transformer mem).
    Conditions action generation on multi-camera observations enriched with causal
    short-video temporal memory across past frames (OpenPI architecture).
    """
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: Union[DDIMScheduler, DDPMScheduler],
            obs_encoder: MemMultiImageObsEncoder,
            horizon: int, 
            n_action_steps: int, 
            n_obs_steps: int = 1,
            num_inference_steps: Optional[int] = None,
            # arch
            n_layer: int = 8,
            n_cond_layers: int = 0,
            n_head: int = 4,
            n_emb: int = 256,
            p_drop_emb: float = 0.0,
            p_drop_attn: float = 0.3,
            causal_attn: bool = True,
            time_as_cond: bool = True,
            obs_as_cond: bool = True,
            pred_action_steps_only: bool = False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # obs feature dim from MemMultiImageObsEncoder
        obs_feature_dim = obs_encoder.output_shape()[0]

        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
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
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference ============
    def conditional_sample(self, 
            condition_data: torch.Tensor, 
            condition_mask: torch.Tensor,
            cond=None, 
            generator=None,
            **kwargs
            ) -> torch.Tensor:
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond=cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
            ).prev_sample

        # finally ensure condition holds
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include observation keys in shape_meta
        result: must include "action" and "action_pred" keys
        """
        assert 'past_action' not in obs_dict
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim

        device = self.device
        dtype = self.dtype

        # Encode temporal short-video memory observations
        nobs_features = self.obs_encoder(nobs) # (B, obs_feature_dim)

        cond = None
        cond_data = None
        cond_mask = None

        if self.obs_as_cond:
            # Reshape to (B, n_obs_steps, Do)
            cond = nobs_features.reshape(B, self.n_obs_steps, Do)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :self.n_obs_steps, Da:] = nobs_features.reshape(B, self.n_obs_steps, Do)
            cond_mask[:, :self.n_obs_steps, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs
        )
        
        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
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
            transformer_weight_decay: float = 1.0e-3, 
            obs_encoder_weight_decay: float = 1.0e-6,
            learning_rate: float = 1.0e-4, 
            betas: Tuple[float, float] = (0.9, 0.95),
            eps: float = 1.0e-8,
            weight_decay: Optional[float] = None,
            **kwargs
        ) -> torch.optim.Optimizer:
        if weight_decay is not None:
            transformer_weight_decay = weight_decay
            obs_encoder_weight_decay = weight_decay

        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay
        )
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, eps=eps
        )
        return optimizer

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]

        # extract vision memory representation
        nobs_features = self.obs_encoder(nobs) # (B, obs_feature_dim)
        Do = self.obs_feature_dim

        cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_cond:
            cond = nobs_features.reshape(batch_size, self.n_obs_steps, Do)
            if self.pred_action_steps_only:
                trajectory = nactions[:, :self.n_action_steps]
        else:
            cond_data = torch.cat([nactions, nobs_features.reshape(batch_size, 1, Do).expand(-1, nactions.shape[1], -1)], dim=-1)
            trajectory = cond_data.detach()

        # sample noise to add to actions
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        # sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()

        # add noise to clean actions
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        # compute loss mask
        loss_mask = ~self.mask_generator(trajectory.shape)

        # predict noise residual or clean sample
        pred = self.model(noisy_trajectory, timesteps, cond=cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

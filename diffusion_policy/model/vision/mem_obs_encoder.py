from typing import Dict, Tuple, Union, Optional
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.vision.mem_video import MemVideoConfig, MemShortVideoEncoder


class MemMultiImageObsEncoder(ModuleAttrMixin):
    """
    Multi-Image Observation Encoder with Short-Video Memory (Sequence Vision Encoder).
    Takes a temporal sequence of historical image frames [B, T, C, H, W] per camera,
    processes each frame with a spatial CNN/ViT backbone, and enriches the current frame
    representation with causal temporal self-attention across the past frames.
    
    Inspired by OpenPI's mem_video architecture.
    """
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str, nn.Module]],
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            use_group_norm: bool=False,
            share_rgb_model: bool=False,
            imagenet_norm: bool=False,
            # Memory video parameters
            mem_num_frames: int=12,
            mem_frame_stride: int=15,
            mem_temporal_num_heads: int=8,
            mem_temporal_num_layers: int=2,
            mem_temporal_dropout: float=0.0,
            share_mem_encoder: bool=False,
        ):
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_mem_map = nn.ModuleDict()
        key_shape_map = dict()

        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type_name = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type_name == 'rgb':
                rgb_keys.append(key)
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features
                            )
                        )
                    key_model_map[key] = this_model

                # Determine spatial transforms
                # Shape can be [C, H, W] or [T, C, H, W] in shape_meta
                c, h, w = shape[-3], shape[-2], shape[-1]
                input_shape = (c, h, w)

                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        target_h, target_w = resize_shape[key]
                    else:
                        target_h, target_w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(target_h, target_w))
                    input_shape = (c, target_h, target_w)

                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        crop_h, crop_w = crop_shape[key]
                    else:
                        crop_h, crop_w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=crop_h,
                            crop_width=crop_w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(crop_h, crop_w))

                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )

                key_transform_map[key] = nn.Sequential(this_resizer, this_randomizer, this_normalizer)

            elif type_name == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type_name}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.mem_num_frames = mem_num_frames
        self.mem_frame_stride = mem_frame_stride
        self.mem_temporal_num_heads = mem_temporal_num_heads
        self.mem_temporal_num_layers = mem_temporal_num_layers
        self.mem_temporal_dropout = mem_temporal_dropout
        self.share_mem_encoder = share_mem_encoder

        # Infer feature dimension for each rgb model to initialize MemShortVideoEncoder
        with torch.no_grad():
            for key in self.rgb_keys:
                model = self.key_model_map['rgb'] if share_rgb_model else self.key_model_map[key]
                shape = self.key_shape_map[key]
                c, h, w = shape[-3], shape[-2], shape[-1]
                dummy_img = torch.zeros(1, c, h, w)
                dummy_trans = self.key_transform_map[key](dummy_img)
                dummy_feat = model(dummy_trans)
                feat_dim = dummy_feat.shape[-1]

                if share_mem_encoder and 'shared' in key_mem_map:
                    continue

                mem_cfg = MemVideoConfig(
                    hidden_dim=feat_dim,
                    num_frames=mem_num_frames,
                    frame_stride=mem_frame_stride,
                    num_heads=mem_temporal_num_heads,
                    num_layers=mem_temporal_num_layers,
                    dropout=mem_temporal_dropout
                )
                if share_mem_encoder:
                    key_mem_map['shared'] = MemShortVideoEncoder(mem_cfg)
                else:
                    key_mem_map[key] = MemShortVideoEncoder(mem_cfg)

        self.key_mem_map = key_mem_map

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = None
        features = list()

        for key in self.rgb_keys:
            img = obs_dict[key]
            # Handle both 4D [B, C, H, W] and 5D [B, T, C, H, W]
            is_temporal = (img.ndim == 5)
            if is_temporal:
                b, t, c, h, w = img.shape
                flat_imgs = img.reshape(b * t, c, h, w)
            else:
                b, c, h, w = img.shape
                t = 1
                flat_imgs = img

            if batch_size is None:
                batch_size = b
            else:
                assert batch_size == b, f"Mismatched batch size for key {key}"

            # Apply transforms
            transformed_imgs = self.key_transform_map[key](flat_imgs)

            # Spatial backbone
            model = self.key_model_map['rgb'] if self.share_rgb_model else self.key_model_map[key]
            feat = model(transformed_imgs)

            # Shape of feat from backbone: [B*T, D] or [B*T, P, D]
            if feat.ndim == 2:
                d = feat.shape[-1]
                feat_seq = feat.reshape(b, t, d)
            elif feat.ndim == 4: # e.g. [B*T, D, H', W']
                d = feat.shape[1]
                feat_seq = feat.permute(0, 2, 3, 1).reshape(b, t, -1, d)
            elif feat.ndim == 3: # [B*T, P, D]
                p, d = feat.shape[1], feat.shape[2]
                feat_seq = feat.reshape(b, t, p, d)
            else:
                raise RuntimeError(f"Unsupported feature dim {feat.shape} from model for {key}")

            # Apply Temporal Memory Video Encoder if sequence has T > 1
            if is_temporal and t > 1:
                mem_encoder = self.key_mem_map['shared'] if self.share_mem_encoder else self.key_mem_map[key]
                # Returns [B, D] or [B, P, D] for the latest frame conditioned on history
                current_feat = mem_encoder(feat_seq)
                if current_feat.ndim > 2:
                    current_feat = current_feat.reshape(b, -1)
            else:
                # If only 1 frame, take feat as is
                current_feat = feat_seq[:, -1]
                if current_feat.ndim > 2:
                    current_feat = current_feat.reshape(b, -1)

            features.append(current_feat)

        # Process low-dimensional state inputs
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]

            # If low-dim data has temporal dimension [B, T, D], extract the latest state [B, D]
            if data.ndim == 3:
                data = data[:, -1]
            features.append(data)

        # Concatenate all camera and state features into a single observation vector
        result = torch.cat(features, dim=-1)
        return result

    @torch.no_grad()
    def output_shape(self) -> Tuple[int, ...]:
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type_name = attr.get('type', 'low_dim')
            # If shape does not include time, add time if rgb
            if type_name == 'rgb' and len(shape) == 3:
                # shape is [C, H, W], create [B, T, C, H, W]
                full_shape = (batch_size, self.mem_num_frames) + shape
            elif type_name == 'low_dim' and len(shape) == 1:
                full_shape = (batch_size, self.mem_num_frames) + shape
            else:
                full_shape = (batch_size,) + shape

            this_obs = torch.zeros(full_shape, dtype=self.dtype, device=self.device)
            example_obs_dict[key] = this_obs

        example_output = self.forward(example_obs_dict)
        return example_output.shape[1:]

from dataclasses import dataclass
from typing import Optional, Union
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(
    positions: torch.Tensor, 
    embedding_dim: int
) -> torch.Tensor:
    """
    Computes sinusoidal time embedding for 1D positions tensor of shape [T].
    Returns tensor of shape [T, embedding_dim].
    Matches OpenPI's sinusoidal_time_embedding logic.
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    device = positions.device
    dtype = positions.dtype
    half_dim = embedding_dim // 2
    fraction = torch.linspace(0.0, 1.0, half_dim, device=device, dtype=torch.float32)
    period = 1.0 * (10000.0 ** fraction)
    
    sinusoid_input = positions.unsqueeze(-1).to(torch.float32) * (1.0 / period.unsqueeze(0))
    emb = torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)
    
    # Zero out position 0 exactly as in OpenPI
    is_zero = (positions == 0).unsqueeze(-1)
    emb = torch.where(is_zero, torch.zeros_like(emb), emb)
    return emb.to(dtype=dtype)


@dataclass
class MemVideoConfig:
    hidden_dim: int
    num_frames: int = 12
    frame_stride: int = 15
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.0


class TemporalBlock(nn.Module):
    """
    Causal temporal self-attention block across sequence of frames.
    Operates on token tensor of shape [B, T, P, D] where:
      B: batch size
      T: number of frames (temporal dimension)
      P: number of spatial patches / tokens
      D: feature dimension (hidden_dim)
    """
    def __init__(self, config: MemVideoConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = config.dropout

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.mlp_in = nn.Linear(self.hidden_dim, 4 * self.hidden_dim)
        self.mlp_out = nn.Linear(4 * self.hidden_dim, self.hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t, p, d = tokens.shape
        x = self.norm1(tokens)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B*P, H, T, C) so causal attention runs along T for every spatial token
        q = einops.rearrange(q, "b t p (h c) -> (b p) h t c", h=self.num_heads)
        k = einops.rearrange(k, "b t p (h c) -> (b p) h t c", h=self.num_heads)
        v = einops.rearrange(v, "b t p (h c) -> (b p) h t c", h=self.num_heads)

        # PyTorch FlashAttention / SDPA with is_causal=True
        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            dropout_p=self.dropout if self.training else 0.0
        )
        attn_out = einops.rearrange(attn_out, "(b p) h t c -> b t p (h c)", b=b, p=p)
        tokens = tokens + self.out_proj(attn_out)

        y = self.norm2(tokens)
        y = self.mlp_in(y)
        y = F.gelu(y)
        y = self.mlp_out(y)
        return tokens + y


class MemShortVideoEncoder(nn.Module):
    """
    Sequence Vision Memory Encoder for Diffusion Policy.
    Conditioning the current observation on a short-video memory of past frames
    spaced `frame_stride` ticks apart.
    """
    def __init__(self, config: MemVideoConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            TemporalBlock(config) for _ in range(config.num_layers)
        ])

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        """
        Input:
          frame_tokens: [B, T, P, D] or [B, T, D]
        Output:
          [B, P, D] (if 4D input) or [B, D] (if 3D input) representing the
          latest frame t=-1 enriched by temporal attention over past frames.
        """
        is_3d = (frame_tokens.ndim == 3)
        if is_3d:
            # [B, T, D] -> [B, T, 1, D]
            frame_tokens = frame_tokens.unsqueeze(2)

        b, t, p, d = frame_tokens.shape
        num_frames = self.config.num_frames
        frame_stride = self.config.frame_stride

        # Construct positions ending at 0.0: [-(num_frames - 1)*stride, ..., -stride, 0.0]
        # If input T differs from config.num_frames, adjust accordingly
        positions = torch.arange(
            -(t - 1) * frame_stride, 
            frame_stride, 
            frame_stride, 
            device=frame_tokens.device, 
            dtype=torch.float32
        )[:t]
        positions[-1] = 0.0

        pos_emb = sinusoidal_time_embedding(positions, d).to(dtype=frame_tokens.dtype)
        # [1, T, 1, D]
        frame_tokens = frame_tokens + pos_emb.unsqueeze(0).unsqueeze(2)

        for block in self.blocks:
            frame_tokens = block(frame_tokens)

        # Return latest frame representation
        out = frame_tokens[:, -1]
        if is_3d:
            out = out.squeeze(1) # [B, D]
        return out

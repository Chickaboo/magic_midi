from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks.rope import RotaryEmbedding


class GQABlock(nn.Module):
    """Grouped-query causal attention with RoPE and (B,S,D)->(B,S,D) contract."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        kv = int(num_kv_heads) if num_kv_heads is not None else int(num_heads)
        kv = max(1, kv)
        if self.num_heads % kv != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({kv})"
            )
        self.num_kv_heads = int(kv)
        self.group_size = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            self.d_model, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_dropout = nn.Dropout(float(dropout))
        self.rope = RotaryEmbedding(dim=self.head_dim, base=float(rope_base))

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """Run one causal GQA update."""

        if x.ndim != 3:
            raise ValueError(f"x must be (batch, seq, d_model), got {tuple(x.shape)}")
        batch_size, seq_len, dim = x.shape
        if dim != self.d_model:
            raise ValueError(f"last dim must be {self.d_model}, got {dim}")

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )

        # Expand grouped K/V to full query heads.
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        q, k = self.rope.apply(q, k, offset=int(max(0, position_offset)))

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.out_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        out = self.out_proj(out)
        out = self.out_dropout(out)
        return out


class CausalSelfAttentionRoPE(nn.Module):
    """Standard causal multi-head attention with RoPE and full Q=K=V heads."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.core = GQABlock(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            dropout=dropout,
            rope_base=rope_base,
        )

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        return self.core(x, position_offset=position_offset)


__all__ = ["GQABlock", "CausalSelfAttentionRoPE"]

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionBias(nn.Module):
    def __init__(self, max_distance: int, num_heads: int) -> None:
        super().__init__()
        if max_distance <= 0:
            raise ValueError("max_distance must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")

        self.max_distance = int(max_distance)
        self.num_heads = int(num_heads)
        self.embeddings = nn.Embedding(2 * self.max_distance + 1, self.num_heads)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        positions = torch.arange(seq_len, device=device)
        relative = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative = relative.clamp(-self.max_distance, self.max_distance)
        relative = relative + self.max_distance
        bias = self.embeddings(relative)
        return bias.permute(2, 0, 1)


class MusicAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_relative_distance: int = 128,
        dropout: float = 0.1,
        use_relative_bias: bool = True,
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
        self.dropout = float(dropout)
        self.use_relative_bias = bool(use_relative_bias)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.out_dropout = nn.Dropout(self.dropout)

        if self.use_relative_bias:
            self.rel_bias = RelativePositionBias(max_relative_distance, self.num_heads)
        else:
            self.rel_bias = None

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Dropout(self.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must be (batch, seq_len, d_model), got {tuple(x.shape)}"
            )

        batch_size, seq_len, dim = x.shape
        if dim != self.d_model:
            raise ValueError(f"last dim must be d_model={self.d_model}, got {dim}")

        normed = self.norm1(x)
        qkv = self.qkv_proj(normed)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        causal = torch.zeros(
            (1, 1, seq_len, seq_len), device=x.device, dtype=normed.dtype
        )
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        causal = causal.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        combined_bias = causal
        if self.use_relative_bias and self.rel_bias is not None:
            bias = self.rel_bias(seq_len, x.device).to(dtype=normed.dtype)
            combined_bias = combined_bias + bias.unsqueeze(0)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=combined_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        attn_out = self.out_dropout(self.out_proj(attn_out))

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

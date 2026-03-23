from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_alibi_slopes(num_heads: int, device: torch.device) -> torch.Tensor:
    """Build ALiBi slope values for each attention head."""

    head_ids = torch.arange(1, num_heads + 1, device=device, dtype=torch.float32)
    return torch.pow(2.0, (-8.0 / float(num_heads)) * head_ids)


class RelativePositionBias(nn.Module):
    """Learned relative position bias with shape (heads, seq, seq)."""

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
        """Return learned relative position bias with shape `(heads, seq, seq)`."""

        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        positions = torch.arange(seq_len, device=device)
        relative = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative = relative.clamp(-self.max_distance, self.max_distance)
        relative = relative + self.max_distance
        bias = self.embeddings(relative)
        return bias.permute(2, 0, 1)


class ALiBiPositionBias(nn.Module):
    """Deterministic ALiBi attention bias (heads, seq, seq)."""

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        self.num_heads = int(num_heads)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return deterministic ALiBi bias with shape `(heads, seq, seq)`."""

        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        distance = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1))
        slopes = _build_alibi_slopes(self.num_heads, device)
        return -slopes.view(self.num_heads, 1, 1) * distance.unsqueeze(0)


class MusicAttentionBlock(nn.Module):
    """Causal multi-head attention block with residual FFN sublayer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_relative_distance: int = 128,
        dropout: float = 0.1,
        use_relative_bias: bool = True,
        bias_type: str = "learned",
        ffn_expansion: int = 2,
        residual_scale: float = 1.0,
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
        if ffn_expansion <= 0:
            raise ValueError("ffn_expansion must be > 0")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be > 0")

        normalized_bias_type = str(bias_type).strip().lower()
        if normalized_bias_type not in {"learned", "alibi"}:
            raise ValueError(
                f"bias_type must be either 'learned' or 'alibi', got '{bias_type}'"
            )

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.dropout = float(dropout)
        self.residual_scale = float(residual_scale)
        self.use_relative_bias = bool(use_relative_bias)
        self.bias_type = normalized_bias_type

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.out_dropout = nn.Dropout(self.dropout)

        if self.use_relative_bias:
            if self.bias_type == "alibi":
                self.rel_bias: nn.Module | None = ALiBiPositionBias(self.num_heads)
            else:
                self.rel_bias = RelativePositionBias(
                    max_relative_distance,
                    self.num_heads,
                )
        else:
            self.rel_bias = None

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * int(ffn_expansion)),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * int(ffn_expansion), self.d_model),
            nn.Dropout(self.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run causal attention + FFN residual updates on sequence features."""

        # Entry shape contract: x is (batch, seq_len, d_model).
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
            (1, 1, seq_len, seq_len),
            device=x.device,
            dtype=normed.dtype,
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

        x = x + (attn_out * float(self.residual_scale))
        x = x + (self.ffn(self.norm2(x)) * float(self.residual_scale))
        # Exit shape contract: output is (batch, seq_len, d_model).
        return x

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by swapping halves."""

    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """Minimal RoPE helper for attention q/k tensors."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")
        if base <= 1.0:
            raise ValueError("base must be > 1")

        self.dim = int(dim)
        self.base = float(base)
        self._seq_len_cached = 0
        self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0), persistent=False)

    def _build_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / float(self.dim)
            )
        )
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = torch.cos(emb).to(dtype=dtype)
        sin = torch.sin(emb).to(dtype=dtype)

        self._cos_cached = cos
        self._sin_cached = sin
        self._seq_len_cached = int(seq_len)

    def _get_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        required = int(seq_len + max(0, int(offset)))
        if (
            self._cos_cached.numel() == 0
            or self._sin_cached.numel() == 0
            or self._seq_len_cached < required
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._build_cache(required, device=device, dtype=dtype)

        start = int(max(0, offset))
        end = start + int(seq_len)
        cos = self._cos_cached[start:end]
        sin = self._sin_cached[start:end]
        return cos, sin

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to q/k with shape (batch, heads, seq, head_dim)."""

        if q.ndim != 4 or k.ndim != 4:
            raise ValueError("q and k must be rank-4 tensors")
        if q.shape[-1] != self.dim or k.shape[-1] != self.dim:
            raise ValueError(
                f"RoPE head dim mismatch: expected {self.dim}, got {q.shape[-1]} and {k.shape[-1]}"
            )

        seq_len = int(q.shape[-2])
        if int(k.shape[-2]) != seq_len:
            raise ValueError(
                f"q/k seq mismatch: {int(q.shape[-2])} vs {int(k.shape[-2])}"
            )

        cos, sin = self._get_cos_sin(
            seq_len=seq_len,
            device=q.device,
            dtype=q.dtype,
            offset=int(offset),
        )
        cos = cos.view(1, 1, seq_len, self.dim)
        sin = sin.view(1, 1, seq_len, self.dim)

        q_out = (q * cos) + (_rotate_half(q) * sin)
        k_out = (k * cos) + (_rotate_half(k) * sin)
        return q_out, k_out


__all__ = ["RotaryEmbedding"]

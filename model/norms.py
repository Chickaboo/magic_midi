from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned scale."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError("d_model must be > 0")
        self.d_model = int(d_model)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        y = x.float()
        inv_rms = torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = y * inv_rms
        return (y.to(dtype=input_dtype) * self.weight.to(dtype=input_dtype))


def round_multiple(value: float, multiple: int) -> int:
    """Round a positive value to the nearest positive multiple."""

    m = int(max(1, multiple))
    rounded = int(round(float(value) / float(m)) * m)
    return int(max(m, rounded))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward projection used by decoder-only transformer blocks."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        multiple_of: int = 64,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError("d_model must be > 0")
        self.d_model = int(d_model)
        if hidden_dim is None:
            hidden_dim = round_multiple(float(self.d_model) * 8.0 / 3.0, multiple_of)
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        self.w1 = nn.Linear(self.d_model, self.hidden_dim, bias=bool(bias))
        self.w3 = nn.Linear(self.d_model, self.hidden_dim, bias=bool(bias))
        self.drop = nn.Dropout(float(dropout))
        self.w2 = nn.Linear(self.hidden_dim, self.d_model, bias=bool(bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(F.silu(self.w1(x)) * self.w3(x)))


__all__ = ["RMSNorm", "SwiGLU", "round_multiple"]

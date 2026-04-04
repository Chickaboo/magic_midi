from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


GDN_AVAILABLE = False
_GatedDeltaNet = None


def try_import_fla():
    global GDN_AVAILABLE, _GatedDeltaNet
    try:
        from fla.layers import GatedDeltaNet as _GDN

        _GatedDeltaNet = _GDN
        GDN_AVAILABLE = True
        return True
    except Exception:
        _GatedDeltaNet = None
        GDN_AVAILABLE = False
        return False


try_import_fla()


class _GatedDeltaFallback(nn.Module):
    """Fallback approximation used when fla GatedDeltaNet is unavailable."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mix = nn.Linear(d_model, d_model * 2, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, g = self.mix(x).chunk(2, dim=-1)
        y = F.silu(u) * torch.sigmoid(g)
        y = self.out(y)
        y = self.dropout(y)
        return y


class GatedDeltaNetBlock(nn.Module):
    """Thin GDN wrapper exposing strict `(B,S,D)->(B,S,D)` behavior."""

    def __init__(
        self,
        d_model: int,
        inner_dim: int = 320,
        num_heads: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if inner_dim <= 0:
            raise ValueError("inner_dim must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if inner_dim % num_heads != 0:
            raise ValueError(
                f"inner_dim ({inner_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = int(d_model)
        self.inner_dim = int(inner_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.inner_dim // self.num_heads

        self.in_proj = (
            nn.Identity()
            if self.inner_dim == self.d_model
            else nn.Linear(self.d_model, self.inner_dim, bias=False)
        )
        self.out_proj = (
            nn.Identity()
            if self.inner_dim == self.d_model
            else nn.Linear(self.inner_dim, self.d_model, bias=False)
        )

        if GDN_AVAILABLE and _GatedDeltaNet is not None:
            self.core = _GatedDeltaNet(
                hidden_size=self.inner_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mode="chunk",
                use_short_conv=True,
            )
            self.using_fallback = False
        else:
            self.core = _GatedDeltaFallback(self.inner_dim, dropout=dropout)
            self.using_fallback = True
            warnings.warn(
                "flash-linear-attention GatedDeltaNet is unavailable; using fallback "
                "approximation block for GDN-based variants. Install "
                "`flash-linear-attention` for true GDN behavior."
            )

        self.post_dropout = nn.Dropout(float(dropout))

    def _run_core(self, x: torch.Tensor) -> torch.Tensor:
        if self.using_fallback:
            return self.core(x)

        out = self.core(x)
        if isinstance(out, tuple):
            y = out[0]
        else:
            y = out
        if not isinstance(y, torch.Tensor):
            raise TypeError(
                "Unexpected GatedDeltaNet output type: "
                f"{type(y).__name__}. Expected Tensor or tuple[Tensor, ...]."
            )
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GDN core and project back to model width."""

        if x.ndim != 3:
            raise ValueError(
                f"GatedDeltaNetBlock expects (batch, seq, d_model), got {tuple(x.shape)}"
            )
        if int(x.shape[-1]) != self.d_model:
            raise ValueError(
                f"Expected feature dim {self.d_model}, got {int(x.shape[-1])}"
            )

        y = self.in_proj(x)
        y = self._run_core(y)
        y = self.out_proj(y)
        y = self.post_dropout(y)
        return y


__all__ = ["GDN_AVAILABLE", "GatedDeltaNetBlock", "try_import_fla"]

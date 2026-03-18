from __future__ import annotations

import warnings

import torch
import torch.nn as nn

MAMBA_AVAILABLE = False

try:
    from mamba_ssm import Mamba as _Mamba

    MAMBA_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _Mamba = None
    warnings.warn(
        "mamba-ssm not available (requires CUDA). Using GRU fallback for local "
        "development. Install mamba-ssm on Colab for full performance."
    )
    warnings.warn(f"mamba-ssm import details: {exc}")


class MambaFallback(nn.Module):
    """CPU-friendly approximation of Mamba using a bidirectional GRU."""

    def __init__(self, d_model: int, expand: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_size = max(d_model * expand // 2, 1)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.out_proj = nn.Linear(2 * hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float = 0.1,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.debug = debug
        self.norm = nn.LayerNorm(d_model)

        use_real_mamba = bool(MAMBA_AVAILABLE and torch.cuda.is_available())
        self.using_fallback = not use_real_mamba

        if use_real_mamba and _Mamba is not None:
            self.core = _Mamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
        else:
            self.core = MambaFallback(d_model=d_model, expand=expand, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug:
            assert x.ndim == 3, (
                f"MambaBlock expects (batch, seq, feat), got {tuple(x.shape)}"
            )
            assert x.shape[-1] == self.d_model, (
                f"MambaBlock feature mismatch: expected {self.d_model}, got {x.shape[-1]}"
            )

        residual = x
        x = self.norm(x)
        x = self.core(x)
        out = residual + x

        if self.debug:
            assert out.shape == residual.shape, (
                f"MambaBlock output shape mismatch: expected {tuple(residual.shape)}, "
                f"got {tuple(out.shape)}"
            )

        return out

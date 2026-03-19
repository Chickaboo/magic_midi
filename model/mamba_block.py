from __future__ import annotations

import warnings

import torch
import torch.nn as nn

try:  # pragma: no cover - optional dependency on CUDA runtimes
    from mamba_ssm import Mamba as _Mamba  # pyright: ignore[reportMissingImports]

    MAMBA_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _Mamba = None
    MAMBA_AVAILABLE = False
    warnings.warn(
        "mamba-ssm not available (requires CUDA). Using causal GRU fallback for local "
        "development. Install mamba-ssm on Colab/Kaggle for full performance."
    )
    warnings.warn(f"mamba-ssm import details: {exc}")


class MambaFallback(nn.Module):
    """CPU-friendly causal approximation of Mamba using a unidirectional GRU."""

    def __init__(self, d_model: int, expand: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.expand = int(max(1, expand))
        hidden_size = max(d_model * max(1, int(expand)), 1)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.out_proj = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.parameter_scale = 0.75

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
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.core = MambaFallback(d_model=d_model, expand=expand, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entry shape contract: x is (batch, seq_len, d_model).
        if self.debug:
            assert x.ndim == 3, (
                f"MambaBlock expects (batch, seq, feat), got {tuple(x.shape)}"
            )
            assert x.shape[-1] == self.d_model, (
                f"MambaBlock feature mismatch: expected {self.d_model}, got {x.shape[-1]}"
            )

        residual = x
        y = self.norm(x)
        y = self.core(y)
        out = residual + y

        if self.debug:
            assert out.shape == residual.shape, (
                f"MambaBlock output shape mismatch: expected {tuple(residual.shape)}, "
                f"got {tuple(out.shape)}"
            )

        # Exit shape contract: output is (batch, seq_len, d_model).
        return out

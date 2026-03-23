from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency on CUDA runtimes
    from mamba_ssm import Mamba as _Mamba  # pyright: ignore[reportMissingImports]

    MAMBA_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _Mamba = None
    MAMBA_AVAILABLE = False
    warnings.warn(
        "mamba-ssm not available (requires CUDA). Using torch reference fallback for "
        "development. Install mamba-ssm on Colab/Kaggle for full performance."
    )
    warnings.warn(f"mamba-ssm import details: {exc}")


class MambaCompatFallback(nn.Module):
    """Torch reference implementation of Mamba for CPU/local runtimes."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(max(1, d_state))
        self.expand = int(max(1, expand))
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16))

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=int(max(1, d_conv)),
            groups=self.d_inner,
            padding=int(max(1, d_conv)) - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + (self.d_state * 2),
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _selective_scan_ref(
        self,
        u: torch.Tensor,
        dt: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Reference selective scan kernel used by fallback Mamba path."""

        # u, dt, z: (batch, seq, d_inner)
        # b, c: (batch, seq, d_state)
        # A_log: (d_inner, d_state), D: (d_inner,)
        batch_size, seq_len, d_inner = u.shape
        d_state = self.d_state

        state = torch.zeros(
            batch_size,
            d_inner,
            d_state,
            device=u.device,
            dtype=u.dtype,
        )
        a = -torch.exp(self.A_log.float()).to(dtype=u.dtype)
        d = self.D.to(dtype=u.dtype)

        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            u_t = u[:, t, :]  # (b, d_inner)
            dt_t = dt[:, t, :]  # (b, d_inner)
            b_t = b[:, t, :]  # (b, d_state)
            c_t = c[:, t, :]  # (b, d_state)

            delta_a = torch.exp(dt_t.unsqueeze(-1) * a.unsqueeze(0))
            delta_b_u = dt_t.unsqueeze(-1) * b_t.unsqueeze(1) * u_t.unsqueeze(-1)
            state = delta_a * state + delta_b_u

            y_t = torch.sum(state * c_t.unsqueeze(1), dim=-1)
            y_t = y_t + d.unsqueeze(0) * u_t
            y_t = y_t * F.silu(z[:, t, :])
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fallback Mamba core over sequence tensor."""

        _, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :seq_len]
        x_branch = F.silu(x_branch.transpose(1, 2))

        proj = self.x_proj(x_branch)
        dt = proj[:, :, : self.dt_rank]
        bc = proj[:, :, self.dt_rank :]
        b = bc[:, :, : self.d_state]
        c = bc[:, :, self.d_state :]

        dt = F.softplus(self.dt_proj(dt))
        y = self._selective_scan_ref(
            u=x_branch,
            dt=dt,
            b=b,
            c=c,
            z=z_branch,
        )
        y = self.out_proj(y)
        y = self.dropout(y)
        return y


class MambaBlock(nn.Module):
    """Residual Mamba block with runtime fallback selection."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        debug: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if d_state <= 0:
            raise ValueError("d_state must be > 0")
        if d_conv <= 0:
            raise ValueError("d_conv must be > 0")
        if expand <= 0:
            raise ValueError("expand must be > 0")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be > 0")

        self.d_model = d_model
        self.debug = debug
        self.residual_scale = float(residual_scale)
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
            self.core = MambaCompatFallback(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one residual Mamba update over sequence features."""

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
        y = y * float(self.residual_scale)
        out = residual + y

        if self.debug:
            assert out.shape == residual.shape, (
                f"MambaBlock output shape mismatch: expected {tuple(residual.shape)}, "
                f"got {tuple(out.shape)}"
            )

        # Exit shape contract: output is (batch, seq_len, d_model).
        return out

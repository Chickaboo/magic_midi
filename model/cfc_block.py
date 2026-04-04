from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, cast

import torch
import torch.nn as nn

from utils.logging_utils import get_project_logger

try:
    from ncps.torch import CfC as _CfC

    CFC_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _CfC = None
    CFC_AVAILABLE = False
    warnings.warn(f"ncps CfC import failed. Using GRU fallback. Details: {exc}")


LOGGER = get_project_logger()


class _CfCFallback(nn.Module):
    """Fallback recurrent block used when ncps CfC is unavailable."""

    def __init__(self, units: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=units,
            hidden_size=units,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run fallback recurrent update and return sequence plus last hidden state."""

        h0 = hidden.unsqueeze(0) if hidden is not None and hidden.dim() == 2 else hidden
        y, h = self.gru(x, h0)
        y = self.dropout(y)
        h_last = h[-1] if h.dim() == 3 else h
        return y, h_last


class CfCBlock(nn.Module):
    """Residual CfC block with LayerNorm and projection wrappers."""

    def __init__(
        self,
        d_model: int,
        cfc_units: int,
        backbone_units: int = 128,
        backbone_layers: int = 2,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        debug: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if cfc_units <= 0:
            raise ValueError("cfc_units must be > 0")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be > 0")

        self.d_model = d_model
        self.cfc_units = cfc_units
        self.residual_scale = float(residual_scale)
        self.debug = debug

        self.norm = nn.LayerNorm(d_model)
        self.input_proj = (
            nn.Linear(d_model, cfc_units) if d_model != cfc_units else nn.Identity()
        )
        self.output_proj = nn.Linear(cfc_units, d_model)
        self.dropout = nn.Dropout(dropout)

        if self.cfc_units != self.d_model:
            LOGGER.warning(
                "CfC units (%d) differ from d_model (%d). "
                "v3 presets are expected to keep these equal.",
                self.cfc_units,
                self.d_model,
            )

        if CFC_AVAILABLE and _CfC is not None:
            cfc_ctor = _CfC
            cfc = None
            creation_errors: list[str] = []

            attempts = [
                {
                    "mode": "pure",
                    "batch_first": True,
                    "return_sequences": True,
                    "backbone_units": backbone_units,
                    "backbone_layers": backbone_layers,
                    "backbone_dropout": dropout,
                },
                {
                    "mode": "pure",
                    "batch_first": True,
                    "backbone_units": backbone_units,
                    "backbone_layers": backbone_layers,
                },
                {
                    "mode": "pure",
                    "batch_first": True,
                },
                {
                    "mode": "pure",
                },
            ]

            for kwargs in attempts:
                try:
                    cfc = cfc_ctor(cfc_units, cfc_units, **kwargs)
                    break
                except (TypeError, ValueError) as exc:
                    creation_errors.append(f"kwargs={kwargs}: {exc}")

            if cfc is None:
                warnings.warn(
                    "Failed to construct ncps CfC; using GRU fallback. Errors: "
                    + " | ".join(creation_errors)
                )
                self.cfc = _CfCFallback(cfc_units, dropout=dropout)
                self.using_fallback = True
                self.cfc_mode = "gru_fallback"
            else:
                self.cfc = cfc
                self.using_fallback = False
                self.cfc_mode = "pure"
        else:
            self.cfc = _CfCFallback(cfc_units, dropout=dropout)
            self.using_fallback = True
            self.cfc_mode = "gru_fallback"

        # Discover ncps CfC call signatures once, then reuse to avoid per-step probing.
        self._call_mode: Optional[str] = None
        self._timespan_call_mode: Optional[str] = None

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Run one residual CfC block and return updated hidden state."""

        if self.debug:
            assert x.ndim == 3, (
                f"CfCBlock expects (batch, seq, feat), got {tuple(x.shape)}"
            )
            assert x.shape[-1] == self.d_model, (
                f"CfCBlock feature mismatch: expected {self.d_model}, got {x.shape[-1]}"
            )

        residual = x
        input_dtype = x.dtype
        x = self.norm(x)
        x = self.input_proj(x)

        if hidden is None:
            hidden = x.new_zeros((x.shape[0], self.cfc_units))

        x_cfc = x.float() if x.dtype != torch.float32 else x
        y, new_hidden = self.call_core(x_cfc, hidden=hidden)

        if y.dtype != input_dtype:
            y = y.to(dtype=input_dtype)
        if isinstance(new_hidden, torch.Tensor) and new_hidden.dtype != input_dtype:
            new_hidden = new_hidden.to(dtype=input_dtype)

        y = self.output_proj(y)
        y = self.dropout(y)
        y = y * float(self.residual_scale)
        out = residual + y

        if self.debug:
            assert out.shape == residual.shape, (
                f"CfCBlock output shape mismatch: expected {tuple(residual.shape)}, "
                f"got {tuple(out.shape)}"
            )

        return out, new_hidden

    def call_core(
        self,
        x: torch.Tensor,
        hidden: Any,
        timespans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Run recurrent core with optional elapsed-time deltas."""

        if timespans is None:
            return self._forward_cfc(x, hidden)

        ts = timespans
        if ts.dtype != x.dtype:
            ts = ts.to(dtype=x.dtype)
        return self._forward_cfc_with_timespans(x, hidden, ts)

    def _forward_cfc(self, x: torch.Tensor, hidden: Any) -> Tuple[torch.Tensor, Any]:
        """Call recurrent core across supported non-timespan signatures."""

        if self.using_fallback:
            out = self.cfc(x, hidden)
            return self._normalize_cfc_output(out, hidden)

        if self._call_mode == "x_hidden":
            out = self.cfc(x, hidden)
            return self._normalize_cfc_output(out, hidden)
        if self._call_mode == "x_hx":
            out = self.cfc(x, hx=hidden)
            return self._normalize_cfc_output(out, hidden)
        if self._call_mode == "x_only":
            out = self.cfc(x)
            return self._normalize_cfc_output(out, hidden)

        attempts = [
            ("x_hidden", lambda: self.cfc(x, hidden)),
            ("x_hx", lambda: self.cfc(x, hx=hidden)),
            ("x_only", lambda: self.cfc(x)),
        ]
        errors: list[str] = []
        for mode, fn in attempts:
            try:
                out = fn()
                self._call_mode = mode
                return self._normalize_cfc_output(out, hidden)
            except TypeError as exc:
                errors.append(f"{mode}: {exc}")

        raise RuntimeError(
            "CfCBlock could not resolve a non-timespan call signature. "
            + " | ".join(errors)
        )

    def _forward_cfc_with_timespans(
        self,
        x: torch.Tensor,
        hidden: Any,
        timespans: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """Call recurrent core across supported elapsed-time signatures."""

        if self.using_fallback:
            out = self.cfc(x, hidden)
            return self._normalize_cfc_output(out, hidden)

        ts2d = timespans
        ts3d = timespans.unsqueeze(-1) if timespans.ndim == 2 else timespans

        def _call_mode(mode: str) -> Any:
            if mode == "hx_ts_2d":
                return self.cfc(x, hx=hidden, timespans=ts2d)
            if mode == "hx_ts_3d":
                return self.cfc(x, hx=hidden, timespans=ts3d)
            if mode == "pos_ts_2d":
                return self.cfc(x, hidden, ts2d)
            if mode == "pos_ts_3d":
                return self.cfc(x, hidden, ts3d)
            if mode == "x_hidden":
                return self.cfc(x, hidden)
            if mode == "x_hx":
                return self.cfc(x, hx=hidden)
            if mode == "x_only":
                return self.cfc(x)
            raise ValueError(f"Unsupported CfC timespan mode: {mode}")

        cached_error: Optional[str] = None
        if self._timespan_call_mode is not None:
            try:
                out = _call_mode(self._timespan_call_mode)
                return self._normalize_cfc_output(out, hidden)
            except (TypeError, RuntimeError, ValueError) as exc:
                # Cached signature can become invalid when a probe succeeded on
                # batch-size=1 but fails on larger batches (common for 2D ts).
                cached_error = f"cached({self._timespan_call_mode}): {exc}"
                self._timespan_call_mode = None

        attempts = [
            "hx_ts_3d",
            "pos_ts_3d",
            "hx_ts_2d",
            "pos_ts_2d",
            "x_hidden",
            "x_hx",
            "x_only",
        ]
        errors: list[str] = [cached_error] if cached_error is not None else []
        for mode in attempts:
            try:
                out = _call_mode(mode)
                # Avoid persisting fragile 2D timespan modes discovered with
                # batch-size=1; they can fail later on larger training batches.
                if not (mode in {"hx_ts_2d", "pos_ts_2d"} and int(x.shape[0]) == 1):
                    self._timespan_call_mode = mode
                return self._normalize_cfc_output(out, hidden)
            except (TypeError, RuntimeError, ValueError) as exc:
                errors.append(f"{mode}: {exc}")

        raise RuntimeError(
            "CfCBlock could not resolve a timespan call signature. "
            + " | ".join(errors)
        )

    @staticmethod
    def _normalize_cfc_output(out: Any, hidden: Any) -> Tuple[torch.Tensor, Any]:
        """Normalize varying CfC return signatures to (sequence, hidden)."""

        if isinstance(out, tuple):
            if len(out) >= 2:
                return cast(torch.Tensor, out[0]), out[1]
            if len(out) == 1:
                return cast(torch.Tensor, out[0]), hidden

        return cast(torch.Tensor, out), hidden

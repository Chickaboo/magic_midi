from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, cast

import torch
import torch.nn as nn

try:
    from ncps.torch import CfC as _CfC

    CFC_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _CfC = None
    CFC_AVAILABLE = False
    warnings.warn(f"ncps CfC import failed. Using GRU fallback. Details: {exc}")


class _CfCFallback(nn.Module):
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
        h0 = hidden.unsqueeze(0) if hidden is not None and hidden.dim() == 2 else hidden
        y, h = self.gru(x, h0)
        y = self.dropout(y)
        h_last = h[-1] if h.dim() == 3 else h
        return y, h_last


class CfCBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        cfc_units: int,
        backbone_units: int = 128,
        backbone_layers: int = 2,
        dropout: float = 0.1,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.cfc_units = cfc_units
        self.debug = debug

        self.norm = nn.LayerNorm(d_model)
        self.input_proj = (
            nn.Linear(d_model, cfc_units) if d_model != cfc_units else nn.Identity()
        )
        self.output_proj = nn.Linear(cfc_units, d_model)
        self.dropout = nn.Dropout(dropout)

        if CFC_AVAILABLE and _CfC is not None:
            cfc_ctor = _CfC
            cfc = None
            mode_used = None
            creation_errors: list[str] = []

            for mode in ("pure_memory", "pure"):
                kwargs = {
                    "mode": mode,
                    "batch_first": True,
                    "backbone_units": backbone_units,
                    "backbone_layers": backbone_layers,
                    "backbone_dropout": dropout,
                }
                try:
                    cfc = cfc_ctor(cfc_units, cfc_units, **kwargs)
                    mode_used = mode
                    break
                except (TypeError, ValueError) as exc:
                    creation_errors.append(f"mode={mode}, full kwargs: {exc}")

                try:
                    cfc = cfc_ctor(
                        cfc_units,
                        cfc_units,
                        mode=mode,
                        batch_first=True,
                    )
                    mode_used = mode
                    break
                except (TypeError, ValueError) as exc:
                    creation_errors.append(f"mode={mode}, batch_first only: {exc}")

                try:
                    cfc = cfc_ctor(cfc_units, cfc_units, mode=mode)
                    mode_used = mode
                    break
                except (TypeError, ValueError) as exc:
                    creation_errors.append(f"mode={mode}, minimal: {exc}")

            if cfc is None:
                warnings.warn(
                    "Failed to construct ncps CfC; using GRU fallback. Errors: "
                    + " | ".join(creation_errors)
                )
                self.cfc = _CfCFallback(cfc_units, dropout=dropout)
                self.using_fallback = True
                self.cfc_mode = "gru_fallback"
            else:
                if mode_used != "pure_memory":
                    warnings.warn(
                        "CfC mode 'pure_memory' is unavailable in this ncps version. "
                        "Falling back to mode='pure'."
                    )
                self.cfc = cfc
                self.using_fallback = False
                self.cfc_mode = mode_used
        else:
            self.cfc = _CfCFallback(cfc_units, dropout=dropout)
            self.using_fallback = True
            self.cfc_mode = "gru_fallback"

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
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
        if self.using_fallback:
            y, new_hidden = self.cfc(x_cfc, hidden)
        else:
            y, new_hidden = self._forward_cfc(x_cfc, hidden)

        if y.dtype != input_dtype:
            y = y.to(dtype=input_dtype)
        if isinstance(new_hidden, torch.Tensor) and new_hidden.dtype != input_dtype:
            new_hidden = new_hidden.to(dtype=input_dtype)

        y = self.output_proj(y)
        y = self.dropout(y)
        out = residual + y

        if self.debug:
            assert out.shape == residual.shape, (
                f"CfCBlock output shape mismatch: expected {tuple(residual.shape)}, "
                f"got {tuple(out.shape)}"
            )

        return out, new_hidden

    def _forward_cfc(self, x: torch.Tensor, hidden: Any) -> Tuple[torch.Tensor, Any]:
        try:
            out = self.cfc(x, hidden)
        except TypeError:
            try:
                out = self.cfc(x, hx=hidden)
            except TypeError:
                out = self.cfc(x)

        if isinstance(out, tuple) and len(out) == 2:
            y = cast(torch.Tensor, out[0])
            h = out[1]
            return y, h

        if isinstance(out, tuple) and len(out) > 2:
            y = cast(torch.Tensor, out[0])
            h = out[1]
            return y, h

        y = cast(torch.Tensor, out)
        return y, hidden

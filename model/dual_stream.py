from __future__ import annotations

import torch
import torch.nn as nn


def _resolve_heads(dim: int, requested_heads: int) -> int:
    """Return a valid head count that divides one embedding dimension."""

    heads = max(1, int(requested_heads))
    while heads > 1 and (dim % heads) != 0:
        heads -= 1
    return heads


class DualStreamSplit(nn.Module):
    """Learned split from token features into harmonic and temporal streams."""

    def __init__(self, d_model: int, harmonic_dim: int, temporal_dim: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if harmonic_dim <= 0:
            raise ValueError("harmonic_dim must be > 0")
        if temporal_dim <= 0:
            raise ValueError("temporal_dim must be > 0")

        self.harmonic_proj = nn.Linear(d_model, harmonic_dim)
        self.temporal_proj = nn.Linear(d_model, temporal_dim)
        self.harmonic_norm = nn.LayerNorm(harmonic_dim)
        self.temporal_norm = nn.LayerNorm(temporal_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split token sequence into two specialized feature streams."""

        if x.ndim != 3:
            raise ValueError(
                f"x must be (batch, seq_len, d_model), got {tuple(x.shape)}"
            )
        harmonic = self.harmonic_norm(self.harmonic_proj(x))
        temporal = self.temporal_norm(self.temporal_proj(x))
        return harmonic, temporal


class CrossStreamAttention(nn.Module):
    """Bidirectional attention exchange between harmonic and temporal streams."""

    def __init__(
        self,
        harmonic_dim: int,
        temporal_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if harmonic_dim <= 0:
            raise ValueError("harmonic_dim must be > 0")
        if temporal_dim <= 0:
            raise ValueError("temporal_dim must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be > 0")

        self.harmonic_dim = int(harmonic_dim)
        self.temporal_dim = int(temporal_dim)
        self.residual_scale = float(residual_scale)

        harmonic_heads = _resolve_heads(self.harmonic_dim, int(num_heads))
        temporal_heads = _resolve_heads(self.temporal_dim, int(num_heads))

        self.pre_norm_h = nn.LayerNorm(self.harmonic_dim)
        self.pre_norm_t = nn.LayerNorm(self.temporal_dim)

        self.harmonic_from_temporal = nn.MultiheadAttention(
            embed_dim=self.harmonic_dim,
            num_heads=harmonic_heads,
            dropout=dropout,
            batch_first=True,
            kdim=self.temporal_dim,
            vdim=self.temporal_dim,
        )
        self.temporal_from_harmonic = nn.MultiheadAttention(
            embed_dim=self.temporal_dim,
            num_heads=temporal_heads,
            dropout=dropout,
            batch_first=True,
            kdim=self.harmonic_dim,
            vdim=self.harmonic_dim,
        )
        self.harmonic_dropout = nn.Dropout(dropout)
        self.temporal_dropout = nn.Dropout(dropout)
        self.harmonic_norm = nn.LayerNorm(self.harmonic_dim)
        self.temporal_norm = nn.LayerNorm(self.temporal_dim)

    def forward(
        self,
        harmonic: torch.Tensor,
        temporal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply bidirectional stream fusion updates."""

        if harmonic.ndim != 3 or temporal.ndim != 3:
            raise ValueError("harmonic and temporal must be rank-3 tensors")
        if harmonic.shape[:2] != temporal.shape[:2]:
            raise ValueError(
                "harmonic and temporal must share (batch, seq_len), "
                f"got {tuple(harmonic.shape)} vs {tuple(temporal.shape)}"
            )
        if int(harmonic.shape[-1]) != self.harmonic_dim:
            raise ValueError(
                f"harmonic dim mismatch: expected {self.harmonic_dim}, got {int(harmonic.shape[-1])}"
            )
        if int(temporal.shape[-1]) != self.temporal_dim:
            raise ValueError(
                f"temporal dim mismatch: expected {self.temporal_dim}, got {int(temporal.shape[-1])}"
            )

        harmonic_norm = self.pre_norm_h(harmonic)
        temporal_norm = self.pre_norm_t(temporal)

        h_update, _ = self.harmonic_from_temporal(
            harmonic_norm,
            temporal_norm,
            temporal_norm,
            need_weights=False,
        )
        harmonic = self.harmonic_norm(
            harmonic + self.harmonic_dropout(h_update * float(self.residual_scale))
        )

        t_update, _ = self.temporal_from_harmonic(
            temporal_norm,
            harmonic_norm,
            harmonic_norm,
            need_weights=False,
        )
        temporal = self.temporal_norm(
            temporal + self.temporal_dropout(t_update * float(self.residual_scale))
        )
        return harmonic, temporal

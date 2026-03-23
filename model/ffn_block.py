from __future__ import annotations

import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """Residual feed-forward block used when CfC is disabled."""

    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        dropout: float = 0.1,
        debug: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if expansion <= 0:
            raise ValueError("expansion must be > 0")

        self.d_model = int(d_model)
        self.hidden_dim = int(d_model * expansion)
        self.debug = bool(debug)

        self.norm = nn.LayerNorm(self.d_model)
        self.fc1 = nn.Linear(self.d_model, self.hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_dim, self.d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one residual FFN update on sequence features."""

        # Entry shape contract: x is (batch, seq_len, d_model).
        if self.debug:
            assert x.ndim == 3, (
                f"FeedForwardBlock expects (batch, seq, feat), got {tuple(x.shape)}"
            )
            assert x.shape[-1] == self.d_model, (
                "FeedForwardBlock feature mismatch: "
                f"expected {self.d_model}, got {x.shape[-1]}"
            )

        residual = x
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        out = residual + y

        if self.debug:
            assert out.shape == residual.shape, (
                f"FeedForwardBlock output shape mismatch: expected {tuple(residual.shape)}, "
                f"got {tuple(out.shape)}"
            )

        # Exit shape contract: output is (batch, seq_len, d_model).
        return out

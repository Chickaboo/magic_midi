from __future__ import annotations

import torch
import torch.nn as nn


class ContinuousTimeEncoding(nn.Module):
    """Encode absolute musical onset time in seconds at multiple timescales."""

    def __init__(self, d_model: int, max_time_seconds: float = 600.0) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")

        self.d_model = int(d_model)
        self.max_time_seconds = float(max_time_seconds)
        if self.max_time_seconds < 1.0:
            self.max_time_seconds = 1.0

        timescales = torch.tensor(
            [
                0.05,
                0.1,
                0.2,
                0.5,
                1.0,
                2.0,
                4.0,
                8.0,
                16.0,
                32.0,
                64.0,
                128.0,
                256.0,
                512.0,
            ],
            dtype=torch.float32,
        )
        self.register_buffer("timescales", timescales)

        self.projection = nn.Linear(int(timescales.numel()) * 2, self.d_model)
        self.output_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)

    def forward(self, onset_times: torch.Tensor) -> torch.Tensor:
        """Return time encoding for onset tensor shaped `(batch, seq_len)`."""

        if onset_times.ndim != 2:
            raise ValueError(
                "onset_times must be shaped (batch, seq_len), "
                f"got {tuple(onset_times.shape)}"
            )

        onset = onset_times.to(dtype=torch.float32)
        onset = torch.relu(onset)
        max_time = float(self.max_time_seconds)
        onset = onset - torch.relu(onset - max_time)

        timescales = torch.reshape(self.timescales, (1, 1, -1))
        scaled = onset.unsqueeze(-1) / timescales
        features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        encoded = self.projection(features)
        scale = torch.sigmoid(self.output_scale)
        return encoded * scale

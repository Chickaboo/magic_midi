from __future__ import annotations

import torch
import torch.nn.functional as F


def next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(
            f"logits must be (batch, seq, vocab), got {tuple(logits.shape)}"
        )
    if targets.ndim != 2:
        raise ValueError(f"targets must be (batch, seq), got {tuple(targets.shape)}")
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            f"Mismatch logits and targets shapes: {tuple(logits.shape[:2])} vs {tuple(targets.shape)}"
        )

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=float(label_smoothing),
    )


def create_targets(
    seed: torch.Tensor,
    continuation: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    if seed.ndim != 2 or continuation.ndim != 2:
        raise ValueError("seed and continuation must be rank-2 tensors")
    if seed.shape[0] != continuation.shape[0]:
        raise ValueError("seed and continuation batch sizes must match")

    full = torch.cat([seed, continuation], dim=1)
    targets = torch.full_like(full, fill_value=ignore_index)

    if full.shape[1] > 1:
        shifted = full[:, 1:]
        targets[:, :-1] = shifted

    targets[:, : seed.shape[1]] = ignore_index
    return targets

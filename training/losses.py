from __future__ import annotations

import torch
import torch.nn.functional as F


def next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    piece_boundary_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute next-token cross-entropy loss with optional label smoothing."""

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

    if piece_boundary_mask is None:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=ignore_index,
            label_smoothing=float(label_smoothing),
        )

    if piece_boundary_mask.shape != targets.shape:
        raise ValueError(
            "piece_boundary_mask must match targets shape, "
            f"got {tuple(piece_boundary_mask.shape)} vs {tuple(targets.shape)}"
        )

    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=float(label_smoothing),
        reduction="none",
    ).view_as(targets)

    valid = (targets != int(ignore_index)) & (~piece_boundary_mask.to(dtype=torch.bool))
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return per_token.new_zeros(())
    return per_token[valid].mean()


def create_targets(
    seed: torch.Tensor,
    continuation: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Create autoregressive targets masking seed region with ignore index."""

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


def build_piece_boundary_mask(
    seed: torch.Tensor,
    continuation: torch.Tensor,
    new_piece: torch.Tensor,
) -> torch.Tensor:
    """Mark invalid next-token targets at piece junction positions."""

    if seed.ndim != 2 or continuation.ndim != 2:
        raise ValueError("seed and continuation must be rank-2 tensors")
    if new_piece.ndim != 1:
        raise ValueError("new_piece must be rank-1 tensor")
    if seed.shape[0] != continuation.shape[0] or seed.shape[0] != new_piece.shape[0]:
        raise ValueError("seed, continuation, and new_piece batch sizes must match")

    total_len = int(seed.shape[1] + continuation.shape[1])
    mask = torch.zeros((seed.shape[0], total_len), dtype=torch.bool, device=seed.device)
    if total_len <= 1:
        return mask

    new_piece_bool = new_piece.to(dtype=torch.bool)
    for batch_idx in range(1, int(new_piece_bool.shape[0])):
        if bool(new_piece_bool[batch_idx]):
            mask[batch_idx - 1, total_len - 1] = True
    return mask

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _build_slot_allowed_mask(
    *,
    seq_len: int,
    vocab_size: int,
    event_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return `(seq_len, vocab_size)` boolean mask of valid token IDs per next-token slot."""

    if int(event_size) != 4 or seq_len <= 0 or vocab_size <= 0:
        return torch.ones((max(0, seq_len), max(0, vocab_size)), dtype=torch.bool, device=device)

    # Next-token slot at position t predicts token t+1.
    slot_ids = (torch.arange(seq_len, device=device, dtype=torch.long) + 1) % 4
    allowed = torch.zeros((seq_len, vocab_size), dtype=torch.bool, device=device)

    ranges = {
        0: (0, 32),    # delta
        1: (32, 120),  # pitch
        2: (120, 152), # duration
        3: (152, 168), # velocity
    }
    for slot, (start, end) in ranges.items():
        valid_start = max(0, int(start))
        valid_end = min(int(vocab_size), int(end))
        if valid_end <= valid_start:
            continue
        slot_rows = slot_ids == int(slot)
        if bool(slot_rows.any().item()):
            allowed[slot_rows, valid_start:valid_end] = True

    # Safety: never allow an empty class row.
    row_has_class = allowed.any(dim=-1)
    if bool((~row_has_class).any().item()):
        allowed[~row_has_class, :] = True
    return allowed


def _apply_slot_aware_logits(
    logits: torch.Tensor,
    *,
    event_size: int,
) -> torch.Tensor:
    """Mask logits to valid classes per event slot for next-token training."""

    if logits.ndim != 3:
        return logits

    allowed = _build_slot_allowed_mask(
        seq_len=int(logits.shape[1]),
        vocab_size=int(logits.shape[2]),
        event_size=int(max(1, event_size)),
        device=logits.device,
    )
    if logits.dtype.is_floating_point:
        fill_value = torch.finfo(logits.dtype).min
    else:
        fill_value = -1e9
    return logits.masked_fill(~allowed.unsqueeze(0), fill_value)


def next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    piece_boundary_mask: torch.Tensor | None = None,
    slot_aware: bool = False,
    event_size: int = 4,
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

    masked_logits = (
        _apply_slot_aware_logits(logits, event_size=int(max(1, event_size)))
        if bool(slot_aware)
        else logits
    )

    if piece_boundary_mask is None:
        return F.cross_entropy(
            masked_logits.reshape(-1, masked_logits.shape[-1]),
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
        masked_logits.reshape(-1, masked_logits.shape[-1]),
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


def next_token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    piece_boundary_mask: torch.Tensor | None = None,
    slot_aware: bool = False,
    event_size: int = 4,
) -> float:
    """Return token accuracy over valid next-token positions."""

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

    eval_logits = (
        _apply_slot_aware_logits(logits, event_size=int(max(1, event_size)))
        if bool(slot_aware)
        else logits
    )
    pred = torch.argmax(eval_logits, dim=-1)

    valid = targets != int(ignore_index)
    if piece_boundary_mask is not None:
        if piece_boundary_mask.shape != targets.shape:
            raise ValueError(
                "piece_boundary_mask must match targets shape, "
                f"got {tuple(piece_boundary_mask.shape)} vs {tuple(targets.shape)}"
            )
        valid = valid & (~piece_boundary_mask.to(dtype=torch.bool))

    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return 0.0
    correct = (pred == targets) & valid
    return float(correct.sum().item() / float(valid_count))


def next_token_slot_accuracies(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    piece_boundary_mask: torch.Tensor | None = None,
    event_size: int = 4,
    slot_aware: bool = False,
) -> Dict[str, float]:
    """Return per-slot token accuracy for quad-event tokenization."""

    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[:2] != targets.shape:
        raise ValueError("logits and targets must be shaped (batch, seq, vocab)/(batch, seq)")

    eval_logits = (
        _apply_slot_aware_logits(logits, event_size=int(max(1, event_size)))
        if bool(slot_aware)
        else logits
    )
    pred = torch.argmax(eval_logits, dim=-1)

    valid = targets != int(ignore_index)
    if piece_boundary_mask is not None:
        if piece_boundary_mask.shape != targets.shape:
            raise ValueError(
                "piece_boundary_mask must match targets shape, "
                f"got {tuple(piece_boundary_mask.shape)} vs {tuple(targets.shape)}"
            )
        valid = valid & (~piece_boundary_mask.to(dtype=torch.bool))

    e_size = int(max(1, event_size))
    slot_ids = (torch.arange(targets.shape[1], device=targets.device, dtype=torch.long) + 1) % e_size
    slot_ids = slot_ids.unsqueeze(0).expand_as(targets)

    if e_size == 4:
        slot_names = ["delta", "pitch", "duration", "velocity"]
    else:
        slot_names = [f"slot_{i}" for i in range(e_size)]

    results: Dict[str, float] = {}
    for slot_index, slot_name in enumerate(slot_names):
        slot_valid = valid & (slot_ids == int(slot_index))
        denom = int(slot_valid.sum().item())
        if denom <= 0:
            results[slot_name] = 0.0
            continue
        slot_correct = ((pred == targets) & slot_valid).sum().item()
        results[slot_name] = float(slot_correct / float(denom))
    return results

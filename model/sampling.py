from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SamplingDiagnostics:
    """Per-step diagnostics emitted by sampling distribution builders."""

    raw_top1_prob: torch.Tensor
    final_top1_prob: torch.Tensor
    candidate_count: torch.Tensor


def _apply_repetition_penalty(
    logits: torch.Tensor,
    context_tokens: torch.Tensor,
    repetition_penalty: float,
    recent_window: int,
) -> torch.Tensor:
    if repetition_penalty <= 1.0:
        return logits
    if recent_window <= 0:
        return logits
    if context_tokens.ndim != 2:
        raise ValueError(
            f"context_tokens must be (batch, seq_len), got {tuple(context_tokens.shape)}"
        )

    adjusted = logits.clone()
    recent = context_tokens[:, -min(recent_window, context_tokens.shape[1]) :]
    batch_size = adjusted.shape[0]
    for batch_idx in range(batch_size):
        token_ids = torch.unique(recent[batch_idx])
        token_logits = adjusted[batch_idx, token_ids]
        adjusted[batch_idx, token_ids] = torch.where(
            token_logits < 0,
            token_logits * repetition_penalty,
            token_logits / repetition_penalty,
        )
    return adjusted


def _enforce_min_candidates(
    candidate_mask: torch.Tensor,
    logits: torch.Tensor,
    min_tokens_to_keep: int,
) -> torch.Tensor:
    if min_tokens_to_keep <= 0:
        return candidate_mask

    batch_size, vocab_size = logits.shape
    keep_n = min(max(1, int(min_tokens_to_keep)), vocab_size)
    current_counts = candidate_mask.sum(dim=-1)
    needs_fix = current_counts < keep_n
    if not bool(needs_fix.any()):
        return candidate_mask

    fixed = candidate_mask.clone()
    top_idx = torch.topk(logits, k=keep_n, dim=-1).indices
    for batch_idx in range(batch_size):
        if bool(needs_fix[batch_idx]):
            fixed[batch_idx, top_idx[batch_idx]] = True
    return fixed


def _apply_topk_topp_filter(
    logits: torch.Tensor,
    top_k: Optional[int],
    top_p: Optional[float],
    min_tokens_to_keep: int,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must be (batch, vocab), got {tuple(logits.shape)}")

    batch_size, vocab_size = logits.shape
    keep_k = vocab_size
    if top_k is not None and top_k > 0:
        keep_k = min(max(int(top_k), int(min_tokens_to_keep)), vocab_size)

    topk_indices = torch.topk(logits, k=keep_k, dim=-1).indices
    candidate_mask = torch.zeros_like(logits, dtype=torch.bool)
    candidate_mask.scatter_(dim=-1, index=topk_indices, value=True)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative_probs > float(top_p)
        if min_tokens_to_keep > 0:
            remove_mask[..., : int(min_tokens_to_keep)] = False

        nucleus_keep = ~remove_mask
        nucleus_mask = torch.zeros_like(candidate_mask)
        nucleus_mask.scatter_(dim=-1, index=sorted_indices, src=nucleus_keep)
        candidate_mask = candidate_mask & nucleus_mask

    candidate_mask = _enforce_min_candidates(
        candidate_mask,
        logits,
        min_tokens_to_keep=min_tokens_to_keep,
    )

    filtered = logits.masked_fill(~candidate_mask, float("-inf"))
    return filtered


def _cap_top1_probability(
    probs: torch.Tensor,
    cap: Optional[float],
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    if cap is None:
        return probs
    if not (0.0 < cap < 1.0):
        return probs

    adjusted = probs.clone()
    batch_size = adjusted.shape[0]
    for batch_idx in range(batch_size):
        row = adjusted[batch_idx]
        pmax = float(row.max().item())
        if pmax <= cap:
            continue

        active = candidate_mask[batch_idx]
        active_count = int(active.sum().item())
        if active_count <= 1:
            continue

        uniform_prob = 1.0 / float(active_count)
        if pmax <= uniform_prob + 1e-12:
            continue

        alpha = (pmax - cap) / max(pmax - uniform_prob, 1e-12)
        alpha = max(0.0, min(1.0, alpha))

        uniform = torch.zeros_like(row)
        uniform[active] = uniform_prob
        row = (1.0 - alpha) * row + alpha * uniform
        row = row / row.sum().clamp_min(1e-12)

        if float(row.max().item()) > cap + 1e-6:
            row = uniform

        adjusted[batch_idx] = row

    return adjusted


def build_sampling_distribution(
    logits: torch.Tensor,
    context_tokens: torch.Tensor,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: float,
    recent_window: int,
    min_tokens_to_keep: int,
    top1_cap: Optional[float] = 0.95,
) -> tuple[torch.Tensor, SamplingDiagnostics]:
    """Build filtered sampling distribution and diagnostics for next-token draw."""

    if logits.ndim != 2:
        raise ValueError(f"logits must be (batch, vocab), got {tuple(logits.shape)}")

    temperature = max(float(temperature), 0.1)
    min_tokens_to_keep = max(int(min_tokens_to_keep), 1)

    penalized = _apply_repetition_penalty(
        logits=logits,
        context_tokens=context_tokens,
        repetition_penalty=float(repetition_penalty),
        recent_window=int(recent_window),
    )
    scaled = penalized / temperature

    raw_probs = torch.softmax(scaled, dim=-1)
    raw_top1_prob = raw_probs.max(dim=-1).values

    filtered_logits = _apply_topk_topp_filter(
        logits=scaled,
        top_k=top_k,
        top_p=top_p,
        min_tokens_to_keep=min_tokens_to_keep,
    )
    candidate_mask = torch.isfinite(filtered_logits)

    probs = torch.softmax(filtered_logits, dim=-1)
    probs = _cap_top1_probability(
        probs=probs,
        cap=top1_cap,
        candidate_mask=candidate_mask,
    )
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    diagnostics = SamplingDiagnostics(
        raw_top1_prob=raw_top1_prob,
        final_top1_prob=probs.max(dim=-1).values,
        candidate_count=candidate_mask.sum(dim=-1),
    )
    return probs, diagnostics


def sample_next_token(
    logits: torch.Tensor,
    context_tokens: torch.Tensor,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: float,
    recent_window: int,
    min_tokens_to_keep: int,
    top1_cap: Optional[float] = 0.95,
) -> tuple[torch.Tensor, SamplingDiagnostics]:
    """Sample one next token from filtered distribution and return diagnostics."""

    probs, diagnostics = build_sampling_distribution(
        logits=logits,
        context_tokens=context_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        recent_window=recent_window,
        min_tokens_to_keep=min_tokens_to_keep,
        top1_cap=top1_cap,
    )
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token, diagnostics

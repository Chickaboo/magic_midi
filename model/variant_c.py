from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from model.blocks.gqa_block import CausalSelfAttentionRoPE
from model.sampling import sample_next_token


@dataclass
class VariantCConfig:
    vocab_size: int = 171
    d_model: int = 512
    n_layers: int = 4
    max_sequence_length: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.1
    tie_embeddings: bool = True
    embedding_init_std: float = 0.02
    output_logit_scale: Optional[float] = None

    # Attn (MHA+RoPE) -> FFN (4x, GELU) repeating block.
    num_attention_heads: int = 8
    ffn_expansion: int = 4

    # Trainer compatibility gate (lets Trainer pass onset_times unchanged).
    use_v2_architecture: bool = True


class _VariantCFFN(nn.Module):
    def __init__(self, d_model: int, expansion: int, dropout: float) -> None:
        super().__init__()
        inner = int(d_model * expansion)
        self.fc1 = nn.Linear(d_model, inner)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(inner, d_model)
        self.drop2 = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class _VariantCBlock(nn.Module):
    """One repeating Variant-C block: Attn(MHA+RoPE) -> FFN(4x,GELU)."""

    def __init__(self, cfg: VariantCConfig) -> None:
        super().__init__()
        d = int(cfg.d_model)

        self.norm_attn = nn.LayerNorm(d)
        self.attn = CausalSelfAttentionRoPE(
            d_model=d,
            num_heads=int(cfg.num_attention_heads),
            dropout=float(cfg.attention_dropout),
        )

        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = _VariantCFFN(
            d_model=d,
            expansion=int(cfg.ffn_expansion),
            dropout=float(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, position_offset: int) -> torch.Tensor:
        x = x + self.attn(
            self.norm_attn(x), position_offset=int(max(0, position_offset))
        )
        x = x + self.ffn(self.norm_ffn(x))
        return x


class VariantCModel(nn.Module):
    """Ablation Variant C model (attention + FFN control)."""

    def __init__(self, config: Optional[VariantCConfig] = None) -> None:
        super().__init__()
        self.config = config or VariantCConfig()
        cfg = self.config

        self.vocab_size = int(cfg.vocab_size)
        self.d_model = int(cfg.d_model)
        self.max_sequence_length = int(cfg.max_sequence_length)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_sequence_length, self.d_model)
        self.dropout = nn.Dropout(float(cfg.dropout))

        self.layers = nn.ModuleList(
            [_VariantCBlock(cfg) for _ in range(int(cfg.n_layers))]
        )

        self.final_norm = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if bool(cfg.tie_embeddings):
            self.lm_head.weight = self.token_embedding.weight

        if cfg.output_logit_scale is None:
            self.output_logit_scale = 1.0 / math.sqrt(float(self.d_model))
        else:
            self.output_logit_scale = float(cfg.output_logit_scale)

        self._reset_parameters()
        self.last_generation_stats: Dict[str, Any] = {}

    def _reset_parameters(self) -> None:
        std = float(max(1e-6, self.config.embedding_init_std))
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=std)
        if self.lm_head.weight.data_ptr() != self.token_embedding.weight.data_ptr():
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

    @staticmethod
    def _to_seed_tensor(
        seed_tokens: Sequence[int] | torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(seed_tokens, torch.Tensor):
            if seed_tokens.ndim == 1:
                seed = seed_tokens.unsqueeze(0)
            elif seed_tokens.ndim == 2 and int(seed_tokens.shape[0]) == 1:
                seed = seed_tokens
            else:
                raise ValueError("seed tensor must be shape (seq,) or (1, seq)")
            return seed.to(device=device, dtype=torch.long)
        vals = [int(t) for t in seed_tokens]
        if not vals:
            raise ValueError("seed_tokens cannot be empty")
        return torch.tensor(vals, dtype=torch.long, device=device).unsqueeze(0)

    @staticmethod
    def _triplet_slot(index: int) -> int:
        return int(index % 4)

    @staticmethod
    def _allowed_ids_for_slot(slot: int, vocab_size: int) -> torch.Tensor:
        if slot == 0:
            return torch.arange(0, 32, dtype=torch.long)
        if slot == 1:
            return torch.arange(32, 120, dtype=torch.long)
        if slot == 2:
            return torch.arange(120, 152, dtype=torch.long)
        if slot == 3:
            return torch.arange(152, 168, dtype=torch.long)
        return torch.arange(0, vocab_size, dtype=torch.long)

    def _mask_logits_to_triplet_slot(
        self,
        logits: torch.Tensor,
        slot: int,
    ) -> torch.Tensor:
        mask = torch.full_like(logits, fill_value=-float("inf"))
        allowed = self._allowed_ids_for_slot(slot, logits.shape[-1]).to(logits.device)
        mask[:, allowed] = logits[:, allowed]
        return mask

    @staticmethod
    def _delta_from_token_events(
        token_id: int,
        token_id_to_events: Any,
        default_step: float,
    ) -> float:
        if callable(token_id_to_events):
            try:
                events = token_id_to_events(int(token_id))
                if isinstance(events, str):
                    events = [events]
                if isinstance(events, (list, tuple)):
                    for ev in events:
                        text = str(ev)
                        if text.startswith("Delta_"):
                            return float(max(1e-4, float(text.split("_", 1)[1])))
            except Exception:
                pass

        if 0 <= int(token_id) <= 31:
            bins = torch.cat(
                [
                    torch.tensor([0.0], dtype=torch.float32),
                    torch.logspace(
                        math.log10(1e-4),
                        math.log10(2.0),
                        steps=31,
                        dtype=torch.float32,
                    ),
                ]
            )
            return float(max(1e-4, bins[int(token_id)].item()))
        return float(max(1e-4, default_step))

    def forward(
        self,
        token_ids: torch.Tensor,
        onset_times: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        memory: Optional[Any] = None,
        return_memory: bool = False,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Any]] | torch.Tensor:
        del memory

        if token_ids.ndim != 2 or onset_times.ndim != 2:
            raise ValueError("token_ids and onset_times must be rank-2")
        if token_ids.shape != onset_times.shape:
            raise ValueError("token_ids and onset_times must have same shape")
        if durations is not None and durations.shape != token_ids.shape:
            raise ValueError("durations must match token_ids shape")

        bsz, seq_len = token_ids.shape
        positions = torch.arange(
            int(max(0, position_offset)),
            int(max(0, position_offset)) + int(seq_len),
            device=token_ids.device,
        )
        positions = torch.clamp(positions, max=self.max_sequence_length - 1)
        positions = positions.unsqueeze(0).expand(bsz, -1)

        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, position_offset=int(max(0, position_offset)))

        logits = self.lm_head(self.final_norm(x)) * float(self.output_logit_scale)
        if return_memory:
            return logits, None
        return logits

    @torch.no_grad()
    def generate(
        self,
        seed_tokens: Sequence[int] | torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        repetition_window: int = 64,
        min_tokens_to_keep: int = 3,
        seed_onset_times: Sequence[float] | torch.Tensor | None = None,
        step_seconds: float = 0.1,
        token_id_to_events: Any = None,
        max_consecutive_zero_deltas: int = 8,
    ) -> List[int]:
        self.eval()
        device = next(self.parameters()).device

        tokens = self._to_seed_tensor(seed_tokens, device=device)
        if seed_onset_times is None:
            onsets = (
                torch.arange(tokens.shape[1], device=device, dtype=torch.float32)
                * float(max(1e-4, step_seconds))
            ).unsqueeze(0)
        else:
            if isinstance(seed_onset_times, torch.Tensor):
                on = seed_onset_times
                if on.ndim == 1:
                    on = on.unsqueeze(0)
                onsets = on.to(device=device, dtype=torch.float32)
            else:
                onsets = torch.tensor(
                    [float(v) for v in seed_onset_times],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
        if onsets.shape != tokens.shape:
            raise ValueError("seed_onset_times shape must match seed token shape")

        final_top1_probs: List[float] = []
        raw_top1_probs: List[float] = []
        candidate_counts: List[int] = []
        zero_delta_streak = 0
        max_zero_delta = max(0, int(max_consecutive_zero_deltas))

        for _ in range(int(max_new_tokens)):
            context_tokens = tokens[:, -self.max_sequence_length :]
            context_onsets = onsets[:, -self.max_sequence_length :]
            context_offset = max(0, int(tokens.shape[1] - context_tokens.shape[1]))

            logits, _ = self.forward(
                token_ids=context_tokens,
                onset_times=context_onsets,
                memory=None,
                return_memory=True,
                position_offset=context_offset,
            )

            next_slot = self._triplet_slot(int(tokens.shape[1]))
            masked_logits = self._mask_logits_to_triplet_slot(
                logits[:, -1, :],
                next_slot,
            )
            if (
                next_slot == 0
                and max_zero_delta > 0
                and zero_delta_streak >= max_zero_delta
                and masked_logits.shape[-1] > 1
            ):
                # Prevent pathological near-simultaneous note piles by forcing a
                # non-zero delta token after too many consecutive zero-delta events.
                valid_non_zero = bool(torch.isfinite(masked_logits[:, 1:]).any())
                if valid_non_zero:
                    masked_logits = masked_logits.clone()
                    masked_logits[:, 0] = float("-inf")
            next_token, diagnostics = sample_next_token(
                logits=masked_logits,
                context_tokens=context_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                recent_window=repetition_window,
                min_tokens_to_keep=max(4, min_tokens_to_keep),
                top1_cap=0.95,
            )

            final_top1_probs.extend(
                [float(v) for v in diagnostics.final_top1_prob.detach().cpu().tolist()]
            )
            raw_top1_probs.extend(
                [float(v) for v in diagnostics.raw_top1_prob.detach().cpu().tolist()]
            )
            candidate_counts.extend(
                [int(v) for v in diagnostics.candidate_count.detach().cpu().tolist()]
            )

            tokens = torch.cat([tokens, next_token], dim=1)
            slot = self._triplet_slot(int(tokens.shape[1] - 1))
            tok = int(next_token.view(-1)[0].item())
            delta = float(max(1e-4, step_seconds))
            if slot == 0:
                if tok == 0:
                    zero_delta_streak += 1
                else:
                    zero_delta_streak = 0
                delta = self._delta_from_token_events(
                    token_id=tok,
                    token_id_to_events=token_id_to_events,
                    default_step=step_seconds,
                )
            next_onset = onsets[:, -1:] + (delta if slot == 0 else 0.0)
            onsets = torch.cat([onsets, next_onset], dim=1)

        self.last_generation_stats = {
            "raw_top1_max": max(raw_top1_probs) if raw_top1_probs else 0.0,
            "final_top1_max": max(final_top1_probs) if final_top1_probs else 0.0,
            "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
            "generated_tokens": int(max_new_tokens),
        }
        return tokens.squeeze(0).tolist()

    @torch.no_grad()
    def generation_health_check(
        self,
        seed_tokens: Sequence[int] | torch.Tensor,
        steps: int = 20,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        repetition_window: int = 64,
        min_tokens_to_keep: int = 3,
        top1_threshold: float = 0.95,
        raise_on_failure: bool = True,
    ) -> Dict[str, float | bool]:
        self.eval()
        device = next(self.parameters()).device
        tokens = self._to_seed_tensor(seed_tokens, device=device)
        onsets = (
            torch.arange(tokens.shape[1], device=device, dtype=torch.float32) * 0.1
        ).unsqueeze(0)

        final_top1_probs: List[float] = []
        raw_top1_probs: List[float] = []
        candidate_counts: List[int] = []

        for _ in range(int(steps)):
            context_tokens = tokens[:, -self.max_sequence_length :]
            context_onsets = onsets[:, -self.max_sequence_length :]
            context_offset = max(0, int(tokens.shape[1] - context_tokens.shape[1]))

            logits, _ = self.forward(
                token_ids=context_tokens,
                onset_times=context_onsets,
                memory=None,
                return_memory=True,
                position_offset=context_offset,
            )

            next_slot = self._triplet_slot(int(tokens.shape[1]))
            masked_logits = self._mask_logits_to_triplet_slot(
                logits[:, -1, :],
                next_slot,
            )
            next_token, diagnostics = sample_next_token(
                logits=masked_logits,
                context_tokens=context_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                recent_window=repetition_window,
                min_tokens_to_keep=max(4, min_tokens_to_keep),
                top1_cap=top1_threshold,
            )

            final_top1_probs.extend(
                [float(v) for v in diagnostics.final_top1_prob.detach().cpu().tolist()]
            )
            raw_top1_probs.extend(
                [float(v) for v in diagnostics.raw_top1_prob.detach().cpu().tolist()]
            )
            candidate_counts.extend(
                [int(v) for v in diagnostics.candidate_count.detach().cpu().tolist()]
            )

            tokens = torch.cat([tokens, next_token], dim=1)
            slot = self._triplet_slot(int(tokens.shape[1] - 1))
            tok = int(next_token.view(-1)[0].item())
            delta = self._delta_from_token_events(tok, None, 0.1)
            next_onset = onsets[:, -1:] + (delta if slot == 0 else 0.0)
            onsets = torch.cat([onsets, next_onset], dim=1)

        max_final_top1 = max(final_top1_probs) if final_top1_probs else 0.0
        passed = bool(max_final_top1 <= float(top1_threshold) + 1e-6)
        result: Dict[str, float | bool] = {
            "passed": passed,
            "max_final_top1_prob": float(max_final_top1),
            "mean_final_top1_prob": float(
                sum(final_top1_probs) / max(1, len(final_top1_probs))
            ),
            "max_raw_top1_prob": float(max(raw_top1_probs) if raw_top1_probs else 0.0),
            "mean_raw_top1_prob": float(
                sum(raw_top1_probs) / max(1, len(raw_top1_probs))
            ),
            "min_candidate_count": float(
                min(candidate_counts) if candidate_counts else 0
            ),
        }
        if raise_on_failure and not passed:
            raise AssertionError(
                "Generation health check failed: "
                f"max_final_top1_prob={max_final_top1:.4f} exceeds threshold={top1_threshold:.4f}."
            )
        return result


__all__ = ["VariantCConfig", "VariantCModel"]

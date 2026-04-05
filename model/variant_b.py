from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from model.blocks.gqa_block import CausalSelfAttentionRoPE
from model.cfc_block import CfCBlock
from model.sampling import sample_next_token


@dataclass
class VariantBConfig:
    vocab_size: int = 171
    d_model: int = 512
    n_layers: int = 4
    max_sequence_length: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.1
    tie_embeddings: bool = True
    embedding_init_std: float = 0.02
    output_logit_scale: Optional[float] = None

    # Attn (MHA+RoPE) -> CfC repeating block.
    num_attention_heads: int = 8
    cfc_units: int = 512
    cfc_backbone_units: int = 384
    cfc_backbone_layers: int = 2

    # Trainer compatibility gate (lets Trainer pass onset_times unchanged).
    use_v2_architecture: bool = True


class _VariantBBlock(nn.Module):
    """One repeating Variant-B block: Attn(MHA+RoPE) -> CfC."""

    def __init__(self, cfg: VariantBConfig) -> None:
        super().__init__()
        d = int(cfg.d_model)

        self.norm_attn = nn.LayerNorm(d)
        self.attn = CausalSelfAttentionRoPE(
            d_model=d,
            num_heads=int(cfg.num_attention_heads),
            dropout=float(cfg.attention_dropout),
        )

        self.norm_cfc = nn.LayerNorm(d)
        self.cfc = CfCBlock(
            d_model=d,
            cfc_units=d,
            backbone_units=int(cfg.cfc_backbone_units),
            backbone_layers=int(cfg.cfc_backbone_layers),
            dropout=float(cfg.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cfc_hidden: Any,
        timespans: torch.Tensor,
        position_offset: int,
    ) -> Tuple[torch.Tensor, Any]:
        x = x + self.attn(
            self.norm_attn(x), position_offset=int(max(0, position_offset))
        )

        cfc_in = self.norm_cfc(x)
        cfc_dtype = cfc_in.dtype
        cfc_x = cfc_in.float() if cfc_in.dtype != torch.float32 else cfc_in
        cfc_x = self.cfc.input_proj(cfc_x)
        ts = timespans.to(dtype=cfc_x.dtype)

        cfc_out, new_hidden = self.cfc.call_core(
            cfc_x,
            hidden=cfc_hidden,
            timespans=ts,
        )

        if cfc_out.dtype != cfc_dtype:
            cfc_out = cfc_out.to(dtype=cfc_dtype)
        cfc_out = self.cfc.output_proj(cfc_out)
        cfc_out = self.cfc.dropout(cfc_out)
        x = x + cfc_out
        return x, new_hidden


class VariantBModel(nn.Module):
    """Ablation Variant B model."""

    def __init__(self, config: Optional[VariantBConfig] = None) -> None:
        super().__init__()
        self.config = config or VariantBConfig()
        cfg = self.config

        self.vocab_size = int(cfg.vocab_size)
        self.d_model = int(cfg.d_model)
        self.max_sequence_length = int(cfg.max_sequence_length)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_sequence_length, self.d_model)
        self.dropout = nn.Dropout(float(cfg.dropout))

        self.layers = nn.ModuleList(
            [_VariantBBlock(cfg) for _ in range(int(cfg.n_layers))]
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
    def _unwrap(model: Any) -> Any:
        if isinstance(model, torch.nn.DataParallel):
            return model.module
        return model

    def _prepare_generation_device(self) -> torch.device:
        current_device = next(self.parameters()).device
        if torch.cuda.is_available() and current_device.type == "cuda":
            target_device = torch.device("cuda:0")
        else:
            target_device = current_device

        if current_device != target_device:
            self.to(target_device)
        return target_device

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
    def _timespan_deltas(onset_times: torch.Tensor) -> torch.Tensor:
        """Compute elapsed-time deltas for CfC and clamp min to 1e-4."""

        x = onset_times.to(dtype=torch.float32)
        deltas = torch.zeros_like(x)
        if int(x.shape[1]) > 1:
            deltas[:, 1:] = x[:, 1:] - x[:, :-1]
        return torch.clamp(deltas, min=1e-4)

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
    ) -> Tuple[torch.Tensor, Optional[List[Any]]] | torch.Tensor:
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

        timespans = self._timespan_deltas(onset_times)
        hidden_in: List[Any]
        if isinstance(memory, list) and len(memory) == len(self.layers):
            hidden_in = list(memory)
        else:
            hidden_in = [None] * len(self.layers)

        new_hidden: List[Any] = []
        for i, layer in enumerate(self.layers):
            x, h = layer(
                x=x,
                cfc_hidden=hidden_in[i],
                timespans=timespans,
                position_offset=int(max(0, position_offset)),
            )
            new_hidden.append(h)

        logits = self.lm_head(self.final_norm(x)) * float(self.output_logit_scale)
        if return_memory:
            return logits, new_hidden
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
    ) -> List[int]:
        self.eval()
        device = self._prepare_generation_device()

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

        context_tokens = tokens[:, -self.max_sequence_length :]
        context_onsets = onsets[:, -self.max_sequence_length :]
        context_offset = max(0, int(tokens.shape[1] - context_tokens.shape[1]))
        logits, hidden_any = self.forward(
            token_ids=context_tokens,
            onset_times=context_onsets,
            memory=None,
            return_memory=True,
            position_offset=context_offset,
        )
        hidden_list = (
            hidden_any if isinstance(hidden_any, list) else [None] * len(self.layers)
        )

        for step in range(int(max_new_tokens)):
            next_slot = self._triplet_slot(int(tokens.shape[1]))
            masked_logits = self._mask_logits_to_triplet_slot(
                logits[:, -1, :],
                next_slot,
            )
            next_token, diagnostics = sample_next_token(
                logits=masked_logits,
                context_tokens=tokens[:, -self.max_sequence_length :],
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
                delta = self._delta_from_token_events(
                    token_id=tok,
                    token_id_to_events=token_id_to_events,
                    default_step=step_seconds,
                )
            next_onset = onsets[:, -1:] + (delta if slot == 0 else 0.0)
            onsets = torch.cat([onsets, next_onset], dim=1)

            if step == int(max_new_tokens) - 1:
                break

            next_pos = int(tokens.shape[1] - 1)
            logits, hidden_any = self.forward(
                token_ids=next_token,
                onset_times=next_onset,
                memory=hidden_list,
                return_memory=True,
                position_offset=next_pos,
            )
            hidden_list = (
                hidden_any
                if isinstance(hidden_any, list)
                else [None] * len(self.layers)
            )
            hidden_list = [
                h.detach() if isinstance(h, torch.Tensor) else h for h in hidden_list
            ]

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
        device = self._prepare_generation_device()
        tokens = self._to_seed_tensor(seed_tokens, device=device)
        onsets = (
            torch.arange(tokens.shape[1], device=device, dtype=torch.float32) * 0.1
        ).unsqueeze(0)

        final_top1_probs: List[float] = []
        raw_top1_probs: List[float] = []
        candidate_counts: List[int] = []

        context_tokens = tokens[:, -self.max_sequence_length :]
        context_onsets = onsets[:, -self.max_sequence_length :]
        context_offset = max(0, int(tokens.shape[1] - context_tokens.shape[1]))
        logits, hidden_any = self.forward(
            token_ids=context_tokens,
            onset_times=context_onsets,
            memory=None,
            return_memory=True,
            position_offset=context_offset,
        )
        hidden_list = (
            hidden_any if isinstance(hidden_any, list) else [None] * len(self.layers)
        )

        for step in range(int(steps)):
            next_slot = self._triplet_slot(int(tokens.shape[1]))
            masked_logits = self._mask_logits_to_triplet_slot(
                logits[:, -1, :],
                next_slot,
            )
            next_token, diagnostics = sample_next_token(
                logits=masked_logits,
                context_tokens=tokens[:, -self.max_sequence_length :],
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

            if step == int(steps) - 1:
                break

            next_pos = int(tokens.shape[1] - 1)
            logits, hidden_any = self.forward(
                token_ids=next_token,
                onset_times=next_onset,
                memory=hidden_list,
                return_memory=True,
                position_offset=next_pos,
            )
            hidden_list = (
                hidden_any
                if isinstance(hidden_any, list)
                else [None] * len(self.layers)
            )
            hidden_list = [
                h.detach() if isinstance(h, torch.Tensor) else h for h in hidden_list
            ]

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


__all__ = ["VariantBConfig", "VariantBModel"]

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from model.blocks.gdn_block import GatedDeltaNetBlock
from model.blocks.gqa_block import GQABlock
from model.sampling import sample_next_token
from model.time_encoding import ContinuousTimeEncoding


def _resolve_divisible_heads(
    width: int,
    requested_heads: int,
    *,
    require_even_head_dim: bool = False,
) -> int:
    w = int(max(1, width))
    heads = max(1, min(int(requested_heads), w))
    while heads > 1:
        if (w % heads) == 0:
            head_dim = w // heads
            if not require_even_head_dim or (head_dim % 2 == 0):
                return int(heads)
        heads -= 1

    if require_even_head_dim and (w % 2) != 0:
        return 1
    return 1


def _resolve_hybrid_dims(d_model: int, gdn_ratio: float) -> Tuple[int, int]:
    d = int(max(4, d_model))
    ratio = float(min(0.9, max(0.1, gdn_ratio)))
    gdn_dim = int(round(float(d) * ratio))
    gdn_dim = max(1, min(d - 1, gdn_dim))
    gqa_dim = int(d - gdn_dim)

    if (gqa_dim % 2) != 0:
        if gdn_dim > 1:
            gdn_dim -= 1
            gqa_dim += 1
        else:
            gdn_dim += 1
            gqa_dim -= 1

    gdn_dim = max(1, gdn_dim)
    gqa_dim = max(2, gqa_dim)
    if (gqa_dim % 2) != 0:
        gqa_dim -= 1
        gdn_dim += 1

    return int(gdn_dim), int(gqa_dim)


@dataclass
class VariantEConfig:
    vocab_size: int = 374
    d_model: int = 512
    n_layers: int = 6
    max_sequence_length: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.1
    tie_embeddings: bool = True
    embedding_init_std: float = 0.02
    output_logit_scale: Optional[float] = None

    # Parallel hybrid-head block (no CfC): split width into GDN + dense GQA paths.
    gdn_path_ratio: float = 0.5
    gdn_inner_dim: int = 256
    gdn_num_heads: int = 4
    gqa_num_heads: int = 8
    gqa_groups: int = 4
    # Kept for runner/checkpoint compatibility; Variant E beta always uses dense GQA.
    attention_every_n_layers: int = 2
    full_attention: bool = False

    # Optional continuous-time conditioning.
    use_continuous_time: bool = True
    max_time_seconds: float = 1200.0

    # Trainer compatibility gate.
    use_v2_architecture: bool = True


class _VariantEBlock(nn.Module):
    """Parallel hybrid-head block: GDN path + dense GQA path with fused output."""

    def __init__(self, cfg: VariantEConfig) -> None:
        super().__init__()
        d = int(cfg.d_model)

        self.gdn_dim, self.gqa_dim = _resolve_hybrid_dims(
            d_model=d,
            gdn_ratio=float(cfg.gdn_path_ratio),
        )

        self.norm_in = nn.LayerNorm(d)
        self.gdn_in_proj = nn.Linear(d, self.gdn_dim, bias=False)
        self.gqa_in_proj = nn.Linear(d, self.gqa_dim, bias=False)

        gdn_inner_dim = max(int(self.gdn_dim), int(cfg.gdn_inner_dim))
        gdn_heads = _resolve_divisible_heads(
            width=int(gdn_inner_dim),
            requested_heads=int(cfg.gdn_num_heads),
            require_even_head_dim=False,
        )

        self.norm_gdn = nn.LayerNorm(self.gdn_dim)
        self.gdn = GatedDeltaNetBlock(
            d_model=int(self.gdn_dim),
            inner_dim=int(gdn_inner_dim),
            num_heads=int(gdn_heads),
            dropout=float(cfg.dropout),
        )

        gqa_heads = _resolve_divisible_heads(
            width=int(self.gqa_dim),
            requested_heads=int(cfg.gqa_num_heads),
            require_even_head_dim=True,
        )
        gqa_groups = max(1, min(int(cfg.gqa_groups), int(gqa_heads)))
        while gqa_groups > 1 and (gqa_heads % gqa_groups) != 0:
            gqa_groups -= 1
        kv_heads = max(1, int(gqa_heads) // int(gqa_groups))

        self.norm_gqa = nn.LayerNorm(self.gqa_dim)
        self.gqa = GQABlock(
            d_model=int(self.gqa_dim),
            num_heads=int(gqa_heads),
            num_kv_heads=int(kv_heads),
            dropout=float(cfg.attention_dropout),
        )

        self.fuse = nn.Sequential(
            nn.LayerNorm(int(self.gdn_dim + self.gqa_dim)),
            nn.Linear(int(self.gdn_dim + self.gqa_dim), d, bias=False),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(d, d, bias=False),
        )

    def forward(self, x: torch.Tensor, position_offset: int) -> torch.Tensor:
        h = self.norm_in(x)

        gdn_state = self.gdn_in_proj(h)
        gdn_state = gdn_state + self.gdn(self.norm_gdn(gdn_state))

        gqa_state = self.gqa_in_proj(h)
        gqa_state = gqa_state + self.gqa(
            self.norm_gqa(gqa_state), position_offset=int(max(0, position_offset))
        )

        fused = self.fuse(torch.cat([gdn_state, gqa_state], dim=-1))
        return x + fused


class VariantEModel(nn.Module):
    """Variant E beta model with parallel GDN + dense GQA hybrid-head blocks."""

    def __init__(self, config: Optional[VariantEConfig] = None) -> None:
        super().__init__()
        self.config = config or VariantEConfig()
        cfg = self.config

        self.vocab_size = int(cfg.vocab_size)
        self.d_model = int(cfg.d_model)
        self.max_sequence_length = int(cfg.max_sequence_length)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_sequence_length, self.d_model)
        self.time_encoding = (
            ContinuousTimeEncoding(
                d_model=self.d_model,
                max_time_seconds=float(max(1.0, cfg.max_time_seconds)),
            )
            if bool(cfg.use_continuous_time)
            else None
        )
        self.dropout = nn.Dropout(float(cfg.dropout))

        n_layers = int(cfg.n_layers)
        self.layers = nn.ModuleList([_VariantEBlock(cfg) for _ in range(n_layers)])

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
            return torch.arange(0, 128, dtype=torch.long)
        if slot == 1:
            return torch.arange(128, 216, dtype=torch.long)
        if slot == 2:
            return torch.arange(216, 344, dtype=torch.long)
        if slot == 3:
            return torch.arange(344, 360, dtype=torch.long)
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

        if 0 <= int(token_id) <= 127:
            if int(token_id) == 0:
                return 0.0
            bins = torch.logspace(
                math.log10(1e-4),
                math.log10(8.0),
                steps=127,
                dtype=torch.float32,
            )
            idx = max(0, min(126, int(token_id) - 1))
            return float(max(1e-4, bins[idx].item()))
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
        if self.time_encoding is not None:
            x = x + self.time_encoding(onset_times)
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
                delta = self._delta_from_token_events(
                    token_id=tok,
                    token_id_to_events=token_id_to_events,
                    default_step=step_seconds,
                )
            next_onset = onsets[:, -1:] + (delta if slot == 0 else 0.0)
            onsets = torch.cat([onsets, next_onset], dim=1)

        self.last_generation_stats = {
            "steps": int(max_new_tokens),
            "mean_final_top1_prob": float(sum(final_top1_probs) / max(1, len(final_top1_probs))),
            "mean_raw_top1_prob": float(sum(raw_top1_probs) / max(1, len(raw_top1_probs))),
            "mean_candidate_count": float(sum(candidate_counts) / max(1, len(candidate_counts))),
        }

        return [int(t) for t in tokens[0].tolist()]

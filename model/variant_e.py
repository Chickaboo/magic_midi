from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from model.blocks.gdn_block import GatedDeltaNetBlock
from model.blocks.gqa_block import GQABlock
from model.sampling import sample_next_token


@dataclass
class VariantEConfig:
    vocab_size: int = 155
    d_model: int = 512
    n_layers: int = 6
    max_sequence_length: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.1
    tie_embeddings: bool = True
    embedding_init_std: float = 0.02
    output_logit_scale: Optional[float] = None

    # Gated Delta stack (no CfC) with sparse attention anchor layers.
    gdn_inner_dim: int = 256
    gdn_num_heads: int = 4
    gqa_num_heads: int = 8
    gqa_groups: int = 4
    attention_every_n_layers: int = 2

    # Trainer compatibility gate.
    use_v2_architecture: bool = True


class _VariantEBlock(nn.Module):
    """One repeating Variant-E block: GDN -> GDN -> (optional) GQA."""

    def __init__(self, cfg: VariantEConfig, use_attention: bool) -> None:
        super().__init__()
        d = int(cfg.d_model)

        self.norm_gdn1 = nn.LayerNorm(d)
        self.gdn1 = GatedDeltaNetBlock(
            d_model=d,
            inner_dim=int(cfg.gdn_inner_dim),
            num_heads=int(cfg.gdn_num_heads),
            dropout=float(cfg.dropout),
        )

        self.norm_gdn2 = nn.LayerNorm(d)
        self.gdn2 = GatedDeltaNetBlock(
            d_model=d,
            inner_dim=int(cfg.gdn_inner_dim),
            num_heads=int(cfg.gdn_num_heads),
            dropout=float(cfg.dropout),
        )

        self.use_attention = bool(use_attention)
        if self.use_attention:
            kv_heads = max(1, int(cfg.gqa_num_heads) // max(1, int(cfg.gqa_groups)))
            self.norm_gqa = nn.LayerNorm(d)
            self.gqa = GQABlock(
                d_model=d,
                num_heads=int(cfg.gqa_num_heads),
                num_kv_heads=kv_heads,
                dropout=float(cfg.attention_dropout),
            )
        else:
            self.norm_gqa = None
            self.gqa = None

    def forward(self, x: torch.Tensor, position_offset: int) -> torch.Tensor:
        x = x + self.gdn1(self.norm_gdn1(x))
        x = x + self.gdn2(self.norm_gdn2(x))

        if self.use_attention and self.gqa is not None and self.norm_gqa is not None:
            x = x + self.gqa(
                self.norm_gqa(x), position_offset=int(max(0, position_offset))
            )
        return x


class VariantEModel(nn.Module):
    """Ablation Variant E model (GDN + sparse attention, no CfC)."""

    def __init__(self, config: Optional[VariantEConfig] = None) -> None:
        super().__init__()
        self.config = config or VariantEConfig()
        cfg = self.config

        self.vocab_size = int(cfg.vocab_size)
        self.d_model = int(cfg.d_model)
        self.max_sequence_length = int(cfg.max_sequence_length)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_sequence_length, self.d_model)
        self.dropout = nn.Dropout(float(cfg.dropout))

        n_layers = int(cfg.n_layers)
        attn_stride = max(1, int(cfg.attention_every_n_layers))

        def _use_attention(layer_index: int) -> bool:
            is_last = layer_index == (n_layers - 1)
            return is_last or ((layer_index + 1) % attn_stride == 0)

        self.layers = nn.ModuleList(
            [_VariantEBlock(cfg, use_attention=_use_attention(i)) for i in range(n_layers)]
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
        return int(index % 3)

    @staticmethod
    def _allowed_ids_for_slot(slot: int, vocab_size: int) -> torch.Tensor:
        if slot == 0:
            return torch.arange(0, 32, dtype=torch.long)
        if slot == 1:
            return torch.arange(32, 120, dtype=torch.long)
        if slot == 2:
            return torch.arange(120, 152, dtype=torch.long)
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
                min_tokens_to_keep=max(3, min_tokens_to_keep),
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

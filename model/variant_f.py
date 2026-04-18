from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention_block import MusicAttentionBlock
from model.blocks.gdn_block import GatedDeltaNetBlock
from model.blocks.gqa_block import GQABlock
from model.cfc_block import CfCBlock
from model.dual_stream import CrossStreamAttention
from model.phrase_memory import EpisodicThemeMemory, PhraseSummarizer
from model.sampling import sample_next_token
from model.time_encoding import ContinuousTimeEncoding


def _resolve_divisible_heads(width: int, requested_heads: int) -> int:
    heads = max(1, min(int(requested_heads), int(width)))
    while heads > 1 and (int(width) % heads) != 0:
        heads -= 1
    return max(1, heads)


def _resolve_rope_heads(width: int, requested_heads: int) -> int:
    heads = _resolve_divisible_heads(width, requested_heads)
    while heads > 1:
        head_dim = int(width) // int(heads)
        if head_dim % 2 == 0:
            return int(heads)
        heads -= 1
        while heads > 1 and (int(width) % heads) != 0:
            heads -= 1
    return 1


class _TemporalCfCLayer(nn.Module):
    """CfC temporal layer that always uses elapsed-time deltas when available."""

    def __init__(
        self,
        d_model: int,
        backbone_units: int,
        backbone_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(d_model))
        self.core = CfCBlock(
            d_model=int(d_model),
            cfc_units=int(d_model),
            backbone_units=int(max(1, backbone_units)),
            backbone_layers=int(max(1, backbone_layers)),
            dropout=float(dropout),
        )

    @property
    def using_fallback(self) -> bool:
        return bool(getattr(self.core, "using_fallback", False))

    def forward(
        self,
        x: torch.Tensor,
        timespans: torch.Tensor,
        hidden: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfc_in = self.norm(x)
        input_dtype = cfc_in.dtype

        cfc_x = cfc_in.float() if cfc_in.dtype != torch.float32 else cfc_in
        cfc_x = self.core.input_proj(cfc_x)

        if hidden is None:
            hidden = cfc_x.new_zeros((cfc_x.shape[0], self.core.cfc_units))

        ts = timespans.to(dtype=cfc_x.dtype)
        cfc_out, new_hidden = self.core.call_core(cfc_x, hidden=hidden, timespans=ts)

        if cfc_out.dtype != input_dtype:
            cfc_out = cfc_out.to(dtype=input_dtype)
        if isinstance(new_hidden, torch.Tensor) and new_hidden.dtype != input_dtype:
            new_hidden = new_hidden.to(dtype=input_dtype)

        cfc_out = self.core.output_proj(cfc_out)
        cfc_out = self.core.dropout(cfc_out)
        return x + cfc_out, new_hidden


@dataclass
class VariantFConfig:
    vocab_size: int = 171
    d_model: int = 896
    n_layers: int = 16
    max_sequence_length: int = 2048
    event_size: int = 4

    dropout: float = 0.1
    attention_dropout: float = 0.1
    tie_embeddings: bool = True
    embedding_init_std: float = 0.02
    output_logit_scale: Optional[float] = None

    # Event-hierarchical tri-path allocation.
    harmonic_ratio: float = 0.40
    temporal_ratio: float = 0.30

    # Harmonic path (GDN).
    gdn_inner_ratio: float = 0.50
    gdn_num_heads: int = 4

    # Temporal path (CfC).
    temporal_cfc_backbone_units: int = 448
    temporal_cfc_backbone_layers: int = 2

    # Structural path (sparse global anchors).
    structural_num_heads: int = 8
    structural_gqa_groups: int = 4

    # Inter-path fusion cadence.
    cross_stream_every_n_layers: int = 2

    # Phrase memory over event latents.
    tokens_per_phrase: int = 8
    phrase_dim: Optional[int] = None
    memory_size: int = 96
    theme_memory_heads: int = 8

    # Continuous-time signal.
    use_continuous_time: bool = True
    max_time_seconds: float = 1200.0

    # Trainer compatibility gate.
    use_v2_architecture: bool = True


class VariantFModel(nn.Module):
    """Event-hierarchical tri-path model with phrase-level temporal CfC."""

    def __init__(self, config: Optional[VariantFConfig] = None) -> None:
        super().__init__()
        self.config = config or VariantFConfig()
        cfg = self.config

        self.vocab_size = int(cfg.vocab_size)
        self.d_model = int(cfg.d_model)
        self.max_sequence_length = int(cfg.max_sequence_length)
        self.event_size = int(max(1, cfg.event_size))
        self.max_event_positions = int(
            math.ceil(float(self.max_sequence_length) / float(self.event_size))
        )

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.slot_embedding = nn.Embedding(self.event_size, self.d_model)
        self.event_position_embedding = nn.Embedding(
            self.max_event_positions,
            self.d_model,
        )
        self.time_encoding = (
            ContinuousTimeEncoding(
                d_model=self.d_model,
                max_time_seconds=float(max(1.0, cfg.max_time_seconds)),
            )
            if bool(cfg.use_continuous_time)
            else None
        )
        self.input_dropout = nn.Dropout(float(cfg.dropout))

        self.event_pack_norm = nn.LayerNorm(self.d_model * self.event_size)
        self.event_pack = nn.Sequential(
            nn.Linear(self.d_model * self.event_size, self.d_model),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.harmonic_dim, self.temporal_dim, self.structural_dim = self._resolve_stream_dims(
            d_model=self.d_model,
            harmonic_ratio=float(cfg.harmonic_ratio),
            temporal_ratio=float(cfg.temporal_ratio),
        )

        self.harmonic_in = nn.Sequential(
            nn.Linear(self.d_model, self.harmonic_dim),
            nn.LayerNorm(self.harmonic_dim),
        )
        self.temporal_in = nn.Sequential(
            nn.Linear(self.d_model, self.temporal_dim),
            nn.LayerNorm(self.temporal_dim),
        )
        self.structural_in = nn.Sequential(
            nn.Linear(self.d_model, self.structural_dim),
            nn.LayerNorm(self.structural_dim),
        )

        n_layers = int(max(1, cfg.n_layers))

        gdn_inner_dim = max(
            128,
            int(round(float(self.harmonic_dim) * float(max(0.1, cfg.gdn_inner_ratio)))),
        )
        gdn_heads = _resolve_divisible_heads(gdn_inner_dim, int(cfg.gdn_num_heads))
        if gdn_inner_dim % gdn_heads != 0:
            gdn_heads = 1

        structural_heads = _resolve_rope_heads(
            self.structural_dim,
            int(cfg.structural_num_heads),
        )
        structural_groups = max(1, int(cfg.structural_gqa_groups))
        structural_kv_heads = max(1, structural_heads // structural_groups)
        while structural_kv_heads > 1 and (structural_heads % structural_kv_heads) != 0:
            structural_kv_heads -= 1

        self.harmonic_norms = nn.ModuleList(
            [nn.LayerNorm(self.harmonic_dim) for _ in range(n_layers)]
        )
        self.harmonic_layers = nn.ModuleList(
            [
                GatedDeltaNetBlock(
                    d_model=self.harmonic_dim,
                    inner_dim=int(gdn_inner_dim),
                    num_heads=int(gdn_heads),
                    dropout=float(cfg.dropout),
                )
                for _ in range(n_layers)
            ]
        )

        self.temporal_layers = nn.ModuleList(
            [
                _TemporalCfCLayer(
                    d_model=self.temporal_dim,
                    backbone_units=int(cfg.temporal_cfc_backbone_units),
                    backbone_layers=int(cfg.temporal_cfc_backbone_layers),
                    dropout=float(cfg.dropout),
                )
                for _ in range(n_layers)
            ]
        )

        self.structural_norms = nn.ModuleList(
            [nn.LayerNorm(self.structural_dim) for _ in range(n_layers)]
        )
        self.structural_layers = nn.ModuleList(
            [
                GQABlock(
                    d_model=self.structural_dim,
                    num_heads=int(structural_heads),
                    num_kv_heads=int(structural_kv_heads),
                    dropout=float(cfg.attention_dropout),
                )
                for _ in range(n_layers)
            ]
        )

        self.cross_stream_every_n_layers = int(max(1, cfg.cross_stream_every_n_layers))
        cross_layers = max(
            1,
            (n_layers + self.cross_stream_every_n_layers - 1)
            // self.cross_stream_every_n_layers,
        )

        cross_heads = _resolve_divisible_heads(
            min(self.harmonic_dim, self.temporal_dim),
            4,
        )
        self.harmonic_temporal_cross = nn.ModuleList(
            [
                CrossStreamAttention(
                    harmonic_dim=self.harmonic_dim,
                    temporal_dim=self.temporal_dim,
                    num_heads=int(cross_heads),
                    dropout=float(cfg.attention_dropout),
                )
                for _ in range(cross_layers)
            ]
        )

        self.events_per_phrase = int(max(1, cfg.tokens_per_phrase))
        self.temporal_phrase_summarizer = PhraseSummarizer(
            d_model=self.d_model,
            phrase_dim=self.temporal_dim,
            tokens_per_phrase=self.events_per_phrase,
        )

        self.ht_to_struct = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.harmonic_dim + self.temporal_dim),
                    nn.Linear(self.harmonic_dim + self.temporal_dim, self.structural_dim),
                    nn.GELU(),
                    nn.Dropout(float(cfg.dropout)),
                )
                for _ in range(cross_layers)
            ]
        )
        self.struct_to_ht = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.structural_dim),
                    nn.Linear(self.structural_dim, self.harmonic_dim + self.temporal_dim),
                    nn.GELU(),
                    nn.Dropout(float(cfg.dropout)),
                )
                for _ in range(cross_layers)
            ]
        )
        self.struct_to_ht_gate = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.structural_dim),
                    nn.Linear(self.structural_dim, self.harmonic_dim + self.temporal_dim),
                    nn.Sigmoid(),
                )
                for _ in range(cross_layers)
            ]
        )

        self.stream_merge = nn.Sequential(
            nn.LayerNorm(self.harmonic_dim + self.temporal_dim + self.structural_dim),
            nn.Linear(self.harmonic_dim + self.temporal_dim + self.structural_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        phrase_dim = int(cfg.phrase_dim or self.d_model)
        self.phrase_summarizer = PhraseSummarizer(
            d_model=self.d_model,
            phrase_dim=phrase_dim,
            tokens_per_phrase=self.events_per_phrase,
        )

        memory_heads = _resolve_divisible_heads(int(phrase_dim), int(cfg.theme_memory_heads))
        self.theme_memory = EpisodicThemeMemory(
            phrase_dim=int(phrase_dim),
            memory_size=int(max(1, cfg.memory_size)),
            num_heads=int(max(1, memory_heads)),
        )

        phrase_attn_heads = _resolve_rope_heads(int(phrase_dim), int(cfg.theme_memory_heads))
        self.phrase_attention = MusicAttentionBlock(
            d_model=int(phrase_dim),
            num_heads=int(max(1, phrase_attn_heads)),
            max_relative_distance=128,
            dropout=float(cfg.attention_dropout),
            use_relative_bias=True,
            bias_type="learned",
            ffn_expansion=2,
        )

        self.phrase_to_event = nn.Linear(int(phrase_dim), self.d_model)
        self.final_norm = nn.LayerNorm(self.d_model)

        self.event_to_token = nn.Linear(self.d_model, self.d_model)
        self.slot_decode_embedding = nn.Embedding(self.event_size, self.d_model)
        self.slot_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Dropout(float(cfg.dropout)),
                    nn.Linear(self.d_model, self.vocab_size),
                )
                for _ in range(self.event_size)
            ]
        )

        if cfg.output_logit_scale is None:
            self.output_logit_scale = 1.0 / math.sqrt(float(self.d_model))
        else:
            self.output_logit_scale = float(cfg.output_logit_scale)

        self._reset_parameters()
        self.last_generation_stats: Dict[str, Any] = {}

    @staticmethod
    def _resolve_stream_dims(
        *,
        d_model: int,
        harmonic_ratio: float,
        temporal_ratio: float,
    ) -> Tuple[int, int, int]:
        h = int(round(float(d_model) * float(harmonic_ratio)))
        t = int(round(float(d_model) * float(temporal_ratio)))

        h = max(64, h)
        t = max(64, t)

        if h + t >= int(d_model):
            overflow = (h + t) - int(d_model) + 64
            trim_h = int(math.ceil(overflow / 2.0))
            trim_t = int(math.floor(overflow / 2.0))
            h = max(64, h - trim_h)
            t = max(64, t - trim_t)

        s = int(d_model) - h - t
        if s < 64:
            need = 64 - s
            take_h = min(max(0, h - 64), int(math.ceil(need / 2.0)))
            h -= take_h
            need -= take_h
            take_t = min(max(0, t - 64), need)
            t -= take_t
            s = int(d_model) - h - t

        if s <= 0:
            s = 64
            if h > t:
                h = max(64, h - 64)
            else:
                t = max(64, t - 64)

        # RoPE in structural attention requires an even head dimension.
        # Keep the total width constant while nudging one channel if needed.
        if (s % 2) != 0:
            if s > 64:
                s -= 1
                if h <= t:
                    h += 1
                else:
                    t += 1
            elif h > 64:
                h -= 1
                s += 1
            elif t > 64:
                t -= 1
                s += 1

        return int(h), int(t), int(s)

    def _reset_parameters(self) -> None:
        std = float(max(1e-6, self.config.embedding_init_std))
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.slot_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.event_position_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.slot_decode_embedding.weight, mean=0.0, std=std)
        nn.init.zeros_(self.phrase_to_event.weight)
        if self.phrase_to_event.bias is not None:
            nn.init.zeros_(self.phrase_to_event.bias)

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
    def _triplet_slot(index: int, event_size: int = 4) -> int:
        return int(index % max(1, int(event_size)))

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

    def _mask_logits_to_triplet_slot(self, logits: torch.Tensor, slot: int) -> torch.Tensor:
        mask = torch.full_like(logits, fill_value=-float("inf"))
        allowed = self._allowed_ids_for_slot(int(slot), logits.shape[-1]).to(logits.device)
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

    @staticmethod
    def _timespan_deltas(onset_times: torch.Tensor) -> torch.Tensor:
        x = onset_times.to(dtype=torch.float32)
        deltas = torch.zeros_like(x)
        if int(x.shape[1]) > 1:
            deltas[:, 1:] = x[:, 1:] - x[:, :-1]
        return torch.clamp(deltas, min=1e-4)

    def _phrase_onsets(self, onset_times: torch.Tensor) -> torch.Tensor:
        """Reduce event-level onset times to one onset per phrase."""

        if onset_times.ndim != 2:
            raise ValueError(
                f"onset_times must be rank-2, got {tuple(onset_times.shape)}"
            )

        batch_size, seq_len = onset_times.shape
        pad_len = (
            self.events_per_phrase - (int(seq_len) % self.events_per_phrase)
        ) % self.events_per_phrase
        if pad_len > 0:
            if seq_len > 0:
                onset_pad = onset_times[:, -1:].expand(-1, pad_len)
            else:
                onset_pad = onset_times.new_zeros((batch_size, pad_len))
            onset_times = torch.cat([onset_times, onset_pad], dim=1)

        return onset_times.view(batch_size, -1, self.events_per_phrase)[:, :, 0]

    def _repeat_phrase_context(
        self,
        phrase_context: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Broadcast phrase-level context back to the event sequence length."""

        if phrase_context.ndim != 3:
            raise ValueError(
                f"phrase_context must be rank-3, got {tuple(phrase_context.shape)}"
            )
        expanded = phrase_context.repeat_interleave(self.events_per_phrase, dim=1)
        return expanded[:, :seq_len, :]

    def _prepare_embeddings(
        self,
        token_ids: torch.Tensor,
        onset_times: torch.Tensor,
        position_offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch_size, seq_len = token_ids.shape

        raw_positions = torch.arange(
            int(max(0, position_offset)),
            int(max(0, position_offset)) + int(seq_len),
            device=token_ids.device,
        )
        slot_ids = (raw_positions % int(self.event_size)).to(dtype=torch.long)
        event_positions = torch.clamp(
            raw_positions // int(self.event_size),
            max=int(self.max_event_positions - 1),
        )

        x = self.token_embedding(token_ids)
        x = x + self.slot_embedding(slot_ids).unsqueeze(0)
        x = x + self.event_position_embedding(event_positions).unsqueeze(0)
        if self.time_encoding is not None:
            x = x + self.time_encoding(onset_times)
        x = self.input_dropout(x)

        pad_len = (int(self.event_size) - (int(seq_len) % int(self.event_size))) % int(
            self.event_size
        )
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len), value=0.0)
            if seq_len > 0:
                onset_pad = onset_times[:, -1:].expand(-1, pad_len)
            else:
                onset_pad = onset_times.new_zeros((batch_size, pad_len))
            onset_times = torch.cat([onset_times, onset_pad], dim=1)

        num_events = int(x.shape[1] // int(self.event_size))
        event_tokens = x.view(batch_size, num_events, int(self.event_size), self.d_model)
        packed = event_tokens.reshape(batch_size, num_events, self.d_model * int(self.event_size))
        event_latents = self.event_pack(self.event_pack_norm(packed))

        onset_events = onset_times.view(batch_size, num_events, int(self.event_size))[:, :, 0]
        return event_latents, onset_events, slot_ids, seq_len

    def _decode_slot_logits(
        self,
        event_latents: torch.Tensor,
        slot_ids: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        batch_size, num_events, _ = event_latents.shape

        token_base = self.event_to_token(event_latents)
        token_base = token_base.unsqueeze(2).expand(-1, -1, int(self.event_size), -1)
        token_base = token_base + self.slot_decode_embedding.weight.view(
            1,
            1,
            int(self.event_size),
            self.d_model,
        )

        token_features = token_base.reshape(
            batch_size,
            num_events * int(self.event_size),
            self.d_model,
        )
        token_features = token_features[:, :seq_len, :]

        stacked = torch.stack(
            [head(token_features) for head in self.slot_heads],
            dim=2,
        )

        slot_index = slot_ids.view(1, seq_len, 1, 1).expand(
            batch_size,
            seq_len,
            1,
            self.vocab_size,
        )
        logits = torch.gather(stacked, dim=2, index=slot_index).squeeze(2)
        return logits

    def forward(
        self,
        token_ids: torch.Tensor,
        onset_times: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        return_memory: bool = False,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        del durations

        if token_ids.ndim != 2 or onset_times.ndim != 2:
            raise ValueError("token_ids and onset_times must be rank-2")
        if token_ids.shape != onset_times.shape:
            raise ValueError("token_ids and onset_times must have same shape")

        event_latents, event_onsets, slot_ids, seq_len = self._prepare_embeddings(
            token_ids=token_ids,
            onset_times=onset_times,
            position_offset=int(max(0, position_offset)),
        )

        h = self.harmonic_in(event_latents)
        s = self.structural_in(event_latents)

        temporal_phrase_onsets = self._phrase_onsets(event_onsets)
        temporal_phrase = self.temporal_phrase_summarizer(event_latents)
        temporal_timespans = self._timespan_deltas(temporal_phrase_onsets)
        temporal_hidden: List[Optional[torch.Tensor]] = [None] * int(
            self.config.n_layers
        )
        for i in range(int(self.config.n_layers)):
            temporal_phrase, temporal_hidden[i] = self.temporal_layers[i](
                temporal_phrase,
                timespans=temporal_timespans,
                hidden=temporal_hidden[i],
            )

        temporal_context = self._repeat_phrase_context(
            temporal_phrase,
            seq_len=int(event_latents.shape[1]),
        )

        cross_idx = 0
        event_offset = int(max(0, position_offset) // max(1, int(self.event_size)))

        t = self.temporal_in(event_latents) + temporal_context

        for i in range(int(self.config.n_layers)):
            h = h + self.harmonic_layers[i](self.harmonic_norms[i](h))
            s = s + self.structural_layers[i](
                self.structural_norms[i](s),
                position_offset=event_offset,
            )

            if (i + 1) % int(self.cross_stream_every_n_layers) == 0:
                if cross_idx >= len(self.harmonic_temporal_cross):
                    cross_idx = len(self.harmonic_temporal_cross) - 1
                h, t = self.harmonic_temporal_cross[cross_idx](h, t)

                ht = torch.cat([h, t], dim=-1)
                s = s + self.ht_to_struct[cross_idx](ht)

                ht_delta = self.struct_to_ht[cross_idx](s)
                ht_gate = self.struct_to_ht_gate[cross_idx](s)
                ht_delta = ht_delta * ht_gate

                h = h + ht_delta[:, :, : self.harmonic_dim]
                t = t + ht_delta[:, :, self.harmonic_dim :]
                cross_idx += 1

        merged = self.stream_merge(torch.cat([h, t, s], dim=-1))

        phrase_repr = self.phrase_summarizer(merged)
        phrase_repr, new_memory = self.theme_memory(phrase_repr, memory=memory)
        phrase_repr = self.phrase_attention(phrase_repr)

        phrase_expanded = phrase_repr.repeat_interleave(self.events_per_phrase, dim=1)
        phrase_expanded = phrase_expanded[:, : merged.shape[1], :]
        merged = merged + self.phrase_to_event(phrase_expanded)

        event_out = self.final_norm(merged)
        logits = self._decode_slot_logits(event_out, slot_ids=slot_ids, seq_len=seq_len)
        logits = logits * float(self.output_logit_scale)

        if return_memory:
            return logits, new_memory
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
        max_consecutive_zero_deltas: int = 12,
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
                onset_tensor = seed_onset_times
                if onset_tensor.ndim == 1:
                    onset_tensor = onset_tensor.unsqueeze(0)
                onsets = onset_tensor.to(device=device, dtype=torch.float32)
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
        zero_delta_run = 0

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

            next_slot = self._triplet_slot(int(tokens.shape[1]), event_size=self.event_size)
            masked_logits = self._mask_logits_to_triplet_slot(logits[:, -1, :], next_slot)

            next_token, diagnostics = sample_next_token(
                logits=masked_logits,
                context_tokens=context_tokens,
                temperature=float(max(0.1, temperature)),
                top_p=float(min(1.0, max(0.0, top_p))),
                top_k=int(max(1, top_k)),
                repetition_penalty=float(max(1.0, repetition_penalty)),
                recent_window=int(max(1, repetition_window)),
                min_tokens_to_keep=max(4, int(min_tokens_to_keep)),
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

            token_value = int(next_token.view(-1)[0].item())
            delta = float(max(1e-4, step_seconds))
            if next_slot == 0:
                delta = self._delta_from_token_events(
                    token_id=token_value,
                    token_id_to_events=token_id_to_events,
                    default_step=step_seconds,
                )
                if delta <= 1e-4:
                    zero_delta_run += 1
                else:
                    zero_delta_run = 0
            if zero_delta_run >= int(max(1, max_consecutive_zero_deltas)):
                break

            tokens = torch.cat([tokens, next_token], dim=1)
            next_onset = onsets[:, -1:] + (delta if next_slot == 0 else 0.0)
            onsets = torch.cat([onsets, next_onset], dim=1)

        self.last_generation_stats = {
            "steps": int(max_new_tokens),
            "mean_final_top1_prob": float(
                sum(final_top1_probs) / max(1, len(final_top1_probs))
            ),
            "mean_raw_top1_prob": float(
                sum(raw_top1_probs) / max(1, len(raw_top1_probs))
            ),
            "mean_candidate_count": float(
                sum(candidate_counts) / max(1, len(candidate_counts))
            ),
        }

        return [int(t) for t in tokens[0].tolist()]

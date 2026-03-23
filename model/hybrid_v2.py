from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from config import ModelConfig
from model.attention_block import MusicAttentionBlock
from model.cfc_block import CfCBlock
from model.dual_stream import CrossStreamAttention, DualStreamSplit
from model.mamba_block import MambaBlock
from model.phrase_memory import EpisodicThemeMemory, PhraseSummarizer
from model.sampling import sample_next_token
from model.time_encoding import ContinuousTimeEncoding


class IttyBittyPianoV2(nn.Module):
    """Version 2 hybrid model with dual streams and episodic theme memory."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.residual_scale = 1.0 / math.sqrt(max(1.0, 2.0 * float(config.n_layers)))
        if config.output_logit_scale is None:
            self.output_logit_scale = 1.0
        else:
            self.output_logit_scale = float(config.output_logit_scale)

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.time_encoding = ContinuousTimeEncoding(
            d_model=config.d_model,
            max_time_seconds=float(config.max_time_seconds),
        )

        harmonic_ratio = float(getattr(config, "harmonic_ratio", 0.5))
        harmonic_ratio = max(0.2, min(0.8, harmonic_ratio))

        if config.stream_dim is None:
            total_stream_dim = int(config.d_model)
        else:
            # `stream_dim` in v2 presets is legacy per-stream width.
            total_stream_dim = int(config.stream_dim) * 2
        if total_stream_dim <= 1:
            raise ValueError("stream_dim must be > 1")

        harmonic_dim = int(round(total_stream_dim * harmonic_ratio))
        harmonic_dim = max(1, min(total_stream_dim - 1, harmonic_dim))
        temporal_dim = int(total_stream_dim - harmonic_dim)
        if temporal_dim <= 0:
            temporal_dim = 1
            harmonic_dim = max(1, total_stream_dim - 1)

        self.harmonic_dim = int(harmonic_dim)
        self.temporal_dim = int(temporal_dim)
        self.stream_dim = int(self.harmonic_dim + self.temporal_dim)

        self.stream_split = DualStreamSplit(
            config.d_model,
            harmonic_dim=self.harmonic_dim,
            temporal_dim=self.temporal_dim,
        )

        self.harmonic_layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=self.harmonic_dim,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                    dropout=config.dropout,
                    residual_scale=self.residual_scale,
                    debug=config.debug,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.temporal_layers = nn.ModuleList(
            [
                CfCBlock(
                    d_model=self.temporal_dim,
                    cfc_units=self.temporal_dim,
                    backbone_units=max(1, int(config.cfc_backbone_units)),
                    backbone_layers=max(1, int(config.cfc_backbone_layers)),
                    dropout=config.dropout,
                    residual_scale=self.residual_scale,
                    debug=config.debug,
                )
                for _ in range(config.n_layers)
            ]
        )

        cross_every = getattr(config, "cross_attention_every_n", None)
        if cross_every is None:
            cross_every = config.cross_stream_every_n_layers
        self.cross_stream_every_n_layers = max(1, int(cross_every))
        cross_heads = max(1, int(config.num_attention_heads))
        self.cross_stream_layers = nn.ModuleList(
            [
                CrossStreamAttention(
                    harmonic_dim=self.harmonic_dim,
                    temporal_dim=self.temporal_dim,
                    num_heads=cross_heads,
                    dropout=config.attention_dropout,
                    residual_scale=self.residual_scale,
                )
                for _ in range(
                    max(
                        1,
                        (config.n_layers + self.cross_stream_every_n_layers - 1)
                        // self.cross_stream_every_n_layers,
                    )
                )
            ]
        )

        self.stream_merge = nn.Sequential(
            nn.LayerNorm(self.stream_dim),
            nn.Linear(self.stream_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        phrase_dim = int(config.phrase_dim or config.d_model)
        self.tokens_per_phrase = max(1, int(config.tokens_per_phrase))
        self.phrase_summarizer = PhraseSummarizer(
            d_model=config.d_model,
            phrase_dim=phrase_dim,
            tokens_per_phrase=self.tokens_per_phrase,
            residual_scale=self.residual_scale,
        )
        memory_heads = max(1, int(config.theme_memory_heads))
        while phrase_dim % memory_heads != 0 and memory_heads > 1:
            memory_heads -= 1
        self.theme_memory = EpisodicThemeMemory(
            phrase_dim=phrase_dim,
            memory_size=max(1, int(config.memory_size)),
            num_heads=memory_heads,
            residual_scale=self.residual_scale,
        )
        phrase_attn_heads = max(1, int(config.num_attention_heads))
        while phrase_dim % phrase_attn_heads != 0 and phrase_attn_heads > 1:
            phrase_attn_heads -= 1
        self.phrase_attention = MusicAttentionBlock(
            d_model=phrase_dim,
            num_heads=phrase_attn_heads,
            max_relative_distance=config.max_relative_distance,
            dropout=config.attention_dropout,
            use_relative_bias=config.use_relative_attention,
            bias_type=config.attention_bias_type,
            residual_scale=self.residual_scale,
        )
        self.phrase_to_token = nn.Linear(phrase_dim, config.d_model)

        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if bool(config.tie_embeddings):
            self.output_proj.weight = self.token_embedding.weight

        self._reset_parameters()
        self.last_generation_stats: Dict[str, Any] = {}
        self._tokenizer = None

    def bind_tokenizer(self, tokenizer: Any) -> None:
        """Attach tokenizer for token-id semantic decoding during generation."""

        self._tokenizer = tokenizer

    def _reset_parameters(self) -> None:
        """Initialize token/output projections for stable random-logit scale."""

        init_std = float(max(1e-6, getattr(self.config, "embedding_init_std", 0.02)))
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=init_std)
        if not bool(self.config.tie_embeddings):
            nn.init.normal_(self.output_proj.weight, mean=0.0, std=init_std)

        nn.init.zeros_(self.phrase_to_token.weight)
        if self.phrase_to_token.bias is not None:
            nn.init.zeros_(self.phrase_to_token.bias)

    def get_num_params(self) -> int:
        """Return trainable parameter count for compatibility with v1 helpers."""

        return int(sum(p.numel() for p in self.parameters()))

    def forward(
        self,
        token_ids: torch.Tensor,
        onset_times: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        return_memory: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        """Forward pass for v2 model."""

        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must be (batch, seq_len), got {tuple(token_ids.shape)}"
            )
        if onset_times.ndim != 2:
            raise ValueError(
                f"onset_times must be (batch, seq_len), got {tuple(onset_times.shape)}"
            )
        if token_ids.shape != onset_times.shape:
            raise ValueError(
                "token_ids and onset_times must share shape, "
                f"got {tuple(token_ids.shape)} vs {tuple(onset_times.shape)}"
            )
        if durations is not None and durations.shape != token_ids.shape:
            raise ValueError(
                "durations must match token_ids shape when provided, "
                f"got {tuple(durations.shape)} vs {tuple(token_ids.shape)}"
            )

        if durations is not None:
            onset_times = onset_times + 0.0 * durations

        x = self.token_embedding(token_ids)
        x = x + self.time_encoding(onset_times)

        harmonic, temporal = self.stream_split(x)

        hidden_states: List[Any] = [None] * len(self.temporal_layers)
        cross_idx = 0
        for i in range(self.config.n_layers):
            harmonic = self.harmonic_layers[i](harmonic)
            temporal, hidden_states[i] = self.temporal_layers[i](
                temporal,
                hidden=hidden_states[i],
            )

            if (i + 1) % self.cross_stream_every_n_layers == 0:
                if cross_idx >= len(self.cross_stream_layers):
                    cross_idx = len(self.cross_stream_layers) - 1
                harmonic, temporal = self.cross_stream_layers[cross_idx](
                    harmonic,
                    temporal,
                )
                cross_idx += 1

        merged = self.stream_merge(torch.cat([harmonic, temporal], dim=-1))

        phrase_repr = self.phrase_summarizer(merged)
        phrase_repr, new_memory = self.theme_memory(phrase_repr, memory=memory)
        phrase_repr = self.phrase_attention(phrase_repr)

        seq_len = merged.shape[1]
        phrase_expanded = phrase_repr.repeat_interleave(self.tokens_per_phrase, dim=1)
        phrase_expanded = phrase_expanded[:, :seq_len, :]
        phrase_context = self.phrase_to_token(phrase_expanded)

        out = self.final_norm(merged + phrase_context)
        logits = self.output_proj(out) * float(self.output_logit_scale)

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
        step_seconds: float = 0.5,
        token_id_to_events: Optional[Callable[..., Any]] = None,
    ) -> List[int]:
        """Generate continuation using v2 forward signature and memory state."""

        self.eval()
        device = next(self.parameters()).device

        if isinstance(seed_tokens, torch.Tensor):
            if seed_tokens.ndim == 1:
                tokens = seed_tokens.unsqueeze(0).to(device=device, dtype=torch.long)
            else:
                tokens = seed_tokens.to(device=device, dtype=torch.long)
        else:
            tokens = torch.tensor(
                [int(t) for t in seed_tokens], dtype=torch.long, device=device
            ).unsqueeze(0)

        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("generate expects a single-sequence seed")

        if seed_onset_times is None:
            base_step = float(max(1e-4, step_seconds))
            onset_times = (
                torch.arange(tokens.shape[1], device=device, dtype=torch.float32)
                * base_step
            ).unsqueeze(0)
        else:
            if isinstance(seed_onset_times, torch.Tensor):
                if seed_onset_times.ndim == 1:
                    onset_times = seed_onset_times.unsqueeze(0)
                else:
                    onset_times = seed_onset_times
                onset_times = onset_times.to(device=device, dtype=torch.float32)
            else:
                onset_times = torch.tensor(
                    [float(v) for v in seed_onset_times],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

        if onset_times.shape != tokens.shape:
            raise ValueError(
                f"seed_onset_times shape {tuple(onset_times.shape)} does not match seed_tokens {tuple(tokens.shape)}"
            )

        memory: Optional[torch.Tensor] = None
        final_top1_probs: List[float] = []
        raw_top1_probs: List[float] = []
        candidate_counts: List[int] = []

        bar_start = onset_times[:, -1:].clone()

        def _next_onset(prev: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
            token_id = int(token.item())
            events = self._token_events_from_id(token_id, token_id_to_events)

            position_value = self._first_numeric_token(events, prefix="Position")
            if position_value is not None:
                return bar_start + (
                    float(position_value) * float(max(1e-4, step_seconds)) / 4.0
                )

            duration_value = self._first_duration_token(events)
            if duration_value is not None:
                return prev + float(max(1e-4, duration_value))

            has_bar = any(ev.startswith("Bar") for ev in events)
            has_pitch = any(ev.startswith("Pitch") for ev in events)
            if has_bar:
                return prev + float(max(1e-4, step_seconds))
            if has_pitch:
                return prev + float(max(1e-4, step_seconds))
            return prev

        for _ in range(int(max_new_tokens)):
            context_tokens = tokens[:, -int(self.config.max_sequence_length) :]
            context_onsets = onset_times[:, -int(self.config.max_sequence_length) :]
            logits, memory = self.forward(
                context_tokens,
                context_onsets,
                memory=memory,
                return_memory=True,
            )
            next_token, diagnostics = sample_next_token(
                logits=logits[:, -1, :],
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
            next_onset = _next_onset(onset_times[:, -1:], next_token.view(-1)[0])
            next_events = self._token_events_from_id(
                int(next_token.view(-1)[0].item()),
                token_id_to_events,
            )
            if any(ev.startswith("Bar") for ev in next_events):
                bar_start = next_onset.clone()
            onset_times = torch.cat([onset_times, next_onset], dim=1)

            if memory is not None:
                memory = memory.detach()

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
        """Short generation diagnostic for collapse detection."""

        self.eval()
        device = next(self.parameters()).device

        if isinstance(seed_tokens, torch.Tensor):
            if seed_tokens.ndim == 1:
                tokens = seed_tokens.unsqueeze(0)
            else:
                tokens = seed_tokens
            tokens = tokens.to(device=device, dtype=torch.long)
        else:
            tokens = torch.tensor(
                [int(t) for t in seed_tokens], dtype=torch.long, device=device
            ).unsqueeze(0)

        token_id_to_events: Optional[Callable[..., Any]] = None
        if self._tokenizer is not None:
            maybe_cb = getattr(self._tokenizer, "decode_token_id_events", None)
            if callable(maybe_cb):
                token_id_to_events = maybe_cb  # type: ignore[assignment]

        onset_times = (
            torch.arange(tokens.shape[1], device=device, dtype=torch.float32) * 0.5
        ).unsqueeze(0)
        bar_start = onset_times[:, -1:].clone()
        memory: Optional[torch.Tensor] = None

        final_top1_probs: List[float] = []
        raw_top1_probs: List[float] = []
        candidate_counts: List[int] = []

        for _ in range(int(steps)):
            context_tokens = tokens[:, -int(self.config.max_sequence_length) :]
            context_onsets = onset_times[:, -int(self.config.max_sequence_length) :]
            logits, memory = self.forward(
                context_tokens,
                context_onsets,
                memory=memory,
                return_memory=True,
            )
            next_token, diagnostics = sample_next_token(
                logits=logits[:, -1, :],
                context_tokens=context_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                recent_window=repetition_window,
                min_tokens_to_keep=max(3, min_tokens_to_keep),
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
            next_events = self._token_events_from_id(
                int(next_token.view(-1)[0].item()),
                token_id_to_events,
            )
            next_onset = onset_times[:, -1:]

            position_value = self._first_numeric_token(next_events, prefix="Position")
            if position_value is not None:
                next_onset = bar_start + (float(position_value) * 0.5 / 4.0)
            else:
                duration_value = self._first_duration_token(next_events)
                if duration_value is not None:
                    next_onset = next_onset + float(max(1e-4, duration_value))
                elif any(ev.startswith("Bar") for ev in next_events):
                    next_onset = next_onset + 0.5
                elif any(ev.startswith("Pitch") for ev in next_events):
                    next_onset = next_onset + 0.5

            if any(ev.startswith("Bar") for ev in next_events):
                bar_start = next_onset.clone()

            onset_times = torch.cat([onset_times, next_onset], dim=1)

            if memory is not None:
                memory = memory.detach()

        max_final_top1 = max(final_top1_probs) if final_top1_probs else 0.0
        tolerance = 1e-6
        passed = bool(max_final_top1 <= float(top1_threshold) + tolerance)
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

    def _token_events_from_id(
        self,
        token_id: int,
        token_id_to_events: Optional[Callable[..., Any]],
    ) -> List[str]:
        """Decode one token id into semantic events when callback is available."""

        callback = token_id_to_events
        if callback is None and self._tokenizer is not None:
            maybe_cb = getattr(self._tokenizer, "decode_token_id_events", None)
            if callable(maybe_cb):
                callback = maybe_cb

        if callback is None:
            return []

        try:
            events = callback(int(token_id))
            if events is None:
                return []
            if isinstance(events, str):
                return [str(events)]
            if isinstance(events, torch.Tensor):
                return [str(v) for v in events.detach().cpu().tolist()]
            if not isinstance(events, (list, tuple)):
                return []
            return [str(ev) for ev in events]
        except Exception:
            return []

    @staticmethod
    def _first_numeric_token(events: Sequence[str], prefix: str) -> Optional[float]:
        """Return numeric suffix from first event matching token prefix."""

        head = f"{prefix}_"
        for token_name in events:
            if not token_name.startswith(head):
                continue
            part = token_name.split("_", 1)[1]
            try:
                return float(part)
            except Exception:
                continue
        return None

    @staticmethod
    def _first_duration_token(events: Sequence[str]) -> Optional[float]:
        """Return duration seconds estimate from first duration event in one token."""

        for token_name in events:
            if not token_name.startswith("Duration_"):
                continue
            part = token_name.split("_", 1)[1]
            nums = [p for p in part.split(".") if p.isdigit()]
            if not nums:
                continue
            if len(nums) == 1:
                return max(1e-4, float(nums[0]))
            numerator = float(nums[-2])
            denominator = float(max(1, int(nums[-1])))
            return max(1e-4, numerator / denominator)
        return None

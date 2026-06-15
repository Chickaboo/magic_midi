from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.norms import RMSNorm, SwiGLU, round_multiple
from model.sampling import sample_next_token


def _flash_attention_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


FLASH_ATTENTION_AVAILABLE = _flash_attention_available()


@dataclass
class DensePianoTransformerConfig:
    vocab_size: int = 30000
    d_model: int = 576
    n_layers: int = 12
    max_sequence_length: int = 8192
    dropout: float = 0.1
    attention_dropout: float = 0.1
    tie_embeddings: bool = False
    embedding_init_std: float = 0.02
    output_logit_scale: Optional[float] = None

    num_attention_heads: int = 9
    head_dim: int = 64
    window_schedule: Tuple[int, ...] = field(
        default_factory=lambda: (512, 512, 512, 512, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048)
    )
    max_relative_distance: int = 2048
    global_anchor_count: int = 256
    global_anchor_start_layer: int = 9

    ffn_multiple_of: int = 64
    rms_norm_eps: float = 1e-6
    gradient_checkpointing: bool = False

    # Trainer compatibility gate.
    use_v2_architecture: bool = True


class _DenseSlidingWindowAttention(nn.Module):
    """Dense causal local attention with learned relative bias and optional anchors."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        head_dim: int,
        window_size: int,
        max_relative_distance: int,
        dropout: float,
        use_global_anchors: bool,
    ) -> None:
        super().__init__()
        if int(d_model) != int(num_heads) * int(head_dim):
            raise ValueError("d_model must equal num_heads * head_dim")
        if int(window_size) <= 0:
            raise ValueError("window_size must be > 0")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.window_size = int(window_size)
        self.max_relative_distance = int(max_relative_distance)
        self.dropout = float(dropout)
        self.use_global_anchors = bool(use_global_anchors)

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_dropout = nn.Dropout(self.dropout)
        self.relative_bias = nn.Parameter(
            torch.zeros(self.num_heads, self.max_relative_distance + 1)
        )

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return (
            x.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _build_attention_mask_and_bias(
        self,
        *,
        seq_len: int,
        anchor_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        q_pos = torch.arange(seq_len, device=device)
        k_pos = torch.arange(seq_len, device=device)
        relative = q_pos[:, None] - k_pos[None, :]
        token_allowed = (relative >= 0) & (relative < int(self.window_size))

        clipped = relative.clamp(min=0, max=int(self.max_relative_distance)).to(torch.long)
        token_bias = self.relative_bias[:, clipped].to(dtype=dtype)
        token_bias = token_bias.masked_fill(
            ~token_allowed.unsqueeze(0),
            torch.finfo(dtype).min if dtype.is_floating_point else -1e9,
        )

        if anchor_count <= 0:
            return token_bias.unsqueeze(0)

        anchor_bias = torch.zeros(
            (self.num_heads, seq_len, anchor_count),
            device=device,
            dtype=dtype,
        )
        return torch.cat([anchor_bias, token_bias], dim=-1).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        global_anchors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be (batch, seq, d_model), got {tuple(x.shape)}")
        batch_size, seq_len, dim = x.shape
        if dim != self.d_model:
            raise ValueError(f"last dim must be {self.d_model}, got {dim}")

        q = self._shape_heads(self.q_proj(x))
        k = self._shape_heads(self.k_proj(x))
        v = self._shape_heads(self.v_proj(x))

        anchor_count = 0
        if self.use_global_anchors and global_anchors is not None:
            if global_anchors.ndim != 2 or global_anchors.shape[-1] != self.d_model:
                raise ValueError(
                    "global_anchors must be (anchor_count, d_model), "
                    f"got {tuple(global_anchors.shape)}"
                )
            anchor_count = int(global_anchors.shape[0])
            anchor = global_anchors.unsqueeze(0).expand(batch_size, -1, -1)
            anchor_k = self._shape_heads(self.k_proj(anchor))
            anchor_v = self._shape_heads(self.v_proj(anchor))
            k = torch.cat([anchor_k, k], dim=2)
            v = torch.cat([anchor_v, v], dim=2)

        attn_bias = self._build_attention_mask_and_bias(
            seq_len=int(seq_len),
            anchor_count=int(anchor_count),
            device=x.device,
            dtype=q.dtype,
        )

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = (
            attn.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.out_dropout(self.out_proj(out))


class _DensePianoTransformerBlock(nn.Module):
    def __init__(
        self,
        cfg: DensePianoTransformerConfig,
        *,
        layer_index: int,
        window_size: int,
        use_global_anchors: bool,
    ) -> None:
        super().__init__()
        d = int(cfg.d_model)
        self.layer_index = int(layer_index)
        self.window_size = int(window_size)
        self.use_global_anchors = bool(use_global_anchors)

        self.norm_attn = RMSNorm(d, eps=float(cfg.rms_norm_eps))
        self.attn = _DenseSlidingWindowAttention(
            d_model=d,
            num_heads=int(cfg.num_attention_heads),
            head_dim=int(cfg.head_dim),
            window_size=int(window_size),
            max_relative_distance=int(cfg.max_relative_distance),
            dropout=float(cfg.attention_dropout),
            use_global_anchors=bool(use_global_anchors),
        )

        hidden_dim = round_multiple(
            float(d) * 8.0 / 3.0,
            int(cfg.ffn_multiple_of),
        )
        self.norm_ffn = RMSNorm(d, eps=float(cfg.rms_norm_eps))
        self.ffn = SwiGLU(
            d_model=d,
            hidden_dim=int(hidden_dim),
            dropout=float(cfg.dropout),
            multiple_of=int(cfg.ffn_multiple_of),
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        global_anchors: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), global_anchors=global_anchors)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class DensePianoTransformer(nn.Module):
    """Decoder-only dense sliding-window transformer for REMI-BPE piano tokens."""

    def __init__(self, config: Optional[DensePianoTransformerConfig] = None) -> None:
        super().__init__()
        self.config = config or DensePianoTransformerConfig()
        cfg = self.config

        if int(cfg.d_model) != int(cfg.num_attention_heads) * int(cfg.head_dim):
            raise ValueError(
                "DensePianoTransformer requires d_model == num_attention_heads * head_dim "
                f"({cfg.d_model} != {cfg.num_attention_heads} * {cfg.head_dim})"
            )

        schedule = tuple(int(v) for v in cfg.window_schedule)
        if len(schedule) != int(cfg.n_layers):
            raise ValueError(
                f"window_schedule length must equal n_layers ({len(schedule)} != {cfg.n_layers})"
            )
        if max(schedule) > int(cfg.max_relative_distance):
            raise ValueError("max_relative_distance must cover all window sizes")

        self.vocab_size = int(cfg.vocab_size)
        self.d_model = int(cfg.d_model)
        self.max_sequence_length = int(cfg.max_sequence_length)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.global_anchors = nn.Parameter(
            torch.empty(int(cfg.global_anchor_count), self.d_model)
        )

        anchor_start = int(max(1, cfg.global_anchor_start_layer))
        self.layers = nn.ModuleList(
            [
                _DensePianoTransformerBlock(
                    cfg,
                    layer_index=i,
                    window_size=int(schedule[i]),
                    use_global_anchors=(i + 1) >= anchor_start,
                )
                for i in range(int(cfg.n_layers))
            ]
        )

        self.final_norm = RMSNorm(self.d_model, eps=float(cfg.rms_norm_eps))
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
        nn.init.normal_(self.global_anchors, mean=0.0, std=std)
        if self.lm_head.weight.data_ptr() != self.token_embedding.weight.data_ptr():
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

    def get_num_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

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

    def _run_layer(
        self,
        layer: _DensePianoTransformerBlock,
        x: torch.Tensor,
    ) -> torch.Tensor:
        anchors = self.global_anchors if bool(layer.use_global_anchors) else None
        return layer(x, anchors)

    def forward(
        self,
        token_ids: torch.Tensor,
        onset_times: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        memory: Optional[Any] = None,
        return_memory: bool = False,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Any]] | torch.Tensor:
        del onset_times, durations, memory, position_offset

        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be (batch, seq), got {tuple(token_ids.shape)}")
        if int(token_ids.shape[1]) <= 0:
            raise ValueError("input sequence length must be > 0")
        if int(token_ids.shape[1]) > int(self.max_sequence_length):
            token_ids = token_ids[:, -int(self.max_sequence_length) :]

        x = self.dropout(self.token_embedding(token_ids))

        for layer in self.layers:
            if bool(self.config.gradient_checkpointing) and self.training:
                x = checkpoint(
                    lambda hidden, block=layer: self._run_layer(block, hidden),
                    x,
                    use_reentrant=False,
                )
            else:
                x = self._run_layer(layer, x)

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
        **_: Any,
    ) -> List[int]:
        self.eval()
        device = next(self.parameters()).device
        tokens = self._to_seed_tensor(seed_tokens, device=device)

        final_top1_probs: List[float] = []
        raw_top1_probs: List[float] = []
        candidate_counts: List[int] = []

        for _step in range(int(max_new_tokens)):
            context_tokens = tokens[:, -self.max_sequence_length :]
            logits, _ = self.forward(context_tokens, return_memory=True)
            next_token, diagnostics = sample_next_token(
                logits=logits[:, -1, :],
                context_tokens=context_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                recent_window=repetition_window,
                min_tokens_to_keep=max(3, int(min_tokens_to_keep)),
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
        _ = self.generate(
            seed_tokens=seed_tokens,
            max_new_tokens=int(steps),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            min_tokens_to_keep=min_tokens_to_keep,
        )
        max_final = float(self.last_generation_stats.get("final_top1_max", 0.0))
        passed = bool(max_final <= float(top1_threshold) + 1e-6)
        result: Dict[str, float | bool] = {
            "passed": passed,
            "max_final_top1_prob": max_final,
            "max_raw_top1_prob": float(self.last_generation_stats.get("raw_top1_max", 0.0)),
            "min_candidate_count": float(
                self.last_generation_stats.get("candidate_count_min", 0)
            ),
        }
        if raise_on_failure and not passed:
            raise AssertionError(
                "Generation health check failed: "
                f"max_final_top1_prob={max_final:.4f} exceeds threshold={top1_threshold:.4f}."
            )
        return result


__all__ = [
    "DensePianoTransformer",
    "DensePianoTransformerConfig",
    "FLASH_ATTENTION_AVAILABLE",
]

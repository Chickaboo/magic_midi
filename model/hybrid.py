from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from config import ModelConfig
from model.attention_block import MusicAttentionBlock
from model.cfc_block import CfCBlock
from model.ffn_block import FeedForwardBlock
from model.mamba_block import MAMBA_AVAILABLE, MambaBlock
from model.sampling import sample_next_token
from utils.logging_utils import get_project_logger


LOGGER = get_project_logger()


class PianoHybridModel(nn.Module):
    """Hybrid autoregressive model combining Mamba, CfC/FFN, and sparse attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.use_mamba = bool(config.use_mamba)
        self.use_cfc = bool(config.use_cfc)
        self.max_sequence_length = int(config.max_sequence_length)

        if config.attention_every_n_layers <= 0:
            raise ValueError("attention_every_n_layers must be > 0")
        if config.cfc_every_n_layers <= 0:
            raise ValueError("cfc_every_n_layers must be > 0")

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length,
            config.d_model,
        )
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList()
        self._cfc_layer_indices: List[int] = []
        for i in range(config.n_layers):
            use_cfc_this_layer = bool(
                self.use_cfc and (i % config.cfc_every_n_layers == 0)
            )
            cfc_module: nn.Module
            ffn_module: nn.Module
            if use_cfc_this_layer:
                cfc_module = CfCBlock(
                    d_model=config.d_model,
                    cfc_units=config.cfc_units,
                    backbone_units=config.cfc_backbone_units,
                    backbone_layers=config.cfc_backbone_layers,
                    dropout=config.dropout,
                    debug=config.debug,
                )
                ffn_module = nn.Identity()
                self._cfc_layer_indices.append(i)
            elif not self.use_cfc:
                cfc_module = nn.Identity()
                ffn_module = FeedForwardBlock(
                    d_model=config.d_model,
                    expansion=config.ffn_expansion,
                    dropout=config.dropout,
                    debug=config.debug,
                )
            else:
                cfc_module = nn.Identity()
                ffn_module = nn.Identity()

            layer_group = nn.ModuleDict(
                {
                    "mamba": (
                        MambaBlock(
                            d_model=config.d_model,
                            d_state=config.d_state,
                            d_conv=config.d_conv,
                            expand=config.expand,
                            dropout=config.dropout,
                            debug=config.debug,
                        )
                        if self.use_mamba
                        else nn.Identity()
                    ),
                    "cfc": cfc_module,
                    "ffn": ffn_module,
                }
            )
            self.layers.append(layer_group)

            if (i + 1) % config.attention_every_n_layers == 0:
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "attention": MusicAttentionBlock(
                                d_model=config.d_model,
                                num_heads=config.num_attention_heads,
                                max_relative_distance=config.max_relative_distance,
                                dropout=config.attention_dropout,
                                use_relative_bias=config.use_relative_attention,
                                bias_type=config.attention_bias_type,
                            )
                        }
                    )
                )

        self._num_cfc_layers = len(self._cfc_layer_indices)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.tie_embeddings = bool(config.tie_embeddings)
        if self.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        if config.output_logit_scale is None:
            self.output_logit_scale = 1.0 / math.sqrt(float(config.d_model))
        else:
            self.output_logit_scale = float(config.output_logit_scale)

        self._reset_parameters()
        self.last_generation_stats: Dict[str, Any] = {}

    def _reset_parameters(self) -> None:
        """Initialize embeddings and untied output projection."""

        init_std = float(max(1e-6, self.config.embedding_init_std))
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=init_std)
        if not self.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: Optional[List[Any]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[List[Any]]]:
        """Run a forward pass.

        Args:
            input_ids: Token IDs with shape `(batch, seq_len)`.
            hidden_states: Optional CfC hidden states, one entry per active CfC layer.
            position_offset: Absolute offset for position embedding indexing.

        Returns:
            Tuple of `(logits, new_hidden_states)`.
        """

        # Entry shape contract: input_ids is (batch, seq_len).
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape (batch, seq_len), got {tuple(input_ids.shape)}"
            )

        batch_size, seq_len = input_ids.shape
        if seq_len <= 0:
            raise ValueError("input sequence length must be > 0")

        if self.config.debug:
            assert input_ids.dtype in (torch.int64, torch.int32), (
                f"input_ids must be integer, got {input_ids.dtype}"
            )

        if self.config.use_absolute_positions:
            positions = torch.arange(
                position_offset,
                position_offset + seq_len,
                device=input_ids.device,
            )
            positions = torch.clamp(positions, max=self.config.max_sequence_length - 1)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        else:
            positions = torch.zeros_like(input_ids)

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        if self.use_cfc:
            if hidden_states is None:
                hidden_states = [None] * self._num_cfc_layers
            elif len(hidden_states) != self._num_cfc_layers:
                raise ValueError(
                    f"hidden_states length {len(hidden_states)} does not match "
                    f"expected CfC layers {self._num_cfc_layers}"
                )

        new_hidden: List[Any] = []
        cfc_idx = 0

        for i, layer_module in enumerate(self.layers):
            if not isinstance(layer_module, nn.ModuleDict):
                raise TypeError(
                    f"Expected ModuleDict in self.layers[{i}], got {type(layer_module).__name__}"
                )
            layer = layer_module

            if self.config.debug:
                assert x.ndim == 3 and x.shape[-1] == self.config.d_model, (
                    f"Layer {i} expected (batch, seq, {self.config.d_model}), got {tuple(x.shape)}"
                )

            if "attention" in layer:
                x = layer["attention"](x)
                continue

            if self.use_mamba and "mamba" in layer:
                x = layer["mamba"](x)

            if (
                self.use_cfc
                and "cfc" in layer
                and not isinstance(layer["cfc"], nn.Identity)
            ):
                h_i = hidden_states[cfc_idx] if hidden_states is not None else None
                x, new_h = layer["cfc"](x, hidden=h_i)
                new_hidden.append(new_h)
                cfc_idx += 1

            if (
                (not self.use_cfc)
                and "ffn" in layer
                and not isinstance(layer["ffn"], nn.Identity)
            ):
                x = layer["ffn"](x)

        x = self.final_norm(x)
        logits = self.lm_head(x) * float(self.output_logit_scale)

        if self.config.debug:
            assert logits.shape == (batch_size, seq_len, self.config.vocab_size), (
                f"Logits shape mismatch. Expected {(batch_size, seq_len, self.config.vocab_size)}, "
                f"got {tuple(logits.shape)}"
            )

        # Exit shape contract: logits is (batch, seq_len, vocab_size).
        return logits, (new_hidden if len(new_hidden) > 0 else None)

    def get_num_params(self) -> int:
        """Log and return total number of trainable parameters."""

        embedding_params = (
            self.token_embedding.weight.numel() + self.position_embedding.weight.numel()
        )
        mamba_params = 0
        cfc_params = 0
        ffn_params = 0
        attention_params = 0

        for i, layer_module in enumerate(self.layers):
            if not isinstance(layer_module, nn.ModuleDict):
                raise TypeError(
                    f"Expected ModuleDict in self.layers[{i}], got {type(layer_module).__name__}"
                )
            layer = layer_module

            if "attention" in layer:
                attention_params += sum(
                    p.numel() for p in layer["attention"].parameters()
                )
                continue

            if "mamba" in layer and not isinstance(layer["mamba"], nn.Identity):
                mamba_params += sum(p.numel() for p in layer["mamba"].parameters())

            if "cfc" in layer and not isinstance(layer["cfc"], nn.Identity):
                cfc_params += sum(p.numel() for p in layer["cfc"].parameters())
            if "ffn" in layer and not isinstance(layer["ffn"], nn.Identity):
                ffn_params += sum(p.numel() for p in layer["ffn"].parameters())

        final_norm_params = sum(p.numel() for p in self.final_norm.parameters())
        output_projection_params = self.lm_head.weight.numel()
        output_tied = (
            self.lm_head.weight.data_ptr() == self.token_embedding.weight.data_ptr()
        )

        total_params = sum(p.numel() for p in self.parameters())

        using_fallback = False
        for i, layer_module in enumerate(self.layers):
            if not isinstance(layer_module, nn.ModuleDict):
                raise TypeError(
                    f"Expected ModuleDict in self.layers[{i}], got {type(layer_module).__name__}"
                )
            layer = layer_module

            if "mamba" in layer and hasattr(layer["mamba"], "using_fallback"):
                using_fallback = bool(getattr(layer["mamba"], "using_fallback", False))
                break

        LOGGER.info("PianoHybridModel parameter breakdown")
        LOGGER.info("  embedding params: %s", f"{embedding_params:,}")
        LOGGER.info("  attention params: %s", f"{attention_params:,}")
        LOGGER.info("  mamba params: %s", f"{mamba_params:,}")
        LOGGER.info("  cfc params: %s", f"{cfc_params:,}")
        LOGGER.info("  ffn params: %s", f"{ffn_params:,}")
        LOGGER.info("  final norm params: %s", f"{final_norm_params:,}")
        if output_tied:
            LOGGER.info(
                "  output projection params: %s",
                f"{output_projection_params:,} (weight-tied to token embedding)",
            )
        else:
            LOGGER.info(
                "  output projection params: %s", f"{output_projection_params:,}"
            )
        LOGGER.info("  total params: %s", f"{total_params:,}")
        LOGGER.info(
            "  mamba backend: "
            f"{'mamba-ssm' if (MAMBA_AVAILABLE and not using_fallback) else 'GRU fallback'}"
        )
        LOGGER.info("  CfC enabled: %s", self.use_cfc)
        LOGGER.info("  Mamba enabled: %s", self.use_mamba)
        LOGGER.info("  Relative attention: %s", self.config.use_relative_attention)
        LOGGER.info("  Attention bias: %s", self.config.attention_bias_type)
        LOGGER.info("  Output logit scale: %.6f", self.output_logit_scale)
        return total_params

    @staticmethod
    def _to_seed_tensor(
        seed_tokens: Sequence[int] | torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert input seed tokens to a `(1, seq_len)` long tensor."""

        if isinstance(seed_tokens, torch.Tensor):
            if seed_tokens.ndim == 1:
                seed = seed_tokens.unsqueeze(0)
            elif seed_tokens.ndim == 2:
                if seed_tokens.shape[0] != 1:
                    raise ValueError("generate supports batch size 1 seed tensor")
                seed = seed_tokens
            else:
                raise ValueError(
                    f"Unsupported seed tensor shape: {tuple(seed_tokens.shape)}"
                )
            return seed.to(device=device, dtype=torch.long)

        seed_list = [int(t) for t in seed_tokens]
        if not seed_list:
            raise ValueError("seed_tokens cannot be empty")
        return torch.tensor(seed_list, device=device, dtype=torch.long).unsqueeze(0)

    @staticmethod
    def _repetition_ratio(tokens: Sequence[int]) -> float:
        """Compute max-token frequency ratio for repetition diagnostics."""

        if not tokens:
            return 0.0
        counts: Dict[int, int] = {}
        for token in tokens:
            token_i = int(token)
            counts[token_i] = counts.get(token_i, 0) + 1
        most_common = max(counts.values())
        return float(most_common) / float(len(tokens))

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
    ) -> List[int]:
        """Generate a continuation from a seed sequence."""

        self.eval()
        device = next(self.parameters()).device

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")

        generated = self._to_seed_tensor(seed_tokens, device=device)
        if max_new_tokens == 0:
            return generated.squeeze(0).tolist()

        final_top1_probs: List[float] = []
        raw_top1_probs: List[float] = []
        candidate_counts: List[int] = []

        if self.use_cfc:
            context = generated[:, -self.max_sequence_length :]
            pos_offset = max(0, generated.shape[1] - context.shape[1])
            logits, hidden_states = self.forward(
                context,
                hidden_states=None,
                position_offset=pos_offset,
            )

            for step in range(max_new_tokens):
                sample_context = generated[:, -self.max_sequence_length :]
                next_token, diagnostics = sample_next_token(
                    logits=logits[:, -1, :],
                    context_tokens=sample_context,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    recent_window=repetition_window,
                    min_tokens_to_keep=max(3, min_tokens_to_keep),
                    top1_cap=0.95,
                )
                final_top1_probs.extend(
                    [
                        float(v)
                        for v in diagnostics.final_top1_prob.detach().cpu().tolist()
                    ]
                )
                raw_top1_probs.extend(
                    [
                        float(v)
                        for v in diagnostics.raw_top1_prob.detach().cpu().tolist()
                    ]
                )
                candidate_counts.extend(
                    [
                        int(v)
                        for v in diagnostics.candidate_count.detach().cpu().tolist()
                    ]
                )
                generated = torch.cat([generated, next_token], dim=1)

                if step == max_new_tokens - 1:
                    break

                next_pos = generated.shape[1] - 1
                logits, hidden_states = self.forward(
                    next_token,
                    hidden_states=hidden_states,
                    position_offset=next_pos,
                )
        else:
            for _ in range(max_new_tokens):
                sample_context = generated[:, -self.max_sequence_length :]
                pos_offset = max(0, generated.shape[1] - sample_context.shape[1])
                logits, _ = self.forward(
                    sample_context,
                    hidden_states=None,
                    position_offset=pos_offset,
                )
                next_token, diagnostics = sample_next_token(
                    logits=logits[:, -1, :],
                    context_tokens=sample_context,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    recent_window=repetition_window,
                    min_tokens_to_keep=max(3, min_tokens_to_keep),
                    top1_cap=0.95,
                )
                final_top1_probs.extend(
                    [
                        float(v)
                        for v in diagnostics.final_top1_prob.detach().cpu().tolist()
                    ]
                )
                raw_top1_probs.extend(
                    [
                        float(v)
                        for v in diagnostics.raw_top1_prob.detach().cpu().tolist()
                    ]
                )
                candidate_counts.extend(
                    [
                        int(v)
                        for v in diagnostics.candidate_count.detach().cpu().tolist()
                    ]
                )
                generated = torch.cat([generated, next_token], dim=1)

        generated_list = generated.squeeze(0).tolist()
        new_tokens = generated_list[-max_new_tokens:]
        repetition_ratio = self._repetition_ratio(new_tokens)
        if repetition_ratio > 0.60:
            warnings.warn(
                "Generation repetition warning: "
                f"{repetition_ratio * 100:.1f}% of generated tokens are identical"
            )

        self.last_generation_stats = {
            "raw_top1_max": max(raw_top1_probs) if raw_top1_probs else 0.0,
            "final_top1_max": max(final_top1_probs) if final_top1_probs else 0.0,
            "final_top1_mean": (
                float(sum(final_top1_probs) / len(final_top1_probs))
                if final_top1_probs
                else 0.0
            ),
            "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
            "candidate_count_mean": (
                float(sum(candidate_counts) / len(candidate_counts))
                if candidate_counts
                else 0.0
            ),
            "repetition_ratio": repetition_ratio,
            "generated_tokens": int(max_new_tokens),
        }
        return generated_list

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
        """Run short generation diagnostics and report confidence statistics."""

        self.eval()
        device = next(self.parameters()).device

        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")

        generated = self._to_seed_tensor(seed_tokens, device=device)
        final_top1_probs: List[float] = []
        raw_top1_probs: List[float] = []
        candidate_counts: List[int] = []

        if self.use_cfc:
            context = generated[:, -self.max_sequence_length :]
            pos_offset = max(0, generated.shape[1] - context.shape[1])
            logits, hidden_states = self.forward(
                context,
                hidden_states=None,
                position_offset=pos_offset,
            )

            for step in range(int(steps)):
                sample_context = generated[:, -self.max_sequence_length :]
                next_token, diagnostics = sample_next_token(
                    logits=logits[:, -1, :],
                    context_tokens=sample_context,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    recent_window=repetition_window,
                    min_tokens_to_keep=max(3, min_tokens_to_keep),
                    top1_cap=top1_threshold,
                )
                final_top1_probs.extend(
                    [
                        float(v)
                        for v in diagnostics.final_top1_prob.detach().cpu().tolist()
                    ]
                )
                raw_top1_probs.extend(
                    [
                        float(v)
                        for v in diagnostics.raw_top1_prob.detach().cpu().tolist()
                    ]
                )
                candidate_counts.extend(
                    [
                        int(v)
                        for v in diagnostics.candidate_count.detach().cpu().tolist()
                    ]
                )
                generated = torch.cat([generated, next_token], dim=1)

                if step == int(steps) - 1:
                    break

                next_pos = generated.shape[1] - 1
                logits, hidden_states = self.forward(
                    next_token,
                    hidden_states=hidden_states,
                    position_offset=next_pos,
                )
        else:
            for _ in range(int(steps)):
                sample_context = generated[:, -self.max_sequence_length :]
                pos_offset = max(0, generated.shape[1] - sample_context.shape[1])
                logits, _ = self.forward(
                    sample_context,
                    hidden_states=None,
                    position_offset=pos_offset,
                )
                next_token, diagnostics = sample_next_token(
                    logits=logits[:, -1, :],
                    context_tokens=sample_context,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    recent_window=repetition_window,
                    min_tokens_to_keep=max(3, min_tokens_to_keep),
                    top1_cap=top1_threshold,
                )
                final_top1_probs.extend(
                    [
                        float(v)
                        for v in diagnostics.final_top1_prob.detach().cpu().tolist()
                    ]
                )
                raw_top1_probs.extend(
                    [
                        float(v)
                        for v in diagnostics.raw_top1_prob.detach().cpu().tolist()
                    ]
                )
                candidate_counts.extend(
                    [
                        int(v)
                        for v in diagnostics.candidate_count.detach().cpu().tolist()
                    ]
                )
                generated = torch.cat([generated, next_token], dim=1)

        max_final_top1 = max(final_top1_probs) if final_top1_probs else 0.0
        max_raw_top1 = max(raw_top1_probs) if raw_top1_probs else 0.0
        passed = bool(max_final_top1 <= float(top1_threshold))

        result: Dict[str, float | bool] = {
            "passed": passed,
            "max_final_top1_prob": float(max_final_top1),
            "mean_final_top1_prob": float(
                sum(final_top1_probs) / max(1, len(final_top1_probs))
            ),
            "max_raw_top1_prob": float(max_raw_top1),
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
                f"max_final_top1_prob={max_final_top1:.4f} exceeds threshold={top1_threshold:.4f}. "
                "Reduce confidence by increasing temperature/top-k or checking model calibration."
            )
        return result

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from config import ModelConfig
from model.attention_block import MusicAttentionBlock
from model.cfc_block import CfCBlock
from model.mamba_block import MAMBA_AVAILABLE, MambaBlock


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 50,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        top_vals, _ = torch.topk(logits, k=k, dim=-1)
        cutoff = top_vals[..., -1, None]
        logits = torch.where(
            logits < cutoff, torch.full_like(logits, -float("inf")), logits
        )

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(remove_mask, -float("inf"))
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class PianoHybridModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.use_mamba = bool(config.use_mamba)
        self.use_cfc = bool(config.use_cfc)

        if config.attention_every_n_layers <= 0:
            raise ValueError("attention_every_n_layers must be > 0")
        if config.cfc_every_n_layers <= 0:
            raise ValueError("cfc_every_n_layers must be > 0")

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.d_model
        )
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList()
        self._cfc_layer_indices: List[int] = []
        for i in range(config.n_layers):
            use_cfc_this_layer = bool(
                self.use_cfc and (i % config.cfc_every_n_layers == 0)
            )
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
                    "cfc": (
                        CfCBlock(
                            d_model=config.d_model,
                            cfc_units=config.cfc_units,
                            backbone_units=config.cfc_backbone_units,
                            backbone_layers=config.cfc_backbone_layers,
                            dropout=config.dropout,
                            debug=config.debug,
                        )
                        if use_cfc_this_layer
                        else nn.Identity()
                    ),
                }
            )
            self.layers.append(layer_group)
            if use_cfc_this_layer:
                self._cfc_layer_indices.append(i)

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
                            )
                        }
                    )
                )

        self._num_cfc_layers = len(self._cfc_layer_indices)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: Optional[List[Any]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[List[Any]]]:
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

        positions = torch.arange(
            position_offset,
            position_offset + seq_len,
            device=input_ids.device,
        )
        positions = torch.clamp(positions, max=self.config.max_sequence_length - 1)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

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

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if self.config.debug:
            assert logits.shape == (batch_size, seq_len, self.config.vocab_size), (
                f"Logits shape mismatch. Expected {(batch_size, seq_len, self.config.vocab_size)}, "
                f"got {tuple(logits.shape)}"
            )

        return logits, (new_hidden if len(new_hidden) > 0 else None)

    def get_num_params(self) -> int:
        embedding_params = (
            self.token_embedding.weight.numel() + self.position_embedding.weight.numel()
        )
        mamba_params = 0
        cfc_params = 0
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

        print("PianoHybridModel parameter breakdown")
        print(f"  embedding params: {embedding_params:,}")
        print(f"  attention params: {attention_params:,}")
        print(f"  mamba params: {mamba_params:,}")
        print(f"  cfc params: {cfc_params:,}")
        print(f"  final norm params: {final_norm_params:,}")
        if output_tied:
            print(
                "  output projection params: "
                f"{output_projection_params:,} (weight-tied to token embedding)"
            )
        else:
            print(f"  output projection params: {output_projection_params:,}")
        print(f"  total params: {total_params:,}")
        print(
            "  mamba backend: "
            f"{'mamba-ssm' if (MAMBA_AVAILABLE and not using_fallback) else 'GRU fallback'}"
        )
        print(f"  CfC enabled: {self.use_cfc}")
        print(f"  CfC cadence: every {self.config.cfc_every_n_layers} layer(s)")
        print(f"  Mamba enabled: {self.use_mamba}")
        print(f"  Relative attention: {self.config.use_relative_attention}")
        return total_params

    @torch.no_grad()
    def generate(
        self,
        seed_tokens: Sequence[int] | torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> List[int]:
        self.eval()
        device = next(self.parameters()).device

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")

        if isinstance(seed_tokens, torch.Tensor):
            if seed_tokens.ndim == 1:
                seed = seed_tokens.unsqueeze(0)
            elif seed_tokens.ndim == 2:
                if seed_tokens.shape[0] != 1:
                    raise ValueError("generate supports batch size 1 seed tensor.")
                seed = seed_tokens
            else:
                raise ValueError(
                    f"Unsupported seed tensor shape: {tuple(seed_tokens.shape)}"
                )
            seed = seed.to(device=device, dtype=torch.long)
        else:
            seed_list = [int(t) for t in seed_tokens]
            if not seed_list:
                raise ValueError("seed_tokens cannot be empty")
            seed = torch.tensor(seed_list, device=device, dtype=torch.long).unsqueeze(0)

        generated = seed.clone()
        if max_new_tokens == 0:
            return generated.squeeze(0).tolist()

        for _ in range(max_new_tokens):
            logits, _ = self.forward(
                generated,
                hidden_states=None,
                position_offset=0,
            )
            next_token = _sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            generated = torch.cat([generated, next_token], dim=1)

        return generated.squeeze(0).tolist()

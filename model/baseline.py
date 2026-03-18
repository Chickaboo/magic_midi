from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 50,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-8)

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


class PianoBaselineModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_sequence_length: int = 1024,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_sequence_length, d_model)
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids should be (batch, seq), got {tuple(input_ids.shape)}"
            )

        bsz, seq_len = input_ids.shape
        positions = torch.arange(
            position_offset,
            position_offset + seq_len,
            device=input_ids.device,
        )
        positions = torch.clamp(positions, max=self.max_sequence_length - 1)
        positions = positions.unsqueeze(0).expand(bsz, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        y, h = self.gru(x, hidden_states)
        y = self.final_norm(y)
        logits = self.lm_head(y)
        return logits, h

    def get_num_params(self) -> int:
        embedding_params = (
            self.token_embedding.weight.numel() + self.position_embedding.weight.numel()
        )
        gru_params = sum(p.numel() for p in self.gru.parameters())
        final_params = sum(p.numel() for p in self.final_norm.parameters())
        total = sum(p.numel() for p in self.parameters())

        print("PianoBaselineModel parameter breakdown")
        print(f"  embedding params: {embedding_params:,}")
        print(f"  gru params: {gru_params:,}")
        print(f"  final norm params: {final_params:,}")
        print(f"  total params: {total:,}")
        return total

    @torch.no_grad()
    def generate(
        self,
        seed_tokens: Sequence[int] | torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> List[int]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")

        self.eval()
        device = next(self.parameters()).device

        if isinstance(seed_tokens, torch.Tensor):
            seed = seed_tokens.to(device=device, dtype=torch.long)
            if seed.ndim == 1:
                seed = seed.unsqueeze(0)
        else:
            seed = torch.tensor(
                [int(t) for t in seed_tokens], dtype=torch.long, device=device
            ).unsqueeze(0)

        generated = seed.clone()
        logits, hidden = self.forward(generated, hidden_states=None, position_offset=0)

        for _ in range(max_new_tokens):
            next_token = _sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            generated = torch.cat([generated, next_token], dim=1)
            pos_offset = generated.shape[1] - 1
            logits, hidden = self.forward(
                generated[:, -1:],
                hidden_states=hidden,
                position_offset=pos_offset,
            )

        return generated.squeeze(0).tolist()

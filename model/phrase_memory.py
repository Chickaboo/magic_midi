from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhraseSummarizer(nn.Module):
    """Compress token features into phrase representations via attentive pooling."""

    def __init__(
        self,
        d_model: int,
        phrase_dim: int,
        tokens_per_phrase: int = 16,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if phrase_dim <= 0:
            raise ValueError("phrase_dim must be > 0")
        if tokens_per_phrase <= 0:
            raise ValueError("tokens_per_phrase must be > 0")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be > 0")

        self.tokens_per_phrase = int(tokens_per_phrase)
        self.residual_scale = float(residual_scale)
        self.phrase_attention = nn.Linear(d_model, 1)
        self.boundary_score = nn.Linear(d_model, 1)
        self.phrase_proj = nn.Linear(d_model, phrase_dim)
        self.norm = nn.LayerNorm(phrase_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return phrase tensor shaped `(batch, num_phrases, phrase_dim)`."""

        if x.ndim != 3:
            raise ValueError(
                f"x must be (batch, seq_len, d_model), got {tuple(x.shape)}"
            )

        batch, seq_len, d_model = x.shape
        pad_len = (
            self.tokens_per_phrase - (seq_len % self.tokens_per_phrase)
        ) % self.tokens_per_phrase
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        num_phrases = x.shape[1] // self.tokens_per_phrase
        phrases = x.view(batch, num_phrases, self.tokens_per_phrase, d_model)

        pos = torch.linspace(
            0.0,
            1.0,
            steps=self.tokens_per_phrase,
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, self.tokens_per_phrase)
        edge_prior = 1.0 + 0.5 * (2.0 * torch.abs(pos - 0.5))
        boundary = torch.sigmoid(self.boundary_score(phrases).squeeze(-1))
        raw_attn = self.phrase_attention(phrases).squeeze(-1)
        raw_attn = raw_attn + torch.log(edge_prior.clamp_min(1e-6))
        raw_attn = raw_attn + torch.log(boundary.clamp_min(1e-6))

        attn = torch.softmax(raw_attn, dim=-1)
        pooled = (attn.unsqueeze(-1) * phrases).sum(dim=2)
        out = self.norm(self.phrase_proj(pooled))
        return out * float(self.residual_scale)


class EpisodicThemeMemory(nn.Module):
    """Content-addressable phrase memory for long-range thematic callbacks."""

    def __init__(
        self,
        phrase_dim: int,
        memory_size: int = 64,
        num_heads: int = 4,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if phrase_dim <= 0:
            raise ValueError("phrase_dim must be > 0")
        if memory_size <= 0:
            raise ValueError("memory_size must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if phrase_dim % num_heads != 0:
            raise ValueError(
                f"phrase_dim ({phrase_dim}) must be divisible by num_heads ({num_heads})"
            )
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be > 0")

        self.memory_size = int(memory_size)
        self.phrase_dim = int(phrase_dim)
        self.residual_scale = float(residual_scale)

        self.query_proj = nn.Linear(phrase_dim, phrase_dim)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=phrase_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.pre_norm = nn.LayerNorm(phrase_dim)
        self.write_gate = nn.Sequential(
            nn.Linear(phrase_dim, max(1, phrase_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(1, phrase_dim // 2), 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(phrase_dim)

    @staticmethod
    def _normalize_for_storage(x: torch.Tensor) -> torch.Tensor:
        """Normalize phrase vectors before memory writes."""

        return F.normalize(x, dim=-1)

    def _select_memory_candidates(
        self,
        memory: torch.Tensor,
        candidates: torch.Tensor,
        write_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Keep top-scoring phrase vectors when memory capacity is exceeded."""

        existing_score = torch.full(
            size=(memory.shape[0], memory.shape[1]),
            fill_value=0.5,
            dtype=write_scores.dtype,
            device=write_scores.device,
        )
        all_values = torch.cat([memory, candidates], dim=1)
        all_scores = torch.cat([existing_score, write_scores.squeeze(-1)], dim=1)

        keep = min(self.memory_size, int(all_values.shape[1]))
        if keep <= 0:
            return all_values[:, :0, :]

        top_idx = torch.topk(all_scores, k=keep, dim=1).indices
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, all_values.shape[-1])
        kept_values = torch.gather(all_values, dim=1, index=gather_idx)

        sorted_idx = torch.sort(top_idx, dim=1).indices
        reorder_idx = sorted_idx.unsqueeze(-1).expand(-1, -1, all_values.shape[-1])
        return torch.gather(kept_values, dim=1, index=reorder_idx)

    def forward(
        self,
        phrases: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enhance phrases with memory context and update memory state."""

        if phrases.ndim != 3:
            raise ValueError(
                f"phrases must be (batch, num_phrases, phrase_dim), got {tuple(phrases.shape)}"
            )

        write_scores = self.write_gate(phrases)
        write_mask = (write_scores > 0.5).to(dtype=phrases.dtype)
        candidates = self._normalize_for_storage(phrases) * write_mask * write_scores

        if memory is None or memory.numel() == 0 or memory.shape[1] == 0:
            initial = self._select_memory_candidates(
                memory=phrases[:, :0, :],
                candidates=candidates,
                write_scores=write_scores,
            )
            if initial.shape[1] == 0:
                initial = self._normalize_for_storage(phrases[:, -1:, :])
            return phrases, initial

        normalized_phrases = self.pre_norm(phrases)
        queries = self.query_proj(normalized_phrases)
        memory_context, _ = self.memory_attention(queries, memory, memory)
        enhanced = self.norm(phrases + (memory_context * float(self.residual_scale)))

        merged = self._select_memory_candidates(
            memory=memory,
            candidates=candidates,
            write_scores=write_scores,
        )
        return enhanced, merged

    def reset(self) -> None:
        """Return empty state marker for start-of-piece resets."""

        return None

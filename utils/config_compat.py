from __future__ import annotations

from typing import Any, Dict


def normalize_model_config_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fill backward-compatible defaults for checkpoints from older versions."""

    normalized = dict(payload)

    if "attention_bias_type" not in normalized:
        # v1 used learned relative bias tables.
        normalized["attention_bias_type"] = "learned"
    if "use_absolute_positions" not in normalized:
        normalized["use_absolute_positions"] = True
    if "ffn_expansion" not in normalized:
        normalized["ffn_expansion"] = 4
    if "tie_embeddings" not in normalized:
        normalized["tie_embeddings"] = True
    if "embedding_init_std" not in normalized:
        normalized["embedding_init_std"] = 0.02
    if "output_logit_scale" not in normalized:
        normalized["output_logit_scale"] = None

    if "use_v2_architecture" not in normalized:
        normalized["use_v2_architecture"] = False
    if "use_continuous_time_encoding" not in normalized:
        normalized["use_continuous_time_encoding"] = False
    if "max_time_seconds" not in normalized:
        normalized["max_time_seconds"] = 600.0
    if "stream_dim" not in normalized:
        normalized["stream_dim"] = None
    if "harmonic_ratio" not in normalized:
        normalized["harmonic_ratio"] = 0.5
    if "cross_stream_every_n_layers" not in normalized:
        normalized["cross_stream_every_n_layers"] = 2
    if "cross_attention_every_n" not in normalized:
        normalized["cross_attention_every_n"] = None
    if "tokens_per_phrase" not in normalized:
        normalized["tokens_per_phrase"] = 16
    if "phrase_dim" not in normalized:
        normalized["phrase_dim"] = None
    if "memory_size" not in normalized:
        normalized["memory_size"] = 64
    if "theme_memory_heads" not in normalized:
        normalized["theme_memory_heads"] = 4

    return normalized

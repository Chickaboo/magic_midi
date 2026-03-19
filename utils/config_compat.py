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

    return normalized

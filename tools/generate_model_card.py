from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DataConfig
from scale_config import SCALE_PRESETS
from utils import checkpoint_loading as ckpt_utils


def _load_state_payload(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    if path.suffix != ".pt":
        return {}
    try:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def _tokenizer_search_paths(checkpoint_path: Path) -> list[Path]:
    return [
        checkpoint_path.parent / "custom_tokenizer.json",
        checkpoint_path.parent / "tokenizer.json",
        checkpoint_path.parent / "tokenizer" / "custom_tokenizer.json",
        checkpoint_path.parent / "tokenizer" / "tokenizer.json",
        Path("tokenizer.json"),
    ]


def _config_value(model_config: Dict[str, Any], key: str, default: Any = "n/a") -> Any:
    value = model_config.get(key, default)
    if value is None:
        return default
    return value


def _optional_arch_line(
    model_config: Dict[str, Any],
    key: str,
    label: str,
) -> Optional[str]:
    if key not in model_config:
        return None
    return f"- {label}: `{model_config.get(key)}`"


def _load_tokenizer_for_card(
    checkpoint_metadata: Dict[str, Any],
    checkpoint_path: Path,
) -> tuple[Any | None, Optional[Path]]:
    try:
        tokenizer, tokenizer_meta = ckpt_utils.load_tokenizer_for_checkpoint(
            checkpoint_metadata,
            search_paths=_tokenizer_search_paths(checkpoint_path),
        )
    except FileNotFoundError:
        return None, None

    tokenizer_path = tokenizer_meta.get("tokenizer_path")
    if isinstance(tokenizer_path, str) and tokenizer_path.strip():
        return tokenizer, Path(tokenizer_path)
    return tokenizer, None


def _output_logit_scale_text(model: Any) -> str:
    raw = getattr(model, "output_logit_scale", None)
    if raw is None:
        return "n/a"
    try:
        return f"{float(raw):.6f}"
    except Exception:
        return str(raw)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_seed_and_continuation_lengths(data_cfg: DataConfig) -> tuple[int, int]:
    seed_length = max(1, _to_int(getattr(data_cfg, "seed_length", 128), 128))
    continuation = max(
        1,
        _to_int(getattr(data_cfg, "continuation_length", 128), 128),
    )
    return seed_length, continuation


def _history_metrics(state: Dict[str, Any]) -> tuple[list[Any], list[Any], list[Any]]:
    history = state.get("history") if isinstance(state.get("history"), dict) else {}
    train_loss = history.get("train_loss") if isinstance(history, dict) else []
    val_loss = history.get("val_loss") if isinstance(history, dict) else []
    gen_health = (
        history.get("gen_health_max_final_top1") if isinstance(history, dict) else []
    )
    return (
        train_loss if isinstance(train_loss, list) else [],
        val_loss if isinstance(val_loss, list) else [],
        gen_health if isinstance(gen_health, list) else [],
    )


def _model_architecture_name(model_config: Dict[str, Any]) -> str:
    if not model_config:
        return "unknown"
    return ckpt_utils.infer_model_architecture(model_config)


def _build_architecture_lines(model: Any, model_config: Dict[str, Any]) -> list[str]:
    lines: list[str] = [
        f"- Model class: `{type(model).__name__}`",
        f"- Inferred architecture: `{_model_architecture_name(model_config)}`",
        f"- d_model: `{_config_value(model_config, 'd_model')}`",
        f"- layers: `{_config_value(model_config, 'n_layers')}`",
    ]

    for key, label in (
        ("use_mamba", "Mamba enabled"),
        ("use_cfc", "CfC enabled"),
        ("ffn_expansion", "FFN expansion"),
        ("num_attention_heads", "Attention heads"),
        ("attention_every_n_layers", "Attention cadence"),
        ("attention_bias_type", "Attention bias"),
        ("tie_embeddings", "Tied embeddings"),
    ):
        line = _optional_arch_line(model_config, key, label)
        if line is not None:
            lines.append(line)

    lines.append(f"- Output logit scale: `{_output_logit_scale_text(model)}`")
    return lines


def _build_data_lines(data_cfg: DataConfig, model_config: Dict[str, Any]) -> list[str]:
    seed_length, continuation = _safe_seed_and_continuation_lengths(data_cfg)
    vocab_size = _to_int(_config_value(model_config, "vocab_size", data_cfg.vocab_size), data_cfg.vocab_size)
    return [
        f"- Vocabulary size: `{vocab_size}`",
        f"- Tokenization strategy: `{getattr(data_cfg, 'tokenization_strategy', 'n/a')}`",
        f"- Seed length: `{seed_length}`",
        f"- Continuation length: `{continuation}`",
    ]


def _build_history_lines(state: Dict[str, Any]) -> tuple[list[str], list[Any]]:
    train_loss, val_loss, gen_health = _history_metrics(state)
    lines = [
        f"- Epoch in checkpoint: `{state.get('epoch', 'n/a')}`",
        f"- Last val loss in checkpoint: `{state.get('val_loss', 'n/a')}`",
        f"- Best val loss tracked: `{state.get('best_val_loss', 'n/a')}`",
        f"- Train loss entries: `{len(train_loss)}`",
        f"- Val loss entries: `{len(val_loss)}`",
        f"- Generation health entries: `{len(gen_health)}`",
    ]
    return lines, gen_health


def _checkpoint_diagnostic_lines(bundle: ckpt_utils.LoadedModelBundle) -> list[str]:
    return [
        f"- Missing keys: `{int(bundle.missing_keys)}`",
        f"- Unexpected keys: `{int(bundle.unexpected_keys)}`",
    ]


def _preview_lines(preview_tokens: List[int], preview_unique: int) -> list[str]:
    return [
        f"- Preview token count: `{len(preview_tokens)}`",
        f"- Preview unique token count: `{preview_unique}`",
        "- First 32 tokens:",
        "",
        "```text",
        " ".join(str(int(t)) for t in preview_tokens[:32]),
        "```",
    ]


def _resolve_data_config(checkpoint_metadata: Dict[str, Any]) -> DataConfig:
    data_cfg_payload = ckpt_utils.extract_data_config(checkpoint_metadata)
    if isinstance(data_cfg_payload, dict) and data_cfg_payload:
        return DataConfig(**data_cfg_payload)
    return DataConfig()


def _validate_tokenizer_vocab(
    tokenizer: Any | None,
    model_config: Dict[str, Any],
) -> None:
    if tokenizer is None:
        return
    model_vocab = _to_int(model_config.get("vocab_size", -1), -1)
    tokenizer_vocab = _to_int(getattr(tokenizer, "vocab_size", -1), -1)
    if model_vocab > 0 and tokenizer_vocab > 0 and model_vocab != tokenizer_vocab:
        raise RuntimeError(
            "Tokenizer/model vocab mismatch while generating model card: "
            f"tokenizer vocab={tokenizer_vocab}, model vocab={model_vocab}."
        )


def _preview_generation_lengths(data_cfg: DataConfig) -> tuple[int, int]:
    seed_length, continuation = _safe_seed_and_continuation_lengths(data_cfg)
    return seed_length, min(128, continuation)


def _state_path_for_card(checkpoint_path: Path) -> Optional[Path]:
    return ckpt_utils.resolve_sidecar_path(checkpoint_path)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _make_preview_tokens(
    model: Any,
    tokenizer: Optional[Any],
    seed_length: int,
    max_new_tokens: int,
) -> List[int]:
    if tokenizer is None:
        seed = torch.randint(0, model.config.vocab_size, (seed_length,))
        return model.generate(
            seed,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            repetition_window=64,
            min_tokens_to_keep=3,
        )

    seed = torch.randint(0, model.config.vocab_size, (seed_length,))

    return model.generate(
        seed_tokens=seed,
        max_new_tokens=max_new_tokens,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        repetition_window=64,
        min_tokens_to_keep=3,
    )


def build_model_card(checkpoint_path: Path, output_path: Path) -> Path:
    """Build markdown model card from checkpoint and sidecar metadata."""

    state_path = _state_path_for_card(checkpoint_path)
    state = _load_state_payload(state_path)

    bundle = ckpt_utils.load_model_from_checkpoint(
        model_path=checkpoint_path,
        sidecar_path=state_path,
        device="cpu",
        strict=True,
    )

    model = bundle.model
    model_config = dict(bundle.model_config)
    total_params = sum(p.numel() for p in model.parameters())

    data_cfg = _resolve_data_config(bundle.checkpoint_metadata)

    tokenizer, tokenizer_path = _load_tokenizer_for_card(
        checkpoint_metadata=bundle.checkpoint_metadata,
        checkpoint_path=checkpoint_path,
    )
    _validate_tokenizer_vocab(tokenizer, model_config)

    seed_length, preview_new_tokens = _preview_generation_lengths(data_cfg)
    preview_tokens = _make_preview_tokens(
        model=model,
        tokenizer=tokenizer,
        seed_length=seed_length,
        max_new_tokens=preview_new_tokens,
    )
    preview_array = np.asarray(preview_tokens, dtype=np.int64)
    preview_unique = int(len(np.unique(preview_array))) if preview_array.size else 0

    history_lines, gen_health = _build_history_lines(state)

    default_small_cfg = SCALE_PRESETS["small"]["model"]
    if not model_config:
        model_config = dict(getattr(default_small_cfg, "__dict__", {}))

    card_lines = [
        f"# Model Card: {checkpoint_path.name}",
        "",
        "## Project",
        "- Name: `Itty Bitty Piano`",
        "",
        "## Architecture",
    ]
    card_lines.extend(_build_architecture_lines(model, model_config))
    card_lines.extend(
        [
        f"- Total parameters (measured): `{total_params:,}`",
        "",
        "## Data / Tokenization",
        ]
    )
    card_lines.extend(_build_data_lines(data_cfg, model_config))

    if tokenizer_path is not None:
        card_lines.append(f"- Tokenizer path: `{tokenizer_path}`")

    card_lines.extend(
        [
            "",
            "## Training History",
        ]
    )
    card_lines.extend(history_lines)

    if gen_health:
        card_lines.append(
            f"- Last generation max top-1 prob: `{_safe_float(gen_health[-1]):.4f}`"
        )

    card_lines.extend(
        [
            "",
            "## Checkpoint Load Diagnostics",
        ]
    )
    card_lines.extend(_checkpoint_diagnostic_lines(bundle))
    card_lines.extend(["", "## Generation Preview"])
    card_lines.extend(_preview_lines(preview_tokens, preview_unique))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(card_lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    """CLI entrypoint for model card generation."""

    parser = argparse.ArgumentParser(description="Generate a markdown model card")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", default="MODEL_CARD.md", type=str)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out = build_model_card(
        checkpoint_path=checkpoint_path,
        output_path=Path(args.output),
    )
    print(f"Model card written: {out}")


if __name__ == "__main__":
    main()

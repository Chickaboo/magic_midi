from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from safetensors.torch import load_file as safetensors_load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DataConfig, ModelConfig
from data.tokenizer import PianoTokenizer
from model.hybrid import PianoHybridModel
from scale_config import SCALE_PRESETS
from utils.config_compat import normalize_model_config_payload


def _load_state_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
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


def _find_state_sidecar(checkpoint_path: Path) -> Optional[Path]:
    if checkpoint_path.suffix == ".pt":
        return checkpoint_path if checkpoint_path.exists() else None
    if checkpoint_path.name.endswith("_model.safetensors"):
        candidate = checkpoint_path.with_name(
            checkpoint_path.name.replace("_model.safetensors", "_state.pt")
        )
        if candidate.exists():
            return candidate
    latest_state = checkpoint_path.parent / "latest_state.pt"
    if latest_state.exists():
        return latest_state
    best_state = checkpoint_path.parent / "best_state.pt"
    if best_state.exists():
        return best_state
    return None


def _resolve_tokenizer_path(
    state: Dict[str, Any], fallback_root: Path
) -> Optional[Path]:
    data_cfg = state.get("data_config")
    if isinstance(data_cfg, dict):
        tok = data_cfg.get("tokenizer_path")
        if isinstance(tok, str):
            p = Path(tok)
            if p.exists():
                return p
    candidates = [
        fallback_root / "tokenizer.json",
        fallback_root / "tokenizer" / "tokenizer.json",
        Path("tokenizer.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _make_preview_tokens(
    model: PianoHybridModel,
    tokenizer: Optional[PianoTokenizer],
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
    state_path = _find_state_sidecar(checkpoint_path)
    state = _load_state_dict(state_path) if state_path is not None else {}

    model_cfg_payload = state.get("model_config")
    if isinstance(model_cfg_payload, dict):
        model_cfg = ModelConfig(
            **normalize_model_config_payload(dict(model_cfg_payload))
        )
    else:
        model_cfg = SCALE_PRESETS["small"]["model"]

    model = PianoHybridModel(model_cfg)
    missing: List[str] = []
    unexpected: List[str] = []
    if checkpoint_path.suffix == ".safetensors" and checkpoint_path.exists():
        raw = safetensors_load_file(str(checkpoint_path), device="cpu")
        missing, unexpected = model.load_state_dict(raw, strict=False)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())

    data_cfg_payload = state.get("data_config")
    if isinstance(data_cfg_payload, dict):
        data_cfg = DataConfig(**data_cfg_payload)
    else:
        data_cfg = DataConfig()

    tokenizer_path = _resolve_tokenizer_path(state, checkpoint_path.parent)
    tokenizer = None
    if tokenizer_path is not None:
        try:
            tokenizer = PianoTokenizer.load(str(tokenizer_path))
        except Exception:
            tokenizer = None

    preview_tokens = _make_preview_tokens(
        model=model,
        tokenizer=tokenizer,
        seed_length=int(data_cfg.seed_length),
        max_new_tokens=min(128, int(data_cfg.continuation_length)),
    )
    preview_array = np.asarray(preview_tokens, dtype=np.int64)
    preview_unique = int(len(np.unique(preview_array))) if preview_array.size else 0

    history = state.get("history") if isinstance(state.get("history"), dict) else {}
    train_loss = history.get("train_loss") if isinstance(history, dict) else []
    val_loss = history.get("val_loss") if isinstance(history, dict) else []
    gen_health = (
        history.get("gen_health_max_final_top1") if isinstance(history, dict) else []
    )

    card_lines = [
        f"# Model Card: {checkpoint_path.name}",
        "",
        "## Project",
        "- Name: `Itty Bitty Piano`",
        "",
        "## Architecture",
        f"- Model class: `PianoHybridModel`",
        f"- d_model: `{model_cfg.d_model}`",
        f"- layers: `{model_cfg.n_layers}`",
        f"- Mamba enabled: `{model_cfg.use_mamba}`",
        f"- CfC enabled: `{model_cfg.use_cfc}`",
        f"- FFN expansion: `{model_cfg.ffn_expansion}`",
        f"- Attention heads: `{model_cfg.num_attention_heads}`",
        f"- Attention cadence: every `{model_cfg.attention_every_n_layers}` layers",
        f"- Attention bias: `{model_cfg.attention_bias_type}`",
        f"- Output logit scale: `{model.output_logit_scale:.6f}`",
        f"- Tied embeddings: `{model_cfg.tie_embeddings}`",
        f"- Total parameters (measured): `{total_params:,}`",
        "",
        "## Data / Tokenization",
        f"- Vocabulary size: `{model_cfg.vocab_size}`",
        f"- Tokenization strategy: `{data_cfg.tokenization_strategy}`",
        f"- Seed length: `{data_cfg.seed_length}`",
        f"- Continuation length: `{data_cfg.continuation_length}`",
    ]

    if tokenizer_path is not None:
        card_lines.append(f"- Tokenizer path: `{tokenizer_path}`")

    card_lines.extend(
        [
            "",
            "## Training History",
            f"- Epoch in checkpoint: `{state.get('epoch', 'n/a')}`",
            f"- Last val loss in checkpoint: `{state.get('val_loss', 'n/a')}`",
            f"- Best val loss tracked: `{state.get('best_val_loss', 'n/a')}`",
            f"- Train loss entries: `{len(train_loss) if isinstance(train_loss, list) else 0}`",
            f"- Val loss entries: `{len(val_loss) if isinstance(val_loss, list) else 0}`",
            f"- Generation health entries: `{len(gen_health) if isinstance(gen_health, list) else 0}`",
        ]
    )

    if isinstance(gen_health, list) and gen_health:
        card_lines.append(
            f"- Last generation max top-1 prob: `{_safe_float(gen_health[-1]):.4f}`"
        )

    card_lines.extend(
        [
            "",
            "## Checkpoint Load Diagnostics",
            f"- Missing keys: `{len(missing)}`",
            f"- Unexpected keys: `{len(unexpected)}`",
            "",
            "## Generation Preview",
            f"- Preview token count: `{len(preview_tokens)}`",
            f"- Preview unique token count: `{preview_unique}`",
            "- First 32 tokens:",
            "",
            "```text",
            " ".join(str(int(t)) for t in preview_tokens[:32]),
            "```",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(card_lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
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

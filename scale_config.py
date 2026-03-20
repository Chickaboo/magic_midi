from __future__ import annotations

import copy
import math
from typing import Any, Dict

from config import DataConfig, ModelConfig, TrainConfig


TARGET_PARAM_COUNTS: Dict[str, int] = {
    "small": 15_000_000,
    "medium": 40_000_000,
    "large": 100_000_000,
}


SCALE_PRESETS: Dict[str, Dict[str, Any]] = {
    "small": {
        "description": "~15M params - primary Kaggle free-tier target",
        "params": "~15M",
        "model": ModelConfig(
            d_model=320,
            n_layers=8,
            d_state=16,
            d_conv=4,
            expand=2,
            cfc_units=320,
            cfc_backbone_units=160,
            cfc_backbone_layers=2,
            cfc_every_n_layers=1,
            num_attention_heads=5,
            attention_every_n_layers=3,
            attention_dropout=0.1,
            use_relative_attention=True,
            attention_bias_type="alibi",
            max_relative_distance=128,
            use_absolute_positions=True,
            dropout=0.1,
            use_cfc=False,
            use_mamba=True,
            ffn_expansion=4,
            tie_embeddings=True,
            embedding_init_std=0.02,
            output_logit_scale=None,
        ),
        "train": TrainConfig(
            batch_size=8,
            grad_accumulation_steps=2,
            learning_rate=3e-4,
            warmup_steps=200,
            max_grad_norm=1.5,
            label_smoothing=0.1,
            save_every_n_epochs=5,
            keep_every_n_epochs=10,
        ),
        "data": DataConfig(
            max_pieces=None,
            tokenization_strategy="remi",
            seed_length=256,
            continuation_length=768,
            max_sequence_length=1024,
        ),
    },
    "medium": {
        "description": "~40M params - stretch Kaggle target",
        "params": "~40M",
        "model": ModelConfig(
            d_model=448,
            n_layers=10,
            d_state=16,
            d_conv=4,
            expand=2,
            cfc_units=448,
            cfc_backbone_units=224,
            cfc_backbone_layers=2,
            cfc_every_n_layers=1,
            num_attention_heads=7,
            attention_every_n_layers=2,
            attention_dropout=0.1,
            use_relative_attention=True,
            attention_bias_type="alibi",
            max_relative_distance=128,
            use_absolute_positions=True,
            dropout=0.1,
            use_cfc=False,
            use_mamba=True,
            ffn_expansion=4,
            tie_embeddings=True,
            embedding_init_std=0.02,
            output_logit_scale=None,
        ),
        "train": TrainConfig(
            batch_size=4,
            grad_accumulation_steps=4,
            learning_rate=2.5e-4,
            warmup_steps=2000,
            max_grad_norm=1.5,
            label_smoothing=0.1,
            save_every_n_epochs=5,
            keep_every_n_epochs=10,
        ),
        "data": DataConfig(
            max_pieces=None,
            tokenization_strategy="octuple",
            seed_length=256,
            continuation_length=768,
            max_sequence_length=1024,
        ),
    },
    "large": {
        "description": "~100M params - future Kaggle Pro / pro GPU target",
        "params": "~100M",
        "model": ModelConfig(
            d_model=640,
            n_layers=14,
            d_state=32,
            d_conv=4,
            expand=2,
            cfc_units=640,
            cfc_backbone_units=320,
            cfc_backbone_layers=2,
            cfc_every_n_layers=1,
            num_attention_heads=10,
            attention_every_n_layers=3,
            attention_dropout=0.1,
            use_relative_attention=True,
            attention_bias_type="alibi",
            max_relative_distance=128,
            use_absolute_positions=True,
            dropout=0.1,
            use_cfc=False,
            use_mamba=True,
            ffn_expansion=4,
            tie_embeddings=True,
            embedding_init_std=0.02,
            output_logit_scale=None,
        ),
        "train": TrainConfig(
            batch_size=2,
            grad_accumulation_steps=8,
            learning_rate=2e-4,
            warmup_steps=2000,
            max_grad_norm=1.5,
            label_smoothing=0.1,
            save_every_n_epochs=5,
            keep_every_n_epochs=10,
        ),
        "data": DataConfig(
            max_pieces=None,
            tokenization_strategy="octuple",
            seed_length=256,
            continuation_length=768,
            max_sequence_length=1024,
        ),
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    key = name.strip().lower()
    if key not in SCALE_PRESETS:
        print("Unknown preset. Available presets:")
        print(", ".join(sorted(SCALE_PRESETS.keys())))
        raise ValueError(f"Unknown scale preset: {name}")

    preset = copy.deepcopy(SCALE_PRESETS[key])
    print(f"[{key}] {preset['description']}")
    return preset


def list_presets() -> None:
    print("Scale     Params    Description")
    print("------    ------    -----------")
    for name in ("small", "medium", "large"):
        if name not in SCALE_PRESETS:
            continue
        p = SCALE_PRESETS[name]
        params = str(p.get("params", "n/a"))
        desc = str(p.get("description", ""))
        print(f"{name:<8}  {params:<8}  {desc}")


def _estimate_real_mamba_params(model_cfg: ModelConfig) -> int:
    d = int(model_cfg.d_model)
    vocab = int(model_cfg.vocab_size)
    max_seq = int(model_cfg.max_sequence_length)
    n_layers = int(model_cfg.n_layers)
    d_state = int(model_cfg.d_state)
    d_conv = int(model_cfg.d_conv)
    expand = int(model_cfg.expand)
    attn_every = int(model_cfg.attention_every_n_layers)
    heads = int(model_cfg.num_attention_heads)

    total = vocab * d + max_seq * d

    for layer_idx in range(n_layers):
        d_inner = expand * d
        dt_rank = math.ceil(d / 16)
        mamba_core = (
            3 * d * d_inner
            + d_inner * d_conv
            + 2 * d_inner * dt_rank
            + 3 * d_inner * d_state
            + 3 * d_inner
        )
        mamba_block = mamba_core + 2 * d
        total += mamba_block

        if not bool(model_cfg.use_cfc):
            ff_h = int(d * model_cfg.ffn_expansion)
            ffn_block = 2 * d + (d * ff_h + ff_h) + (ff_h * d + d)
            total += ffn_block

        if (layer_idx + 1) % attn_every == 0:
            attn_block = 4 * d
            attn_block += 3 * d * d
            attn_block += d * d + d
            attn_ff_h = 2 * d
            attn_block += d * attn_ff_h + attn_ff_h
            attn_block += attn_ff_h * d + d
            if (
                bool(model_cfg.use_relative_attention)
                and str(model_cfg.attention_bias_type).lower() != "alibi"
            ):
                attn_block += (2 * int(model_cfg.max_relative_distance) + 1) * heads
            total += attn_block

    total += 2 * d
    return int(total)


def verify_preset_params(tolerance: float = 0.10) -> Dict[str, Dict[str, float | bool]]:
    from model.hybrid import PianoHybridModel

    print("Preset parameter verification")
    print("----------------------------")

    results: Dict[str, Dict[str, float | bool]] = {}
    for name in ("small", "medium", "large"):
        preset = SCALE_PRESETS[name]
        target = TARGET_PARAM_COUNTS[name]

        model_cfg = copy.deepcopy(preset["model"])
        model = PianoHybridModel(model_cfg)
        measured_runtime = sum(p.numel() for p in model.parameters())
        estimated_real_mamba = _estimate_real_mamba_params(model_cfg)

        delta_ratio = abs(estimated_real_mamba - target) / float(target)
        within = delta_ratio <= tolerance

        print(
            f"{name:<8} target={target / 1e6:>6.1f}M | "
            f"real_mamba_est={estimated_real_mamba / 1e6:>7.2f}M "
            f"({estimated_real_mamba:,}) | "
            f"runtime_measured={measured_runtime / 1e6:>7.2f}M "
            f"({measured_runtime:,}) | "
            f"delta={delta_ratio * 100:>5.1f}% | "
            f"{'OK' if within else 'OUT_OF_RANGE'}"
        )

        results[name] = {
            "target": float(target),
            "estimated_real_mamba": float(estimated_real_mamba),
            "runtime_measured": float(measured_runtime),
            "delta_ratio": float(delta_ratio),
            "within_tolerance": bool(within),
        }
        del model

    return results


if __name__ == "__main__":
    verify_preset_params()

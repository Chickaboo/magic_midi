from __future__ import annotations

import copy
import math
from typing import Any, Dict

from config import DataConfig, ModelConfig, TrainConfig
from model.factory import build_model


TARGET_PARAM_COUNTS: Dict[str, int] = {
    "small": 25_600_000,
    "medium": 60_000_000,
    "large": 100_000_000,
    "large_v2": 100_000_000,
}


# Runtime-measured reference from v2 Kaggle run (real mamba-ssm backend):
# d_model=320, n_layers=6 with CfC enabled -> ~25.6M parameters.
# v3 presets below are calibrated around this baseline.
SCALE_PRESETS: Dict[str, Dict[str, Any]] = {
    "small": {
        "description": "~25M params - verified from v2 training run",
        "params": "~25M",
        "model": ModelConfig(
            d_model=528,
            n_layers=6,
            d_state=16,
            d_conv=4,
            expand=2,
            cfc_units=528,
            cfc_backbone_units=264,
            cfc_backbone_layers=2,
            cfc_every_n_layers=1,
            num_attention_heads=8,
            attention_every_n_layers=3,
            attention_dropout=0.1,
            use_relative_attention=True,
            attention_bias_type="alibi",
            max_relative_distance=128,
            use_absolute_positions=True,
            dropout=0.1,
            use_cfc=True,
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
            weight_decay=0.01,
            warmup_steps=200,
            max_grad_norm=1.0,
            label_smoothing=0.1,
            save_every_n_epochs=10,
            keep_every_n_epochs=25,
            max_checkpoints=8,
        ),
        "data": DataConfig(
            max_pieces=None,
            tokenization_strategy="custom_delta",
            seed_length=256,
            continuation_length=768,
            max_sequence_length=1024,
            use_multi_dataset=True,
            dataset_weights={
                "maestro": 1.5,
                "giant_midi": 1.2,
                "aria_midi": 1.0,
                "adl_piano": 1.3,
            },
            min_duration_seconds=30.0,
            quality_filter_velocity=True,
        ),
    },
    "medium": {
        "description": "~60M params - serious Kaggle run",
        "params": "~60M",
        "model": ModelConfig(
            d_model=736,
            n_layers=8,
            d_state=16,
            d_conv=4,
            expand=2,
            cfc_units=736,
            cfc_backbone_units=368,
            cfc_backbone_layers=2,
            cfc_every_n_layers=1,
            num_attention_heads=8,
            attention_every_n_layers=3,
            attention_dropout=0.1,
            use_relative_attention=True,
            attention_bias_type="alibi",
            max_relative_distance=128,
            use_absolute_positions=True,
            dropout=0.1,
            use_cfc=True,
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
            weight_decay=0.01,
            warmup_steps=400,
            max_grad_norm=1.0,
            label_smoothing=0.1,
            save_every_n_epochs=10,
            keep_every_n_epochs=25,
            max_checkpoints=8,
        ),
        "data": DataConfig(
            max_pieces=None,
            tokenization_strategy="custom_delta",
            seed_length=256,
            continuation_length=768,
            max_sequence_length=1024,
            use_multi_dataset=True,
            dataset_weights={
                "maestro": 1.5,
                "giant_midi": 1.2,
                "aria_midi": 1.0,
                "adl_piano": 1.3,
            },
            min_duration_seconds=30.0,
            quality_filter_velocity=True,
        ),
    },
    "large": {
        "description": "~100M params - v3 target",
        "params": "~100M",
        "model": ModelConfig(
            # Target v3 shape. Validate/adjust on Kaggle using `calibrate_on_kaggle()`.
            d_model=840,
            n_layers=10,
            d_state=16,
            d_conv=4,
            expand=2,
            cfc_units=840,
            cfc_backbone_units=420,
            cfc_backbone_layers=2,
            cfc_every_n_layers=1,
            num_attention_heads=10,
            attention_every_n_layers=3,
            attention_dropout=0.1,
            use_relative_attention=True,
            attention_bias_type="alibi",
            max_relative_distance=256,
            use_absolute_positions=True,
            dropout=0.1,
            use_cfc=True,
            use_mamba=True,
            ffn_expansion=4,
            tie_embeddings=True,
            embedding_init_std=0.02,
            output_logit_scale=None,
        ),
        "train": TrainConfig(
            batch_size=4,
            grad_accumulation_steps=8,
            learning_rate=2e-4,
            weight_decay=0.01,
            max_epochs=5000,
            warmup_steps=500,
            max_grad_norm=1.0,
            label_smoothing=0.1,
            save_every_n_epochs=10,
            keep_every_n_epochs=25,
            max_checkpoints=8,
            use_wandb=False,
            seed=42,
            device="auto",
        ),
        "data": DataConfig(
            max_pieces=None,
            tokenization_strategy="custom_delta",
            seed_length=256,
            continuation_length=768,
            max_sequence_length=1024,
            use_multi_dataset=True,
            dataset_weights={
                "maestro": 1.5,
                "giant_midi": 1.2,
                "aria_midi": 1.0,
                "adl_piano": 1.3,
            },
            min_duration_seconds=30.0,
            quality_filter_velocity=True,
        ),
    },
    "large_v2": {
        "description": "~100M params - official v2 dual-stream architecture",
        "params": "~100M",
        "model": ModelConfig(
            vocab_size=374,
            d_model=1152,
            n_layers=17,
            d_state=16,
            d_conv=4,
            expand=2,
            cfc_units=576,
            cfc_backbone_units=288,
            cfc_backbone_layers=3,
            cfc_every_n_layers=1,
            num_attention_heads=18,
            attention_every_n_layers=3,
            attention_dropout=0.1,
            use_relative_attention=True,
            attention_bias_type="alibi",
            max_relative_distance=256,
            use_absolute_positions=False,
            dropout=0.1,
            use_cfc=True,
            use_mamba=True,
            ffn_expansion=4,
            tie_embeddings=True,
            embedding_init_std=0.02,
            output_logit_scale=None,
            max_sequence_length=1024,
            use_v2_architecture=True,
            use_continuous_time_encoding=True,
            max_time_seconds=600.0,
            stream_dim=576,
            harmonic_ratio=0.6,
            cross_stream_every_n_layers=3,
            cross_attention_every_n=3,
            tokens_per_phrase=32,
            phrase_dim=1152,
            memory_size=128,
            theme_memory_heads=4,
        ),
        "train": TrainConfig(
            batch_size=8,
            grad_accumulation_steps=8,
            learning_rate=1e-4,
            lr_schedule="cosine",
            min_lr_ratio=0.1,
            weight_decay=0.01,
            warmup_steps=500,
            max_grad_norm=1.0,
            label_smoothing=0.1,
            save_every_n_epochs=10,
            keep_every_n_epochs=25,
            max_checkpoints=8,
            max_epochs=5000,
            theme_memory_reset_on_piece=True,
            use_amp=True,
            val_generation_check=True,
            val_generation_steps=20,
            use_wandb=False,
            seed=42,
            device="auto",
        ),
        "data": DataConfig(
            max_pieces=None,
            tokenization_strategy="custom_delta",
            seed_length=256,
            continuation_length=768,
            max_sequence_length=1024,
            use_multi_dataset=True,
            dataset_weights={
                "maestro": 1.5,
                "giant_midi": 1.2,
                "aria_midi": 1.0,
                "adl_piano": 1.3,
            },
            dataset_profiles={
                "maestro": {
                    "min_duration_seconds": 30.0,
                    "filter_velocity": True,
                },
                "giant_midi": {
                    "min_duration_seconds": 30.0,
                    "filter_velocity": True,
                },
                "aria_midi": {
                    "min_duration_seconds": 20.0,
                    "filter_velocity": True,
                },
                "adl_piano": {
                    "min_duration_seconds": 15.0,
                    "filter_velocity": False,
                },
            },
            min_duration_seconds=30.0,
            quality_filter_velocity=True,
            min_note_count=100,
            min_distinct_pitches=12,
            piano_dominance_threshold=0.70,
            use_continuous_time=True,
            time_feature_fallback_step_seconds=0.5,
        ),
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """Return a deep copy of a named scale preset."""

    key = name.strip().lower()
    if key not in SCALE_PRESETS:
        raise ValueError(
            f"Unknown scale preset '{name}'. Use one of: {', '.join(sorted(SCALE_PRESETS))}."
        )
    return copy.deepcopy(SCALE_PRESETS[key])


def list_presets() -> None:
    """Print available scale presets."""

    print("Scale     Params    Description")
    print("------    ------    -----------")
    for name in sorted(SCALE_PRESETS):
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

        if bool(model_cfg.use_cfc):
            cfc_units = int(model_cfg.cfc_units)
            bb_units = int(model_cfg.cfc_backbone_units)
            bb_layers = int(model_cfg.cfc_backbone_layers)
            # Approximation for ncps.torch.CfC(mode='pure') + residual projections.
            cfc_proj = (d * cfc_units if d != cfc_units else 0) + (cfc_units * d + d)
            cfc_cell = (cfc_units * cfc_units * 2) + (2 * cfc_units)
            cfc_backbone = bb_layers * (
                (cfc_units * bb_units + bb_units) + (bb_units * cfc_units + cfc_units)
            )
            total += cfc_proj + cfc_cell + cfc_backbone + (2 * d)
        else:
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
    """Build each preset and compare measured parameters against target counts."""

    print("Preset parameter verification")
    print("----------------------------")

    results: Dict[str, Dict[str, float | bool]] = {}
    for name in ("small", "medium", "large", "large_v2"):
        preset = SCALE_PRESETS[name]
        target = TARGET_PARAM_COUNTS[name]

        model_cfg = copy.deepcopy(preset["model"])
        model = build_model(model_cfg)
        measured_runtime = sum(p.numel() for p in model.parameters())
        estimated_real_mamba = _estimate_real_mamba_params(model_cfg)

        using_mamba_fallback = any(
            bool(getattr(module, "using_fallback", False))
            for module in model.modules()
            if hasattr(module, "using_fallback")
        )

        if bool(getattr(model_cfg, "use_v2_architecture", False)):
            reference_count = measured_runtime
            reference_label = "runtime_measured"
            backend_label = "dual-stream runtime"
        elif bool(model_cfg.use_mamba) and using_mamba_fallback:
            reference_count = estimated_real_mamba
            reference_label = "real_mamba_est"
            backend_label = "GRU fallback"
        else:
            reference_count = measured_runtime
            reference_label = "runtime_measured"
            backend_label = "native runtime"

        runtime_delta_ratio = abs(measured_runtime - target) / float(target)
        reference_delta_ratio = abs(reference_count - target) / float(target)
        within = reference_delta_ratio <= tolerance

        print(
            f"{name:<8} target={target / 1e6:>6.1f}M | "
            f"real_mamba_est={estimated_real_mamba / 1e6:>7.2f}M "
            f"({estimated_real_mamba:,}) | "
            f"runtime_measured={measured_runtime / 1e6:>7.2f}M "
            f"({measured_runtime:,}) | "
            f"backend={backend_label} | "
            f"{reference_label}_delta={reference_delta_ratio * 100:>5.1f}% | "
            f"{'OK' if within else 'OUT_OF_RANGE'}"
        )

        results[name] = {
            "target": float(target),
            "estimated_real_mamba": float(estimated_real_mamba),
            "runtime_measured": float(measured_runtime),
            "delta_ratio": float(runtime_delta_ratio),
            "reference_delta_ratio": float(reference_delta_ratio),
            "reference_is_real_mamba_estimate": bool(
                bool(model_cfg.use_mamba) and using_mamba_fallback
            ),
            "within_tolerance": bool(within),
        }
        del model

    return results


if __name__ == "__main__":
    verify_preset_params()

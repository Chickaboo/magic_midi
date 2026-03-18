from __future__ import annotations

import copy
from typing import Any, Dict

from config import DataConfig, ModelConfig, TrainConfig


TARGET_PARAM_COUNTS: Dict[str, int] = {
    "nano": 3_000_000,
    "micro": 8_000_000,
    "small": 22_000_000,
    "medium": 60_000_000,
}


SCALE_PRESETS: Dict[str, Dict[str, Any]] = {
    "nano": {
        "description": "~3M params - 30min Colab session, pipeline validation",
        "params": "~3M",
        "model": ModelConfig(
            d_model=192,  # local calibration: 3.46M (CPU fallback runtime)
            n_layers=4,
            d_state=16,
            d_conv=4,
            expand=2,
            cfc_units=192,
            cfc_backbone_units=96,
            cfc_backbone_layers=2,
            cfc_every_n_layers=2,
            num_attention_heads=4,
            attention_every_n_layers=2,
            attention_dropout=0.1,
            use_relative_attention=True,
            max_relative_distance=128,
            dropout=0.1,
            # Calibrate with session.calibrate_preset("nano") on Colab runtime.
        ),
        "train": TrainConfig(
            batch_size=4,
            grad_accumulation_steps=4,
            learning_rate=5e-4,
            warmup_steps=200,
            save_every_n_epochs=1,
            keep_every_n_epochs=10,
        ),
        "data": DataConfig(
            max_pieces=400,
            seed_length=128,
            continuation_length=256,
            max_sequence_length=512,
        ),
    },
    "micro": {
        "description": "~8M params - meaningful learning, ~2 epochs per 30min session",
        "params": "~8M",
        "model": ModelConfig(
            d_model=320,  # local calibration: 9.50M (CPU fallback runtime)
            n_layers=4,
            d_state=16,
            d_conv=4,
            expand=2,
            cfc_units=320,
            cfc_backbone_units=160,
            cfc_backbone_layers=2,
            cfc_every_n_layers=1,
            num_attention_heads=8,
            attention_every_n_layers=2,
            attention_dropout=0.1,
            use_relative_attention=True,
            max_relative_distance=128,
            dropout=0.1,
            # Calibrate with session.calibrate_preset("micro") on Colab runtime.
        ),
        "train": TrainConfig(
            batch_size=4,
            grad_accumulation_steps=4,
            learning_rate=3e-4,
            warmup_steps=300,
            save_every_n_epochs=1,
            keep_every_n_epochs=10,
        ),
        "data": DataConfig(
            max_pieces=600,
            seed_length=128,
            continuation_length=512,
            max_sequence_length=768,
        ),
    },
    "small": {
        "description": "~22M params - full architecture, ~1 epoch per 30min session",
        "params": "~22M",
        "model": ModelConfig(
            d_model=512,  # local calibration: 32.17M (CPU fallback runtime)
            n_layers=6,
            d_state=16,
            d_conv=4,
            expand=2,
            cfc_units=512,
            cfc_backbone_units=256,
            cfc_backbone_layers=2,
            cfc_every_n_layers=1,
            num_attention_heads=8,
            attention_every_n_layers=3,
            attention_dropout=0.1,
            use_relative_attention=True,
            max_relative_distance=128,
            dropout=0.1,
            # Calibrate with session.calibrate_preset("small") on Colab runtime.
        ),
        "train": TrainConfig(
            batch_size=8,
            grad_accumulation_steps=2,
            learning_rate=3e-4,
            warmup_steps=500,
            save_every_n_epochs=1,
            keep_every_n_epochs=10,
        ),
        "data": DataConfig(
            max_pieces=None,
            seed_length=256,
            continuation_length=768,
            max_sequence_length=1024,
        ),
    },
    "medium": {
        "description": "~60M params - serious model, needs Colab Pro or Kaggle",
        "params": "~60M",
        "model": ModelConfig(
            d_model=640,  # local calibration: 70.90M (CPU fallback runtime)
            n_layers=8,
            d_state=32,
            d_conv=4,
            expand=2,
            cfc_units=640,
            cfc_backbone_units=320,
            cfc_backbone_layers=3,
            cfc_every_n_layers=1,
            num_attention_heads=10,
            attention_every_n_layers=2,
            attention_dropout=0.1,
            use_relative_attention=True,
            max_relative_distance=128,
            dropout=0.1,
            # Calibrate with session.calibrate_preset("medium") on Colab runtime.
        ),
        "train": TrainConfig(
            batch_size=4,
            grad_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_steps=1000,
            save_every_n_epochs=1,
            keep_every_n_epochs=10,
        ),
        "data": DataConfig(
            max_pieces=None,
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
    for name in ("nano", "micro", "small", "medium"):
        if name not in SCALE_PRESETS:
            continue
        p = SCALE_PRESETS[name]
        params = str(p.get("params", "n/a"))
        desc = str(p.get("description", ""))
        print(f"{name:<8}  {params:<8}  {desc}")


def verify_preset_params(tolerance: float = 0.15) -> Dict[str, Dict[str, float | bool]]:
    from model.hybrid import PianoHybridModel

    print("Preset parameter verification")
    print("----------------------------")

    results: Dict[str, Dict[str, float | bool]] = {}
    for name in ("nano", "micro", "small", "medium"):
        preset = SCALE_PRESETS[name]
        target = TARGET_PARAM_COUNTS[name]

        model_cfg = copy.deepcopy(preset["model"])
        model = PianoHybridModel(model_cfg)
        actual = sum(p.numel() for p in model.parameters())

        delta_ratio = abs(actual - target) / float(target)
        within = delta_ratio <= tolerance

        print(
            f"{name:<8} target={target / 1e6:>5.1f}M | "
            f"actual={actual / 1e6:>6.2f}M ({actual:,}) | "
            f"delta={delta_ratio * 100:>5.1f}% | "
            f"{'OK' if within else 'OUT_OF_RANGE'}"
        )

        results[name] = {
            "target": float(target),
            "actual": float(actual),
            "delta_ratio": float(delta_ratio),
            "within_tolerance": bool(within),
        }
        del model

    return results


if __name__ == "__main__":
    verify_preset_params()

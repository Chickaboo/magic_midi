from __future__ import annotations

from typing import Any

from config import ModelConfig
from model.hybrid import PianoHybridModel
from model.hybrid_v2 import IttyBittyPianoV2
from model.variant_a import VariantAConfig, VariantAModel
from model.variant_b import VariantBConfig, VariantBModel
from model.variant_c import VariantCConfig, VariantCModel
from model.variant_d import VariantDConfig, VariantDModel
from model.variant_e import VariantEConfig, VariantEModel
from model.variant_f import VariantFConfig, VariantFModel


def build_model(config: ModelConfig):
    """Build v1 or v2 model based on config flag."""

    if bool(getattr(config, "use_v2_architecture", False)):
        return IttyBittyPianoV2(config)
    return PianoHybridModel(config)


def build_named_model(name: str, config: Any):
    """Build one concrete model family by stable name."""

    key = str(name).strip().lower()
    aliases = {
        "variant_a": "variant_a",
        "a": "variant_a",
        "variant_b": "variant_b",
        "b": "variant_b",
        "variant_c": "variant_c",
        "c": "variant_c",
        "variant_d": "variant_d",
        "d": "variant_d",
        "variant_e": "variant_e",
        "e": "variant_e",
        "variant_f": "variant_f",
        "f": "variant_f",
        "hybrid": "hybrid",
        "hybrid_v2": "hybrid_v2",
        "v2": "hybrid_v2",
    }
    key = aliases.get(key, key)

    if key == "variant_a":
        cfg = config if isinstance(config, VariantAConfig) else VariantAConfig(**dict(config))
        return VariantAModel(cfg)
    if key == "variant_b":
        cfg = config if isinstance(config, VariantBConfig) else VariantBConfig(**dict(config))
        return VariantBModel(cfg)
    if key == "variant_c":
        cfg = config if isinstance(config, VariantCConfig) else VariantCConfig(**dict(config))
        return VariantCModel(cfg)
    if key == "variant_d":
        cfg = config if isinstance(config, VariantDConfig) else VariantDConfig(**dict(config))
        return VariantDModel(cfg)
    if key == "variant_e":
        cfg = config if isinstance(config, VariantEConfig) else VariantEConfig(**dict(config))
        return VariantEModel(cfg)
    if key == "variant_f":
        cfg = config if isinstance(config, VariantFConfig) else VariantFConfig(**dict(config))
        return VariantFModel(cfg)
    if key in {"hybrid", "hybrid_v2"}:
        cfg = config if isinstance(config, ModelConfig) else ModelConfig(**dict(config))
        if key == "hybrid_v2":
            cfg.use_v2_architecture = True
        elif key == "hybrid":
            cfg.use_v2_architecture = False
        return build_model(cfg)

    raise ValueError(f"Unsupported model family '{name}'.")

from __future__ import annotations

from config import ModelConfig
from model.hybrid import PianoHybridModel
from model.hybrid_v2 import IttyBittyPianoV2


def build_model(config: ModelConfig):
    """Build v1 or v2 model based on config flag."""

    if bool(getattr(config, "use_v2_architecture", False)):
        return IttyBittyPianoV2(config)
    return PianoHybridModel(config)

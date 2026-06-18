from __future__ import annotations

from typing import Any

__all__ = [
    "CustomDeltaTokenizer",
    "create_tokenizer",
    "load_tokenizer",
    "PianoDataset",
    "create_dataloaders",
    "MultiDatasetPreprocessor",
    "create_seed_pairs",
    "preprocess_maestro",
]


def __getattr__(name: str) -> Any:
    if name in {"CustomDeltaTokenizer", "create_tokenizer", "load_tokenizer"}:
        from .tokenizer import CustomDeltaTokenizer, create_tokenizer, load_tokenizer

        values = {
            "CustomDeltaTokenizer": CustomDeltaTokenizer,
            "create_tokenizer": create_tokenizer,
            "load_tokenizer": load_tokenizer,
        }
        return values[name]

    if name in {"PianoDataset", "create_dataloaders"}:
        from .dataset import PianoDataset, create_dataloaders

        values = {
            "PianoDataset": PianoDataset,
            "create_dataloaders": create_dataloaders,
        }
        return values[name]

    if name in {"MultiDatasetPreprocessor", "create_seed_pairs", "preprocess_maestro"}:
        from .preprocess import MultiDatasetPreprocessor, create_seed_pairs, preprocess_maestro

        values = {
            "MultiDatasetPreprocessor": MultiDatasetPreprocessor,
            "create_seed_pairs": create_seed_pairs,
            "preprocess_maestro": preprocess_maestro,
        }
        return values[name]

    raise AttributeError(f"module 'data' has no attribute {name!r}")

"""
Quick generation tool. Run locally without Colab.
Usage: python tools/generate_sample.py --checkpoint path/to/checkpoint.safetensors
                                       --seed path/to/seed.mid
                                       --output output.mid
                                       --temperature 0.9
                                       --scale small
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file as safetensors_load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.tokenizer import PianoTokenizer
from generation.generate import GenerationConfig, generate_continuation
from model.factory import build_model
from config import DataConfig, ModelConfig
from scale_config import get_preset
from utils.config_compat import normalize_model_config_payload


def _load_tokenizer(tokenizer_path: Path) -> PianoTokenizer:
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    return PianoTokenizer.load(str(tokenizer_path))


def _resolve_state_path(checkpoint_path: Path) -> Optional[Path]:
    candidates = []
    if checkpoint_path.suffix == ".safetensors":
        if checkpoint_path.name.endswith("_model.safetensors"):
            candidates.append(
                checkpoint_path.with_name(
                    checkpoint_path.name.replace("_model.safetensors", "_state.pt")
                )
            )
        candidates.append(checkpoint_path.with_suffix(".pt"))
        candidates.append(checkpoint_path.parent / "latest_state.pt")
        candidates.append(checkpoint_path.parent / "best_state.pt")
    else:
        candidates.append(checkpoint_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    state_path = _resolve_state_path(checkpoint_path)
    if state_path is None or state_path.suffix != ".pt":
        return {}

    try:
        return torch.load(state_path, map_location="cpu")
    except Exception:
        return {}


def _resolve_configs(
    args_scale: str,
    checkpoint_metadata: Dict[str, Any],
) -> tuple[ModelConfig, DataConfig]:
    preset = get_preset(args_scale)
    model_cfg = ModelConfig(**dict(getattr(preset["model"], "__dict__", {})))
    data_cfg = DataConfig(**dict(getattr(preset["data"], "__dict__", {})))

    if isinstance(checkpoint_metadata.get("model_config"), dict):
        try:
            model_cfg = ModelConfig(
                **normalize_model_config_payload(
                    dict(checkpoint_metadata["model_config"])
                )
            )
        except Exception:
            pass

    if isinstance(checkpoint_metadata.get("data_config"), dict):
        try:
            data_cfg = DataConfig(**checkpoint_metadata["data_config"])
        except Exception:
            pass

    return model_cfg, data_cfg


def main() -> None:
    """CLI entrypoint for one-off sample generation."""

    parser = argparse.ArgumentParser(
        description="Generate continuation from checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output", default="generated.mid")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--repetition_window", type=int, default=64)
    parser.add_argument("--min_tokens_to_keep", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--scale", default="small")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    seed_path = Path(args.seed)
    output_path = Path(args.output)
    tokenizer_path = Path(args.tokenizer)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed MIDI not found: {seed_path}")

    checkpoint_metadata = _load_checkpoint_metadata(checkpoint_path)
    model_cfg, data_cfg = _resolve_configs(args.scale, checkpoint_metadata)
    if len(model_cfg.__dict__):
        pass

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading model from {checkpoint_path}...")
    # Prefer checkpoint config when available, but allow longer local generation.
    model = build_model(model_cfg)
    state = safetensors_load_file(str(checkpoint_path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys while loading checkpoint: {len(unexpected)}")

    model.eval()
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total:,} parameters")

    if args.tokenizer is None:
        tokenizer_candidates = []
        if isinstance(checkpoint_metadata.get("data_config"), dict):
            tok = checkpoint_metadata["data_config"].get("tokenizer_path")
            if tok:
                tokenizer_candidates.append(Path(str(tok)))
        tokenizer_candidates.extend(
            [
                checkpoint_path.parent / "tokenizer.json",
                checkpoint_path.parent.parent / "tokenizer" / "tokenizer.json",
                Path("tokenizer.json"),
            ]
        )
        for candidate in tokenizer_candidates:
            if candidate.exists():
                tokenizer_path = candidate
                break
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = _load_tokenizer(tokenizer_path)

    gen_config = GenerationConfig(
        max_new_tokens=max(args.max_new_tokens, 2048),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        min_tokens_to_keep=args.min_tokens_to_keep,
        num_samples=1,
    )

    print(f"Generating continuation from {seed_path}...")
    outputs = generate_continuation(
        model=model,
        tokenizer=tokenizer,
        seed_midi_path=seed_path,
        output_path=output_path,
        config=data_cfg,
        generation_config=gen_config,
    )
    print(f"Saved to {outputs[0]}")


if __name__ == "__main__":
    main()

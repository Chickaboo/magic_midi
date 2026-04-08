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
from pathlib import Path
from typing import Any, Dict, Optional

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DataConfig
from generation.generate import GenerationConfig, generate_continuation
from scale_config import get_preset
from utils import checkpoint_loading as ckpt_utils


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


def _resolve_data_config(
    args_scale: str,
    checkpoint_metadata: Dict[str, Any],
) -> DataConfig:
    preset = get_preset(args_scale)
    data_cfg = DataConfig(**dict(getattr(preset["data"], "__dict__", {})))

    if isinstance(checkpoint_metadata.get("data_config"), dict):
        try:
            data_cfg = DataConfig(**checkpoint_metadata["data_config"])
        except Exception:
            pass

    return data_cfg


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
    tokenizer_path = Path(args.tokenizer) if args.tokenizer is not None else None

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed MIDI not found: {seed_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading model from {checkpoint_path}...")
    bundle = ckpt_utils.load_model_from_checkpoint(
        model_path=checkpoint_path,
        sidecar_path=_resolve_state_path(checkpoint_path),
        device=device,
        strict=True,
    )
    model = bundle.model
    checkpoint_metadata = dict(bundle.checkpoint_metadata)
    data_cfg = _resolve_data_config(args.scale, checkpoint_metadata)

    total = sum(p.numel() for p in model.parameters())
    print(
        f"Model loaded: {total:,} parameters | class={bundle.model_class} "
        f"| missing={bundle.missing_keys} unexpected={bundle.unexpected_keys}"
    )

    if args.tokenizer is None:
        tokenizer_candidates = [
            checkpoint_path.parent / "custom_tokenizer.json",
            checkpoint_path.parent / "tokenizer.json",
            checkpoint_path.parent.parent / "tokenizer" / "custom_tokenizer.json",
            checkpoint_path.parent.parent / "tokenizer" / "tokenizer.json",
            Path("app/tokenizer/custom_tokenizer.json"),
            Path("app/tokenizer/tokenizer.json"),
            Path("tokenizer.json"),
        ]
        tokenizer, tokenizer_meta = ckpt_utils.load_tokenizer_for_checkpoint(
            checkpoint_metadata,
            search_paths=tokenizer_candidates,
        )
        tokenizer_path = Path(str(tokenizer_meta.get("tokenizer_path", "")))
    else:
        tokenizer_data_cfg = ckpt_utils.extract_data_config(checkpoint_metadata)
        tokenizer_path = Path(args.tokenizer)
        tokenizer, tokenizer_meta = ckpt_utils.load_tokenizer(
            tokenizer_path,
            tokenizer_data_cfg,
        )

    model_vocab = int(bundle.model_config.get("vocab_size", tokenizer.vocab_size) or 0)
    if model_vocab and int(tokenizer.vocab_size) != model_vocab:
        raise RuntimeError(
            "Tokenizer/model vocab mismatch: "
            f"tokenizer_vocab={tokenizer.vocab_size} model_vocab={model_vocab}. "
            "Use tokenizer from the same checkpoint bundle."
        )

    data_cfg.vocab_size = int(tokenizer.vocab_size)
    data_cfg.tokenizer_path = str(tokenizer_path)

    print(f"Loading tokenizer from {tokenizer_path}...")
    print(
        f"Tokenizer kind: {tokenizer_meta.get('tokenizer_kind', 'unknown')} "
        f"| vocab={getattr(tokenizer, 'vocab_size', 'n/a')}"
    )

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

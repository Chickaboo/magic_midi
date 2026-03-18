"""
Quick generation tool. Run locally without Colab.
Usage: python tools/generate_sample.py --checkpoint path/to/checkpoint.safetensors
                                       --seed path/to/seed.mid
                                       --output output.mid
                                       --temperature 0.9
                                       --scale nano
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as safetensors_load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.tokenizer import PianoTokenizer
from generation.generate import GenerationConfig, generate_continuation
from model.hybrid import PianoHybridModel
from scale_config import get_preset


def _load_tokenizer(tokenizer_path: Path) -> PianoTokenizer:
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    return PianoTokenizer.load(str(tokenizer_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate continuation from checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output", default="generated.mid")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--scale", default="nano")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    seed_path = Path(args.seed)
    output_path = Path(args.output)
    tokenizer_path = Path(args.tokenizer)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed MIDI not found: {seed_path}")

    device = torch.device("cpu")
    preset = get_preset(args.scale)

    print(f"Loading model from {checkpoint_path}...")
    model = PianoHybridModel(preset["model"])
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

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = _load_tokenizer(tokenizer_path)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_samples=1,
    )

    print(f"Generating continuation from {seed_path}...")
    outputs = generate_continuation(
        model=model,
        tokenizer=tokenizer,
        seed_midi_path=seed_path,
        output_path=output_path,
        config=preset["data"],
        generation_config=gen_config,
    )
    print(f"Saved to {outputs[0]}")


if __name__ == "__main__":
    main()

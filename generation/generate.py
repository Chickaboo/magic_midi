from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from config import DataConfig
from data.tokenizer import PianoTokenizer
from utils.midi_utils import midi_duration


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    repetition_window: int = 64
    min_tokens_to_keep: int = 3
    num_samples: int = 1


def generate_continuation(
    model,
    tokenizer: PianoTokenizer,
    seed_midi_path: str | Path,
    output_path: str | Path,
    config: DataConfig,
    generation_config: GenerationConfig,
) -> List[Path]:
    seed_midi_path = Path(seed_midi_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokens = tokenizer.encode(seed_midi_path)
    if len(tokens) < config.seed_length:
        print(
            f"Seed file {seed_midi_path} has only {len(tokens)} tokens; "
            f"using the available seed length instead of configured {config.seed_length}."
        )

    seed_tokens = tokens[: min(len(tokens), config.seed_length)]
    out_paths: List[Path] = []

    for i in range(generation_config.num_samples):
        generated_tokens = model.generate(
            seed_tokens=seed_tokens,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            repetition_penalty=generation_config.repetition_penalty,
            repetition_window=generation_config.repetition_window,
            min_tokens_to_keep=generation_config.min_tokens_to_keep,
        )

        out_file = output_path
        if generation_config.num_samples > 1:
            out_file = output_path.with_name(
                f"{output_path.stem}_sample{i + 1}{output_path.suffix}"
            )

        tokenizer.decode(generated_tokens, out_file)
        out_paths.append(out_file)

        seed_dur = midi_duration(seed_midi_path)
        gen_dur = max(0.0, midi_duration(out_file) - seed_dur)
        total_dur = midi_duration(out_file)
        print(
            f"Generated {out_file.name}: seed={seed_dur:.2f}s, "
            f"generated={gen_dur:.2f}s, total={total_dur:.2f}s"
        )

    return out_paths


def batch_generate(
    model,
    tokenizer: PianoTokenizer,
    seed_dir: str | Path,
    output_dir: str | Path,
    config: DataConfig,
    generation_config: GenerationConfig,
) -> Dict[str, List[Path]]:
    seed_dir = Path(seed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(list(seed_dir.glob("*.mid")) + list(seed_dir.glob("*.midi")))
    if not midi_files:
        raise RuntimeError(f"No MIDI files found in seed_dir={seed_dir.resolve()}")

    outputs: Dict[str, List[Path]] = {}
    for seed_path in midi_files:
        out_name = f"{seed_path.stem}_continuation.mid"
        out_path = output_dir / out_name
        generated = generate_continuation(
            model=model,
            tokenizer=tokenizer,
            seed_midi_path=seed_path,
            output_path=out_path,
            config=config,
            generation_config=generation_config,
        )
        outputs[str(seed_path)] = generated

    return outputs

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

from config import DataConfig
from data.tokenizer import PianoTokenizer
from utils.midi_utils import midi_duration


@dataclass
class GenerationConfig:
    """Sampling and output controls used by generation helpers."""

    max_new_tokens: int = 1024
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    repetition_window: int = 64
    min_tokens_to_keep: int = 3
    num_samples: int = 1


def generate_continuation(
    model: Any,
    tokenizer: PianoTokenizer,
    seed_midi_path: str | Path,
    output_path: str | Path,
    config: DataConfig,
    generation_config: GenerationConfig,
) -> List[Path]:
    """Generate one or more continuations from a seed MIDI file."""

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
    seed_onset_times = [
        float(i) * float(max(1e-4, config.time_feature_fallback_step_seconds))
        for i in range(len(seed_tokens))
    ]
    out_paths: List[Path] = []

    supports_time = bool(
        getattr(getattr(model, "config", None), "use_v2_architecture", False)
    )

    bind_tokenizer = getattr(model, "bind_tokenizer", None)
    if callable(bind_tokenizer):
        try:
            bind_tokenizer(tokenizer)
        except Exception:
            pass

    if supports_time:
        try:
            token_ids, onset_times, _durations = tokenizer.encode_with_time_features(
                seed_midi_path
            )
            seed_tokens = token_ids[: min(len(token_ids), config.seed_length)]
            seed_onset_times = onset_times[: len(seed_tokens)]
        except Exception:
            pass

    token_id_to_events = getattr(tokenizer, "decode_token_id_events", None)
    if not callable(token_id_to_events):
        token_id_to_events = None

    for i in range(generation_config.num_samples):
        if supports_time:
            generated_tokens = model.generate(
                seed_tokens=seed_tokens,
                seed_onset_times=torch.tensor(seed_onset_times, dtype=torch.float32),
                max_new_tokens=generation_config.max_new_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                repetition_penalty=generation_config.repetition_penalty,
                repetition_window=generation_config.repetition_window,
                min_tokens_to_keep=generation_config.min_tokens_to_keep,
                step_seconds=float(
                    max(1e-4, config.time_feature_fallback_step_seconds)
                ),
                token_id_to_events=token_id_to_events,
            )
        else:
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
    model: Any,
    tokenizer: PianoTokenizer,
    seed_dir: str | Path,
    output_dir: str | Path,
    config: DataConfig,
    generation_config: GenerationConfig,
) -> Dict[str, List[Path]]:
    """Generate continuations for every MIDI seed file in a directory."""

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

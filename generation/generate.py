from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Sequence

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
    max_consecutive_zero_deltas: int = 8
    save_continuation_only: bool = True


def _unwrap_parallel_model(model: Any) -> Any:
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def _build_seed_batch(
    base_model: Any,
    seed_tokens: Sequence[int] | torch.Tensor,
    seed_onset_times: Sequence[float] | torch.Tensor | None,
    batch_size: int,
    device: torch.device,
    step_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed = base_model._to_seed_tensor(seed_tokens, device=device)
    if seed.shape[0] != 1:
        if seed.shape[0] != batch_size:
            raise ValueError(
                f"seed_tokens batch size must be 1 or {batch_size}, got {int(seed.shape[0])}"
            )
        tokens = seed
    else:
        tokens = seed.repeat(batch_size, 1)

    if seed_onset_times is None:
        onsets = (
            torch.arange(tokens.shape[1], device=device, dtype=torch.float32)
            * float(max(1e-4, step_seconds))
        ).unsqueeze(0)
        onsets = onsets.repeat(batch_size, 1)
    else:
        if isinstance(seed_onset_times, torch.Tensor):
            on = seed_onset_times
            if on.ndim == 1:
                on = on.unsqueeze(0)
            onsets = on.to(device=device, dtype=torch.float32)
        else:
            onsets = torch.tensor(
                [float(v) for v in seed_onset_times],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
        if onsets.shape[0] == 1 and batch_size > 1:
            onsets = onsets.repeat(batch_size, 1)

    if onsets.shape != tokens.shape:
        raise ValueError("seed_onset_times shape must match seed token shape")

    return tokens, onsets


@torch.inference_mode()
def _generate_batched_sequences(
    model: Any,
    base_model: Any,
    seed_tokens: Sequence[int] | torch.Tensor,
    seed_onset_times: Sequence[float] | torch.Tensor | None,
    generation_config: GenerationConfig,
    *,
    step_seconds: float,
    token_id_to_events: Any,
) -> List[List[int]]:
    batch_size = max(1, int(generation_config.num_samples))
    device = next(base_model.parameters()).device

    forward_model = model
    if (
        batch_size > 1
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
        and not isinstance(forward_model, torch.nn.DataParallel)
    ):
        forward_model = torch.nn.DataParallel(base_model)

    tokens, onsets = _build_seed_batch(
        base_model=base_model,
        seed_tokens=seed_tokens,
        seed_onset_times=seed_onset_times,
        batch_size=batch_size,
        device=device,
        step_seconds=step_seconds,
    )

    final_top1_probs: List[float] = []
    raw_top1_probs: List[float] = []
    candidate_counts: List[int] = []
    zero_delta_streaks = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    max_zero_delta = max(0, int(generation_config.max_consecutive_zero_deltas))

    for _ in range(int(generation_config.max_new_tokens)):
        context_tokens = tokens[:, -base_model.max_sequence_length :]
        context_onsets = onsets[:, -base_model.max_sequence_length :]
        context_offset = max(0, int(tokens.shape[1] - context_tokens.shape[1]))

        logits, _ = forward_model(
            token_ids=context_tokens,
            onset_times=context_onsets,
            memory=None,
            return_memory=True,
            position_offset=context_offset,
        )

        next_slot = base_model._triplet_slot(int(tokens.shape[1]))
        masked_logits = base_model._mask_logits_to_triplet_slot(
            logits[:, -1, :],
            next_slot,
        )
        if next_slot == 0 and max_zero_delta > 0 and masked_logits.shape[-1] > 1:
            over_limit = zero_delta_streaks >= max_zero_delta
            if bool(over_limit.any()):
                valid_non_zero = torch.isfinite(masked_logits[:, 1:]).any(dim=-1)
                rows = torch.where(over_limit & valid_non_zero)[0]
                if rows.numel() > 0:
                    masked_logits = masked_logits.clone()
                    masked_logits[rows, 0] = float("-inf")
        next_token, diagnostics = sample_next_token(
            logits=masked_logits,
            context_tokens=context_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            repetition_penalty=generation_config.repetition_penalty,
            recent_window=generation_config.repetition_window,
            min_tokens_to_keep=max(4, generation_config.min_tokens_to_keep),
            top1_cap=0.95,
        )

        final_top1_probs.extend(
            [float(v) for v in diagnostics.final_top1_prob.detach().cpu().tolist()]
        )
        raw_top1_probs.extend(
            [float(v) for v in diagnostics.raw_top1_prob.detach().cpu().tolist()]
        )
        candidate_counts.extend(
            [int(v) for v in diagnostics.candidate_count.detach().cpu().tolist()]
        )

        tokens = torch.cat([tokens, next_token], dim=1)
        slot = base_model._triplet_slot(int(tokens.shape[1] - 1))
        if slot == 0:
            sampled = next_token.view(-1)
            zero_delta_streaks = torch.where(
                sampled == 0,
                zero_delta_streaks + 1,
                torch.zeros_like(zero_delta_streaks),
            )
            delta_values = [
                float(
                    max(
                        1e-4,
                        base_model._delta_from_token_events(
                            token_id=int(token_id),
                            token_id_to_events=token_id_to_events,
                            default_step=step_seconds,
                        ),
                    )
                )
                for token_id in next_token.view(-1).tolist()
            ]
            delta_tensor = torch.tensor(
                delta_values,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1)
        else:
            delta_tensor = torch.zeros(
                (tokens.shape[0], 1),
                dtype=torch.float32,
                device=device,
            )

        onsets = torch.cat([onsets, onsets[:, -1:] + delta_tensor], dim=1)

    base_model.last_generation_stats = {
        "raw_top1_max": max(raw_top1_probs) if raw_top1_probs else 0.0,
        "final_top1_max": max(final_top1_probs) if final_top1_probs else 0.0,
        "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
        "generated_tokens": int(generation_config.max_new_tokens),
        "generated_samples": int(batch_size),
    }

    return [tokens[idx].tolist() for idx in range(tokens.shape[0])]


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

    base_model = _unwrap_parallel_model(model)

    tokens = tokenizer.encode(seed_midi_path)
    if len(tokens) < config.seed_length:
        print(
            f"Seed file {seed_midi_path} has only {len(tokens)} tokens; "
            f"using the available seed length instead of configured {config.seed_length}."
        )

    seed_token_count = min(len(tokens), int(config.seed_length))
    event_size = int(getattr(tokenizer, "event_size", 0) or 0)
    if event_size > 0 and seed_token_count > 0:
        aligned_count = seed_token_count - (seed_token_count % event_size)
        if aligned_count > 0:
            seed_token_count = aligned_count
        else:
            seed_token_count = min(len(tokens), event_size)

    # Continue from the end of the seed clip instead of replaying the opening bars.
    seed_tokens = tokens[-seed_token_count:]
    seed_onset_times = [
        float(i) * float(max(1e-4, config.time_feature_fallback_step_seconds))
        for i in range(len(seed_tokens))
    ]
    out_paths: List[Path] = []

    supports_time = bool(
        getattr(getattr(base_model, "config", None), "use_v2_architecture", False)
    )

    if supports_time:
        try:
            token_ids, onset_times, _durations = tokenizer.encode_with_time_features(
                seed_midi_path
            )
            seed_token_count = min(len(token_ids), int(config.seed_length))
            event_size = int(getattr(tokenizer, "event_size", 0) or 0)
            if event_size > 0 and seed_token_count > 0:
                aligned_count = seed_token_count - (seed_token_count % event_size)
                if aligned_count > 0:
                    seed_token_count = aligned_count
                else:
                    seed_token_count = min(len(token_ids), event_size)

            seed_tokens = token_ids[-seed_token_count:]
            seed_onset_times = onset_times[-len(seed_tokens):]
        except Exception:
            pass

    prompt_duration: float | None = None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "seed_prompt.mid"
            tokenizer.decode(seed_tokens, prompt_path)
            prompt_duration = midi_duration(prompt_path)
    except Exception:
        prompt_duration = None

    token_id_to_events = getattr(tokenizer, "decode_token_id_events", None)
    if not callable(token_id_to_events):
        token_id_to_events = None

    bind_tokenizer = getattr(base_model, "bind_tokenizer", None)
    if callable(bind_tokenizer):
        try:
            bind_tokenizer(tokenizer)
        except Exception:
            pass

    generate_optional_kwargs: Dict[str, Any] = {}
    try:
        generate_params = inspect.signature(base_model.generate).parameters
    except (TypeError, ValueError):
        generate_params = {}
    if "max_consecutive_zero_deltas" in generate_params:
        generate_optional_kwargs["max_consecutive_zero_deltas"] = max(
            0, int(generation_config.max_consecutive_zero_deltas)
        )

    batch_supported = bool(
        generation_config.num_samples > 1
        and hasattr(base_model, "_triplet_slot")
        and hasattr(base_model, "_mask_logits_to_triplet_slot")
        and hasattr(base_model, "_delta_from_token_events")
    )

    if batch_supported:
        generated_batches = _generate_batched_sequences(
            model=model,
            base_model=base_model,
            seed_tokens=seed_tokens,
            seed_onset_times=seed_onset_times,
            generation_config=generation_config,
            step_seconds=float(max(1e-4, config.time_feature_fallback_step_seconds)),
            token_id_to_events=token_id_to_events,
        )

        for i, generated_tokens in enumerate(generated_batches):
            out_file = output_path
            if generation_config.num_samples > 1:
                out_file = output_path.with_name(
                    f"{output_path.stem}_sample{i + 1}{output_path.suffix}"
                )

            tokenizer.decode(generated_tokens, out_file)
            out_paths.append(out_file)

            if generation_config.save_continuation_only:
                continuation_tokens = generated_tokens[len(seed_tokens) :]
                if continuation_tokens:
                    continuation_file = out_file.with_name(
                        f"{out_file.stem}_new{out_file.suffix}"
                    )
                    tokenizer.decode(continuation_tokens, continuation_file)

            seed_dur = prompt_duration if prompt_duration is not None else midi_duration(seed_midi_path)
            gen_dur = max(0.0, midi_duration(out_file) - seed_dur)
            total_dur = midi_duration(out_file)
            print(
                f"Generated {out_file.name}: seed={seed_dur:.2f}s, "
                f"generated={gen_dur:.2f}s, total={total_dur:.2f}s"
            )

        return out_paths

    for i in range(generation_config.num_samples):
        if supports_time:
            generated_tokens = base_model.generate(
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
                **generate_optional_kwargs,
            )
        else:
            generated_tokens = base_model.generate(
                seed_tokens=seed_tokens,
                max_new_tokens=generation_config.max_new_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                repetition_penalty=generation_config.repetition_penalty,
                repetition_window=generation_config.repetition_window,
                min_tokens_to_keep=generation_config.min_tokens_to_keep,
                **generate_optional_kwargs,
            )

        out_file = output_path
        if generation_config.num_samples > 1:
            out_file = output_path.with_name(
                f"{output_path.stem}_sample{i + 1}{output_path.suffix}"
            )

        tokenizer.decode(generated_tokens, out_file)
        out_paths.append(out_file)

        if generation_config.save_continuation_only:
            continuation_tokens = generated_tokens[len(seed_tokens) :]
            if continuation_tokens:
                continuation_file = out_file.with_name(
                    f"{out_file.stem}_new{out_file.suffix}"
                )
                tokenizer.decode(continuation_tokens, continuation_file)

        seed_dur = prompt_duration if prompt_duration is not None else midi_duration(seed_midi_path)
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

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pretty_midi
import torch

from config import DataConfig


def _load_notes(midi_path: str | Path):
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    return midi, notes


def pitch_class_histogram(midi_path: str | Path) -> np.ndarray:
    _, notes = _load_notes(midi_path)
    hist = np.zeros(12, dtype=np.float64)
    if not notes:
        return hist

    for note in notes:
        hist[note.pitch % 12] += 1.0

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def pitch_class_entropy(histogram: np.ndarray) -> float:
    p = np.asarray(histogram, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def note_density(midi_path: str | Path) -> float:
    midi, notes = _load_notes(midi_path)
    duration = float(midi.get_end_time())
    if duration <= 0:
        return 0.0
    return float(len(notes) / duration)


def rhythmic_regularity(midi_path: str | Path) -> float:
    _, notes = _load_notes(midi_path)
    if len(notes) < 3:
        return 0.0

    onsets = np.asarray(sorted(note.start for note in notes), dtype=np.float64)
    iois = np.diff(onsets)
    iois = iois[iois > 1e-4]
    if iois.size == 0:
        return 0.0

    mean_ioi = float(np.mean(iois))
    if mean_ioi <= 1e-8:
        return 0.0
    std_ioi = float(np.std(iois))
    return float(std_ioi / mean_ioi)


def _mean_velocity(midi_path: str | Path) -> float:
    _, notes = _load_notes(midi_path)
    if not notes:
        return 0.0
    return float(np.mean([n.velocity for n in notes]))


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compare_seed_vs_continuation(
    seed_path: str | Path,
    continuation_path: str | Path,
) -> Dict[str, float]:
    seed_hist = pitch_class_histogram(seed_path)
    cont_hist = pitch_class_histogram(continuation_path)

    seed_density = note_density(seed_path)
    cont_density = note_density(continuation_path)

    seed_vel = _mean_velocity(seed_path)
    cont_vel = _mean_velocity(continuation_path)

    return {
        "pitch_class_cosine": _cosine_similarity(seed_hist, cont_hist),
        "seed_entropy": pitch_class_entropy(seed_hist),
        "continuation_entropy": pitch_class_entropy(cont_hist),
        "seed_note_density": seed_density,
        "continuation_note_density": cont_density,
        "note_density_ratio": cont_density / max(seed_density, 1e-8),
        "seed_mean_velocity": seed_vel,
        "continuation_mean_velocity": cont_vel,
        "velocity_ratio": cont_vel / max(seed_vel, 1e-8),
        "seed_rhythmic_regularity": rhythmic_regularity(seed_path),
        "continuation_rhythmic_regularity": rhythmic_regularity(continuation_path),
    }


@torch.no_grad()
def evaluate_dataset(
    model,
    tokenizer,
    test_loader,
    config: DataConfig,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    device = next(model.parameters()).device

    metric_rows: List[Dict[str, float]] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        for batch_idx, (seed, _continuation) in enumerate(test_loader):
            seed = seed.to(device)

            for i in range(seed.shape[0]):
                seed_tokens = seed[i].detach().cpu().tolist()
                generated_tokens = model.generate(
                    seed_tokens=seed_tokens,
                    max_new_tokens=config.continuation_length,
                    temperature=0.9,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.1,
                    repetition_window=64,
                    min_tokens_to_keep=3,
                )

                seed_path = tmp_dir_path / f"seed_{batch_idx}_{i}.mid"
                continuation_path = tmp_dir_path / f"continuation_{batch_idx}_{i}.mid"

                generated_continuation = generated_tokens[len(seed_tokens) :]

                tokenizer.decode(seed_tokens, seed_path)
                tokenizer.decode(generated_continuation, continuation_path)

                metrics = compare_seed_vs_continuation(seed_path, continuation_path)
                metrics["generated_pitch_entropy"] = pitch_class_entropy(
                    pitch_class_histogram(continuation_path)
                )
                metrics["generated_note_density"] = note_density(continuation_path)
                metrics["generated_rhythmic_regularity"] = rhythmic_regularity(
                    continuation_path
                )
                metric_rows.append(metrics)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not metric_rows:
        return {}

    keys = sorted(metric_rows[0].keys())
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = np.asarray([row[key] for row in metric_rows], dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    return summary

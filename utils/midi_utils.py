from __future__ import annotations

import wave
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional dependency
    plt = None  # type: ignore
    warnings.warn(
        f"matplotlib import failed. Visualization functions will be disabled. Details: {exc}"
    )

import numpy as np
import pretty_midi


def midi_duration(midi_path: str | Path) -> float:
    """Return MIDI duration in seconds."""

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    return float(midi.get_end_time())


def render_midi_audio(
    midi_path: str | Path,
    wav_path: str | Path,
    *,
    sample_rate: int = 22050,
) -> Path:
    """Render a MIDI file to a small WAV preview using pretty_midi's built-in synth."""

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    try:
        waveform = midi.synthesize(fs=int(sample_rate))
    except Exception:
        duration = max(0.5, float(midi.get_end_time()))
        waveform = np.zeros(int(duration * int(sample_rate)), dtype=np.float32)

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.size == 0:
        waveform = np.zeros(int(sample_rate * 0.5), dtype=np.float32)

    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform = 0.95 * waveform / peak

    pcm16 = np.clip(waveform, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)

    out_path = Path(wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())
    return out_path


def _extract_note_events(midi_path: str | Path) -> List[Tuple[float, float, int, int]]:
    """Extract note events `(start, end, pitch, velocity)` for piano range."""

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    events: List[Tuple[float, float, int, int]] = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if 21 <= n.pitch <= 108:
                events.append((n.start, n.end, n.pitch, n.velocity))
    return events


def visualize_pianoroll(
    midi_path: str | Path,
    title: str = "",
    save_path: Optional[str | Path] = None,
) -> None:
    """Render a pianoroll plot for one MIDI file."""

    if plt is None:  # pragma: no cover - optional dependency
        warnings.warn(
            "visualize_pianoroll called but matplotlib is not installed; skipping visualization"
        )
        return

    events = _extract_note_events(midi_path)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(14, 5))
    for start, end, pitch, velocity in events:
        color = cmap(velocity / 127.0)
        ax.hlines(y=pitch, xmin=start, xmax=end, linewidth=2.0, color=color)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_ylim(21, 108)
    ax.set_title(title or f"Piano Roll: {Path(midi_path).name}")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if save_path is None:
        plt.show()
    else:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def compare_pianorolls(
    seed_path: str | Path,
    continuation_path: str | Path,
    save_path: Optional[str | Path] = None,
) -> None:
    """Render side-by-side timeline comparison for seed and continuation."""

    if plt is None:  # pragma: no cover - optional dependency
        warnings.warn(
            "compare_pianorolls called but matplotlib is not installed; skipping visualization"
        )
        return

    seed_events = _extract_note_events(seed_path)
    cont_events = _extract_note_events(continuation_path)
    seed_cmap = plt.get_cmap("Blues")
    cont_cmap = plt.get_cmap("Oranges")

    seed_dur = midi_duration(seed_path)

    fig, ax = plt.subplots(figsize=(16, 5))

    for start, end, pitch, velocity in seed_events:
        color = seed_cmap(0.3 + 0.7 * velocity / 127.0)
        ax.hlines(y=pitch, xmin=start, xmax=end, linewidth=2.0, color=color)

    for start, end, pitch, velocity in cont_events:
        start += seed_dur
        end += seed_dur
        color = cont_cmap(0.3 + 0.7 * velocity / 127.0)
        ax.hlines(y=pitch, xmin=start, xmax=end, linewidth=2.0, color=color)

    ax.axvline(seed_dur, linestyle="--", linewidth=2, color="black", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_ylim(21, 108)
    ax.set_title("Seed | Continuation")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if save_path is None:
        plt.show()
    else:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

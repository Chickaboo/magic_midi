from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np


def _import_pretty_midi() -> Any:
    try:
        import pretty_midi

        return pretty_midi
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pretty_midi import failed. Install pretty_midi to use CustomDeltaTokenizer."
        ) from exc


@dataclass(frozen=True)
class _TokenSpec:
    delta_start: int = 0
    delta_end: int = 31
    pitch_start: int = 32
    pitch_end: int = 119
    duration_start: int = 120
    duration_end: int = 151
    pad_id: int = 152
    bos_id: int = 153
    eos_id: int = 154

    @property
    def vocab_size(self) -> int:
        return 155


class CustomDeltaTokenizer:
    """Triplet tokenizer for solo piano: [delta_onset, pitch, duration]."""

    def __init__(
        self,
        *,
        default_velocity: int = 88,
        include_special_tokens: bool = False,
    ) -> None:
        self.spec = _TokenSpec()
        self.default_velocity = int(max(1, min(127, default_velocity)))
        self.include_special_tokens = bool(include_special_tokens)

        # Delta onset bins: 0..2.0 seconds with log spacing.
        # Bin 0 is exact zero. Remaining 31 bins are log-spaced > 0.
        self._delta_bins = np.concatenate(
            [
                np.asarray([0.0], dtype=np.float64),
                np.logspace(math.log10(1e-4), math.log10(2.0), num=31),
            ],
            axis=0,
        )

        # Duration bins: 0.05..4.0 seconds with log spacing.
        self._duration_bins = np.logspace(
            math.log10(0.05),
            math.log10(4.0),
            num=32,
        ).astype(np.float64)

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size (fixed at 155)."""

        return self.spec.vocab_size

    @property
    def pad_id(self) -> int:
        return self.spec.pad_id

    @property
    def bos_id(self) -> int:
        return self.spec.bos_id

    @property
    def eos_id(self) -> int:
        return self.spec.eos_id

    def train(self, midi_paths: List[Path], vocab_size: int | None = None) -> None:
        """No-op for compatibility; tokenizer is deterministic and fixed-vocab."""

        if not midi_paths:
            raise ValueError("No MIDI files provided.")
        if vocab_size is not None and int(vocab_size) != self.vocab_size:
            raise ValueError(
                f"CustomDeltaTokenizer has fixed vocab_size={self.vocab_size}, "
                f"got vocab_size={int(vocab_size)}"
            )

    def save(self, path: str) -> None:
        """Persist tokenizer config to disk."""

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "CustomDeltaTokenizer",
            "version": 1,
            "vocab_size": int(self.vocab_size),
            "default_velocity": int(self.default_velocity),
            "include_special_tokens": bool(self.include_special_tokens),
            "token_ids": {
                "delta": [self.spec.delta_start, self.spec.delta_end],
                "pitch": [self.spec.pitch_start, self.spec.pitch_end],
                "duration": [self.spec.duration_start, self.spec.duration_end],
                "pad": self.spec.pad_id,
                "bos": self.spec.bos_id,
                "eos": self.spec.eos_id,
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "CustomDeltaTokenizer":
        """Load tokenizer config from disk."""

        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {in_path}")

        payload = json.loads(in_path.read_text(encoding="utf-8"))
        if str(payload.get("type", "")) != "CustomDeltaTokenizer":
            raise ValueError(
                "Unsupported tokenizer payload. Expected type='CustomDeltaTokenizer'."
            )

        return cls(
            default_velocity=int(payload.get("default_velocity", 88)),
            include_special_tokens=bool(payload.get("include_special_tokens", False)),
        )

    def _note_events(self, midi_path: Path) -> List[Tuple[float, int, float]]:
        pretty_midi = _import_pretty_midi()
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        events: List[Tuple[float, int, float]] = []

        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                onset = float(max(0.0, note.start))
                duration = float(max(1e-4, note.end - note.start))
                pitch = int(note.pitch)
                if pitch < 21 or pitch > 108:
                    continue
                events.append((onset, pitch, duration))

        # Deterministic ordering: onset -> pitch -> duration.
        events.sort(key=lambda x: (x[0], x[1], x[2]))
        return events

    @staticmethod
    def _nearest_bin(value: float, bins: np.ndarray) -> int:
        idx = int(np.argmin(np.abs(bins - float(value))))
        return idx

    def _quantize_delta(self, delta_seconds: float) -> int:
        clamped = float(max(0.0, min(2.0, delta_seconds)))
        idx = self._nearest_bin(clamped, self._delta_bins)
        return int(self.spec.delta_start + idx)

    def _quantize_duration(self, duration_seconds: float) -> int:
        clamped = float(max(0.05, min(4.0, duration_seconds)))
        idx = self._nearest_bin(clamped, self._duration_bins)
        return int(self.spec.duration_start + idx)

    def _quantize_pitch(self, pitch: int) -> int:
        pitch_i = int(max(21, min(108, pitch)))
        return int(self.spec.pitch_start + (pitch_i - 21))

    def _dequantize_delta(self, token_id: int) -> float:
        idx = int(token_id) - self.spec.delta_start
        idx = max(0, min(31, idx))
        return float(self._delta_bins[idx])

    def _dequantize_duration(self, token_id: int) -> float:
        idx = int(token_id) - self.spec.duration_start
        idx = max(0, min(31, idx))
        return float(self._duration_bins[idx])

    def _dequantize_pitch(self, token_id: int) -> int:
        idx = int(token_id) - self.spec.pitch_start
        idx = max(0, min(87, idx))
        return int(21 + idx)

    def _encode_triplets(
        self,
        midi_path: Path,
    ) -> Tuple[List[int], List[float], List[float]]:
        events = self._note_events(midi_path)

        token_ids: List[int] = []
        onset_times: List[float] = []
        durations: List[float] = []
        prev_onset = 0.0

        if self.include_special_tokens:
            token_ids.append(self.spec.bos_id)
            onset_times.append(0.0)
            durations.append(1e-4)

        for onset, pitch, duration in events:
            delta = float(max(0.0, onset - prev_onset))
            prev_onset = onset

            d_tok = self._quantize_delta(delta)
            p_tok = self._quantize_pitch(pitch)
            u_tok = self._quantize_duration(duration)
            token_ids.extend([d_tok, p_tok, u_tok])

            # Repeat onset/duration for all 3 tokens in the triplet.
            onset_times.extend([float(onset), float(onset), float(onset)])
            durations.extend([float(duration), float(duration), float(duration)])

        if self.include_special_tokens:
            end_onset = float(onset_times[-1]) if onset_times else 0.0
            token_ids.append(self.spec.eos_id)
            onset_times.append(end_onset)
            durations.append(1e-4)

        if len(token_ids) != len(onset_times):
            raise AssertionError(
                "CustomDeltaTokenizer alignment error: "
                f"len(ids)={len(token_ids)} len(onsets)={len(onset_times)}"
            )
        return token_ids, onset_times, durations

    def encode(self, midi_path: Path) -> List[int]:
        """Encode one MIDI file into flat delta/pitch/duration token IDs."""

        token_ids, _onsets, _durations = self._encode_triplets(Path(midi_path))
        return token_ids

    def encode_with_time_features(
        self,
        midi_path: Path,
    ) -> Tuple[List[int], List[float], List[float]]:
        """Encode one MIDI and return token IDs + aligned onset/duration arrays."""

        return self._encode_triplets(Path(midi_path))

    def _is_delta(self, token_id: int) -> bool:
        return self.spec.delta_start <= int(token_id) <= self.spec.delta_end

    def _is_pitch(self, token_id: int) -> bool:
        return self.spec.pitch_start <= int(token_id) <= self.spec.pitch_end

    def _is_duration(self, token_id: int) -> bool:
        return self.spec.duration_start <= int(token_id) <= self.spec.duration_end

    def _is_special(self, token_id: int) -> bool:
        token = int(token_id)
        return token in {self.spec.pad_id, self.spec.bos_id, self.spec.eos_id}

    def decode_token_id_events(self, token_id: int) -> List[str]:
        """Decode token ID into semantic label(s) for compatibility hooks."""

        token = int(token_id)
        if self._is_delta(token):
            return [f"Delta_{self._dequantize_delta(token):.6f}"]
        if self._is_pitch(token):
            return [f"Pitch_{self._dequantize_pitch(token)}"]
        if self._is_duration(token):
            return [f"Duration_{self._dequantize_duration(token):.6f}"]
        if token == self.spec.pad_id:
            return ["PAD_None"]
        if token == self.spec.bos_id:
            return ["BOS_None"]
        if token == self.spec.eos_id:
            return ["EOS_None"]
        return []

    def decode(
        self,
        token_ids: Sequence[int],
        output_path: Path | str | None = None,
    ) -> Any:
        """Decode token IDs into a PrettyMIDI object.

        If `output_path` is provided, writes MIDI to disk as well.
        """

        pretty_midi = _import_pretty_midi()
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)

        tokens = [int(t) for t in token_ids]
        i = 0
        onset = 0.0

        while i < len(tokens):
            tok = tokens[i]

            if tok == self.spec.eos_id:
                break
            if self._is_special(tok):
                i += 1
                continue

            if not self._is_delta(tok):
                i += 1
                continue
            if i + 2 >= len(tokens):
                break

            p_tok = tokens[i + 1]
            d_tok = tokens[i + 2]
            if (not self._is_pitch(p_tok)) or (not self._is_duration(d_tok)):
                i += 1
                continue

            delta = self._dequantize_delta(tok)
            pitch = self._dequantize_pitch(p_tok)
            duration = self._dequantize_duration(d_tok)

            onset = float(max(0.0, onset + max(0.0, delta)))
            end = float(max(onset + 1e-4, onset + duration))
            note = pretty_midi.Note(
                velocity=int(self.default_velocity),
                pitch=int(pitch),
                start=float(onset),
                end=float(end),
            )
            piano.notes.append(note)
            i += 3

        midi.instruments.append(piano)

        if output_path is not None:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            midi.write(str(out_path))

        return midi

    def verify_roundtrip(self, midi_path: Path) -> bool:
        """Basic encode/decode health check for compatibility with old API."""

        try:
            ids = self.encode(Path(midi_path))
            _ = self.decode(ids)
            return len(ids) > 0
        except Exception:
            return False


__all__ = ["CustomDeltaTokenizer"]

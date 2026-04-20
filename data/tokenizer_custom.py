from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    delta_end: int = 127
    pitch_start: int = 128
    pitch_end: int = 215
    duration_start: int = 216
    duration_end: int = 343
    velocity_start: int = 344
    velocity_end: int = 359
    pad_id: int = 360
    bos_id: int = 361
    eos_id: int = 362
    density_start: int = 363
    density_end: int = 366
    voices_start: int = 367
    voices_end: int = 370
    register_start: int = 371
    register_end: int = 373
    event_size: int = 4

    @property
    def vocab_size(self) -> int:
        return int(self.register_end + 1)


class CustomDeltaTokenizer:
    """Unified frozen quad tokenizer for solo piano with structural meta-prefix context."""

    def __init__(
        self,
        *,
        default_velocity: int = 88,
        include_special_tokens: bool = False,
        include_structural_meta_tokens: bool = True,
        prepend_start_token: bool = True,
        density_quartiles: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.spec = _TokenSpec()
        self.default_velocity = int(max(1, min(127, default_velocity)))
        self.include_special_tokens = bool(include_special_tokens)
        self.include_structural_meta_tokens = bool(include_structural_meta_tokens)
        self.prepend_start_token = bool(prepend_start_token)

        self._density_labels = ("v_low", "low", "med", "high")
        self._voices_labels = ("mono", "poly_small", "poly_med", "poly_large")
        self._register_labels = ("bass", "mid", "treble")
        self._density_quartiles = self._sanitize_density_quartiles(density_quartiles)

        self._density_token_to_label: Dict[int, str] = {
            int(self.spec.density_start + i): label
            for i, label in enumerate(self._density_labels)
        }
        self._voices_token_to_label: Dict[int, str] = {
            int(self.spec.voices_start + i): label
            for i, label in enumerate(self._voices_labels)
        }
        self._register_token_to_label: Dict[int, str] = {
            int(self.spec.register_start + i): label
            for i, label in enumerate(self._register_labels)
        }

        # High-resolution timing bins with floor-style quantization.
        # Delta has one exact-zero bin plus 127 positive bins.
        self._delta_min_positive_seconds = 1e-4
        self._delta_max_seconds = 8.0
        self._duration_min_seconds = 1.0 / 64.0
        self._duration_max_seconds = 8.0

        self._delta_positive_bin_count = int(self.spec.delta_end - self.spec.delta_start)
        self._duration_bin_count = int(self.spec.duration_end - self.spec.duration_start + 1)
        self._velocity_bin_count = int(self.spec.velocity_end - self.spec.velocity_start + 1)

        self._delta_edges = np.logspace(
            math.log10(self._delta_min_positive_seconds),
            math.log10(self._delta_max_seconds),
            num=self._delta_positive_bin_count + 1,
        ).astype(np.float64)
        self._duration_edges = np.logspace(
            math.log10(self._duration_min_seconds),
            math.log10(self._duration_max_seconds),
            num=self._duration_bin_count + 1,
        ).astype(np.float64)

        self._delta_bins = self._bin_representatives_from_edges(
            edges=self._delta_edges,
            include_zero=True,
        )
        self._duration_bins = self._bin_representatives_from_edges(
            edges=self._duration_edges,
            include_zero=False,
        )

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size for the current frozen tokenizer spec."""

        return self.spec.vocab_size

    @property
    def event_size(self) -> int:
        """Return token group size for one note event."""

        return int(self.spec.event_size)

    @property
    def pad_id(self) -> int:
        return self.spec.pad_id

    @property
    def bos_id(self) -> int:
        return self.spec.bos_id

    @property
    def eos_id(self) -> int:
        return self.spec.eos_id

    @property
    def density_quartiles(self) -> Tuple[float, float, float]:
        return self._density_quartiles

    @staticmethod
    def _sanitize_density_quartiles(
        quartiles: Optional[Tuple[float, float, float]],
    ) -> Tuple[float, float, float]:
        if quartiles is None:
            values = [1.0, 2.5, 5.0]
        else:
            raw = [float(v) for v in quartiles]
            if len(raw) != 3:
                raise ValueError("density_quartiles must contain exactly three values")
            values = sorted(max(1e-4, float(v)) for v in raw)

        q1 = float(values[0])
        q2 = float(max(values[1], q1 + 1e-4))
        q3 = float(max(values[2], q2 + 1e-4))
        return (q1, q2, q3)

    def train(self, midi_paths: List[Path], vocab_size: int | None = None) -> None:
        """Calibrate unsupervised density quartiles while keeping fixed vocab."""

        if not midi_paths:
            raise ValueError("No MIDI files provided.")
        if vocab_size is not None and int(vocab_size) != self.vocab_size:
            raise ValueError(
                f"CustomDeltaTokenizer has fixed vocab_size={self.vocab_size}, "
                f"got vocab_size={int(vocab_size)}"
            )

        densities: List[float] = []
        for midi_path in midi_paths:
            try:
                events = self._note_events(Path(midi_path))
            except Exception:
                continue
            density = self._estimate_piece_density(events)
            if density > 0.0:
                densities.append(float(density))

        if len(densities) >= 8:
            q1, q2, q3 = np.quantile(
                np.asarray(densities, dtype=np.float64),
                [0.25, 0.5, 0.75],
            ).tolist()
            self._density_quartiles = self._sanitize_density_quartiles(
                (float(q1), float(q2), float(q3))
            )

    def save(self, path: str) -> None:
        """Persist tokenizer config to disk."""

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "CustomDeltaTokenizer",
            "version": 3,
            "spec_version": 3,
            "frozen": True,
            "vocab_size": int(self.vocab_size),
            "default_velocity": int(self.default_velocity),
            "event_size": int(self.event_size),
            "include_special_tokens": bool(self.include_special_tokens),
            "include_structural_meta_tokens": bool(self.include_structural_meta_tokens),
            "prepend_start_token": bool(self.prepend_start_token),
            "quantization": {
                "delta": {
                    "bins": int(self.spec.delta_end - self.spec.delta_start + 1),
                    "scheme": "logspace_floor",
                    "zero_bin": True,
                    "min_positive_seconds": float(self._delta_min_positive_seconds),
                    "max_seconds": float(self._delta_max_seconds),
                },
                "duration": {
                    "bins": int(self.spec.duration_end - self.spec.duration_start + 1),
                    "scheme": "logspace_floor",
                    "min_seconds": float(self._duration_min_seconds),
                    "max_seconds": float(self._duration_max_seconds),
                },
                "velocity": {
                    "bins": int(self._velocity_bin_count),
                    "scheme": "uniform_floor",
                    "midi_range": [0, 127],
                },
            },
            "density_quartiles": [
                float(self._density_quartiles[0]),
                float(self._density_quartiles[1]),
                float(self._density_quartiles[2]),
            ],
            "token_ids": {
                "delta": [self.spec.delta_start, self.spec.delta_end],
                "pitch": [self.spec.pitch_start, self.spec.pitch_end],
                "duration": [self.spec.duration_start, self.spec.duration_end],
                "velocity": [self.spec.velocity_start, self.spec.velocity_end],
                "density": [self.spec.density_start, self.spec.density_end],
                "voices": [self.spec.voices_start, self.spec.voices_end],
                "register": [self.spec.register_start, self.spec.register_end],
                "pad": self.spec.pad_id,
                "bos": self.spec.bos_id,
                "eos": self.spec.eos_id,
            },
            "token_labels": {
                "density": list(self._density_labels),
                "voices": list(self._voices_labels),
                "register": list(self._register_labels),
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _encode_event_tuples(
        self,
        events: Iterable[Tuple[float, int, float, int]],
    ) -> Tuple[List[int], List[float], List[float]]:
        event_list = list(events)
        token_ids: List[int] = []
        onset_times: List[float] = []
        durations: List[float] = []
        prev_onset = 0.0

        if self.include_structural_meta_tokens:
            density_tok, voices_tok, register_tok = self._derive_structural_meta_tokens(
                event_list
            )
            token_ids.extend([density_tok, voices_tok, register_tok])
            onset_times.extend([0.0, 0.0, 0.0])
            durations.extend([1e-4, 1e-4, 1e-4])

        if self.prepend_start_token or self.include_special_tokens:
            token_ids.append(self.spec.bos_id)
            onset_times.append(0.0)
            durations.append(1e-4)

        for onset, pitch, duration, velocity in event_list:
            delta = float(max(0.0, onset - prev_onset))
            prev_onset = onset

            d_tok = self._quantize_delta(delta)
            p_tok = self._quantize_pitch(pitch)
            u_tok = self._quantize_duration(duration)
            v_tok = self._quantize_velocity(velocity)
            token_ids.extend([d_tok, p_tok, u_tok, v_tok])

            # Repeat onset/duration for all 4 tokens in the event quad.
            onset_times.extend(
                [float(onset), float(onset), float(onset), float(onset)]
            )
            durations.extend(
                [float(duration), float(duration), float(duration), float(duration)]
            )

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

        token_ids = payload.get("token_ids")
        token_ids = token_ids if isinstance(token_ids, dict) else {}
        legacy_no_meta = (
            int(payload.get("vocab_size", 0)) <= 171
            and "density" not in token_ids
            and "voices" not in token_ids
            and "register" not in token_ids
        )

        include_structural_default = not legacy_no_meta
        prepend_start_default = (
            bool(payload.get("include_special_tokens", False))
            if legacy_no_meta
            else True
        )

        quartiles_payload = payload.get("density_quartiles")
        quartiles: Optional[Tuple[float, float, float]] = None
        if isinstance(quartiles_payload, (list, tuple)) and len(quartiles_payload) == 3:
            quartiles = (
                float(quartiles_payload[0]),
                float(quartiles_payload[1]),
                float(quartiles_payload[2]),
            )

        return cls(
            default_velocity=int(payload.get("default_velocity", 88)),
            include_special_tokens=bool(payload.get("include_special_tokens", False)),
            include_structural_meta_tokens=bool(
                payload.get(
                    "include_structural_meta_tokens",
                    include_structural_default,
                )
            ),
            prepend_start_token=bool(
                payload.get("prepend_start_token", prepend_start_default)
            ),
            density_quartiles=quartiles,
        )

    @staticmethod
    def _estimate_piece_density(events: Sequence[Tuple[float, int, float, int]]) -> float:
        if not events:
            return 0.0

        starts = [float(ev[0]) for ev in events]
        ends = [float(ev[0] + max(1e-4, float(ev[2]))) for ev in events]
        span = float(max(1e-3, max(ends) - min(starts)))
        return float(len(events) / span)

    @staticmethod
    def _estimate_polyphony(
        events: Sequence[Tuple[float, int, float, int]],
    ) -> Tuple[float, float]:
        if not events:
            return (1.0, 1.0)

        boundaries: List[Tuple[float, int]] = []
        for onset, _pitch, duration, _velocity in events:
            start = float(max(0.0, onset))
            end = float(max(start + 1e-4, start + float(duration)))
            boundaries.append((start, +1))
            boundaries.append((end, -1))

        boundaries.sort(key=lambda item: (item[0], -item[1]))
        active = 0
        max_active = 0
        weighted_active = 0.0
        total_time = 0.0
        last_t = float(boundaries[0][0])

        for t, delta in boundaries:
            t_f = float(t)
            dt = float(max(0.0, t_f - last_t))
            if dt > 0.0:
                weighted_active += float(max(0, active)) * dt
                total_time += dt
            active += int(delta)
            max_active = max(max_active, active)
            last_t = t_f

        mean_active = (
            float(weighted_active / total_time) if total_time > 0.0 else float(max_active)
        )
        return (float(mean_active), float(max(1, max_active)))

    def _density_token(self, density: float) -> int:
        q1, q2, q3 = self._density_quartiles
        if float(density) <= q1:
            idx = 0
        elif float(density) <= q2:
            idx = 1
        elif float(density) <= q3:
            idx = 2
        else:
            idx = 3
        return int(self.spec.density_start + idx)

    def _voices_token(
        self,
        mean_polyphony: float,
        peak_polyphony: float,
    ) -> int:
        mean_v = float(max(1.0, mean_polyphony))
        peak_v = float(max(1.0, peak_polyphony))
        if peak_v <= 1.05 and mean_v < 1.20:
            idx = 0
        elif mean_v < 2.00 and peak_v <= 3.00:
            idx = 1
        elif mean_v < 3.50 and peak_v <= 6.00:
            idx = 2
        else:
            idx = 3
        return int(self.spec.voices_start + idx)

    def _register_token(self, median_pitch: float) -> int:
        pitch = float(median_pitch)
        if pitch < 48.0:
            idx = 0
        elif pitch <= 72.0:
            idx = 1
        else:
            idx = 2
        return int(self.spec.register_start + idx)

    def _derive_structural_meta_tokens(
        self,
        events: Sequence[Tuple[float, int, float, int]],
    ) -> Tuple[int, int, int]:
        if not events:
            return (
                int(self.spec.density_start),
                int(self.spec.voices_start),
                int(self.spec.register_start + 1),
            )

        density = self._estimate_piece_density(events)
        mean_polyphony, peak_polyphony = self._estimate_polyphony(events)
        pitches = np.asarray([float(ev[1]) for ev in events], dtype=np.float64)
        median_pitch = float(np.median(pitches)) if int(pitches.size) > 0 else 60.0

        return (
            int(self._density_token(density)),
            int(self._voices_token(mean_polyphony, peak_polyphony)),
            int(self._register_token(median_pitch)),
        )

    def _note_events(self, midi_path: Path) -> List[Tuple[float, int, float, int]]:
        pretty_midi = _import_pretty_midi()
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        events: List[Tuple[float, int, float, int]] = []

        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                onset = float(max(0.0, note.start))
                duration = float(max(1e-4, note.end - note.start))
                pitch = int(note.pitch)
                velocity = int(max(0, min(127, int(note.velocity))))
                if pitch < 21 or pitch > 108:
                    continue
                events.append((onset, pitch, duration, velocity))

        # Deterministic ordering: onset -> pitch -> duration -> velocity.
        events.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        return events

    @staticmethod
    def _bin_representatives_from_edges(
        edges: np.ndarray,
        *,
        include_zero: bool,
    ) -> np.ndarray:
        centers = np.sqrt(edges[:-1] * edges[1:])
        if include_zero:
            return np.concatenate(
                [np.asarray([0.0], dtype=np.float64), centers.astype(np.float64)],
                axis=0,
            )
        return centers.astype(np.float64)

    def _quantize_delta(self, delta_seconds: float) -> int:
        clamped = float(max(0.0, min(self._delta_max_seconds, float(delta_seconds))))
        if clamped <= 0.0:
            idx = 0
        else:
            positive = float(max(self._delta_min_positive_seconds, clamped))
            pos_idx = int(np.searchsorted(self._delta_edges, positive, side="right") - 1)
            pos_idx = max(0, min(self._delta_positive_bin_count - 1, pos_idx))
            idx = 1 + int(pos_idx)
        return int(self.spec.delta_start + idx)

    def _quantize_duration(self, duration_seconds: float) -> int:
        clamped = float(
            max(self._duration_min_seconds, min(self._duration_max_seconds, float(duration_seconds)))
        )
        idx = int(np.searchsorted(self._duration_edges, clamped, side="right") - 1)
        idx = max(0, min(self._duration_bin_count - 1, idx))
        return int(self.spec.duration_start + idx)

    def _quantize_pitch(self, pitch: int) -> int:
        pitch_i = int(max(21, min(108, pitch)))
        return int(self.spec.pitch_start + (pitch_i - 21))

    def _quantize_velocity(self, velocity: int) -> int:
        vel = int(max(0, min(127, int(velocity))))
        # Floor-style quantization keeps bucket boundaries stable across runs.
        bin_idx = int((float(vel) / 128.0) * float(self._velocity_bin_count))
        bin_idx = max(0, min(self._velocity_bin_count - 1, bin_idx))
        return int(self.spec.velocity_start + bin_idx)

    def _dequantize_delta(self, token_id: int) -> float:
        idx = int(token_id) - self.spec.delta_start
        idx = max(0, min(int(self._delta_bins.shape[0]) - 1, idx))
        return float(self._delta_bins[idx])

    def _dequantize_duration(self, token_id: int) -> float:
        idx = int(token_id) - self.spec.duration_start
        idx = max(0, min(int(self._duration_bins.shape[0]) - 1, idx))
        return float(self._duration_bins[idx])

    def _dequantize_pitch(self, token_id: int) -> int:
        idx = int(token_id) - self.spec.pitch_start
        idx = max(0, min(87, idx))
        return int(21 + idx)

    def _dequantize_velocity(self, token_id: int) -> int:
        idx = int(token_id) - self.spec.velocity_start
        idx = max(0, min(self._velocity_bin_count - 1, idx))
        center = (float(idx) + 0.5) * (128.0 / float(self._velocity_bin_count))
        return int(max(0, min(127, int(center))))

    def _encode_events(
        self,
        midi_path: Path,
    ) -> Tuple[List[int], List[float], List[float]]:
        events = self._note_events(midi_path)
        return self._encode_event_tuples(events)

    def encode(self, midi_path: Path) -> List[int]:
        """Encode one MIDI file into flat delta/pitch/duration/velocity token IDs."""

        token_ids, _onsets, _durations = self._encode_events(Path(midi_path))
        return token_ids

    def encode_with_time_features(
        self,
        midi_path: Path,
    ) -> Tuple[List[int], List[float], List[float]]:
        """Encode one MIDI and return token IDs + aligned onset/duration arrays."""

        return self._encode_events(Path(midi_path))

    def encode_events(
        self,
        events: Iterable[Tuple[float, int, float, int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode already-parsed note events using the frozen tokenizer spec."""

        token_ids, onset_times, durations = self._encode_event_tuples(events)
        return (
            np.asarray(token_ids, dtype=np.int16),
            np.asarray(onset_times, dtype=np.float32),
            np.asarray(durations, dtype=np.float32),
        )

    def _is_delta(self, token_id: int) -> bool:
        return self.spec.delta_start <= int(token_id) <= self.spec.delta_end

    def _is_pitch(self, token_id: int) -> bool:
        return self.spec.pitch_start <= int(token_id) <= self.spec.pitch_end

    def _is_duration(self, token_id: int) -> bool:
        return self.spec.duration_start <= int(token_id) <= self.spec.duration_end

    def _is_velocity(self, token_id: int) -> bool:
        return self.spec.velocity_start <= int(token_id) <= self.spec.velocity_end

    def _is_density(self, token_id: int) -> bool:
        return self.spec.density_start <= int(token_id) <= self.spec.density_end

    def _is_voices(self, token_id: int) -> bool:
        return self.spec.voices_start <= int(token_id) <= self.spec.voices_end

    def _is_register(self, token_id: int) -> bool:
        return self.spec.register_start <= int(token_id) <= self.spec.register_end

    def _is_meta(self, token_id: int) -> bool:
        token = int(token_id)
        return self._is_density(token) or self._is_voices(token) or self._is_register(
            token
        )

    def _is_special(self, token_id: int) -> bool:
        token = int(token_id)
        return token in {self.spec.pad_id, self.spec.bos_id, self.spec.eos_id} or self._is_meta(token)

    def decode_token_id_events(self, token_id: int) -> List[str]:
        """Decode token ID into semantic label(s) for compatibility hooks."""

        token = int(token_id)
        if self._is_delta(token):
            return [f"Delta_{self._dequantize_delta(token):.6f}"]
        if self._is_pitch(token):
            return [f"Pitch_{self._dequantize_pitch(token)}"]
        if self._is_duration(token):
            return [f"Duration_{self._dequantize_duration(token):.6f}"]
        if self._is_velocity(token):
            return [f"Velocity_{self._dequantize_velocity(token)}"]
        if self._is_density(token):
            return [f"Density_{self._density_token_to_label.get(token, 'v_low')}"]
        if self._is_voices(token):
            return [f"Voices_{self._voices_token_to_label.get(token, 'mono')}"]
        if self._is_register(token):
            return [f"Register_{self._register_token_to_label.get(token, 'mid')}"]
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
            if i + 3 >= len(tokens):
                break

            p_tok = tokens[i + 1]
            d_tok = tokens[i + 2]
            v_tok = tokens[i + 3]
            if (
                (not self._is_pitch(p_tok))
                or (not self._is_duration(d_tok))
                or (not self._is_velocity(v_tok))
            ):
                i += 1
                continue

            delta = self._dequantize_delta(tok)
            pitch = self._dequantize_pitch(p_tok)
            duration = self._dequantize_duration(d_tok)
            velocity = self._dequantize_velocity(v_tok)

            onset = float(max(0.0, onset + max(0.0, delta)))
            end = float(max(onset + 1e-4, onset + duration))
            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=float(onset),
                end=float(end),
            )
            piano.notes.append(note)
            i += 4

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

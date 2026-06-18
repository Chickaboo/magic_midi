from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _import_pretty_midi() -> Any:
    try:
        import pretty_midi

        return pretty_midi
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pretty_midi is required to use PianoREMIBPE.") from exc


@dataclass(frozen=True)
class _RemiEvent:
    tick: int
    order: int
    kind: str
    values: Tuple[int, ...]


class PianoREMIBPETokenizer:
    """REMI-style solo-piano tokenizer with integer-domain BPE merges."""

    def __init__(
        self,
        *,
        vocab_size: int = 30000,
        positions_per_bar: int = 16,
        max_duration_bars: int = 4,
        tempo_bins: int = 64,
        min_tempo: int = 30,
        max_tempo: int = 240,
        include_special_tokens: bool = True,
        merges: Optional[List[Tuple[int, int]]] = None,
        token_to_id: Optional[Dict[str, int]] = None,
    ) -> None:
        self.target_vocab_size = int(max(1, vocab_size))
        self.positions_per_bar = int(max(1, positions_per_bar))
        self.max_duration_bars = int(max(1, max_duration_bars))
        self.tempo_bins = int(max(1, tempo_bins))
        self.min_tempo = int(max(1, min_tempo))
        self.max_tempo = int(max(self.min_tempo + 1, max_tempo))
        self.include_special_tokens = bool(include_special_tokens)
        self.event_size = 1

        self._token_to_id = token_to_id or self._build_base_vocab()
        self._id_to_token = {int(v): str(k) for k, v in self._token_to_id.items()}
        self._base_vocab_size = len(self._token_to_id)
        self._merges: List[Tuple[int, int]] = [tuple(map(int, m)) for m in (merges or [])]
        self._merge_to_id: Dict[Tuple[int, int], int] = {}
        self._bpe_id_to_parts: Dict[int, Tuple[int, ...]] = {}
        self._rebuild_bpe_maps()

    @property
    def pad_id(self) -> int:
        return int(self._token_to_id["PAD"])

    @property
    def bos_id(self) -> int:
        return int(self._token_to_id["BOS"])

    @property
    def eos_id(self) -> int:
        return int(self._token_to_id["EOS"])

    @property
    def vocab_size(self) -> int:
        return int(len(self._token_to_id) + len(self._merges))

    def _build_base_vocab(self) -> Dict[str, int]:
        names: List[str] = ["PAD", "BOS", "EOS", "BAR"]
        names.extend(f"POSITION_{i}" for i in range(self.positions_per_bar))
        names.extend(f"NOTE_ON_{i}" for i in range(128))
        names.extend(f"VELOCITY_{i}" for i in range(32))
        duration_count = self.positions_per_bar * self.max_duration_bars
        names.extend(f"DURATION_{i + 1}" for i in range(duration_count))
        names.extend(f"TIME_SHIFT_{i + 1}" for i in range(duration_count))
        names.extend(f"TEMPO_{i}" for i in range(self.tempo_bins))
        names.extend(["PEDAL_ON", "PEDAL_OFF"])
        return {name: idx for idx, name in enumerate(names)}

    def _rebuild_bpe_maps(self) -> None:
        self._merge_to_id = {}
        self._bpe_id_to_parts = {}
        next_id = int(self._base_vocab_size)
        for pair in self._merges:
            merged_id = next_id
            next_id += 1
            left_parts = self._expand_id(pair[0])
            right_parts = self._expand_id(pair[1])
            self._merge_to_id[pair] = int(merged_id)
            self._bpe_id_to_parts[int(merged_id)] = tuple(left_parts + right_parts)

    def _expand_id(self, token_id: int) -> Tuple[int, ...]:
        token = int(token_id)
        if token < self._base_vocab_size:
            return (token,)
        return tuple(self._bpe_id_to_parts.get(token, (token,)))

    def _token_id(self, name: str) -> int:
        return int(self._token_to_id[name])

    def _tempo_to_bin(self, qpm: float) -> int:
        tempo = float(max(self.min_tempo, min(self.max_tempo, qpm)))
        span = float(self.max_tempo - self.min_tempo)
        idx = int(round(((tempo - float(self.min_tempo)) / span) * float(self.tempo_bins - 1)))
        return int(max(0, min(self.tempo_bins - 1, idx)))

    def _bin_to_tempo(self, idx: int) -> float:
        bin_idx = int(max(0, min(self.tempo_bins - 1, idx)))
        if self.tempo_bins <= 1:
            return float(self.min_tempo)
        frac = float(bin_idx) / float(self.tempo_bins - 1)
        return float(self.min_tempo + frac * float(self.max_tempo - self.min_tempo))

    def _ticks_per_bar(self, midi: Any) -> int:
        resolution = int(getattr(midi, "resolution", 220) or 220)
        return int(max(1, resolution * 4))

    def _grid_ticks(self, midi: Any) -> int:
        return int(max(1, round(float(self._ticks_per_bar(midi)) / float(self.positions_per_bar))))

    def _quantize_tick(self, midi: Any, seconds: float) -> int:
        tick = int(round(float(midi.time_to_tick(float(max(0.0, seconds))))))
        grid = self._grid_ticks(midi)
        return int(round(float(tick) / float(grid)) * grid)

    def _duration_units(self, midi: Any, start: float, end: float) -> int:
        grid = self._grid_ticks(midi)
        raw = int(round(float(max(1e-4, end - start)) / max(1e-9, float(midi.tick_to_time(grid) - midi.tick_to_time(0)))))
        max_units = self.positions_per_bar * self.max_duration_bars
        return int(max(1, min(max_units, raw)))

    def _piano_instruments(self, midi: Any) -> List[Any]:
        return [
            inst
            for inst in midi.instruments
            if (not bool(getattr(inst, "is_drum", False))) and 0 <= int(inst.program) <= 7
        ]

    def _base_sequence_with_features(
        self,
        midi_path: Path,
    ) -> Tuple[List[int], List[float], List[float]]:
        pretty_midi = _import_pretty_midi()
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        piano_tracks = self._piano_instruments(midi)
        if not piano_tracks:
            raise RuntimeError("MIDI contains no piano-family tracks (programs 0..7).")

        events: List[_RemiEvent] = []
        for inst in piano_tracks:
            for note in inst.notes:
                start_tick = self._quantize_tick(midi, float(note.start))
                dur_units = self._duration_units(midi, float(note.start), float(note.end))
                velocity_bin = int(max(0, min(31, int(int(note.velocity) * 32 / 128))))
                events.append(
                    _RemiEvent(
                        tick=int(start_tick),
                        order=20,
                        kind="note",
                        values=(int(max(0, min(127, note.pitch))), velocity_bin, dur_units),
                    )
                )
            for cc in getattr(inst, "control_changes", []):
                if int(getattr(cc, "number", -1)) != 64:
                    continue
                tick = self._quantize_tick(midi, float(cc.time))
                is_on = int(getattr(cc, "value", 0)) >= 64
                events.append(
                    _RemiEvent(
                        tick=int(tick),
                        order=30,
                        kind="pedal",
                        values=(1 if is_on else 0,),
                    )
                )

        try:
            tempo_times, tempi = midi.get_tempo_changes()
            for time_value, tempo in zip(tempo_times.tolist(), tempi.tolist()):
                tick = self._quantize_tick(midi, float(time_value))
                events.append(
                    _RemiEvent(
                        tick=int(tick),
                        order=10,
                        kind="tempo",
                        values=(self._tempo_to_bin(float(tempo)),),
                    )
                )
        except Exception:
            pass

        events.sort(key=lambda ev: (ev.tick, ev.order, ev.kind, ev.values))
        if not events:
            raise RuntimeError("No tokenizable piano events found.")

        ticks_per_bar = self._ticks_per_bar(midi)
        grid = self._grid_ticks(midi)
        token_ids: List[int] = []
        onsets: List[float] = []
        durations: List[float] = []

        def append(name: str, tick: int, duration_units: int = 1) -> None:
            token_ids.append(self._token_id(name))
            try:
                onsets.append(float(midi.tick_to_time(int(max(0, tick)))))
                end_tick = int(max(0, tick) + int(duration_units) * grid)
                durations.append(float(max(1e-4, midi.tick_to_time(end_tick) - midi.tick_to_time(int(max(0, tick))))))
            except Exception:
                onsets.append(0.0)
                durations.append(1e-4)

        if self.include_special_tokens:
            append("BOS", 0)

        current_bar = -1
        last_tick = 0
        for event in events:
            bar = int(event.tick // ticks_per_bar)
            while current_bar < bar:
                current_bar += 1
                append("BAR", current_bar * ticks_per_bar)

            position = int((event.tick - (bar * ticks_per_bar)) // grid)
            position = int(max(0, min(self.positions_per_bar - 1, position)))

            gap_units = int(round(float(max(0, event.tick - last_tick)) / float(grid)))
            max_units = self.positions_per_bar * self.max_duration_bars
            while gap_units > max_units:
                append(f"TIME_SHIFT_{max_units}", last_tick, max_units)
                last_tick += max_units * grid
                gap_units -= max_units
            if gap_units > 0:
                append(f"TIME_SHIFT_{gap_units}", last_tick, gap_units)

            append(f"POSITION_{position}", event.tick)

            if event.kind == "tempo":
                append(f"TEMPO_{event.values[0]}", event.tick)
            elif event.kind == "pedal":
                append("PEDAL_ON" if event.values[0] else "PEDAL_OFF", event.tick)
            elif event.kind == "note":
                pitch, velocity_bin, duration_units = event.values
                append(f"NOTE_ON_{pitch}", event.tick, duration_units)
                append(f"VELOCITY_{velocity_bin}", event.tick, duration_units)
                append(f"DURATION_{duration_units}", event.tick, duration_units)
            last_tick = int(event.tick)

        if self.include_special_tokens:
            append("EOS", max(last_tick, 0))
        return token_ids, onsets, durations

    def _apply_bpe_with_groups(
        self,
        base_ids: Sequence[int],
    ) -> Tuple[List[int], List[List[int]]]:
        seq = [int(t) for t in base_ids]
        groups = [[i] for i in range(len(seq))]
        if not self._merges or len(seq) < 2:
            return seq, groups

        if all(int(left) < self._base_vocab_size and int(right) < self._base_vocab_size for left, right in self._merges):
            out_seq: List[int] = []
            out_groups: List[List[int]] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq):
                    pair = (int(seq[i]), int(seq[i + 1]))
                    merged_id = self._merge_to_id.get(pair)
                    if merged_id is not None:
                        out_seq.append(int(merged_id))
                        out_groups.append(groups[i] + groups[i + 1])
                        i += 2
                        continue
                out_seq.append(int(seq[i]))
                out_groups.append(groups[i])
                i += 1
            return out_seq, out_groups

        merge_rank = {pair: rank for rank, pair in enumerate(self._merges)}
        while len(seq) >= 2:
            best_idx = -1
            best_rank = None
            best_pair: Tuple[int, int] | None = None
            for idx in range(len(seq) - 1):
                pair = (int(seq[idx]), int(seq[idx + 1]))
                rank = merge_rank.get(pair)
                if rank is None:
                    continue
                if best_rank is None or int(rank) < int(best_rank):
                    best_rank = int(rank)
                    best_idx = int(idx)
                    best_pair = pair
            if best_pair is None or best_idx < 0:
                break

            merged_id = self._merge_to_id.get(best_pair)
            if merged_id is None:
                break
            seq[best_idx : best_idx + 2] = [int(merged_id)]
            groups[best_idx : best_idx + 2] = [groups[best_idx] + groups[best_idx + 1]]
        return seq, groups

    def train(
        self,
        midi_paths: List[Path],
        vocab_size: int | None = None,
        sample_size: int | None = None,
        mode: str = "iterative",
        progress_every: int = 0,
    ) -> None:
        target = int(vocab_size or self.target_vocab_size)
        paths = [Path(p) for p in midi_paths]
        if sample_size is not None and int(sample_size) > 0:
            paths = paths[: int(sample_size)]
        if not paths:
            raise ValueError("No MIDI files provided for PianoREMIBPE training.")

        sequences: List[List[int]] = []
        for midi_path in paths:
            try:
                ids, _onsets, _durations = self._base_sequence_with_features(midi_path)
            except Exception:
                continue
            if len(ids) >= 4:
                sequences.append(ids)

        self.train_from_base_sequences(
            sequences,
            vocab_size=target,
            mode=mode,
            progress_every=progress_every,
        )

    def train_from_base_sequences(
        self,
        sequences: Sequence[Sequence[int]],
        vocab_size: int | None = None,
        mode: str = "iterative",
        progress_every: int = 0,
    ) -> None:
        target = int(vocab_size or self.target_vocab_size)
        sequences = [[int(token) for token in seq] for seq in sequences if len(seq) >= 4]
        if not sequences:
            raise RuntimeError("No valid MIDI files were available for BPE training.")
        self._merges = []
        self._rebuild_bpe_maps()

        train_mode = str(mode or "iterative").strip().lower()
        if train_mode in {"fast", "pair_counts", "pair-counts"}:
            counts: Counter[Tuple[int, int]] = Counter()
            for index, seq in enumerate(sequences, start=1):
                counts.update(zip(seq, seq[1:]))
                if int(progress_every) > 0 and index % int(progress_every) == 0:
                    print(
                        f"PianoREMIBPE BPE counting: sequences={index:,}/{len(sequences):,} "
                        f"unique_pairs={len(counts):,}",
                        flush=True,
                    )
            max_merges = max(0, int(target) - int(self._base_vocab_size))
            self._merges = [
                (int(pair[0]), int(pair[1]))
                for pair, freq in counts.most_common(max_merges)
                if int(freq) >= 2
            ]
            self._rebuild_bpe_maps()
            print(
                f"PianoREMIBPE fast BPE complete: merges={len(self._merges):,} "
                f"vocab={self.vocab_size:,}",
                flush=True,
            )
            return

        while self.vocab_size < target:
            counts: Counter[Tuple[int, int]] = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if int(freq) < 2:
                break
            pair = (int(pair[0]), int(pair[1]))
            self._merges.append(pair)
            self._rebuild_bpe_maps()
            merged_id = self._merge_to_id[pair]
            new_sequences: List[List[int]] = []
            for seq in sequences:
                out: List[int] = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and (int(seq[i]), int(seq[i + 1])) == pair:
                        out.append(int(merged_id))
                        i += 2
                    else:
                        out.append(int(seq[i]))
                        i += 1
                new_sequences.append(out)
            sequences = new_sequences
            if int(progress_every) > 0 and len(self._merges) % int(progress_every) == 0:
                print(
                    f"PianoREMIBPE iterative BPE: merges={len(self._merges):,} "
                    f"vocab={self.vocab_size:,}/{target:,}",
                    flush=True,
                )

    def encode(self, midi_path: Path) -> List[int]:
        base_ids, _onsets, _durations = self._base_sequence_with_features(Path(midi_path))
        ids, _groups = self._apply_bpe_with_groups(base_ids)
        return [int(t) for t in ids]

    def encode_with_time_features(
        self,
        midi_path: Path,
    ) -> Tuple[List[int], List[float], List[float]]:
        base_ids, base_onsets, base_durations = self._base_sequence_with_features(Path(midi_path))
        ids, groups = self._apply_bpe_with_groups(base_ids)
        onsets: List[float] = []
        durations: List[float] = []
        for group in groups:
            first = int(group[0])
            last = int(group[-1])
            onsets.append(float(base_onsets[first]))
            end = float(base_onsets[last] + base_durations[last])
            durations.append(float(max(1e-4, end - float(base_onsets[first]))))
        return [int(t) for t in ids], onsets, durations

    def decode_token_id_events(self, token_id: int) -> List[str]:
        labels: List[str] = []
        for base_id in self._expand_id(int(token_id)):
            labels.append(str(self._id_to_token.get(int(base_id), f"UNK_{base_id}")))
        return labels

    def decode(
        self,
        token_ids: Sequence[int],
        output_path: Path | str | None = None,
    ) -> Any:
        pretty_midi = _import_pretty_midi()
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        piano = pretty_midi.Instrument(program=0)

        base_tokens: List[str] = []
        for token_id in token_ids:
            base_tokens.extend(self.decode_token_id_events(int(token_id)))

        seconds_per_position = 0.125
        bar_seconds = seconds_per_position * float(self.positions_per_bar)
        current_bar = -1
        current_position = 0
        current_time = 0.0
        pending_pitch: Optional[int] = None
        pending_velocity = 80

        for token in base_tokens:
            if token == "EOS":
                break
            if token in {"PAD", "BOS"}:
                continue
            if token == "BAR":
                current_bar += 1
                current_position = 0
                current_time = float(max(0, current_bar)) * bar_seconds
                continue
            if token.startswith("POSITION_"):
                current_position = int(token.split("_", 1)[1])
                current_time = float(max(0, current_bar)) * bar_seconds + (
                    float(current_position) * seconds_per_position
                )
                continue
            if token.startswith("TIME_SHIFT_"):
                units = int(token.rsplit("_", 1)[1])
                current_time += float(units) * seconds_per_position
                continue
            if token.startswith("TEMPO_"):
                continue
            if token == "PEDAL_ON" or token == "PEDAL_OFF":
                value = 127 if token == "PEDAL_ON" else 0
                piano.control_changes.append(
                    pretty_midi.ControlChange(number=64, value=value, time=float(current_time))
                )
                continue
            if token.startswith("NOTE_ON_"):
                pending_pitch = int(token.rsplit("_", 1)[1])
                pending_velocity = 80
                continue
            if token.startswith("VELOCITY_"):
                bin_idx = int(token.rsplit("_", 1)[1])
                pending_velocity = int(max(0, min(127, (bin_idx * 4) + 2)))
                continue
            if token.startswith("DURATION_") and pending_pitch is not None:
                units = int(token.rsplit("_", 1)[1])
                duration = float(max(1, units)) * seconds_per_position
                piano.notes.append(
                    pretty_midi.Note(
                        velocity=int(pending_velocity),
                        pitch=int(max(0, min(127, pending_pitch))),
                        start=float(current_time),
                        end=float(current_time + max(1e-4, duration)),
                    )
                )
                pending_pitch = None

        midi.instruments.append(piano)
        if output_path is not None:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            midi.write(str(out_path))
        return midi

    def verify_roundtrip(self, midi_path: Path) -> bool:
        try:
            ids = self.encode(Path(midi_path))
            _ = self.decode(ids)
            return len(ids) > 0
        except Exception:
            return False

    def save(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "PianoREMIBPE",
            "version": 1,
            "target_vocab_size": int(self.target_vocab_size),
            "vocab_size": int(self.vocab_size),
            "base_vocab_size": int(self._base_vocab_size),
            "positions_per_bar": int(self.positions_per_bar),
            "max_duration_bars": int(self.max_duration_bars),
            "tempo_bins": int(self.tempo_bins),
            "min_tempo": int(self.min_tempo),
            "max_tempo": int(self.max_tempo),
            "include_special_tokens": bool(self.include_special_tokens),
            "event_size": 1,
            "token_to_id": self._token_to_id,
            "merges": [[int(a), int(b)] for a, b in self._merges],
            "special_tokens": {
                "pad": int(self.pad_id),
                "bos": int(self.bos_id),
                "eos": int(self.eos_id),
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PianoREMIBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(payload.get("type", "")) != "PianoREMIBPE":
            raise ValueError("Expected tokenizer payload type='PianoREMIBPE'.")
        return cls(
            vocab_size=int(payload.get("target_vocab_size", payload.get("vocab_size", 30000))),
            positions_per_bar=int(payload.get("positions_per_bar", 16)),
            max_duration_bars=int(payload.get("max_duration_bars", 4)),
            tempo_bins=int(payload.get("tempo_bins", 64)),
            min_tempo=int(payload.get("min_tempo", 30)),
            max_tempo=int(payload.get("max_tempo", 240)),
            include_special_tokens=bool(payload.get("include_special_tokens", True)),
            merges=[tuple(map(int, pair)) for pair in payload.get("merges", [])],
            token_to_id={str(k): int(v) for k, v in dict(payload.get("token_to_id", {})).items()},
        )


__all__ = ["PianoREMIBPETokenizer"]

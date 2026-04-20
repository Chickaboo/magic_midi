from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from miditok import Octuple, REMI, TokenizerConfig

from data.tokenizer_custom import CustomDeltaTokenizer


TIME_EVENT_TYPES = {
    "bar",
    "position",
    "tempo",
    "time_sig",
    "timesig",
    "program",
    "chord",
    "pedal",
    "pitch",
    "velocity",
    "duration",
}
ROUNDTRIP_TOLERANCE = 0.05


class PianoTokenizer:
    """Wrapper around MidiTok tokenizers with compatibility helpers."""

    def __init__(
        self,
        tokenizer: REMI | Octuple | None = None,
        strategy: str = "remi",
    ) -> None:
        self.strategy = str(strategy).strip().lower()
        if self.strategy not in {"remi", "octuple"}:
            raise ValueError(
                f"Unsupported tokenization strategy '{strategy}'. Use 'remi' or 'octuple'."
            )

        if tokenizer is not None:
            self.tokenizer = tokenizer
            self._token_event_cache: Dict[int, List[str]] = {}
            return

        tokenizer_config = TokenizerConfig(
            num_velocities=32,
            use_tempos=True,
            use_time_signatures=True,
            use_chords=False,
            use_sustain_pedals=True,
            use_programs=False,
        )
        if self.strategy == "octuple":
            self.tokenizer = Octuple(tokenizer_config=tokenizer_config)
        else:
            self.tokenizer = REMI(tokenizer_config=tokenizer_config)
        self._token_event_cache: Dict[int, List[str]] = {}

    def train(self, midi_paths: List[Path], vocab_size: int) -> None:
        """Train BPE tokenizer using available MidiTok API variants."""

        paths = list(midi_paths)
        if not paths:
            raise ValueError("No MIDI files provided for tokenizer training.")

        errors: List[str] = []

        if hasattr(self.tokenizer, "train"):
            try:
                self.tokenizer.train(
                    vocab_size=vocab_size, files_paths=paths, model="BPE"
                )
                return
            except TypeError as exc:
                errors.append(
                    f"train(vocab_size=..., files_paths=..., model='BPE'): {exc}"
                )
            except Exception as exc:  # pragma: no cover
                errors.append(f"train(...) failed: {exc}")

        if hasattr(self.tokenizer, "learn_bpe"):
            try:
                self.tokenizer.learn_bpe(vocab_size=vocab_size, files_paths=paths)
                return
            except TypeError as exc:
                errors.append(f"learn_bpe(vocab_size=..., files_paths=...): {exc}")
            except Exception as exc:  # pragma: no cover
                errors.append(f"learn_bpe(...) failed: {exc}")

            try:
                path_strs = [str(p) for p in paths]
                self.tokenizer.learn_bpe(vocab_size=vocab_size, tokens_paths=path_strs)
                return
            except Exception as exc:  # pragma: no cover
                errors.append(f"learn_bpe(vocab_size=..., tokens_paths=...): {exc}")

        raise RuntimeError(
            "Unable to train BPE tokenizer with current MidiTok API."
            f" Attempted calls: {' | '.join(errors) if errors else 'none'}"
        )

    def encode(self, midi_path: Path) -> List[int]:
        """Encode MIDI file into integer token IDs."""

        encoded = self._encode_raw(midi_path, encode_ids=True)

        token_ids = self._extract_token_ids(encoded)
        if not token_ids:
            raise RuntimeError(f"Tokenizer produced no tokens for {midi_path}.")
        return token_ids

    def encode_with_time_features(
        self,
        midi_path: Path,
    ) -> Tuple[List[int], List[float], List[float]]:
        """Encode one MIDI file and return `(ids, onset_times, durations)` in seconds."""

        pretty_midi = _import_pretty_midi()
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        raw = self._encode_raw(midi_path, encode_ids=False)
        seq = self._coerce_tok_sequence(raw)

        events = list(getattr(seq, "events", []) or [])
        event_onsets, event_durations = self._extract_event_time_features(midi, events)

        if not event_onsets:
            token_ids = self.encode(midi_path)
            fallback = self._fallback_time_features(
                len(token_ids), total_seconds=float(midi.get_end_time())
            )
            return token_ids, fallback[0], fallback[1]

        self._encode_tok_sequence_ids(seq)
        token_ids = [int(i) for i in list(getattr(seq, "ids", []) or [])]
        if not token_ids:
            token_ids = self._extract_token_ids(raw)

        onset_times = self._align_event_features_to_token_ids(
            values=event_onsets,
            token_ids=token_ids,
            monotonic=True,
        )
        durations = self._align_event_features_to_token_ids(
            values=event_durations,
            token_ids=token_ids,
            monotonic=False,
        )
        durations = [max(1e-4, float(d)) for d in durations]
        return token_ids, onset_times, durations

    def _extract_event_time_features(
        self,
        midi: Any,
        events: Sequence[Any],
    ) -> Tuple[List[float], List[float]]:
        """Build event-aligned onset/duration arrays with token-type-aware updates."""

        event_onsets: List[float] = []
        event_durations: List[float] = []
        current_onset = 0.0
        current_duration = 0.5

        for event in events:
            type_name = str(getattr(event, "type_", "") or "").strip().lower()
            if type_name in TIME_EVENT_TYPES:
                tick = self._event_tick(event)
                current_onset = self._tick_to_seconds(midi, tick)

            if type_name == "duration":
                dur = self._event_duration_seconds(midi, event)
                if dur is not None:
                    current_duration = max(1e-4, float(dur))

            event_onsets.append(float(max(0.0, current_onset)))
            event_durations.append(float(max(1e-4, current_duration)))

        return event_onsets, event_durations

    def _align_event_features_to_token_ids(
        self,
        values: Sequence[float],
        token_ids: Sequence[int],
        monotonic: bool,
    ) -> List[float]:
        """Align event-level features to token IDs with BPE-aware compression."""

        target_len = len(token_ids)
        if target_len <= 0:
            return []

        event_count = len(values)
        if event_count <= 0:
            return [0.0] * target_len

        if event_count == target_len:
            arr = np.asarray(values, dtype=np.float64)
        elif event_count > target_len:
            group_lengths = self._bpe_group_lengths_from_ids(token_ids)
            if group_lengths is None:
                arr = np.asarray(
                    self._resample_features(values, target_len, monotonic=False),
                    dtype=np.float64,
                )
            else:
                arr = np.asarray(
                    self._compress_features_by_groups(values, group_lengths),
                    dtype=np.float64,
                )
        else:
            arr = np.asarray(
                self._expand_aligned_features(
                    values,
                    event_count=event_count,
                    target_len=target_len,
                ),
                dtype=np.float64,
            )

        if monotonic:
            arr = np.maximum.accumulate(arr)

        if arr.shape[0] != target_len:
            arr = np.asarray(
                self._resample_features(arr.tolist(), target_len, monotonic=monotonic),
                dtype=np.float64,
            )
        return [float(v) for v in arr.tolist()]

    def _bpe_group_lengths_from_ids(
        self,
        token_ids: Sequence[int],
    ) -> List[int] | None:
        """Infer pre-BPE group lengths by decoding one BPE id at a time."""

        if not token_ids:
            return None

        lengths: List[int] = []
        for token_id in token_ids:
            events = self.decode_token_id_events(int(token_id))
            if not events:
                return None
            lengths.append(int(max(1, len(events))))

        return lengths

    def decode_token_id_events(self, token_id: int) -> List[str]:
        """Decode one token ID into constituent base token events."""

        key = int(token_id)
        cached = self._token_event_cache.get(key)
        if cached is not None:
            return list(cached)

        decode_token_ids = getattr(self.tokenizer, "decode_token_ids", None)
        complete_sequence = getattr(self.tokenizer, "complete_sequence", None)

        if callable(decode_token_ids):
            try:
                from miditok.classes import TokSequence

                seq = TokSequence(ids=[key])
                seq.are_ids_encoded = True
                decode_token_ids(seq)
                if callable(complete_sequence):
                    complete_sequence(seq)
                tokens = [str(t) for t in list(getattr(seq, "tokens", []) or [])]
                if tokens:
                    self._token_event_cache[key] = list(tokens)
                    return tokens
            except Exception:
                pass

        id_to_token = getattr(self.tokenizer, "id_to_token", None)
        if callable(id_to_token):
            try:
                token_name = str(id_to_token(key) or "")
                if token_name:
                    self._token_event_cache[key] = [token_name]
                    return [token_name]
            except Exception:
                pass

        return []

    @staticmethod
    def _compress_features_by_groups(
        values: Sequence[float],
        group_lengths: Sequence[int],
    ) -> List[float]:
        """Compress event features by groups using first event per group."""

        if not group_lengths:
            return []

        seq = list(values)
        if not seq:
            return [0.0] * len(group_lengths)

        out: List[float] = []
        cursor = 0
        last_index = len(seq) - 1
        for group_len in group_lengths:
            idx = min(max(0, cursor), last_index)
            out.append(float(seq[idx]))
            cursor += max(1, int(group_len))
        return out

    def decode(self, tokens: Sequence[int], output_path: Path) -> None:
        """Decode token IDs back to a MIDI file."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        token_ids = [int(t) for t in tokens]

        if not token_ids:
            raise ValueError("Cannot decode empty token list.")

        candidates: List[Any] = [[token_ids], token_ids]

        for token_input in candidates:
            if self._try_decode_with_methods(token_input, output_path):
                return

        raise RuntimeError("Failed to decode tokens with available MidiTok methods.")

    def save(self, path: str) -> None:
        """Persist tokenizer parameters to disk."""

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(self.tokenizer, "save"):
            try:
                self.tokenizer.save(str(save_path))
                return
            except Exception:
                pass

        if hasattr(self.tokenizer, "save_params"):
            self.tokenizer.save_params(str(save_path))
            return

        raise RuntimeError(
            "Tokenizer backend does not support saving with current API."
        )

    @classmethod
    def load(cls, path: str) -> "PianoTokenizer":
        """Load tokenizer from saved parameter file."""

        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {load_path}")

        strategy = "remi"
        try:
            import json

            payload = json.loads(load_path.read_text(encoding="utf-8"))
            tokenization = str(payload.get("tokenization", "REMI")).strip().lower()
            if tokenization.startswith("octuple"):
                strategy = "octuple"
        except Exception:
            strategy = "remi"

        tokenizer_cls = Octuple if strategy == "octuple" else REMI

        try:
            tok = tokenizer_cls(params=str(load_path))
            return cls(tok, strategy=strategy)
        except Exception:
            pass

        tokenizer_config = TokenizerConfig(
            num_velocities=32,
            use_tempos=True,
            use_time_signatures=True,
            use_chords=False,
            use_sustain_pedals=True,
            use_programs=False,
        )
        tok = tokenizer_cls(tokenizer_config=tokenizer_config)
        load_params = getattr(tok, "load_params", None)
        if callable(load_params):
            load_params(str(load_path))
            return cls(tok, strategy=strategy)

        raise RuntimeError(f"Unable to load tokenizer from path: {load_path}")

    @property
    def vocab_size(self) -> int:
        """Return tokenizer vocabulary size."""

        val = getattr(self.tokenizer, "vocab_size", None)
        if callable(val):
            computed = val()
            if isinstance(computed, (int, np.integer)):
                return int(computed)
        if isinstance(val, int):
            return val
        try:
            return len(self.tokenizer)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Unable to determine tokenizer vocab size.") from exc

    def verify_roundtrip(self, midi_path: Path) -> bool:
        """Check encode/decode roundtrip fidelity against source MIDI."""

        try:
            pretty_midi = _import_pretty_midi()
            orig_midi = pretty_midi.PrettyMIDI(str(midi_path))
            orig_notes = sum(
                len(inst.notes) for inst in orig_midi.instruments if not inst.is_drum
            )
            orig_duration = float(orig_midi.get_end_time())

            tokens = self.encode(midi_path)
            with tempfile.TemporaryDirectory() as tmp_dir:
                recon_path = Path(tmp_dir) / "roundtrip.mid"
                self.decode(tokens, recon_path)
                recon_midi = pretty_midi.PrettyMIDI(str(recon_path))

            recon_notes = sum(
                len(inst.notes) for inst in recon_midi.instruments if not inst.is_drum
            )
            recon_duration = float(recon_midi.get_end_time())

            notes_ok = (
                self._relative_diff(orig_notes, recon_notes) <= ROUNDTRIP_TOLERANCE
            )
            dur_ok = (
                self._relative_diff(orig_duration, recon_duration)
                <= ROUNDTRIP_TOLERANCE
            )
            passed = bool(notes_ok and dur_ok)

            if not passed:
                warnings.warn(
                    "Tokenizer roundtrip check failed for "
                    f"{midi_path}. note_count(orig={orig_notes}, recon={recon_notes}), "
                    f"duration(orig={orig_duration:.3f}, recon={recon_duration:.3f})"
                )
            return passed
        except Exception as exc:
            warnings.warn(f"Roundtrip verification failed for {midi_path}: {exc}")
            return False

    def _try_decode_with_methods(self, token_input: Any, output_path: Path) -> bool:
        if hasattr(self.tokenizer, "decode"):
            try:
                maybe_score = self.tokenizer.decode(token_input)
                if self._persist_decoded_output(maybe_score, output_path):
                    return True
            except Exception:
                pass

            try:
                self.tokenizer.decode(token_input, output_path=str(output_path))
                if output_path.exists():
                    return True
            except Exception:
                pass

        if not hasattr(self.tokenizer, "decode") and hasattr(
            self.tokenizer, "tokens_to_midi"
        ):
            try:
                maybe_midi = self.tokenizer.tokens_to_midi(token_input)
                if self._persist_decoded_output(maybe_midi, output_path):
                    return True
            except Exception:
                pass

            try:
                self.tokenizer.tokens_to_midi(token_input, output_path=str(output_path))
                if output_path.exists():
                    return True
            except Exception:
                pass

        return output_path.exists()

    def _encode_raw(self, midi_path: Path, encode_ids: bool) -> Any:
        """Call tokenizer backend with compatibility fallbacks."""

        path_str = str(midi_path)
        if hasattr(self.tokenizer, "encode"):
            try:
                return self.tokenizer.encode(path_str, encode_ids=encode_ids)
            except TypeError:
                try:
                    return self.tokenizer.encode(path_str)
                except Exception:
                    return self.tokenizer.encode(midi_path)

        if callable(self.tokenizer):
            return self.tokenizer(path_str)
        if hasattr(self.tokenizer, "midi_to_tokens"):
            return self.tokenizer.midi_to_tokens(path_str)
        raise RuntimeError("Tokenizer has no recognized encode method.")

    @staticmethod
    def _coerce_tok_sequence(encoded: Any) -> Any:
        """Extract first TokSequence-like object from backend output."""

        if isinstance(encoded, list):
            if not encoded:
                raise RuntimeError("Tokenizer returned empty token sequence list.")
            return encoded[0]
        return encoded

    def _encode_tok_sequence_ids(self, seq: Any) -> None:
        """Apply BPE/id encoding in place for a TokSequence-like object."""

        encode_token_ids = getattr(self.tokenizer, "encode_token_ids", None)
        if callable(encode_token_ids):
            encode_token_ids(seq)
            return

        apply_bpe = getattr(self.tokenizer, "apply_bpe", None)
        if callable(apply_bpe):
            apply_bpe(seq)
            return

    @staticmethod
    def _event_tick(event: Any) -> int:
        """Extract event tick from MidiTok event object."""

        raw = getattr(event, "time", 0)
        try:
            return max(0, int(float(raw)))
        except Exception:
            return 0

    @staticmethod
    def _tick_to_seconds(midi: Any, tick: int) -> float:
        """Convert MIDI tick to seconds using pretty_midi timing map."""

        try:
            return float(midi.tick_to_time(int(max(0, tick))))
        except Exception:
            return float(max(0, tick)) * 0.01

    def _event_duration_seconds(self, midi: Any, event: Any) -> float | None:
        """Estimate one event duration in seconds when available."""

        type_name = str(getattr(event, "type_", "") or "")
        if type_name.lower() != "duration":
            return None

        tick = self._event_tick(event)
        duration_ticks = self._parse_duration_ticks(event)
        if duration_ticks <= 0:
            return None

        start = self._tick_to_seconds(midi, tick)
        end = self._tick_to_seconds(midi, tick + duration_ticks)
        dur = float(max(1e-4, end - start))
        return dur

    @staticmethod
    def _parse_duration_ticks(event: Any) -> int:
        """Parse duration tick count from MidiTok duration event."""

        desc = str(getattr(event, "desc", "") or "")
        if "ticks" in desc:
            for chunk in desc.split():
                if chunk.isdigit():
                    return int(chunk)

        value = str(getattr(event, "value", "") or "")
        parts = value.split(".")
        for chunk in reversed(parts):
            if chunk.isdigit():
                return int(chunk)
        return 0

    @staticmethod
    def _resample_features(
        values: Sequence[float],
        target_len: int,
        monotonic: bool,
    ) -> List[float]:
        """Resample feature array to target length with interpolation."""

        if target_len <= 0:
            return []
        if not values:
            return [0.0] * target_len
        if len(values) == target_len:
            arr = np.asarray(values, dtype=np.float64)
        else:
            xp = np.linspace(0.0, 1.0, num=len(values), dtype=np.float64)
            x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float64)
            arr = np.interp(x, xp, np.asarray(values, dtype=np.float64))

        if monotonic:
            arr = np.maximum.accumulate(arr)
        return [float(v) for v in arr.tolist()]

    @staticmethod
    def _expand_aligned_features(
        values: Sequence[float],
        event_count: int,
        target_len: int,
    ) -> List[float]:
        """Expand event-aligned features to token-id length preserving local timing."""

        if target_len <= 0:
            return []
        if event_count <= 0 or not values:
            return [0.0] * target_len
        if event_count == target_len:
            return [float(v) for v in values]

        base = np.asarray(list(values), dtype=np.float64)
        if event_count > target_len:
            return PianoTokenizer._resample_features(
                base.tolist(), target_len, monotonic=False
            )

        repeats = np.full(event_count, target_len // event_count, dtype=np.int64)
        repeats[: target_len % event_count] += 1
        expanded = np.repeat(base, repeats)

        if expanded.shape[0] != target_len:
            expanded = np.resize(expanded, target_len)

        return [float(v) for v in expanded.tolist()]

    @staticmethod
    def _fallback_time_features(
        token_count: int,
        total_seconds: float,
    ) -> Tuple[List[float], List[float]]:
        """Create fallback onset/duration arrays when event timing is unavailable."""

        if token_count <= 0:
            return [], []

        total = float(max(1e-3, total_seconds))
        step = total / float(max(1, token_count))
        onsets = [float(i) * step for i in range(token_count)]
        durations = [float(max(1e-4, step)) for _ in range(token_count)]
        return onsets, durations

    @staticmethod
    def _persist_decoded_output(decoded: Any, output_path: Path) -> bool:
        if decoded is None:
            return output_path.exists()

        if hasattr(decoded, "dump_midi"):
            decoded.dump_midi(str(output_path))
            return output_path.exists()

        if hasattr(decoded, "write"):
            decoded.write(str(output_path))
            return output_path.exists()

        return False

    @staticmethod
    def _extract_token_ids(encoded: Any) -> List[int]:
        if encoded is None:
            return []

        if hasattr(encoded, "ids"):
            return [int(t) for t in encoded.ids]

        if isinstance(encoded, dict) and "ids" in encoded:
            return [int(t) for t in encoded["ids"]]

        if isinstance(encoded, (list, tuple)):
            if not encoded:
                return []
            if isinstance(encoded[0], (int, np.integer)):
                return [int(t) for t in encoded]

            ids: List[int] = []
            for item in encoded:
                ids.extend(PianoTokenizer._extract_token_ids(item))
            return ids

        raise TypeError(f"Unsupported encoded token container type: {type(encoded)}")

    @staticmethod
    def _relative_diff(a: float, b: float) -> float:
        denom = max(abs(a), 1e-8)
        return abs(a - b) / denom


def create_tokenizer(
    strategy: str = "custom_delta",
    **kwargs: Any,
) -> Any:
    """Create tokenizer instance from the unified tokenizer selector."""

    normalized = str(strategy).strip().lower()
    if normalized not in {"custom_delta", "delta", "quad", "event_quad", "unified"}:
        raise ValueError(
            "Unsupported tokenizer strategy "
            f"'{strategy}'. Only the unified CustomDeltaTokenizer is supported."
        )

    return CustomDeltaTokenizer(
        default_velocity=int(kwargs.get("default_velocity", 88)),
        include_special_tokens=bool(kwargs.get("include_special_tokens", False)),
        include_structural_meta_tokens=bool(
            kwargs.get("include_structural_meta_tokens", True)
        ),
        prepend_start_token=bool(kwargs.get("prepend_start_token", True)),
        density_quartiles=kwargs.get("density_quartiles"),
    )


def load_tokenizer(path: str | Path, strategy: str | None = None) -> Any:
    """Load unified tokenizer from disk."""

    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {load_path}")

    forced = str(strategy).strip().lower() if strategy is not None else ""
    if forced:
        if forced not in {"custom_delta", "delta", "quad", "event_quad", "unified"}:
            raise ValueError(
                "Unsupported tokenizer strategy "
                f"'{strategy}'. Only the unified CustomDeltaTokenizer is supported."
            )
        return CustomDeltaTokenizer.load(str(load_path))

    try:
        import json

        payload = json.loads(load_path.read_text(encoding="utf-8"))
        if str(payload.get("type", "")).strip() == "CustomDeltaTokenizer":
            return CustomDeltaTokenizer.load(str(load_path))
    except Exception:
        raise ValueError(
            "Failed to parse tokenizer payload. Expected a CustomDeltaTokenizer JSON file."
        )

    raise ValueError(
        "Unsupported tokenizer payload. Only CustomDeltaTokenizer is supported in this workspace."
    )


__all__ = [
    "CustomDeltaTokenizer",
    "create_tokenizer",
    "load_tokenizer",
]


def _import_pretty_midi() -> Any:
    try:
        import pretty_midi

        return pretty_midi
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pretty_midi import failed. This can happen if a broken fluidsynth Python "
            "package is installed on your system. Reinstall pretty_midi and/or remove "
            "conflicting fluidsynth bindings."
        ) from exc

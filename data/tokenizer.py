from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
from miditok import Octuple, REMI, TokenizerConfig


ROUNDTRIP_TOLERANCE = 0.05


class PianoTokenizer:
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

    def train(self, midi_paths: List[Path], vocab_size: int) -> None:
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
        path_str = str(midi_path)
        encoded: Any = None

        if hasattr(self.tokenizer, "encode"):
            try:
                encoded = self.tokenizer.encode(path_str)
            except Exception:
                encoded = self.tokenizer.encode(midi_path)
        elif callable(self.tokenizer):
            encoded = self.tokenizer(path_str)
        elif hasattr(self.tokenizer, "midi_to_tokens"):
            encoded = self.tokenizer.midi_to_tokens(path_str)
        else:
            raise RuntimeError("Tokenizer has no recognized encode method.")

        token_ids = self._extract_token_ids(encoded)
        if not token_ids:
            raise RuntimeError(f"Tokenizer produced no tokens for {midi_path}.")
        return token_ids

    def decode(self, tokens: Sequence[int], output_path: Path) -> None:
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


def _import_pretty_midi():
    try:
        import pretty_midi

        return pretty_midi
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pretty_midi import failed. This can happen if a broken fluidsynth Python "
            "package is installed on your system. Reinstall pretty_midi and/or remove "
            "conflicting fluidsynth bindings."
        ) from exc

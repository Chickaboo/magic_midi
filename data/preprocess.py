from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pretty_midi
from tqdm import tqdm

from utils.logging_utils import get_project_logger

try:
    from config import DataConfig
    from data.tokenizer import (
        CustomDeltaTokenizer,
        create_tokenizer,
        load_tokenizer,
    )
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from config import DataConfig
    from data.tokenizer import (
        CustomDeltaTokenizer,
        create_tokenizer,
        load_tokenizer,
    )


LOGGER = get_project_logger()
MAESTRO_DOWNLOAD_URL = "https://magenta.tensorflow.org/datasets/maestro"
SECONDS_PER_HOUR = 3600.0
DEFAULT_DATASET_PROFILES: Dict[str, Dict[str, Any]] = {
    "maestro": {
        "sample_weight": 1.5,
        "reference_raw_share": 0.056,
        "min_duration_seconds": 30.0,
        "filter_velocity": True,
        "min_note_count": 100,
        "min_distinct_pitches": 12,
        "piano_dominance_threshold": 0.70,
    },
    "giant_midi": {
        "sample_weight": 1.2,
        "reference_raw_share": 0.106,
        "min_duration_seconds": 30.0,
        "filter_velocity": True,
        "min_note_count": 100,
        "min_distinct_pitches": 12,
        "piano_dominance_threshold": 0.70,
    },
    "aria_midi": {
        "sample_weight": 1.0,
        "reference_raw_share": 0.796,
        "min_duration_seconds": 20.0,
        "filter_velocity": True,
        "min_note_count": 100,
        "min_distinct_pitches": 12,
        "piano_dominance_threshold": 0.70,
    },
    "adl_piano": {
        "sample_weight": 1.3,
        "reference_raw_share": 0.042,
        "min_duration_seconds": 15.0,
        "filter_velocity": False,
        "min_note_count": 100,
        "min_distinct_pitches": 12,
        "piano_dominance_threshold": 0.70,
    },
}

# DATASET WEIGHT PHILOSOPHY (v2):
#
# Weights are relative; aria_midi=1.0 is the anchor.
#
# aria_midi (1.0): Primary dataset. Largest curated corpus for generative piano.
# adl_piano (1.3): Slight oversample for pop/TV/modern style diversity.
# maestro (1.5): Slight oversample for timing precision quality signal.
# giant_midi (1.2): Slight oversample for composer and repertoire breadth.
#
# Combined effective data distribution (approximate):
#   aria_midi:   ~75%
#   maestro:     ~8%
#   giant_midi:  ~12%
#   adl_piano:   ~5%


@dataclass
class DatasetSpec:
    """Specification for one dataset source used in preprocessing."""

    name: str
    path: str
    type: str


def _to_float(value: Any, fallback: float) -> float:
    """Convert numeric value to float with fallback."""

    try:
        return float(value)
    except Exception:
        return float(fallback)


def _to_int(value: Any, fallback: int) -> int:
    """Convert numeric value to int with fallback."""

    try:
        return int(value)
    except Exception:
        return int(fallback)


def _to_bool(value: Any, fallback: bool) -> bool:
    """Convert bool-like value with fallback."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(fallback)


def _normalize_dataset_name(name: str) -> str:
    """Normalize dataset aliases to canonical profile keys."""

    key = str(name or "").strip().lower()
    alias_map = {
        "maestro": "maestro",
        "giant-midi": "giant_midi",
        "giantmidi": "giant_midi",
        "giant_midi": "giant_midi",
        "aria-midi": "aria_midi",
        "aria": "aria_midi",
        "aria_midi": "aria_midi",
        "adl": "adl_piano",
        "adl-piano": "adl_piano",
        "adl_piano": "adl_piano",
        "piano-e": "adl_piano",
        "pianoe": "adl_piano",
        "piano_e": "adl_piano",
    }
    return alias_map.get(key, key)


def _dataset_profile(config: DataConfig, source_name: str) -> Dict[str, Any]:
    """Return merged dataset profile (defaults + user overrides)."""

    canonical = _normalize_dataset_name(source_name)
    base = dict(
        DEFAULT_DATASET_PROFILES.get(
            canonical,
            {
                "sample_weight": 1.0,
                "min_duration_seconds": float(config.min_duration_seconds),
                "filter_velocity": bool(config.quality_filter_velocity),
                "min_note_count": int(config.min_note_count),
                "min_distinct_pitches": int(config.min_distinct_pitches),
                "piano_dominance_threshold": float(config.piano_dominance_threshold),
            },
        )
    )

    override = (config.dataset_profiles or {}).get(source_name)
    if isinstance(override, dict):
        base.update(override)

    return {
        "sample_weight": _to_float(base.get("sample_weight", 1.0), 1.0),
        "min_duration_seconds": _to_float(
            base.get("min_duration_seconds", config.min_duration_seconds),
            float(config.min_duration_seconds),
        ),
        "filter_velocity": _to_bool(
            base.get("filter_velocity", config.quality_filter_velocity),
            bool(config.quality_filter_velocity),
        ),
        "min_note_count": _to_int(
            base.get("min_note_count", config.min_note_count),
            int(config.min_note_count),
        ),
        "min_distinct_pitches": _to_int(
            base.get("min_distinct_pitches", config.min_distinct_pitches),
            int(config.min_distinct_pitches),
        ),
        "piano_dominance_threshold": _to_float(
            base.get("piano_dominance_threshold", config.piano_dominance_threshold),
            float(config.piano_dominance_threshold),
        ),
    }


def _normalize_rel_path(path_value: str) -> str:
    """Normalize relative MIDI path keys for metadata lookup."""

    return path_value.replace("\\", "/").lstrip("./").lower()


def _safe_int(value: Any) -> Optional[int]:
    """Return integer conversion or None when conversion fails."""

    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    """Return float conversion or fallback when conversion fails."""

    try:
        return float(value)
    except Exception:
        return float(fallback)


def _scan_midi_files(root: Path) -> List[Path]:
    """Recursively scan a root for `.mid` and `.midi` files."""

    return sorted(
        list(root.rglob("*.midi")) + list(root.rglob("*.mid")),
        key=lambda p: str(p),
    )


def _load_maestro_metadata(
    maestro_root: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load MAESTRO metadata indexed by relative path and basename."""

    metadata_by_rel: Dict[str, Dict[str, Any]] = {}
    metadata_by_name: Dict[str, Dict[str, Any]] = {}

    csv_candidates = [
        maestro_root / "maestro-v3.0.0.csv",
        maestro_root.parent / "maestro-v3.0.0.csv",
    ]

    csv_path = None
    for candidate in csv_candidates:
        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        return metadata_by_rel, metadata_by_name

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                midi_filename = str(row.get("midi_filename", "") or "")
                if not midi_filename:
                    continue

                rel_key = _normalize_rel_path(midi_filename)
                base_key = Path(midi_filename).name.lower()
                metadata = {
                    "composer": str(row.get("canonical_composer", "") or "unknown"),
                    "year": _safe_int(row.get("year")),
                    "title": str(row.get("canonical_title", "") or ""),
                }
                metadata_by_rel[rel_key] = metadata
                if base_key and base_key not in metadata_by_name:
                    metadata_by_name[base_key] = metadata
    except Exception as exc:
        LOGGER.warning("Failed to read MAESTRO metadata CSV (%s)", exc)
        return {}, {}

    LOGGER.info(
        "Loaded MAESTRO metadata with %d entries from %s",
        len(metadata_by_rel),
        csv_path.resolve(),
    )
    return metadata_by_rel, metadata_by_name


def _is_piano_only(midi: pretty_midi.PrettyMIDI) -> bool:
    """Return True when file appears to be piano-only performance data."""

    melodic = [inst for inst in midi.instruments if not inst.is_drum]
    if not melodic:
        return False

    if len(melodic) == 1:
        return True

    allowed_programs = {0, 1}
    for inst in melodic:
        if int(inst.program) not in allowed_programs:
            return False
    return True


def _piano_dominance_ratio(midi: pretty_midi.PrettyMIDI) -> float:
    """Compute ratio of piano notes among all non-drum notes."""

    total_notes = 0
    piano_notes = 0
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        note_count = len(inst.notes)
        total_notes += note_count
        if int(inst.program) in {0, 1}:
            piano_notes += note_count
    if total_notes <= 0:
        return 0.0
    return float(piano_notes) / float(total_notes)


def _note_count(midi: pretty_midi.PrettyMIDI) -> int:
    """Count non-drum notes in MIDI."""

    total = 0
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        total += len(inst.notes)
    return int(total)


def _distinct_pitch_count(midi: pretty_midi.PrettyMIDI) -> int:
    """Count distinct non-drum pitches in MIDI."""

    pitches: set[int] = set()
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            pitches.add(int(note.pitch))
    return int(len(pitches))


def _velocity_std(midi: pretty_midi.PrettyMIDI) -> float:
    """Return standard deviation of note velocities for non-drum notes."""

    velocities: List[int] = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        velocities.extend(int(n.velocity) for n in inst.notes)
    if not velocities:
        return 0.0
    return float(np.std(np.asarray(velocities, dtype=np.float64)))


def _quality_check(
    midi_path: Path,
    config: DataConfig,
    profile: Dict[str, Any],
) -> Tuple[bool, float, float, str]:
    """Apply quality filters and return `(keep, duration, vel_std, reason)`."""

    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        return False, 0.0, 0.0, f"parse_error: {exc}"

    duration = float(midi.get_end_time())
    min_duration = float(
        profile.get("min_duration_seconds", config.min_duration_seconds)
    )
    if duration < min_duration:
        return False, duration, 0.0, "too_short_seconds"

    if not _is_piano_only(midi):
        ratio = _piano_dominance_ratio(midi)
        threshold = float(
            profile.get("piano_dominance_threshold", config.piano_dominance_threshold)
        )
        if ratio < threshold:
            return False, duration, 0.0, "not_piano_dominant"

    min_notes = int(profile.get("min_note_count", config.min_note_count))
    if _note_count(midi) < min_notes:
        return False, duration, 0.0, "too_few_notes"

    min_pitches = int(profile.get("min_distinct_pitches", config.min_distinct_pitches))
    if _distinct_pitch_count(midi) < min_pitches:
        return False, duration, 0.0, "too_few_distinct_pitches"

    vel_std = _velocity_std(midi)
    velocity_filter = bool(
        profile.get("filter_velocity", config.quality_filter_velocity)
    )
    if velocity_filter and vel_std < 5.0:
        return False, duration, vel_std, "uniform_velocity"

    return True, duration, vel_std, "ok"


def _manifest_entry(
    *,
    source_name: str,
    source_root: Path,
    midi_path: Path,
    token_path: Path,
    onset_path: Path,
    duration_path: Path,
    token_length: int,
    duration_seconds: float,
    composer: str,
    year: Optional[int],
    title: str,
) -> Dict[str, object]:
    """Create one manifest entry with v3 multi-dataset fields."""

    piece_id = hashlib.sha1(str(midi_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return {
        "piece_id": piece_id,
        "path": str(midi_path.resolve()),
        "source_path": str(midi_path.resolve()),
        "source_root": str(source_root.resolve()),
        "tokens_path": str(token_path.resolve()),
        "onset_times_path": str(onset_path.resolve()),
        "durations_path": str(duration_path.resolve()),
        "tokens": int(token_length),
        "length": int(token_length),
        "composer": composer,
        "year": year,
        "title": title,
        "source": source_name,
        "duration": float(duration_seconds),
        "total_duration": float(duration_seconds),
    }


def create_seed_pairs(
    token_sequence: Sequence[int],
    config: DataConfig,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create `(seed, continuation)` training pairs from one token sequence."""

    seq = np.asarray(token_sequence, dtype=np.int64)
    total_window = config.seed_length + config.continuation_length
    if seq.size < total_window:
        return []

    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    max_start = seq.size - total_window
    for start in range(0, max_start + 1, config.stride):
        seed = seq[start : start + config.seed_length]
        continuation = seq[
            start + config.seed_length : start
            + config.seed_length
            + config.continuation_length
        ]
        pairs.append((seed.copy(), continuation.copy()))
    return pairs


def _build_dataset_specs(config: DataConfig) -> List[DatasetSpec]:
    """Build dataset specifications from config flags."""

    if bool(config.use_multi_dataset):
        specs: List[DatasetSpec] = []
        if config.dataset_paths:
            for name, path in config.dataset_paths.items():
                if not path:
                    continue
                dataset_type = (
                    "maestro" if name.strip().lower() == "maestro" else "generic"
                )
                specs.append(DatasetSpec(name=name, path=path, type=dataset_type))
        if not specs:
            specs.append(
                DatasetSpec(name="maestro", path=config.maestro_path, type="maestro")
            )
        return specs

    return [DatasetSpec(name="maestro", path=config.maestro_path, type="maestro")]


def _infer_piece_onset_duration(
    onsets: Sequence[float],
    durations: Sequence[float],
) -> float:
    """Infer total piece duration from token-aligned onset/duration arrays."""

    if not onsets or not durations:
        return 0.0
    n = min(len(onsets), len(durations))
    if n <= 0:
        return 0.0
    tail = float(onsets[n - 1]) + float(max(1e-4, durations[n - 1]))
    return max(0.0, tail)


def _resolve_composer_year_title(
    midi_path: Path,
    root: Path,
    rel_meta: Dict[str, Dict[str, Any]],
    name_meta: Dict[str, Dict[str, Any]],
) -> Tuple[str, Optional[int], str]:
    """Resolve composer/year/title metadata for one MIDI path."""

    rel_key = ""
    try:
        rel_key = _normalize_rel_path(str(midi_path.relative_to(root)))
    except Exception:
        rel_key = _normalize_rel_path(str(midi_path))

    meta = rel_meta.get(rel_key) or name_meta.get(midi_path.name.lower())
    composer = "unknown"
    year = None
    title = ""
    if meta is not None:
        composer = str(meta.get("composer", "unknown") or "unknown")
        year = _safe_int(meta.get("year"))
        title = str(meta.get("title", "") or "")
    return composer, year, title


def _default_dataset_weights_for_specs(
    specs: Sequence[DatasetSpec],
) -> Dict[str, float]:
    """Build default weight mapping from canonical dataset profiles."""

    weights: Dict[str, float] = {}
    for spec in specs:
        canonical = _normalize_dataset_name(spec.name)
        profile = DEFAULT_DATASET_PROFILES.get(canonical, {})
        weights[spec.name] = float(profile.get("sample_weight", 1.0))
    return weights


def _summarize_rejections(reject_reasons: Dict[str, int]) -> None:
    """Log quality-filter rejection reasons summary."""

    if not reject_reasons:
        return
    LOGGER.info("Filtering rejects by reason:")
    for reason, count in sorted(
        reject_reasons.items(), key=lambda kv: kv[1], reverse=True
    ):
        LOGGER.info("  %-28s %8d", f"{reason}:", int(count))


def _format_dataset_summary_line(
    source: str,
    pieces: int,
    tokens: int,
    hours: float,
    weight: float,
) -> str:
    """Format one dataset summary line with weight annotation."""

    return (
        f"  {source:<10} {pieces:6d} pieces  |  {tokens:12d} tokens"
        f"  |  ~{hours:7.1f} hours  |  weight={weight:.1f}"
    )


class MultiDatasetPreprocessor:
    """Preprocess MAESTRO + optional external MIDI datasets into one manifest."""

    def __init__(
        self,
        config: Optional[DataConfig] = None,
        datasets: Optional[List[Dict[str, str]]] = None,
        dry_run: bool = False,
    ) -> None:
        self.config = config or DataConfig()
        if dry_run:
            self.config.use_multi_dataset = True
            if not self.config.dataset_paths:
                placeholder = str(self.config.maestro_path)
                self.config.dataset_paths = {
                    "aria_midi": placeholder,
                    "maestro": placeholder,
                    "giant_midi": placeholder,
                    "adl_piano": placeholder,
                }

        if datasets is None:
            self.datasets = _build_dataset_specs(self.config)
        else:
            self.datasets = [
                DatasetSpec(
                    name=str(d.get("name", "")).strip(),
                    path=str(d.get("path", "")).strip(),
                    type=str(d.get("type", "generic")).strip().lower() or "generic",
                )
                for d in datasets
            ]
        self.processed_dir = Path(self.config.processed_path)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def print_weight_distribution(self) -> Dict[str, float]:
        """Print approximate effective distribution from configured dataset weights."""

        weights = self._dataset_weights()

        weighted_raw: Dict[str, float] = {}
        for name, weight in sorted(weights.items(), key=lambda kv: kv[0]):
            canonical = _normalize_dataset_name(name)
            profile = DEFAULT_DATASET_PROFILES.get(canonical, {})
            raw_share = float(profile.get("reference_raw_share", 1.0))
            weighted_raw[name] = float(max(0.0, raw_share) * float(weight))

        total = float(sum(weighted_raw.values()))
        if total <= 0.0:
            total = float(sum(weights.values()))
            if total <= 0.0:
                raise ValueError("Total dataset weight must be positive.")
            probs = {
                name: float(weight / total)
                for name, weight in sorted(weights.items(), key=lambda kv: kv[0])
            }
        else:
            probs = {name: float(value / total) for name, value in weighted_raw.items()}

        print("Effective data distribution:")
        for name in ("aria_midi", "maestro", "giant_midi", "adl_piano"):
            if name not in probs:
                continue
            print(f"  {name:<10}: ~{probs[name] * 100.0:5.1f}%")
        return probs

    def _fit_or_load_tokenizer(self, all_midi_paths: List[Path]) -> CustomDeltaTokenizer:
        tokenizer_path = Path(self.config.tokenizer_path)
        strategy = str(getattr(self.config, "tokenization_strategy", "custom_delta")).lower()
        if tokenizer_path.exists():
            LOGGER.info("Loading tokenizer from %s", tokenizer_path.resolve())
            tokenizer = load_tokenizer(
                tokenizer_path,
                strategy=strategy,
            )

            # Auto-upgrade legacy custom-delta payloads that lack structural meta-prefix.
            if (
                strategy in {"custom_delta", "delta", "quad", "event_quad"}
                and isinstance(tokenizer, CustomDeltaTokenizer)
                and not bool(getattr(tokenizer, "include_structural_meta_tokens", False))
            ):
                LOGGER.info(
                    "Loaded legacy CustomDeltaTokenizer without structural meta-prefix; upgrading and recalibrating quartiles."
                )
                tokenizer = CustomDeltaTokenizer(
                    default_velocity=int(getattr(tokenizer, "default_velocity", 88)),
                    include_special_tokens=bool(
                        getattr(tokenizer, "include_special_tokens", False)
                    ),
                    include_structural_meta_tokens=True,
                    prepend_start_token=True,
                    density_quartiles=getattr(tokenizer, "density_quartiles", None),
                )
                tokenizer.train(
                    midi_paths=all_midi_paths,
                    vocab_size=self.config.vocab_size,
                )
                tokenizer.save(str(tokenizer_path))

            if isinstance(tokenizer, CustomDeltaTokenizer):
                q1, q2, q3 = tokenizer.density_quartiles
                LOGGER.info(
                    "CustomDeltaTokenizer quartiles (notes/sec): q25=%.3f q50=%.3f q75=%.3f",
                    q1,
                    q2,
                    q3,
                )
            return tokenizer

        LOGGER.info(
            "Tokenizer not found. Training %s tokenizer on %d files.",
            strategy.upper(),
            len(all_midi_paths),
        )
        tokenizer = create_tokenizer(strategy=strategy)
        tokenizer.train(midi_paths=all_midi_paths, vocab_size=self.config.vocab_size)
        tokenizer.save(str(tokenizer_path))
        LOGGER.info("Saved tokenizer to %s", tokenizer_path.resolve())
        if isinstance(tokenizer, CustomDeltaTokenizer):
            q1, q2, q3 = tokenizer.density_quartiles
            LOGGER.info(
                "CustomDeltaTokenizer quartiles (notes/sec): q25=%.3f q50=%.3f q75=%.3f",
                q1,
                q2,
                q3,
            )
        return tokenizer

    def _dataset_weights(self) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        defaults = _default_dataset_weights_for_specs(self.datasets)
        for spec in self.datasets:
            base = _safe_float(
                self.config.dataset_weights.get(
                    spec.name, defaults.get(spec.name, 1.0)
                ),
                fallback=defaults.get(spec.name, 1.0),
            )
            if base <= 0.0:
                raise ValueError(
                    f"dataset_weights['{spec.name}'] must be > 0, got {base}."
                )
            weights[spec.name] = base
        return weights

    def preprocess(self) -> Dict[str, float]:
        """Run preprocessing and write unified manifest and sampling metadata."""

        if not self.datasets:
            raise ValueError("No datasets configured for preprocessing.")

        expanded: List[Tuple[DatasetSpec, Path]] = []
        for spec in self.datasets:
            if not spec.name:
                raise ValueError("Dataset config is missing non-empty 'name'.")
            root = Path(spec.path)
            if not root.exists():
                if spec.type == "maestro":
                    raise FileNotFoundError(
                        f"MAESTRO path not found: {root.resolve()}. "
                        f"Download MAESTRO from: {MAESTRO_DOWNLOAD_URL}"
                    )
                raise FileNotFoundError(
                    f"Dataset '{spec.name}' path not found: {root.resolve()}. "
                    "Update data.dataset_paths with a valid location."
                )
            expanded.append((spec, root))

        all_midi_paths: List[Path] = []
        scanned_by_dataset: Dict[str, List[Path]] = {}
        for spec, root in expanded:
            midi_paths = _scan_midi_files(root)
            if not midi_paths:
                LOGGER.warning(
                    "Dataset '%s' has no MIDI files under %s", spec.name, root
                )
            scanned_by_dataset[spec.name] = midi_paths
            all_midi_paths.extend(midi_paths)

        if not all_midi_paths:
            raise RuntimeError("No MIDI files found across configured datasets.")

        tokenizer = self._fit_or_load_tokenizer(all_midi_paths)

        manifest: List[Dict[str, object]] = []
        lengths: List[int] = []
        unique_tokens: set[int] = set()
        total_tokens = 0

        by_source_count: Dict[str, int] = {}
        by_source_tokens: Dict[str, int] = {}
        by_source_seconds: Dict[str, float] = {}
        reject_reasons: Dict[str, int] = {}

        metadata_cache: Dict[
            str, Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]
        ] = {}
        for spec, root in expanded:
            if spec.type == "maestro":
                metadata_cache[spec.name] = _load_maestro_metadata(root)

        for spec, root in expanded:
            midi_paths = scanned_by_dataset[spec.name]
            rel_meta, name_meta = metadata_cache.get(spec.name, ({}, {}))
            profile = _dataset_profile(self.config, spec.name)
            desc = f"Tokenizing {spec.name}"
            for midi_path in tqdm(midi_paths, desc=desc):
                keep, duration, _vel_std, reason = _quality_check(
                    midi_path,
                    self.config,
                    profile,
                )
                if not keep:
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                    continue

                try:
                    tokens, onset_times, durations = (
                        tokenizer.encode_with_time_features(midi_path)
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Skipping %s: tokenization failed (%s)",
                        midi_path,
                        exc,
                    )
                    continue

                length = len(tokens)
                if length < int(self.config.min_piece_length):
                    continue

                file_hash = hashlib.sha1(
                    str(midi_path.resolve()).encode("utf-8")
                ).hexdigest()[:16]
                out_file = self.processed_dir / f"{file_hash}.npy"
                np.save(out_file, np.asarray(tokens, dtype=np.int64))
                onset_file = self.processed_dir / f"{file_hash}_onset.npy"
                duration_file = self.processed_dir / f"{file_hash}_duration.npy"
                np.save(onset_file, np.asarray(onset_times, dtype=np.float32))
                np.save(duration_file, np.asarray(durations, dtype=np.float32))

                composer, year, title = _resolve_composer_year_title(
                    midi_path,
                    root,
                    rel_meta,
                    name_meta,
                )

                piece_duration = _infer_piece_onset_duration(onset_times, durations)
                if piece_duration <= 0.0:
                    piece_duration = duration

                entry = _manifest_entry(
                    source_name=spec.name,
                    source_root=root,
                    midi_path=midi_path,
                    token_path=out_file,
                    onset_path=onset_file,
                    duration_path=duration_file,
                    token_length=length,
                    duration_seconds=piece_duration,
                    composer=composer,
                    year=year,
                    title=title,
                )
                manifest.append(entry)

                lengths.append(length)
                total_tokens += length
                unique_tokens.update(int(t) for t in tokens)

                by_source_count[spec.name] = by_source_count.get(spec.name, 0) + 1
                by_source_tokens[spec.name] = (
                    by_source_tokens.get(spec.name, 0) + length
                )
                by_source_seconds[spec.name] = (
                    by_source_seconds.get(spec.name, 0.0) + piece_duration
                )

        if not manifest:
            raise RuntimeError(
                "All pieces were filtered out. Lower quality thresholds or verify dataset integrity."
            )

        weights = self._dataset_weights()
        weighted_sizes: Dict[str, float] = {}
        weighted_total = 0.0
        for source_name in by_source_count:
            weighted = float(by_source_count[source_name]) * float(
                weights.get(source_name, 1.0)
            )
            weighted_sizes[source_name] = weighted
            weighted_total += weighted

        sampling_probs: Dict[str, float] = {}
        if weighted_total > 0:
            for source_name, weighted in weighted_sizes.items():
                sampling_probs[source_name] = float(weighted / weighted_total)
        else:
            uniform = 1.0 / max(1, len(by_source_count))
            for source_name in by_source_count:
                sampling_probs[source_name] = uniform

        manifest_path = self.processed_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        sampling_path = self.processed_dir / "dataset_sampling.json"
        with sampling_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_weights": weights,
                    "dataset_counts": by_source_count,
                    "dataset_profile": {
                        spec.name: _dataset_profile(self.config, spec.name)
                        for spec in self.datasets
                    },
                    "dataset_sampling_probabilities": sampling_probs,
                },
                f,
                indent=2,
            )

        kept_pieces = len(lengths)
        mean_len = float(mean(lengths)) if lengths else 0.0
        min_len = int(min(lengths)) if lengths else 0
        max_len = int(max(lengths)) if lengths else 0
        coverage = float(len(unique_tokens) / max(tokenizer.vocab_size, 1))

        LOGGER.info("Multi-dataset preprocessing complete:")
        for spec, _root in expanded:
            source = spec.name
            pieces = int(by_source_count.get(source, 0))
            tokens = int(by_source_tokens.get(source, 0))
            hours = float(by_source_seconds.get(source, 0.0) / SECONDS_PER_HOUR)
            weight = float(weights.get(source, 1.0))
            LOGGER.info(
                "%s",
                _format_dataset_summary_line(
                    source=f"{source}:",
                    pieces=pieces,
                    tokens=tokens,
                    hours=hours,
                    weight=weight,
                ),
            )
        LOGGER.info(
            "%s",
            _format_dataset_summary_line(
                source="TOTAL:",
                pieces=kept_pieces,
                tokens=total_tokens,
                hours=float(sum(by_source_seconds.values()) / SECONDS_PER_HOUR),
                weight=1.0,
            ),
        )
        _summarize_rejections(reject_reasons)

        LOGGER.info("Preprocessing summary")
        LOGGER.info("  Total MIDI files scanned: %d", len(all_midi_paths))
        LOGGER.info("  Total pieces kept: %d", kept_pieces)
        LOGGER.info("  Total tokens: %d", total_tokens)
        LOGGER.info(
            "  Piece length mean/min/max: %.1f/%d/%d",
            mean_len,
            min_len,
            max_len,
        )
        LOGGER.info(
            "  Vocabulary coverage: %.2f%% (%d/%d)",
            coverage * 100.0,
            len(unique_tokens),
            tokenizer.vocab_size,
        )
        LOGGER.info("  Manifest saved: %s", manifest_path.resolve())
        LOGGER.info("  Sampling metadata saved: %s", sampling_path.resolve())

        return {
            "total_scanned": float(len(all_midi_paths)),
            "total_kept": float(kept_pieces),
            "total_tokens": float(total_tokens),
            "mean_length": mean_len,
            "min_length": float(min_len),
            "max_length": float(max_len),
            "vocab_coverage": coverage,
        }


def preprocess_maestro(config: DataConfig) -> Dict[str, float]:
    """Backward-compatible entrypoint for MAESTRO preprocessing."""

    processor = MultiDatasetPreprocessor(
        config,
        datasets=[
            {
                "name": "maestro",
                "path": str(config.maestro_path),
                "type": "maestro",
            }
        ],
    )
    return processor.preprocess()


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for preprocessing."""

    parser = argparse.ArgumentParser(
        description="Preprocess MIDI datasets into tokenized arrays and manifest."
    )
    parser.add_argument("--maestro_path", type=str, default=DataConfig().maestro_path)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=DataConfig().tokenizer_path,
    )
    parser.add_argument(
        "--processed_path",
        type=str,
        default=DataConfig().processed_path,
    )
    parser.add_argument("--vocab_size", type=int, default=DataConfig().vocab_size)
    parser.add_argument("--seed_length", type=int, default=DataConfig().seed_length)
    parser.add_argument(
        "--continuation_length",
        type=int,
        default=DataConfig().continuation_length,
    )
    parser.add_argument("--stride", type=int, default=DataConfig().stride)
    parser.add_argument(
        "--min_piece_length",
        type=int,
        default=DataConfig().min_piece_length,
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=DataConfig().max_sequence_length,
    )
    parser.add_argument(
        "--min_duration_seconds",
        type=float,
        default=DataConfig().min_duration_seconds,
    )
    parser.add_argument(
        "--disable_velocity_quality_filter",
        action="store_true",
        help="Disable velocity variance quality filtering.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    data_cfg = DataConfig(
        maestro_path=args.maestro_path,
        tokenizer_path=args.tokenizer_path,
        processed_path=args.processed_path,
        vocab_size=args.vocab_size,
        seed_length=args.seed_length,
        continuation_length=args.continuation_length,
        stride=args.stride,
        min_piece_length=args.min_piece_length,
        max_sequence_length=args.max_sequence_length,
        min_duration_seconds=float(args.min_duration_seconds),
        quality_filter_velocity=not bool(args.disable_velocity_quality_filter),
    )
    preprocess_maestro(data_cfg)
    LOGGER.info("Preprocessing completed successfully.")

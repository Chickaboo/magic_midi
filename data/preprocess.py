from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    from config import DataConfig
    from data.tokenizer import PianoTokenizer
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from config import DataConfig
    from data.tokenizer import PianoTokenizer


MAESTRO_DOWNLOAD_URL = "https://magenta.tensorflow.org/datasets/maestro"


def _normalize_rel_path(path_value: str) -> str:
    return path_value.replace("\\", "/").lstrip("./").lower()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _load_maestro_metadata(
    maestro_root: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
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
        print(f"Warning: failed to read MAESTRO metadata CSV ({exc})")
        return {}, {}

    print(
        "Loaded MAESTRO metadata: "
        f"{len(metadata_by_rel)} path entries from {csv_path.resolve()}"
    )
    return metadata_by_rel, metadata_by_name


def create_seed_pairs(
    token_sequence: Sequence[int], config: DataConfig
) -> List[Tuple[np.ndarray, np.ndarray]]:
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


def preprocess_maestro(config: DataConfig) -> Dict[str, float]:
    maestro_root = Path(config.maestro_path)
    if not maestro_root.exists():
        raise FileNotFoundError(
            f"MAESTRO path not found: {maestro_root.resolve()}. "
            f"Download MAESTRO from: {MAESTRO_DOWNLOAD_URL}"
        )

    midi_paths = sorted(
        list(maestro_root.rglob("*.midi")) + list(maestro_root.rglob("*.mid")),
        key=lambda p: str(p),
    )
    if not midi_paths:
        raise RuntimeError(f"No MIDI files found under {maestro_root.resolve()}.")

    tokenizer_path = Path(config.tokenizer_path)
    if tokenizer_path.exists():
        tokenizer = PianoTokenizer.load(str(tokenizer_path))
        print(f"Loaded tokenizer from {tokenizer_path.resolve()}")
    else:
        strategy = str(getattr(config, "tokenization_strategy", "remi")).lower()
        print(f"Tokenizer not found. Training {strategy.upper()}+BPE tokenizer...")
        tokenizer = PianoTokenizer(strategy=strategy)
        tokenizer.train(midi_paths=midi_paths, vocab_size=config.vocab_size)
        tokenizer.save(str(tokenizer_path))
        print(f"Saved tokenizer to {tokenizer_path.resolve()}")

    processed_dir = Path(config.processed_path)
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata_by_rel, metadata_by_name = _load_maestro_metadata(maestro_root)

    manifest: List[Dict[str, object]] = []
    lengths: List[int] = []
    unique_tokens: set[int] = set()
    total_tokens = 0

    for midi_path in tqdm(midi_paths, desc="Tokenizing MAESTRO"):
        try:
            tokens = tokenizer.encode(midi_path)
        except Exception as exc:
            print(f"Skipping {midi_path}: tokenization failed ({exc})")
            continue

        length = len(tokens)
        if length < config.min_piece_length:
            continue

        file_hash = hashlib.sha1(str(midi_path.resolve()).encode("utf-8")).hexdigest()[
            :16
        ]
        out_file = processed_dir / f"{file_hash}.npy"
        np.save(out_file, np.asarray(tokens, dtype=np.int64))

        rel_key = ""
        try:
            rel_key = _normalize_rel_path(str(midi_path.relative_to(maestro_root)))
        except Exception:
            rel_key = _normalize_rel_path(str(midi_path))
        meta = metadata_by_rel.get(rel_key) or metadata_by_name.get(
            midi_path.name.lower()
        )

        composer = "unknown"
        year = None
        title = ""
        if meta is not None:
            composer = str(meta.get("composer", "unknown") or "unknown")
            year = _safe_int(meta.get("year"))
            title = str(meta.get("title", "") or "")

        lengths.append(length)
        total_tokens += length
        unique_tokens.update(int(t) for t in tokens)

        manifest.append(
            {
                "piece_id": file_hash,
                "source_path": str(midi_path.resolve()),
                "tokens_path": str(out_file.resolve()),
                "length": int(length),
                "composer": composer,
                "year": year,
                "title": title,
            }
        )

    manifest_path = processed_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    kept_pieces = len(lengths)
    mean_len = float(mean(lengths)) if lengths else 0.0
    min_len = int(min(lengths)) if lengths else 0
    max_len = int(max(lengths)) if lengths else 0
    coverage = float(len(unique_tokens) / max(tokenizer.vocab_size, 1))

    print("\nPreprocessing summary")
    print(f"  Total MIDI files scanned: {len(midi_paths)}")
    print(f"  Total pieces kept: {kept_pieces}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Piece length mean/min/max: {mean_len:.1f}/{min_len}/{max_len}")
    print(
        f"  Vocabulary coverage: {coverage * 100:.2f}% ({len(unique_tokens)}/{tokenizer.vocab_size})"
    )
    print(f"  Manifest saved: {manifest_path.resolve()}")

    return {
        "total_scanned": float(len(midi_paths)),
        "total_kept": float(kept_pieces),
        "total_tokens": float(total_tokens),
        "mean_length": mean_len,
        "min_length": float(min_len),
        "max_length": float(max_len),
        "vocab_coverage": coverage,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess MAESTRO into tokenized numpy arrays."
    )
    parser.add_argument("--maestro_path", type=str, default=DataConfig().maestro_path)
    parser.add_argument(
        "--tokenizer_path", type=str, default=DataConfig().tokenizer_path
    )
    parser.add_argument(
        "--processed_path", type=str, default=DataConfig().processed_path
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
        "--min_piece_length", type=int, default=DataConfig().min_piece_length
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=DataConfig().max_sequence_length,
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
    )
    preprocess_maestro(data_cfg)
    print("Preprocessing completed successfully.")

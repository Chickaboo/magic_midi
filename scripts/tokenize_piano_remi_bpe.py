from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import hashlib
import json
import os
import shutil
import stat
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from data.tokenizer_remi_bpe import PianoREMIBPETokenizer
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from data.tokenizer_remi_bpe import PianoREMIBPETokenizer


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def is_midi_name(name: str) -> bool:
    lower = str(name).lower()
    return lower.endswith(".mid") or lower.endswith(".midi")


def safe_json_read(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def safe_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_manifest_entry(manifest_jsonl_path: Path, entry: Dict[str, Any]) -> None:
    manifest_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, separators=(",", ":")) + "\n")


def rebuild_manifest_json(manifest_jsonl_path: Path, manifest_json_path: Path) -> int:
    entries: List[Dict[str, Any]] = []
    if manifest_jsonl_path.exists():
        with manifest_jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = line.strip()
                if not row:
                    continue
                try:
                    entries.append(json.loads(row))
                except Exception:
                    continue
    safe_json_write(manifest_json_path, entries)
    return len(entries)


def load_manifest_state(manifest_jsonl_path: Path) -> Tuple[int, Set[int]]:
    if not manifest_jsonl_path.exists():
        return 0, set()
    count = 0
    indices: Set[int] = set()
    with manifest_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue
            count += 1
            try:
                parsed = json.loads(row)
            except Exception:
                continue
            idx = int(parsed.get("index", -1))
            if idx >= 0:
                indices.add(idx)
    return count, indices


def format_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or not np.isfinite(seconds):
        return "unknown"
    sec = int(seconds)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _rmtree_onerror(func: Any, path: str, exc_info: Any) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def remove_tree_strict(path: Path, retries: int = 3) -> None:
    if not path.exists():
        return
    for attempt in range(max(1, int(retries))):
        shutil.rmtree(path, onerror=_rmtree_onerror)
        if not path.exists():
            return
        time.sleep(0.2 * float(attempt + 1))
    if path.exists():
        raise OSError(f"--start-over requested but output root could not be deleted: {path}")


def scan_midi_files(input_dir: Path) -> List[Path]:
    return sorted(
        [p for p in input_dir.rglob("*") if p.is_file() and is_midi_name(p.name)],
        key=lambda p: str(p).lower(),
    )


def load_midi_files_from_jsonl(
    input_dir: Path,
    file_list_jsonl: Path,
    *,
    trust_file_list: bool,
) -> List[Path]:
    midi_files: List[Path] = []
    with file_list_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue
            try:
                parsed = json.loads(row)
            except Exception:
                continue
            rel_path = str(parsed.get("path", "")).strip()
            if not rel_path or not is_midi_name(rel_path):
                continue
            candidate = input_dir / rel_path
            if bool(trust_file_list) or (candidate.exists() and candidate.is_file()):
                midi_files.append(candidate)
    return midi_files


def save_npz_with_retries(
    *,
    out_npz_path: Path,
    tokens: np.ndarray,
    onsets: np.ndarray,
    durations: np.ndarray,
    use_compression: bool,
    retries: int = 5,
) -> Optional[str]:
    attempts = max(1, int(retries))
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            if out_npz_path.exists():
                try:
                    out_npz_path.unlink()
                except Exception:
                    pass
            if bool(use_compression):
                np.savez_compressed(
                    out_npz_path,
                    tokens=tokens,
                    onsets=onsets,
                    durations=durations,
                )
            else:
                np.savez(
                    out_npz_path,
                    tokens=tokens,
                    onsets=onsets,
                    durations=durations,
                )
            return None
        except (PermissionError, OSError) as exc:
            last_error = exc
            time.sleep(min(1.0, 0.05 * float(2**attempt)))
        except Exception as exc:
            last_error = exc
            break
    return f"{type(last_error).__name__}: {last_error}" if last_error else "unknown-write-error"


def build_npz_relative_path(idx: int, source_path: str, shard_size: int = 50000) -> str:
    shard = int(idx) // max(1, int(shard_size))
    md5 = md5_text(source_path)
    return f"data/{shard:05d}/{idx:07d}_{md5}.npz"


_WORKER_TOKENIZER: Optional[PianoREMIBPETokenizer] = None
_BPE_WORKER_TOKENIZER: Optional[PianoREMIBPETokenizer] = None


def _init_worker(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = PianoREMIBPETokenizer.load(tokenizer_path)


def _init_bpe_worker(config: Dict[str, Any]) -> None:
    global _BPE_WORKER_TOKENIZER
    _BPE_WORKER_TOKENIZER = PianoREMIBPETokenizer(
        vocab_size=int(config["vocab_size"]),
        positions_per_bar=int(config["positions_per_bar"]),
        max_duration_bars=int(config["max_duration_bars"]),
        tempo_bins=int(config["tempo_bins"]),
        include_special_tokens=True,
    )


def _worker_base_sequence(midi_path: str) -> Optional[List[int]]:
    tokenizer = _BPE_WORKER_TOKENIZER
    if tokenizer is None:
        return None
    try:
        ids, _onsets, _durations = tokenizer._base_sequence_with_features(Path(str(midi_path)))
    except Exception:
        return None
    if len(ids) < 4:
        return None
    return [int(token) for token in ids]


def _worker_tokenize(payload: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        idx,
        midi_path,
        relative_path,
        min_token_length,
        output_root,
        output_shard_size,
        use_compression,
    ) = payload
    tokenizer = _WORKER_TOKENIZER
    if tokenizer is None:
        return {
            "index": int(idx),
            "source_path": str(relative_path),
            "status": "error",
            "error": "worker-tokenizer-not-initialized",
        }
    try:
        tokens, onsets, durations = tokenizer.encode_with_time_features(Path(str(midi_path)))
    except Exception as exc:
        return {
            "index": int(idx),
            "source_path": str(relative_path),
            "status": "skip",
            "error": f"tokenize-failed: {exc}",
        }

    token_len = int(len(tokens))
    if token_len < int(min_token_length):
        return {
            "index": int(idx),
            "source_path": str(relative_path),
            "status": "skip",
            "error": "short",
        }

    rel_npz_path = build_npz_relative_path(
        idx=int(idx),
        source_path=str(relative_path),
        shard_size=int(output_shard_size),
    )
    out_npz_path = Path(str(output_root)) / rel_npz_path
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    write_error = save_npz_with_retries(
        out_npz_path=out_npz_path,
        tokens=np.asarray(tokens, dtype=np.int32),
        onsets=np.asarray(onsets, dtype=np.float32),
        durations=np.asarray(durations, dtype=np.float32),
        use_compression=bool(use_compression),
    )
    if write_error is not None:
        return {
            "index": int(idx),
            "source_path": str(relative_path),
            "status": "error",
            "error": f"write-failed: {write_error}",
        }

    return {
        "index": int(idx),
        "source_path": str(relative_path),
        "status": "accepted",
        "md5": md5_text(str(relative_path)),
        "npz_path": rel_npz_path,
        "length": int(token_len),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize solo piano MIDI files with PianoREMIBPE into sharded NPZ packs."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory of .mid/.midi files.")
    parser.add_argument(
        "--file-list-jsonl",
        type=str,
        default="",
        help="Optional Godzilla-style JSONL file with relative MIDI paths under --input-dir.",
    )
    parser.add_argument(
        "--trust-file-list",
        action="store_true",
        help="Trust --file-list-jsonl paths without probing every file during startup.",
    )
    parser.add_argument("--output-dir", "--output-root", dest="output_dir", type=str, default="processed/piano_remi_bpe")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--bpe-sample-size", type=int, default=20000)
    parser.add_argument(
        "--bpe-training-mode",
        choices=["fast", "iterative"],
        default="fast",
        help="fast counts high-frequency adjacent integer pairs once; iterative keeps the slower classic BPE loop.",
    )
    parser.add_argument(
        "--bpe-workers",
        type=int,
        default=0,
        help="Worker count for BPE sample parsing. 0 reuses --workers or os.cpu_count().",
    )
    parser.add_argument("--workers", type=int, default=0, help="Worker count. 0 uses os.cpu_count().")
    parser.add_argument("--min-token-length", type=int, default=192)
    parser.add_argument("--positions-per-bar", type=int, default=16)
    parser.add_argument("--max-duration-bars", type=int, default=4)
    parser.add_argument("--tempo-bins", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--output-shard-size", type=int, default=50000)
    parser.add_argument("--compress-output", action="store_true")
    parser.add_argument("--max-files", type=int, default=0, help="Stop after this many accepted files.")
    parser.add_argument("--start-index", type=int, default=-1)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--start-over", action="store_true")
    parser.add_argument("--retrain-tokenizer", action="store_true")
    return parser.parse_args()


def _iter_payloads(
    *,
    midi_files: Sequence[Path],
    input_dir: Path,
    start_index: int,
    end_index_exclusive: int,
    done_indices: Set[int],
    min_token_length: int,
    output_root: Path,
    output_shard_size: int,
    use_compression: bool,
) -> Iterator[Tuple[Any, ...]]:
    for idx in range(int(start_index), int(end_index_exclusive)):
        if idx in done_indices:
            continue
        midi_path = midi_files[idx]
        rel = str(midi_path.relative_to(input_dir)).replace("\\", "/")
        yield (
            int(idx),
            str(midi_path),
            rel,
            int(min_token_length),
            str(output_root),
            int(output_shard_size),
            bool(use_compression),
        )


def _delete_result_npz(output_root: Path, result: Dict[str, Any]) -> None:
    npz_path = str(result.get("npz_path", "")).strip()
    if not npz_path:
        return
    candidate = (output_root / npz_path).resolve()
    root = output_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return
    try:
        if candidate.exists() and candidate.is_file():
            candidate.unlink()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    input_dir = Path(str(args.input_dir)).expanduser()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    output_root = Path(str(args.output_dir)).expanduser()
    if bool(args.start_over):
        remove_tree_strict(output_root)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest_jsonl_path = metadata_dir / "manifest.jsonl"
    manifest_json_path = metadata_dir / "manifest.json"
    checkpoint_path = metadata_dir / "checkpoint.json"
    summary_path = metadata_dir / "summary.json"
    tokenizer_path = metadata_dir / "piano_remi_bpe_tokenizer.json"

    file_list_jsonl = (
        Path(str(args.file_list_jsonl)).expanduser()
        if str(args.file_list_jsonl).strip()
        else None
    )
    if file_list_jsonl is not None:
        if not file_list_jsonl.exists() or not file_list_jsonl.is_file():
            raise FileNotFoundError(f"file list JSONL not found: {file_list_jsonl}")
        print(
            f"Loading file list: {file_list_jsonl.resolve()} "
            f"(trust_file_list={bool(args.trust_file_list)})",
            flush=True,
        )
        midi_files = load_midi_files_from_jsonl(
            input_dir,
            file_list_jsonl,
            trust_file_list=bool(args.trust_file_list),
        )
    else:
        print(f"Scanning MIDI files recursively under {input_dir.resolve()}", flush=True)
        midi_files = scan_midi_files(input_dir)
    if not midi_files:
        raise RuntimeError(f"No MIDI files found under {input_dir.resolve()}")
    print(f"Loaded MIDI file list: {len(midi_files):,} files", flush=True)

    if bool(args.retrain_tokenizer) or not tokenizer_path.exists():
        sample_n = int(max(0, args.bpe_sample_size))
        sample_paths = midi_files[:sample_n] if sample_n > 0 else midi_files
        tokenizer = PianoREMIBPETokenizer(
            vocab_size=int(args.vocab_size),
            positions_per_bar=int(args.positions_per_bar),
            max_duration_bars=int(args.max_duration_bars),
            tempo_bins=int(args.tempo_bins),
            include_special_tokens=True,
        )
        print(
            f"Training PianoREMIBPE vocab_size={int(args.vocab_size):,} "
            f"on sample_files={len(sample_paths):,} mode={str(args.bpe_training_mode)}",
            flush=True,
        )
        bpe_cpu_count = max(1, int(os.cpu_count() or 1))
        bpe_workers = int(args.bpe_workers) if int(args.bpe_workers) > 0 else int(args.workers)
        if bpe_workers <= 0:
            bpe_workers = bpe_cpu_count
        bpe_workers = max(1, int(bpe_workers))
        bpe_config = {
            "vocab_size": int(args.vocab_size),
            "positions_per_bar": int(args.positions_per_bar),
            "max_duration_bars": int(args.max_duration_bars),
            "tempo_bins": int(args.tempo_bins),
        }
        sequences: List[List[int]] = []
        bpe_start = time.time()
        print(f"Parsing BPE sample with workers={bpe_workers}", flush=True)
        with ProcessPoolExecutor(
            max_workers=bpe_workers,
            initializer=_init_bpe_worker,
            initargs=(bpe_config,),
        ) as pool:
            chunk_size = max(1, min(128, bpe_workers * 4))
            for index, ids in enumerate(
                pool.map(_worker_base_sequence, [str(p) for p in sample_paths], chunksize=chunk_size),
                start=1,
            ):
                if ids is not None:
                    sequences.append(ids)
                if index % int(max(1, args.progress_every)) == 0:
                    elapsed = max(1e-6, time.time() - bpe_start)
                    print(
                        f"BPE sample parsed: files={index:,}/{len(sample_paths):,} "
                        f"valid={len(sequences):,} rate={index / elapsed:.2f}/s",
                        flush=True,
                    )
        print(
            f"BPE sample parsing complete: valid={len(sequences):,}/{len(sample_paths):,}",
            flush=True,
        )
        tokenizer.train_from_base_sequences(
            sequences,
            vocab_size=int(args.vocab_size),
            mode=str(args.bpe_training_mode),
            progress_every=int(max(1, args.progress_every)),
        )
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer saved: {tokenizer_path.resolve()} (vocab={tokenizer.vocab_size:,})", flush=True)
    else:
        tokenizer = PianoREMIBPETokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer: {tokenizer_path.resolve()} (vocab={tokenizer.vocab_size:,})", flush=True)

    checkpoint_payload = safe_json_read(checkpoint_path, default={})
    manifest_count, done_indices = load_manifest_state(manifest_jsonl_path)
    accepted = int(max(int(checkpoint_payload.get("accepted", 0)), manifest_count))
    skipped = int(checkpoint_payload.get("skipped", 0))
    last_completed_index = int(checkpoint_payload.get("last_completed_index", -1))
    start_index = max(0, last_completed_index + 1)
    if int(args.start_index) >= 0:
        start_index = max(start_index, int(args.start_index))
    end_index_exclusive = len(midi_files)
    if int(args.end_index) >= 0:
        end_index_exclusive = min(end_index_exclusive, max(0, int(args.end_index)))
    if end_index_exclusive < start_index:
        end_index_exclusive = start_index

    cpu_count = max(1, int(os.cpu_count() or 1))
    workers = int(args.workers) if int(args.workers) > 0 else cpu_count
    workers = max(1, int(workers))
    checkpoint_every = int(max(1, args.checkpoint_every))
    progress_every = int(max(1, args.progress_every))
    min_token_length = int(max(1, args.min_token_length))
    output_shard_size = int(max(1, args.output_shard_size))
    max_files = int(max(0, args.max_files))

    print("PianoREMIBPE tokenization session")
    print(f"  input_dir: {input_dir.resolve()}")
    print(f"  file_list_jsonl: {file_list_jsonl.resolve() if file_list_jsonl is not None else 'recursive scan'}")
    print(f"  output_dir: {output_root.resolve()}")
    print(f"  total_files: {len(midi_files):,}")
    print(f"  start_index: {start_index:,}")
    print(f"  end_index_exclusive: {end_index_exclusive:,}")
    print(f"  accepted_so_far: {accepted:,}")
    print(f"  skipped_so_far: {skipped:,}")
    print(f"  workers: {workers}")
    print(f"  min_token_length: {min_token_length:,}")
    print(f"  output_compression: {'on' if bool(args.compress_output) else 'off'}")
    print(f"  tokenizer_vocab_size: {tokenizer.vocab_size:,}")

    session_start = time.time()
    accepted_at_start = int(accepted)
    processed_this_run = 0
    processed_since_checkpoint = 0

    def _save_checkpoint(*, rebuild_manifest: bool) -> int:
        safe_json_write(
            checkpoint_path,
            {
                "last_completed_index": int(last_completed_index),
                "accepted": int(accepted),
                "skipped": int(skipped),
                "total_files": int(len(midi_files)),
                "tokenizer": {
                    "name": "PianoREMIBPE",
                    "vocab_size": int(tokenizer.vocab_size),
                    "event_size": 1,
                    "tokenizer_path": str(tokenizer_path.resolve()),
                },
                "updated_at": utc_now_iso(),
            },
        )
        if rebuild_manifest:
            return rebuild_manifest_json(manifest_jsonl_path, manifest_json_path)
        return -1

    def _handle_result(result: Dict[str, Any]) -> None:
        nonlocal accepted, skipped, last_completed_index
        nonlocal processed_this_run, processed_since_checkpoint

        idx = int(result.get("index", -1))
        if idx >= 0:
            last_completed_index = idx
        processed_this_run += 1
        processed_since_checkpoint += 1

        status = str(result.get("status", "skip"))
        if status == "accepted":
            if max_files > 0 and accepted >= max_files:
                _delete_result_npz(output_root, result)
                skipped += 1
                return
            row = {
                "index": int(idx),
                "md5": str(result.get("md5", "")),
                "npz_path": str(result.get("npz_path", "")),
                "length": int(result.get("length", 0)),
                "source_path": str(result.get("source_path", "")),
                "tokenizer": "PianoREMIBPE",
                "event_size": 1,
            }
            append_manifest_entry(manifest_jsonl_path, row)
            done_indices.add(int(idx))
            accepted += 1
        else:
            skipped += 1
            if status == "error" or processed_this_run <= 20:
                print(
                    f"Skip/error at index {idx}: {result.get('source_path', '')} "
                    f"({result.get('error', 'unknown')})"
                )

        if processed_this_run % progress_every == 0:
            elapsed = max(1e-6, time.time() - session_start)
            processed_rate = processed_this_run / elapsed
            accepted_rate = max(0, accepted - accepted_at_start) / elapsed
            remaining = max(0, int(end_index_exclusive) - max(last_completed_index + 1, int(start_index)))
            eta = remaining / processed_rate if processed_rate > 0 else None
            print(
                f"idx={last_completed_index:,}/{end_index_exclusive:,} "
                f"accepted={accepted:,} skipped={skipped:,} "
                f"proc_rate={processed_rate:.2f}/s acc_rate={accepted_rate:.2f}/s "
                f"eta={format_eta(eta)}"
            )

        if processed_since_checkpoint >= checkpoint_every:
            manifest_len = _save_checkpoint(rebuild_manifest=True)
            print(
                f"Checkpoint saved at idx={last_completed_index:,} "
                f"accepted={accepted:,} skipped={skipped:,} manifest={manifest_len:,}"
            )
            processed_since_checkpoint = 0

    payload_iter = _iter_payloads(
        midi_files=midi_files,
        input_dir=input_dir,
        start_index=start_index,
        end_index_exclusive=end_index_exclusive,
        done_indices=done_indices,
        min_token_length=min_token_length,
        output_root=output_root,
        output_shard_size=output_shard_size,
        use_compression=bool(args.compress_output),
    )

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(tokenizer_path),),
    ) as pool:
        if max_files > 0:
            payload_source = iter(payload_iter)
            max_pending = max(1, min(128, workers * 4))
            pending: Set[Future[Dict[str, Any]]] = set()
            exhausted = False

            def _fill_pending() -> None:
                nonlocal exhausted
                while not exhausted and accepted < max_files and len(pending) < max_pending:
                    try:
                        payload = next(payload_source)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(pool.submit(_worker_tokenize, payload))

            _fill_pending()
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    _handle_result(future.result())
                if accepted >= max_files:
                    print(f"MAX_FILES reached ({max_files}). Stopping.")
                    for future in pending:
                        future.cancel()
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            if not future.cancelled():
                                _handle_result(future.result())
                    break
                _fill_pending()
        else:
            chunk_size = max(1, min(128, workers * 4))
            for result in pool.map(_worker_tokenize, payload_iter, chunksize=chunk_size):
                _handle_result(result)

    _save_checkpoint(rebuild_manifest=False)
    manifest_len = rebuild_manifest_json(manifest_jsonl_path, manifest_json_path)
    elapsed = max(1e-6, time.time() - session_start)
    summary = {
        "input_dir": str(input_dir.resolve()),
        "file_list_jsonl": str(file_list_jsonl.resolve()) if file_list_jsonl is not None else "",
        "output_dir": str(output_root.resolve()),
        "total_files": int(len(midi_files)),
        "start_index": int(start_index),
        "end_index_exclusive": int(end_index_exclusive),
        "accepted": int(accepted),
        "skipped": int(skipped),
        "manifest_entries": int(manifest_len),
        "processed_this_run": int(processed_this_run),
        "workers": int(workers),
        "tokenizer": {
            "name": "PianoREMIBPE",
            "vocab_size": int(tokenizer.vocab_size),
            "target_vocab_size": int(args.vocab_size),
            "event_size": 1,
            "tokenizer_path": str(tokenizer_path.resolve()),
        },
        "elapsed_seconds": float(elapsed),
        "updated_at": utc_now_iso(),
    }
    safe_json_write(summary_path, summary)

    print("Done.")
    print(f"  elapsed_minutes: {elapsed / 60.0:.2f}")
    print(f"  accepted: {accepted:,}")
    print(f"  skipped: {skipped:,}")
    print(f"  manifest: {manifest_json_path.resolve()}")
    print(f"  tokenizer: {tokenizer_path.resolve()}")
    print(f"  summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()

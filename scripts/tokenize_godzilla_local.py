from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
import stat
import shutil
import tarfile
import time
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from data.tokenizer import CustomDeltaTokenizer
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from data.tokenizer import CustomDeltaTokenizer


def _import_symusic_score() -> Any:
    try:
        from symusic import Score

        return Score
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "symusic is required for local tokenization. Install it with: pip install symusic"
        ) from exc


Score = _import_symusic_score()


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


def format_eta(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    sec = int(seconds)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _rmtree_onerror(func: Any, path: str, exc_info: Any) -> None:
    # Windows can leave read-only files around; make one retry writable before failing.
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
        raise OSError(
            f"--start-over requested, but output root could not be deleted cleanly: {path}"
        )


@dataclass
class SourceIndex:
    source_type: str
    source_path: str
    members: List[str]


class SymusicEventTokenizer(CustomDeltaTokenizer):
    """Symusic parser adapter over the frozen CustomDeltaTokenizer event-quad spec."""

    def __init__(self) -> None:
        super().__init__(default_velocity=88, include_special_tokens=False)

    def parse_events(
        self,
        midi_bytes: bytes,
        strict_piano: bool = True,
    ) -> Optional[List[Tuple[float, int, float, int]]]:
        try:
            score = Score.from_midi(midi_bytes, ttype="second")
        except Exception:
            return None

        events: List[Tuple[float, int, float, int]] = []
        if len(score.tracks) == 0:
            return None

        for track in score.tracks:
            program = int(track.program)
            is_drum = bool(track.is_drum)
            if strict_piano:
                if is_drum:
                    return None
                if program < 0 or program > 7:
                    return None

            for note in track.notes:
                pitch = int(note.pitch)
                if pitch < 21 or pitch > 108:
                    continue
                onset = float(max(0.0, float(note.time)))
                duration = float(max(1e-4, float(note.duration)))
                velocity = int(max(0, min(127, int(note.velocity))))
                events.append((onset, pitch, duration, velocity))

        events.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        return events

def build_source_index(source: Path) -> SourceIndex:
    source_resolved = str(source.resolve())

    if source.is_dir():
        midi_files = sorted(
            list(source.rglob("*.mid")) + list(source.rglob("*.midi")),
            key=lambda p: str(p).lower(),
        )
        members = [str(p.relative_to(source)).replace("\\", "/") for p in midi_files]
        return SourceIndex(
            source_type="directory",
            source_path=source_resolved,
            members=members,
        )

    if source.is_file():
        with tarfile.open(source, mode="r:*") as tar:
            # Preserve archive order for streaming .tar.gz access.
            members = [
                m.name for m in tar.getmembers() if m.isfile() and is_midi_name(m.name)
            ]
        return SourceIndex(
            source_type="tar",
            source_path=source_resolved,
            members=members,
        )

    raise FileNotFoundError(f"Source not found: {source}")


def load_or_create_source_index(
    source: Path,
    index_path: Path,
    rebuild: bool,
) -> SourceIndex:
    if index_path.exists() and not rebuild:
        payload = safe_json_read(index_path, default={})
        source_type = str(payload.get("source_type", "")).strip().lower()
        source_path = str(payload.get("source_path", "")).strip()
        member_order = str(payload.get("member_order", "")).strip().lower()
        members = payload.get("members", [])
        index_compatible = True
        # Older tar indexes used sorted member order, which is extremely slow for
        # random access into .tar.gz. Force rebuild to archive order when needed.
        if source_type == "tar" and member_order != "archive":
            index_compatible = False
        if (
            source_type in {"directory", "tar"}
            and source_path == str(source.resolve())
            and isinstance(members, list)
            and members
            and index_compatible
        ):
            return SourceIndex(
                source_type=source_type,
                source_path=source_path,
                members=[str(x) for x in members],
            )

    index = build_source_index(source)
    safe_json_write(
        index_path,
        {
            "created_at": utc_now_iso(),
            "source_type": index.source_type,
            "source_path": index.source_path,
            "member_order": "archive" if index.source_type == "tar" else "sorted",
            "total_members": int(len(index.members)),
            "members": index.members,
        },
    )
    return index


_WORKER_TOKENIZER: Optional["SymusicEventTokenizer"] = None


def _get_worker_tokenizer() -> "SymusicEventTokenizer":
    global _WORKER_TOKENIZER
    if _WORKER_TOKENIZER is None:
        _WORKER_TOKENIZER = SymusicEventTokenizer()
    return _WORKER_TOKENIZER


def save_npz_with_retries(
    *,
    out_npz_path: Path,
    tokens: np.ndarray,
    onsets: np.ndarray,
    durations: np.ndarray,
    use_compression: bool,
    retries: int = 5,
) -> Optional[str]:
    """Best-effort npz save with retries for transient Windows file locks."""
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
        except PermissionError as exc:
            last_error = exc
        except OSError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
            break

        time.sleep(min(1.0, 0.05 * float(2**attempt)))

    if last_error is None:
        return "unknown-write-error"
    return f"{type(last_error).__name__}: {last_error}"


def build_npz_relative_path(idx: int, member_name: str, shard_size: int = 50000) -> str:
    safe_shard_size = max(1, int(shard_size))
    shard = int(idx) // safe_shard_size
    md5 = md5_text(member_name)
    return f"data/{shard:05d}/{idx:07d}_{md5}.npz"


def _worker_tokenize_and_write(
    *,
    idx: int,
    member_name: str,
    midi_bytes: Optional[bytes],
    strict_piano: bool,
    min_token_length: int,
    output_root: str,
    output_shard_size: int,
    use_compression: bool,
    read_error: str = "",
) -> Dict[str, Any]:
    if read_error:
        return {
            "index": int(idx),
            "source_path": str(member_name),
            "status": "error",
            "error": str(read_error),
        }
    if not midi_bytes:
        return {
            "index": int(idx),
            "source_path": str(member_name),
            "status": "skip",
            "error": "empty-midi-bytes",
        }

    tokenizer = _get_worker_tokenizer()
    events = tokenizer.parse_events(midi_bytes=midi_bytes, strict_piano=bool(strict_piano))
    if not events:
        return {
            "index": int(idx),
            "source_path": str(member_name),
            "status": "skip",
            "error": "parse-or-filter",
        }

    tokens, onsets, durations = tokenizer.encode_events(events)
    token_len = int(tokens.shape[0])
    if token_len < int(min_token_length):
        return {
            "index": int(idx),
            "source_path": str(member_name),
            "status": "skip",
            "error": "short",
        }

    md5 = md5_text(member_name)
    rel_npz_path = build_npz_relative_path(
        idx=idx,
        member_name=member_name,
        shard_size=int(output_shard_size),
    )
    out_npz_path = Path(output_root) / rel_npz_path
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    write_error = save_npz_with_retries(
        out_npz_path=out_npz_path,
        tokens=tokens,
        onsets=onsets,
        durations=durations,
        use_compression=bool(use_compression),
    )
    if write_error is not None:
        return {
            "index": int(idx),
            "source_path": str(member_name),
            "status": "error",
            "error": f"write-failed: {write_error}",
        }

    return {
        "index": int(idx),
        "source_path": str(member_name),
        "status": "accepted",
        "md5": md5,
        "npz_path": rel_npz_path,
        "length": int(token_len),
    }


def _worker_tokenize_from_path(payload: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        idx,
        source_root,
        member_name,
        strict_piano,
        min_token_length,
        output_root,
        output_shard_size,
        use_compression,
    ) = payload
    try:
        midi_bytes = read_midi_bytes_from_directory(
            source_root=Path(str(source_root)),
            relative_name=str(member_name),
        )
    except Exception as exc:
        return _worker_tokenize_and_write(
            idx=int(idx),
            member_name=str(member_name),
            midi_bytes=None,
            strict_piano=bool(strict_piano),
            min_token_length=int(min_token_length),
            output_root=str(output_root),
            output_shard_size=int(output_shard_size),
            use_compression=bool(use_compression),
            read_error=str(exc),
        )

    return _worker_tokenize_and_write(
        idx=int(idx),
        member_name=str(member_name),
        midi_bytes=midi_bytes,
        strict_piano=bool(strict_piano),
        min_token_length=int(min_token_length),
        output_root=str(output_root),
        output_shard_size=int(output_shard_size),
        use_compression=bool(use_compression),
    )


def _worker_tokenize_from_bytes(payload: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        idx,
        member_name,
        midi_bytes,
        strict_piano,
        min_token_length,
        output_root,
        output_shard_size,
        use_compression,
        read_error,
    ) = payload
    return _worker_tokenize_and_write(
        idx=int(idx),
        member_name=str(member_name),
        midi_bytes=midi_bytes,
        strict_piano=bool(strict_piano),
        min_token_length=int(min_token_length),
        output_root=str(output_root),
        output_shard_size=int(output_shard_size),
        use_compression=bool(use_compression),
        read_error=str(read_error),
    )


def read_midi_bytes_from_directory(source_root: Path, relative_name: str) -> bytes:
    return (source_root / relative_name).read_bytes()


def read_midi_bytes_from_tar(
    tar_obj: tarfile.TarFile,
    member_lookup: Dict[str, tarfile.TarInfo],
    member_name: str,
) -> Optional[bytes]:
    info = member_lookup.get(member_name)
    if info is None:
        return None
    handle = tar_obj.extractfile(info)
    if handle is None:
        return None
    return handle.read()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize Godzilla MIDI locally into event-quad .npz packs with resumable checkpoints."
        )
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source MIDI directory or .tar/.tar.gz archive.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="processed/godzilla_tokenized",
        help="Output root where data/ and metadata/ are written.",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=192,
        help="Skip tokenized pieces shorter than this many tokens.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Write checkpoint and rebuild manifest.json every N processed files.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress update every N processed files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Parallel worker count. 0 = auto. "
            "Auto uses a conservative worker cap for .tar/.tar.gz to reduce IPC overhead."
        ),
    )
    parser.add_argument(
        "--compress-output",
        action="store_true",
        help=(
            "Write compressed .npz (smaller files, slower). "
            "Default writes uncompressed .npz for maximum throughput."
        ),
    )
    parser.add_argument(
        "--output-shard-size",
        type=int,
        default=50000,
        help="Number of source indices per output data/ shard folder.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Stop after this many accepted files (0 means no cap).",
    )
    parser.add_argument(
        "--stop-after-seconds",
        type=int,
        default=0,
        help="Soft stop after this many seconds (0 means run until complete).",
    )
    parser.add_argument(
        "--strict-piano",
        dest="strict_piano",
        action="store_true",
        help="Reject files with drums/non-piano programs.",
    )
    parser.add_argument(
        "--allow-mixed-instruments",
        dest="strict_piano",
        action="store_false",
        help="Disable strict piano filtering and keep mixed-instrument files.",
    )
    parser.add_argument(
        "--start-over",
        action="store_true",
        help="Delete existing output root and start from scratch.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of source member index even if it exists.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=-1,
        help=(
            "Optional source member start index (0-based). "
            "Default (-1) resumes from checkpoint."
        ),
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help=(
            "Optional exclusive source member end index (0-based). "
            "Default (-1) processes to source end."
        ),
    )
    parser.set_defaults(strict_piano=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    output_root = Path(args.output_root)
    data_dir = output_root / "data"
    meta_dir = output_root / "metadata"

    if bool(args.start_over) and output_root.exists():
        print(f"START_OVER=True, deleting existing output: {output_root}")
        remove_tree_strict(output_root)

    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    index_path = meta_dir / "source_index.json"
    checkpoint_path = meta_dir / "checkpoint.json"
    manifest_jsonl_path = meta_dir / "manifest.jsonl"
    manifest_json_path = meta_dir / "manifest.json"
    summary_path = meta_dir / "summary.json"

    source_index = load_or_create_source_index(
        source=source,
        index_path=index_path,
        rebuild=bool(args.rebuild_index),
    )
    total_members = len(source_index.members)
    if total_members == 0:
        raise RuntimeError("No MIDI members found in source.")

    manifest_count, done_indices = load_manifest_state(manifest_jsonl_path)
    checkpoint = safe_json_read(
        checkpoint_path,
        default={
            "last_completed_index": -1,
            "accepted": 0,
            "skipped": 0,
            "last_completed_name": "",
            "total_members": int(total_members),
            "source_type": source_index.source_type,
            "source_path": source_index.source_path,
            "member_order": "archive" if source_index.source_type == "tar" else "sorted",
            "updated_at": utc_now_iso(),
        },
    )

    if (
        str(checkpoint.get("source_path", source_index.source_path))
        != source_index.source_path
    ):
        raise RuntimeError(
            "Checkpoint source mismatch. Use --start-over or choose the original source path."
        )

    current_member_order = "archive" if source_index.source_type == "tar" else "sorted"
    checkpoint_member_order = str(checkpoint.get("member_order", "")).strip().lower()
    if checkpoint_member_order and checkpoint_member_order != current_member_order:
        raise RuntimeError(
            "Checkpoint member-order mismatch. Use --start-over to rebuild tokenization output."
        )
    if (
        source_index.source_type == "tar"
        and not checkpoint_member_order
        and manifest_count > 0
        and not bool(args.start_over)
    ):
        raise RuntimeError(
            "Existing output was created before archive-order indexing optimization. "
            "Run once with --start-over to rebuild a consistent manifest."
        )

    accepted = int(max(int(checkpoint.get("accepted", 0)), manifest_count))
    skipped = int(checkpoint.get("skipped", 0))
    last_completed_index = int(checkpoint.get("last_completed_index", -1))
    last_completed_name = str(checkpoint.get("last_completed_name", ""))
    auto_start_index = max(0, last_completed_index + 1)
    requested_start_index = int(args.start_index)
    if requested_start_index >= 0:
        if requested_start_index < auto_start_index:
            print(
                "WARNING: requested --start-index is behind checkpoint progress; "
                f"using checkpoint resume index {auto_start_index:,} instead of {requested_start_index:,}."
            )
            start_index = int(auto_start_index)
        else:
            start_index = int(requested_start_index)
    else:
        start_index = int(auto_start_index)

    requested_end_index = int(args.end_index)
    if requested_end_index >= 0:
        end_index_exclusive = int(min(total_members, max(0, requested_end_index)))
    else:
        end_index_exclusive = int(total_members)

    if end_index_exclusive < start_index:
        end_index_exclusive = int(start_index)

    checkpoint_accepted = int(checkpoint.get("accepted", 0))
    if checkpoint_accepted != manifest_count:
        print(
            "WARNING: checkpoint accepted count and manifest entry count differ; "
            f"checkpoint={checkpoint_accepted:,} manifest={manifest_count:,}. "
            "Using the larger count for reporting."
        )

    strict_piano = bool(args.strict_piano)

    tokenizer_meta = SymusicEventTokenizer()
    min_token_length = int(max(4, args.min_token_length))
    checkpoint_every = int(max(1, args.checkpoint_every))
    progress_every = int(max(1, args.progress_every))
    max_files = int(max(0, args.max_files))
    stop_after_seconds = int(max(0, args.stop_after_seconds))
    use_compression = bool(args.compress_output)
    output_shard_size = int(max(1, args.output_shard_size))

    cpu_count = max(1, int(os.cpu_count() or 1))
    requested_workers = int(max(0, args.workers))
    if requested_workers > 0:
        workers = requested_workers
    else:
        if source_index.source_type == "tar":
            auto_cap = min(8, max(1, cpu_count // 2))
            workers = auto_cap
            if workers > 1:
                workers = max(1, workers - 1)
        else:
            workers = cpu_count
    parallel_mode = bool(workers > 1 and max_files == 0 and stop_after_seconds == 0)

    tokenizer_info = {
        "name": "CustomDeltaTokenizer",
        "version": 1,
        "frozen": True,
        "event_size": int(tokenizer_meta.event_size),
        "vocab_size": int(tokenizer_meta.vocab_size),
    }

    print("Tokenization session")
    print(f"  source_type: {source_index.source_type}")
    print(f"  source_path: {source_index.source_path}")
    print(f"  output_root: {output_root.resolve()}")
    print(f"  total_members: {total_members:,}")
    print(f"  start_index: {start_index:,}")
    print(f"  end_index_exclusive: {end_index_exclusive:,}")
    print(f"  accepted_so_far: {accepted:,}")
    print(f"  skipped_so_far: {skipped:,}")
    print(f"  strict_piano: {strict_piano}")
    print("  tokenizer: CustomDeltaTokenizer v1 (frozen)")
    print(f"  workers: {workers} ({'parallel' if parallel_mode else 'single-process'})")
    print(f"  output_compression: {'on' if use_compression else 'off'}")
    print(f"  output_shard_size: {output_shard_size:,}")
    if source_index.source_type == "tar":
        print(
            "  note: tar archive streaming is often slower than directory tokenization. "
            "For max throughput, extract to SSD and use --source <directory>."
        )
    if workers > 1 and not parallel_mode:
        print(
            "  note: parallel mode is disabled when --max-files or --stop-after-seconds is set."
        )

    session_start = time.time()
    accepted_at_run_start = int(accepted)
    processed_since_checkpoint = 0
    processed_this_run = 0

    tar_obj: Optional[tarfile.TarFile] = None
    tar_lookup: Dict[str, tarfile.TarInfo] = {}
    if source_index.source_type == "tar":
        tar_obj = tarfile.open(source, mode="r:*")
        tar_lookup = {m.name: m for m in tar_obj.getmembers() if m.isfile()}

    def _save_checkpoint(*, rebuild_manifest: bool) -> int:
        checkpoint_payload = {
            "last_completed_index": int(last_completed_index),
            "accepted": int(accepted),
            "skipped": int(skipped),
            "last_completed_name": str(last_completed_name),
            "total_members": int(total_members),
            "source_type": source_index.source_type,
            "source_path": source_index.source_path,
            "member_order": current_member_order,
            "tokenizer": dict(tokenizer_info),
            "updated_at": utc_now_iso(),
        }
        safe_json_write(checkpoint_path, checkpoint_payload)
        if rebuild_manifest:
            return rebuild_manifest_json(manifest_jsonl_path, manifest_json_path)
        return -1

    def _handle_result(result: Dict[str, Any]) -> None:
        nonlocal accepted
        nonlocal skipped
        nonlocal last_completed_index
        nonlocal last_completed_name
        nonlocal processed_since_checkpoint
        nonlocal processed_this_run

        idx = int(result.get("index", -1))
        member_name = str(result.get("source_path", ""))
        if idx >= 0:
            last_completed_index = idx
            last_completed_name = member_name

        processed_since_checkpoint += 1
        processed_this_run += 1

        status = str(result.get("status", "skip"))
        if status == "accepted":
            row = {
                "index": int(idx),
                "md5": str(result.get("md5", "")),
                "npz_path": str(result.get("npz_path", "")),
                "length": int(result.get("length", 0)),
                "source_path": member_name,
            }
            append_manifest_entry(manifest_jsonl_path, row)
            done_indices.add(int(idx))
            accepted += 1
        else:
            skipped += 1
            if status == "error":
                err = str(result.get("error", "unknown-error"))
                if processed_this_run <= 20 or (processed_this_run % progress_every == 0):
                    print(f"Read/tokenize error at index {idx}: {member_name} ({err})")

        if processed_this_run % progress_every == 0:
            elapsed = max(1e-6, time.time() - session_start)
            processed_rate = processed_this_run / elapsed
            accepted_this_run = max(0, accepted - accepted_at_run_start)
            accepted_rate = accepted_this_run / elapsed
            remaining = max(0, int(end_index_exclusive) - max(last_completed_index + 1, int(start_index)))
            eta = (remaining / processed_rate) if processed_rate > 0 else None
            print(
                f"idx={last_completed_index:,}/{end_index_exclusive:,} accepted={accepted:,} skipped={skipped:,} "
                f"proc_rate={processed_rate:.2f}/s acc_rate={accepted_rate:.2f}/s eta={format_eta(eta)}"
            )

        if processed_since_checkpoint >= checkpoint_every:
            manifest_len = _save_checkpoint(rebuild_manifest=True)
            print(
                f"Checkpoint saved at idx={last_completed_index:,} "
                f"accepted={accepted:,} skipped={skipped:,} manifest={manifest_len:,}"
            )
            processed_since_checkpoint = 0

    try:
        if parallel_mode:
            if source_index.source_type == "directory":

                def _iter_dir_payloads() -> Iterator[Tuple[Any, ...]]:
                    nonlocal last_completed_index
                    nonlocal last_completed_name
                    for idx in range(start_index, end_index_exclusive):
                        member_name = source_index.members[idx]
                        if idx in done_indices:
                            last_completed_index = int(idx)
                            last_completed_name = str(member_name)
                            continue
                        yield (
                            int(idx),
                            str(source),
                            str(member_name),
                            bool(strict_piano),
                            int(min_token_length),
                            str(output_root),
                            int(output_shard_size),
                            bool(use_compression),
                        )

                chunk = max(1, min(256, workers * 4))
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    for result in pool.map(
                        _worker_tokenize_from_path,
                        _iter_dir_payloads(),
                        chunksize=chunk,
                    ):
                        _handle_result(result)
            else:
                if tar_obj is None:
                    raise RuntimeError("Internal error: tar source selected but tar object is not open")

                def _iter_tar_payloads() -> Iterator[Tuple[Any, ...]]:
                    nonlocal last_completed_index
                    nonlocal last_completed_name
                    for idx in range(start_index, end_index_exclusive):
                        member_name = source_index.members[idx]
                        if idx in done_indices:
                            last_completed_index = int(idx)
                            last_completed_name = str(member_name)
                            continue
                        midi_bytes: Optional[bytes] = None
                        read_error = ""
                        try:
                            midi_bytes = read_midi_bytes_from_tar(
                                tar_obj=tar_obj,
                                member_lookup=tar_lookup,
                                member_name=member_name,
                            )
                        except Exception as exc:
                            read_error = str(exc)
                        yield (
                            int(idx),
                            str(member_name),
                            midi_bytes,
                            bool(strict_piano),
                            int(min_token_length),
                            str(output_root),
                            int(output_shard_size),
                            bool(use_compression),
                            str(read_error),
                        )

                chunk = max(1, min(128, workers * 2))
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    for result in pool.map(
                        _worker_tokenize_from_bytes,
                        _iter_tar_payloads(),
                        chunksize=chunk,
                    ):
                        _handle_result(result)
        else:
            for idx in range(start_index, end_index_exclusive):
                if max_files > 0 and accepted >= max_files:
                    print(f"MAX_FILES reached ({max_files}). Stopping.")
                    break

                if stop_after_seconds > 0 and (time.time() - session_start) >= stop_after_seconds:
                    print(f"STOP_AFTER_SECONDS reached ({stop_after_seconds}s). Stopping.")
                    break

                member_name = source_index.members[idx]
                last_completed_index = int(idx)
                last_completed_name = member_name
                processed_since_checkpoint += 1
                processed_this_run += 1

                if idx in done_indices:
                    continue

                midi_bytes: Optional[bytes] = None
                try:
                    if source_index.source_type == "directory":
                        midi_bytes = read_midi_bytes_from_directory(
                            source_root=source,
                            relative_name=member_name,
                        )
                    else:
                        if tar_obj is None:
                            raise RuntimeError("Internal error: tar source selected but tar object is not open")
                        midi_bytes = read_midi_bytes_from_tar(
                            tar_obj=tar_obj,
                            member_lookup=tar_lookup,
                            member_name=member_name,
                        )
                except Exception as exc:
                    skipped += 1
                    print(f"Read error at index {idx}: {member_name} ({exc})")
                    midi_bytes = None

                if not midi_bytes:
                    continue

                events = tokenizer_meta.parse_events(midi_bytes=midi_bytes, strict_piano=strict_piano)
                if not events:
                    skipped += 1
                    continue

                tokens, onsets, durations = tokenizer_meta.encode_events(events)
                token_len = int(tokens.shape[0])
                if token_len < min_token_length:
                    skipped += 1
                    continue

                md5 = md5_text(member_name)
                rel_npz_path = build_npz_relative_path(
                    idx=idx,
                    member_name=member_name,
                    shard_size=int(output_shard_size),
                )
                out_npz_path = output_root / rel_npz_path
                out_npz_path.parent.mkdir(parents=True, exist_ok=True)

                write_error = save_npz_with_retries(
                    out_npz_path=out_npz_path,
                    tokens=tokens,
                    onsets=onsets,
                    durations=durations,
                    use_compression=bool(use_compression),
                )
                if write_error is not None:
                    skipped += 1
                    print(
                        f"Write error at index {idx}: {member_name} ({write_error})"
                    )
                    continue

                append_manifest_entry(
                    manifest_jsonl_path,
                    {
                        "index": int(idx),
                        "md5": md5,
                        "npz_path": rel_npz_path,
                        "length": int(token_len),
                        "source_path": member_name,
                    },
                )
                done_indices.add(idx)
                accepted += 1

                if processed_this_run % progress_every == 0:
                    elapsed = max(1e-6, time.time() - session_start)
                    processed_rate = processed_this_run / elapsed
                    accepted_this_run = max(0, accepted - accepted_at_run_start)
                    accepted_rate = accepted_this_run / elapsed
                    remaining = max(0, int(end_index_exclusive) - idx - 1)
                    eta = (remaining / processed_rate) if processed_rate > 0 else None
                    print(
                        f"idx={idx:,}/{end_index_exclusive:,} accepted={accepted:,} skipped={skipped:,} "
                        f"proc_rate={processed_rate:.2f}/s acc_rate={accepted_rate:.2f}/s eta={format_eta(eta)}"
                    )

                if processed_since_checkpoint >= checkpoint_every:
                    manifest_len = _save_checkpoint(rebuild_manifest=True)
                    print(
                        f"Checkpoint saved at idx={last_completed_index:,} "
                        f"accepted={accepted:,} skipped={skipped:,} manifest={manifest_len:,}"
                    )
                    processed_since_checkpoint = 0

        _save_checkpoint(rebuild_manifest=False)
        manifest_len = rebuild_manifest_json(manifest_jsonl_path, manifest_json_path)

        elapsed = max(1e-6, time.time() - session_start)
        summary = {
            "source_type": source_index.source_type,
            "source_path": source_index.source_path,
            "output_root": str(output_root.resolve()),
            "total_members": int(total_members),
            "start_index": int(start_index),
            "end_index_exclusive": int(end_index_exclusive),
            "window_members": int(max(0, end_index_exclusive - start_index)),
            "accepted": int(accepted),
            "skipped": int(skipped),
            "manifest_entries": int(manifest_len),
            "processed_this_run": int(processed_this_run),
            "tokenizer": dict(tokenizer_info),
            "workers": int(workers),
            "parallel_mode": bool(parallel_mode),
            "output_compression": bool(use_compression),
            "elapsed_seconds": float(elapsed),
            "updated_at": utc_now_iso(),
        }
        safe_json_write(summary_path, summary)

        print("Done.")
        print(f"  elapsed_minutes: {elapsed / 60.0:.2f}")
        print(f"  accepted: {accepted:,}")
        print(f"  skipped: {skipped:,}")
        print(f"  manifest: {manifest_json_path.resolve()}")
        print(f"  checkpoint: {checkpoint_path.resolve()}")
        print(f"  summary: {summary_path.resolve()}")
    finally:
        if tar_obj is not None:
            tar_obj.close()


if __name__ == "__main__":
    main()

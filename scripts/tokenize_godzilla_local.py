from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


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


@dataclass
class SourceIndex:
    source_type: str
    source_path: str
    members: List[str]


class SymusicEventTokenizer:
    """Symusic parser + event-quad quantizer compatible with CustomDeltaTokenizer bins."""

    DELTA_START = 0
    DELTA_END = 31
    PITCH_START = 32
    PITCH_END = 119
    DUR_START = 120
    DUR_END = 151
    VEL_START = 152
    VEL_END = 167

    def __init__(self) -> None:
        self.delta_bins = np.concatenate(
            [
                np.asarray([0.0], dtype=np.float64),
                np.logspace(math.log10(1e-4), math.log10(2.0), num=31),
            ],
            axis=0,
        )
        self.duration_bins = np.logspace(
            math.log10(0.05),
            math.log10(4.0),
            num=32,
        ).astype(np.float64)

    @staticmethod
    def _nearest_bin(value: float, bins: np.ndarray) -> int:
        return int(np.argmin(np.abs(bins - float(value))))

    def _quantize_delta(self, delta_seconds: float) -> int:
        clamped = float(max(0.0, min(2.0, delta_seconds)))
        idx = self._nearest_bin(clamped, self.delta_bins)
        return int(self.DELTA_START + idx)

    def _quantize_duration(self, duration_seconds: float) -> int:
        clamped = float(max(0.05, min(4.0, duration_seconds)))
        idx = self._nearest_bin(clamped, self.duration_bins)
        return int(self.DUR_START + idx)

    def _quantize_pitch(self, pitch: int) -> int:
        clamped = int(max(21, min(108, int(pitch))))
        return int(self.PITCH_START + (clamped - 21))

    def _quantize_velocity(self, velocity: int) -> int:
        clamped = int(max(0, min(127, int(velocity))))
        idx = int(round((float(clamped) / 127.0) * 15.0))
        idx = max(0, min(15, idx))
        return int(self.VEL_START + idx)

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

    def encode_events(
        self,
        events: List[Tuple[float, int, float, int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        token_ids: List[int] = []
        onset_times: List[float] = []
        durations: List[float] = []
        prev_onset = 0.0

        for onset, pitch, duration, velocity in events:
            delta = float(max(0.0, onset - prev_onset))
            prev_onset = onset

            d_tok = self._quantize_delta(delta)
            p_tok = self._quantize_pitch(pitch)
            u_tok = self._quantize_duration(duration)
            v_tok = self._quantize_velocity(velocity)

            token_ids.extend([d_tok, p_tok, u_tok, v_tok])
            onset_times.extend(
                [float(onset), float(onset), float(onset), float(onset)]
            )
            durations.extend(
                [float(duration), float(duration), float(duration), float(duration)]
            )

        return (
            np.asarray(token_ids, dtype=np.int16),
            np.asarray(onset_times, dtype=np.float32),
            np.asarray(durations, dtype=np.float32),
        )


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
            members = [m.name for m in tar.getmembers() if m.isfile() and is_midi_name(m.name)]
        members.sort()
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
        members = payload.get("members", [])
        if (
            source_type in {"directory", "tar"}
            and source_path == str(source.resolve())
            and isinstance(members, list)
            and members
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
            "total_members": int(len(index.members)),
            "members": index.members,
        },
    )
    return index


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
        shutil.rmtree(output_root, ignore_errors=True)

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

    accepted = int(max(int(checkpoint.get("accepted", 0)), manifest_count))
    skipped = int(checkpoint.get("skipped", 0))
    last_completed_index = int(checkpoint.get("last_completed_index", -1))
    last_completed_name = str(checkpoint.get("last_completed_name", ""))
    start_index = max(0, last_completed_index + 1)

    strict_piano = bool(args.strict_piano)

    tokenizer = SymusicEventTokenizer()
    min_token_length = int(max(4, args.min_token_length))
    checkpoint_every = int(max(1, args.checkpoint_every))
    progress_every = int(max(1, args.progress_every))
    max_files = int(max(0, args.max_files))
    stop_after_seconds = int(max(0, args.stop_after_seconds))

    print("Tokenization session")
    print(f"  source_type: {source_index.source_type}")
    print(f"  source_path: {source_index.source_path}")
    print(f"  output_root: {output_root.resolve()}")
    print(f"  total_members: {total_members:,}")
    print(f"  start_index: {start_index:,}")
    print(f"  accepted_so_far: {accepted:,}")
    print(f"  skipped_so_far: {skipped:,}")
    print(f"  strict_piano: {strict_piano}")

    session_start = time.time()
    processed_since_checkpoint = 0
    processed_this_run = 0

    tar_obj: Optional[tarfile.TarFile] = None
    tar_lookup: Dict[str, tarfile.TarInfo] = {}
    if source_index.source_type == "tar":
        tar_obj = tarfile.open(source, mode="r:*")
        tar_lookup = {m.name: m for m in tar_obj.getmembers() if m.isfile()}

    try:
        for idx in range(start_index, total_members):
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

            events = tokenizer.parse_events(midi_bytes=midi_bytes, strict_piano=strict_piano)
            if not events:
                skipped += 1
                continue

            tokens, onsets, durations = tokenizer.encode_events(events)
            token_len = int(tokens.shape[0])
            if token_len < min_token_length:
                skipped += 1
                continue

            md5 = md5_text(member_name)
            rel_npz_path = f"data/{idx:07d}_{md5}.npz"
            out_npz_path = output_root / rel_npz_path
            out_npz_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                out_npz_path,
                tokens=tokens,
                onsets=onsets,
                durations=durations,
            )

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
                rate = processed_this_run / elapsed
                remaining = max(0, total_members - idx - 1)
                eta = (remaining / rate) if rate > 0 else None
                print(
                    f"idx={idx:,}/{total_members:,} accepted={accepted:,} skipped={skipped:,} "
                    f"rate={rate:.2f}/s eta={format_eta(eta)}"
                )

            if processed_since_checkpoint >= checkpoint_every:
                checkpoint_payload = {
                    "last_completed_index": int(last_completed_index),
                    "accepted": int(accepted),
                    "skipped": int(skipped),
                    "last_completed_name": str(last_completed_name),
                    "total_members": int(total_members),
                    "source_type": source_index.source_type,
                    "source_path": source_index.source_path,
                    "updated_at": utc_now_iso(),
                }
                safe_json_write(checkpoint_path, checkpoint_payload)
                manifest_len = rebuild_manifest_json(manifest_jsonl_path, manifest_json_path)
                print(
                    f"Checkpoint saved at idx={last_completed_index:,} "
                    f"accepted={accepted:,} skipped={skipped:,} manifest={manifest_len:,}"
                )
                processed_since_checkpoint = 0

        checkpoint_payload = {
            "last_completed_index": int(last_completed_index),
            "accepted": int(accepted),
            "skipped": int(skipped),
            "last_completed_name": str(last_completed_name),
            "total_members": int(total_members),
            "source_type": source_index.source_type,
            "source_path": source_index.source_path,
            "updated_at": utc_now_iso(),
        }
        safe_json_write(checkpoint_path, checkpoint_payload)
        manifest_len = rebuild_manifest_json(manifest_jsonl_path, manifest_json_path)

        elapsed = max(1e-6, time.time() - session_start)
        summary = {
            "source_type": source_index.source_type,
            "source_path": source_index.source_path,
            "output_root": str(output_root.resolve()),
            "total_members": int(total_members),
            "accepted": int(accepted),
            "skipped": int(skipped),
            "manifest_entries": int(manifest_len),
            "processed_this_run": int(processed_this_run),
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

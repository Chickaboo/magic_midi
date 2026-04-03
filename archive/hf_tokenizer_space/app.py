from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import tarfile
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from symusic import Score


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def format_eta(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    sec = int(seconds)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def safe_disk_usage_gb(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                continue
    return total / (1024**3)


@dataclass
class WorkerConfig:
    hf_token: str
    output_repo: str
    source_repo: str
    tar_filename: str
    batch_size: int
    max_files: int

    @staticmethod
    def from_env() -> "WorkerConfig":
        return WorkerConfig(
            hf_token=os.getenv("HF_TOKEN", "").strip(),
            output_repo=os.getenv("OUTPUT_REPO", "").strip(),
            source_repo=os.getenv(
                "SOURCE_REPO", "projectlosangeles/Godzilla-MIDI-Dataset"
            ).strip(),
            tar_filename=os.getenv(
                "TAR_FILENAME", "Godzilla-Piano-MIDI-Dataset-CC-BY-NC-SA.tar.gz"
            ).strip(),
            batch_size=max(1, parse_int_env("BATCH_SIZE", 500)),
            max_files=max(0, parse_int_env("MAX_FILES", 0)),
        )


class SymusicTripletTokenizer:
    """Symusic parser + quantizer matching data/tokenizer_custom.py bins."""

    DELTA_START = 0
    DELTA_END = 31
    PITCH_START = 32
    PITCH_END = 119
    DUR_START = 120
    DUR_END = 151

    def __init__(self) -> None:
        # Exact bin definitions from data/tokenizer_custom.py.
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
        p = int(max(21, min(108, pitch)))
        return int(self.PITCH_START + (p - 21))

    def parse_events(
        self, midi_bytes: bytes
    ) -> Optional[List[Tuple[float, int, float]]]:
        """Return piano events or None if non-piano/drums present."""

        score = Score.from_midi(midi_bytes, ttype="second")
        events: List[Tuple[float, int, float]] = []

        if len(score.tracks) == 0:
            return None

        for track in score.tracks:
            program = int(track.program)
            is_drum = bool(track.is_drum)
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
                events.append((onset, pitch, duration))

        events.sort(key=lambda x: (x[0], x[1], x[2]))
        return events

    def encode_events(
        self,
        events: List[Tuple[float, int, float]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        token_ids: List[int] = []
        onset_times: List[float] = []
        durations: List[float] = []
        prev_onset = 0.0

        for onset, pitch, duration in events:
            delta = float(max(0.0, onset - prev_onset))
            prev_onset = onset

            d_tok = self._quantize_delta(delta)
            p_tok = self._quantize_pitch(pitch)
            u_tok = self._quantize_duration(duration)

            token_ids.extend([d_tok, p_tok, u_tok])
            onset_times.extend([float(onset), float(onset), float(onset)])
            durations.extend([float(duration), float(duration), float(duration)])

        return (
            np.asarray(token_ids, dtype=np.int16),
            np.asarray(onset_times, dtype=np.float32),
            np.asarray(durations, dtype=np.float32),
        )


class TokenizerWorker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.stop_requested = False
        self.running = False
        self.last_error = ""
        self.logs: deque[str] = deque(maxlen=2000)

        self.current_index = -1
        self.total_members = 0
        self.accepted = 0
        self.skipped = 0
        self.last_file = ""
        self.start_time = 0.0
        self.start_index = 0
        self.message = "Idle"

        self.work_dir = Path("/tmp/hf_tokenizer_space")
        self.cache_dir = self.work_dir / "cache"
        self.meta_dir = self.work_dir / "meta"
        self.batch_dir = self.work_dir / "batch"
        self.source_dir = self.work_dir / "source"
        self.verify_dir = self.work_dir / "verify"
        for d in [
            self.work_dir,
            self.cache_dir,
            self.meta_dir,
            self.batch_dir,
            self.source_dir,
            self.verify_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        with self.lock:
            self.logs.append(f"[{stamp}] {text}")

    def _set_progress(
        self,
        *,
        current_index: Optional[int] = None,
        total_members: Optional[int] = None,
        accepted: Optional[int] = None,
        skipped: Optional[int] = None,
        last_file: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        with self.lock:
            if current_index is not None:
                self.current_index = int(current_index)
            if total_members is not None:
                self.total_members = int(total_members)
            if accepted is not None:
                self.accepted = int(accepted)
            if skipped is not None:
                self.skipped = int(skipped)
            if last_file is not None:
                self.last_file = str(last_file)
            if message is not None:
                self.message = str(message)

    def _snapshot(self) -> Tuple[str, str]:
        with self.lock:
            running = self.running
            current_index = self.current_index
            total_members = self.total_members
            accepted = self.accepted
            skipped = self.skipped
            last_file = self.last_file
            start_time = self.start_time
            start_index = self.start_index
            message = self.message
            logs_text = "\n".join(self.logs)
            last_error = self.last_error

        eta: Optional[float] = None
        if (
            running
            and start_time > 0
            and total_members > 0
            and current_index >= start_index
        ):
            elapsed = max(1e-6, time.time() - start_time)
            processed = max(1, current_index - start_index + 1)
            rate = processed / elapsed
            remaining = max(0, total_members - current_index - 1)
            eta = remaining / rate if rate > 0 else None

        status = (
            f"Running: {running}\n"
            f"Current index: {current_index}\n"
            f"Total members: {total_members}\n"
            f"Accepted: {accepted}\n"
            f"Skipped: {skipped}\n"
            f"Last file: {last_file}\n"
            f"ETA: {format_eta(eta)}\n"
            f"Message: {message}"
        )
        if last_error:
            status += f"\nLast error: {last_error}"
        return status, logs_text

    def request_stop(self) -> None:
        with self.lock:
            self.stop_requested = True
        self._log("Stop requested by user.")

    def _should_stop(self) -> bool:
        with self.lock:
            return bool(self.stop_requested)

    def start(self) -> Tuple[str, str]:
        with self.lock:
            if self.running:
                self.logs.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Worker already running."
                )
                return self._snapshot()

            self.running = True
            self.stop_requested = False
            self.last_error = ""
            self.start_time = time.time()
            self.message = "Starting"
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

        return self._snapshot()

    def _retry(
        self,
        desc: str,
        fn,
        attempts: int = 3,
        waits: Tuple[int, ...] = (5, 15, 30),
    ):
        last_exc: Optional[Exception] = None
        for i in range(attempts):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if i == attempts - 1:
                    break
                wait = waits[min(i, len(waits) - 1)]
                self._log(
                    f"{desc} failed ({exc}); retrying in {wait}s (attempt {i + 2}/{attempts})"
                )
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    def _download_tar_with_retry(self, cfg: WorkerConfig) -> Path:
        def _download_once() -> str:
            return hf_hub_download(
                repo_id=cfg.source_repo,
                filename=cfg.tar_filename,
                repo_type="dataset",
                local_dir=str(self.source_dir),
                token=cfg.hf_token or None,
            )

        try:
            local = self._retry("Tar download", _download_once)
            return Path(str(local))
        except Exception as exc:
            self._log(f"Tar download failed ({exc}). Retrying once after 60s...")
            time.sleep(60)
            local = self._retry("Tar download retry", _download_once)
            return Path(str(local))

    def _upload_file(
        self,
        api: HfApi,
        cfg: WorkerConfig,
        local_path: Path,
        path_in_repo: str,
        commit_message: str,
    ) -> None:
        self._retry(
            f"Upload file {path_in_repo}",
            lambda: api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=cfg.output_repo,
                repo_type="dataset",
                token=cfg.hf_token,
                commit_message=commit_message,
            ),
        )

    def _upload_json(
        self,
        api: HfApi,
        cfg: WorkerConfig,
        payload: Dict[str, Any],
        path_in_repo: str,
        commit_message: str,
    ) -> None:
        local_path = self.meta_dir / Path(path_in_repo).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._upload_file(api, cfg, local_path, path_in_repo, commit_message)

    def _download_json_if_exists(
        self, cfg: WorkerConfig, path_in_repo: str
    ) -> Optional[Dict[str, Any]]:
        def _download() -> str:
            return hf_hub_download(
                repo_id=cfg.output_repo,
                filename=path_in_repo,
                repo_type="dataset",
                local_dir=str(self.cache_dir),
                token=cfg.hf_token,
            )

        try:
            local = self._retry(f"Download {path_in_repo}", _download)
        except Exception:
            return None

        try:
            return json.loads(Path(str(local)).read_text(encoding="utf-8"))
        except Exception:
            return None

    def _build_member_index(self, tar_path: Path) -> List[str]:
        names: List[str] = []
        self._log("Building deterministic member index (stream scan)...")
        with tarfile.open(tar_path, mode="r|gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                names.append(member.name)
                if len(names) % 100000 == 0:
                    self._log(f"Indexed members: {len(names):,}")

        names.sort()
        self._log(f"Built sorted member index with {len(names):,} entries.")
        return names

    def _load_or_create_member_index(
        self, api: HfApi, cfg: WorkerConfig, tar_path: Path
    ) -> List[str]:
        payload = self._download_json_if_exists(cfg, "metadata/member_index.json")
        if isinstance(payload, dict) and isinstance(payload.get("member_names"), list):
            names = [str(x) for x in payload["member_names"]]
            self._log(f"Loaded member_index.json from repo ({len(names):,} entries).")
            return names

        names = self._build_member_index(tar_path)
        payload = {
            "created_at": utc_now_iso(),
            "source_repo": cfg.source_repo,
            "tar_filename": cfg.tar_filename,
            "total_members": int(len(names)),
            "member_names": names,
        }
        self._upload_json(
            api,
            cfg,
            payload,
            path_in_repo="metadata/member_index.json",
            commit_message="Add deterministic member index",
        )
        self._log("Uploaded metadata/member_index.json")
        return names

    def _integrity_check(self, cfg: WorkerConfig, checkpoint: Dict[str, Any]) -> bool:
        recent = checkpoint.get("recent_completed", [])
        if not isinstance(recent, list):
            return True

        accepted_recent = [
            e
            for e in recent
            if isinstance(e, dict)
            and bool(e.get("accepted"))
            and isinstance(e.get("npz_path"), str)
        ]
        targets = accepted_recent[-2:]
        if len(targets) < 2:
            self._log(
                "Integrity check: PASSED (insufficient accepted history for 2-entry verification)"
            )
            return True

        for entry in targets:
            npz_repo_path = str(entry.get("npz_path"))
            expected_len = int(entry.get("token_length", -1))

            def _download_npz() -> str:
                return hf_hub_download(
                    repo_id=cfg.output_repo,
                    filename=npz_repo_path,
                    repo_type="dataset",
                    local_dir=str(self.verify_dir),
                    token=cfg.hf_token,
                )

            try:
                local_npz = self._retry(
                    f"Integrity download {npz_repo_path}", _download_npz
                )
                with np.load(local_npz, allow_pickle=False) as pack:
                    actual_len = int(pack["tokens"].shape[0])
                if actual_len != expected_len:
                    self._log(
                        f"Integrity mismatch for {npz_repo_path}: expected len={expected_len}, got len={actual_len}"
                    )
                    return False
            except Exception as exc:
                self._log(f"Integrity check failed for {npz_repo_path}: {exc}")
                return False

        self._log("Integrity check: PASSED")
        return True

    @staticmethod
    def _estimate_eta(
        start_time: float,
        start_index: int,
        current_index: int,
        total_members: int,
    ) -> Optional[float]:
        if current_index < start_index:
            return None
        elapsed = max(1e-6, time.time() - start_time)
        processed = max(1, current_index - start_index + 1)
        rate = processed / elapsed
        if rate <= 0:
            return None
        remaining = max(0, total_members - current_index - 1)
        return remaining / rate

    def _run(self) -> None:
        cfg = WorkerConfig.from_env()
        api = HfApi(token=cfg.hf_token if cfg.hf_token else None)

        with self.lock:
            self.message = "Bootstrapping"

        try:
            if not cfg.hf_token:
                raise RuntimeError("Missing HF_TOKEN environment variable.")
            if not cfg.output_repo:
                raise RuntimeError("Missing OUTPUT_REPO environment variable.")

            self._log(f"Output repo: {cfg.output_repo}")
            self._log(f"Source repo: {cfg.source_repo}")
            self._log(f"Tar filename: {cfg.tar_filename}")
            self._log(f"Batch size: {cfg.batch_size}")
            self._log(
                f"Max files cap: {cfg.max_files if cfg.max_files > 0 else 'unbounded'}"
            )

            self._retry(
                "Create dataset repo",
                lambda: api.create_repo(
                    repo_id=cfg.output_repo,
                    repo_type="dataset",
                    private=True,
                    exist_ok=True,
                    token=cfg.hf_token,
                ),
            )
            self._log("Output dataset repo verified/created (private).")

            tar_path = self._download_tar_with_retry(cfg)
            tar_size_gb = tar_path.stat().st_size / (1024**3)
            self._log(f"Tar downloaded: {tar_path} ({tar_size_gb:.2f} GB)")

            member_names = self._load_or_create_member_index(api, cfg, tar_path)
            total_members = len(member_names)
            self._set_progress(total_members=total_members)

            checkpoint = self._download_json_if_exists(cfg, "metadata/checkpoint.json")
            if checkpoint is None:
                checkpoint = {
                    "last_completed_index": -1,
                    "accepted": 0,
                    "skipped": 0,
                    "last_completed_name": "",
                    "total_members": int(total_members),
                    "recent_completed": [],
                }

            last_completed_index = int(checkpoint.get("last_completed_index", -1))
            last_completed_name = str(checkpoint.get("last_completed_name", ""))
            accepted = int(checkpoint.get("accepted", 0))
            skipped = int(checkpoint.get("skipped", 0))
            recent_completed = checkpoint.get("recent_completed", [])
            if not isinstance(recent_completed, list):
                recent_completed = []

            if last_completed_index >= total_members:
                last_completed_index = total_members - 1

            start_index = max(0, last_completed_index + 1)
            if last_completed_index >= 0:
                self._log(
                    f"Resuming from index {last_completed_index} (last: {last_completed_name})"
                )
                integrity_ok = self._integrity_check(cfg, checkpoint)
                if not integrity_ok:
                    self._log("Integrity check: FAILED")
                    rollback_start = max(0, last_completed_index - 1)
                    for entry in recent_completed:
                        if not isinstance(entry, dict):
                            continue
                        idx = int(entry.get("index", -1))
                        if rollback_start <= idx <= last_completed_index:
                            if bool(entry.get("accepted")):
                                accepted = max(0, accepted - 1)
                            else:
                                skipped = max(0, skipped - 1)
                    start_index = rollback_start
                    self._log(
                        f"Rolling back by 2 entries. New start index: {start_index}"
                    )
                else:
                    self._log("Integrity check: PASSED")

                self._log(f"Accepted so far: {accepted} | Skipped: {skipped}")
            else:
                self._log("No checkpoint found. Starting from index 0.")

            self.start_time = time.time()
            self.start_index = start_index
            self._set_progress(
                current_index=max(-1, start_index - 1),
                total_members=total_members,
                accepted=accepted,
                skipped=skipped,
                last_file=last_completed_name,
                message="Preparing tar member lookup",
            )

            # Random-access tar handle (no full extraction to disk).
            tar = tarfile.open(tar_path, mode="r:gz")
            try:
                self._log("Building tar member lookup map for extraction...")
                member_lookup = {m.name: m for m in tar.getmembers() if m.isfile()}
                self._log(f"Tar member lookup ready ({len(member_lookup):,} files).")

                # Batch state.
                if self.batch_dir.exists():
                    shutil.rmtree(self.batch_dir, ignore_errors=True)
                (self.batch_dir / "data").mkdir(parents=True, exist_ok=True)

                pending_accept_count = 0
                processed_since_checkpoint = 0
                batch_first_index: Optional[int] = None
                batch_last_index: Optional[int] = None
                tokenizer = SymusicTripletTokenizer()

                def write_checkpoint() -> None:
                    cp_payload = {
                        "last_completed_index": int(last_completed_index),
                        "accepted": int(accepted),
                        "skipped": int(skipped),
                        "last_completed_name": str(last_completed_name),
                        "total_members": int(total_members),
                        "recent_completed": recent_completed[-32:],
                        "updated_at": utc_now_iso(),
                    }
                    self._upload_json(
                        api,
                        cfg,
                        cp_payload,
                        path_in_repo="metadata/checkpoint.json",
                        commit_message=(
                            f"Checkpoint idx={last_completed_index} accepted={accepted} skipped={skipped}"
                        ),
                    )

                def upload_pending_data_if_any() -> None:
                    nonlocal pending_accept_count
                    nonlocal batch_first_index
                    nonlocal batch_last_index
                    if pending_accept_count <= 0:
                        return

                    first_idx = int(
                        batch_first_index if batch_first_index is not None else -1
                    )
                    last_idx = int(
                        batch_last_index if batch_last_index is not None else -1
                    )

                    self._retry(
                        "Upload data batch",
                        lambda: api.upload_folder(
                            folder_path=str(self.batch_dir),
                            repo_id=cfg.output_repo,
                            repo_type="dataset",
                            token=cfg.hf_token,
                            commit_message=(
                                f"Upload batch data idx={first_idx}..{last_idx} accepted_total={accepted}"
                            ),
                        ),
                    )

                    eta = self._estimate_eta(
                        start_time=self.start_time,
                        start_index=self.start_index,
                        current_index=last_completed_index,
                        total_members=total_members,
                    )
                    self._log(
                        "Batch upload complete: "
                        f"accepted={accepted:,} skipped={skipped:,} "
                        f"range={first_idx}..{last_idx} ETA={format_eta(eta)}"
                    )

                    shutil.rmtree(self.batch_dir, ignore_errors=True)
                    (self.batch_dir / "data").mkdir(parents=True, exist_ok=True)
                    pending_accept_count = 0
                    batch_first_index = None
                    batch_last_index = None

                for idx in range(start_index, total_members):
                    if self._should_stop():
                        self._log(
                            "Stop acknowledged. Finalizing pending uploads/checkpoint..."
                        )
                        break

                    if cfg.max_files > 0 and accepted >= cfg.max_files:
                        self._log(f"MAX_FILES reached ({cfg.max_files}). Stopping.")
                        break

                    if safe_disk_usage_gb(self.work_dir) > 14.0:
                        self._log("Disk usage exceeded 14GB. Stopping.")
                        break

                    name = member_names[idx]
                    last_completed_index = idx
                    last_completed_name = name
                    processed_since_checkpoint += 1

                    accepted_entry = False
                    entry_npz_path: Optional[str] = None
                    token_len = 0

                    try:
                        if not (
                            name.lower().endswith(".mid")
                            or name.lower().endswith(".midi")
                        ):
                            skipped += 1
                        else:
                            tar_info = member_lookup.get(name)
                            if tar_info is None:
                                skipped += 1
                            else:
                                member_fp = tar.extractfile(tar_info)
                                if member_fp is None:
                                    skipped += 1
                                else:
                                    midi_bytes = member_fp.read()
                                    events = tokenizer.parse_events(midi_bytes)
                                    if events is None:
                                        skipped += 1
                                    else:
                                        tokens, onsets, durations = (
                                            tokenizer.encode_events(events)
                                        )
                                        token_len = int(tokens.shape[0])
                                        if token_len < 64 * 3:
                                            skipped += 1
                                        else:
                                            md5 = md5_text(name)
                                            rel_path = f"data/{idx:07d}_{md5}.npz"
                                            local_npz = self.batch_dir / rel_path
                                            local_npz.parent.mkdir(
                                                parents=True, exist_ok=True
                                            )
                                            np.savez_compressed(
                                                local_npz,
                                                tokens=tokens,
                                                onsets=onsets,
                                                durations=durations,
                                            )
                                            accepted += 1
                                            pending_accept_count += 1
                                            accepted_entry = True
                                            entry_npz_path = rel_path
                                            if batch_first_index is None:
                                                batch_first_index = idx
                                            batch_last_index = idx
                    except Exception as file_exc:
                        skipped += 1
                        self._log(f"File error at index {idx} ({name}): {file_exc}")

                    recent_completed.append(
                        {
                            "index": int(idx),
                            "name": name,
                            "accepted": bool(accepted_entry),
                            "npz_path": entry_npz_path,
                            "token_length": int(token_len),
                        }
                    )
                    if len(recent_completed) > 64:
                        recent_completed = recent_completed[-64:]

                    self._set_progress(
                        current_index=idx,
                        total_members=total_members,
                        accepted=accepted,
                        skipped=skipped,
                        last_file=name,
                        message="Processing",
                    )

                    if (idx + 1) % 100 == 0:
                        eta = self._estimate_eta(
                            start_time=self.start_time,
                            start_index=self.start_index,
                            current_index=idx,
                            total_members=total_members,
                        )
                        self._log(
                            f"Progress idx={idx:,} accepted={accepted:,} skipped={skipped:,} ETA={format_eta(eta)}"
                        )

                    # Push data every BATCH_SIZE accepted files.
                    if pending_accept_count >= cfg.batch_size:
                        upload_pending_data_if_any()
                        write_checkpoint()
                        processed_since_checkpoint = 0

                    # Also checkpoint every BATCH_SIZE processed entries.
                    elif processed_since_checkpoint >= cfg.batch_size:
                        write_checkpoint()
                        processed_since_checkpoint = 0

                # Final flush.
                upload_pending_data_if_any()
                write_checkpoint()

            finally:
                tar.close()

            self._set_progress(message="Completed")
            self._log("Worker finished.")

        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)
                self.message = "Failed"
            self._log(f"Fatal worker error: {exc}")

        finally:
            with self.lock:
                self.running = False
                self.stop_requested = False


WORKER = TokenizerWorker()


def start_worker() -> Tuple[str, str]:
    return WORKER.start()


def stop_worker() -> Tuple[str, str]:
    WORKER.request_stop()
    return WORKER._snapshot()


def poll_status() -> Tuple[str, str]:
    return WORKER._snapshot()


with gr.Blocks(title="Godzilla Piano Tokenizer") as demo:
    gr.Markdown("# Godzilla Piano Tokenizer")
    gr.Markdown(
        "Bulk tokenization worker for Godzilla Piano MIDI tar. "
        "Progress and logs auto-refresh every 10 seconds."
    )

    status_box = gr.Textbox(label="Status", lines=10, interactive=False)
    logs_box = gr.Textbox(label="Live Logs", lines=24, interactive=False)

    with gr.Row():
        start_btn = gr.Button("Start", variant="primary")
        stop_btn = gr.Button("Stop", variant="stop")

    start_btn.click(fn=start_worker, outputs=[status_box, logs_box])
    stop_btn.click(fn=stop_worker, outputs=[status_box, logs_box])
    demo.load(fn=poll_status, outputs=[status_box, logs_box])

    timer = gr.Timer(10.0)
    timer.tick(fn=poll_status, outputs=[status_box, logs_box])


if __name__ == "__main__":
    demo.queue().launch()

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional


class DriveSync:
    def __init__(self, drive_root: str = "/content/drive/MyDrive/piano_model"):
        self.in_colab = self._is_colab_runtime()
        self.requested_drive_root = drive_root

        root = Path(drive_root)
        drive_root_norm = str(drive_root).replace("\\", "/")
        if not self.in_colab and drive_root_norm.startswith("/content/drive"):
            root = Path.cwd() / "local_drive" / "piano_model"

        self.drive_root = root
        self.checkpoints_dir = self.drive_root / "checkpoints"
        self.logs_dir = self.drive_root / "logs"
        self.tokenizer_dir = self.drive_root / "tokenizer"
        self.processed_dir = self.drive_root / "processed"
        self.generated_dir = self.drive_root / "generated"

        self.local_root = (
            Path("/content") if self.in_colab else Path.cwd() / ".session_cache"
        )
        self.local_checkpoints_dir = self.local_root / "checkpoints"
        self.local_logs_dir = self.local_root / "logs"
        self.local_tokenizer_dir = self.local_root / "tokenizer"
        self.local_processed_dir = self.local_root / "processed"
        self.local_generated_dir = self.local_root / "generated"
        self.local_tokenizer_path = self.local_tokenizer_dir / "tokenizer.json"

        for p in [
            self.checkpoints_dir,
            self.logs_dir,
            self.tokenizer_dir,
            self.processed_dir,
            self.generated_dir,
            self.local_checkpoints_dir,
            self.local_logs_dir,
            self.local_tokenizer_dir,
            self.local_processed_dir,
            self.local_generated_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)

        self._sync_threads: List[threading.Thread] = []
        self._sync_lock = threading.Lock()
        self._pending_syncs: set[str] = set()
        self.keep_every_n_epochs: int = 10
        self.last_restored_state_path: Optional[str] = None
        self.free_space_gb = self._check_available_space_gb()

        self._ensure_log_file(self.logs_dir / "training_log.json")
        self._ensure_log_file(Path(__file__).resolve().parent / "training_log.json")

    def mount(self) -> bool:
        if not self.in_colab:
            print(
                "Warning: not running in Colab. Google Drive mount unavailable; "
                f"using local path: {self.drive_root}"
            )
            return False

        if Path("/content/drive/MyDrive").exists():
            print("Drive already mounted.")
            for p in [
                self.checkpoints_dir,
                self.logs_dir,
                self.tokenizer_dir,
                self.processed_dir,
                self.generated_dir,
            ]:
                p.mkdir(parents=True, exist_ok=True)
            self.free_space_gb = self._check_available_space_gb()
            return True

        try:
            from google.colab import drive  # type: ignore[import-not-found]

            drive.mount("/content/drive")
            for p in [
                self.checkpoints_dir,
                self.logs_dir,
                self.tokenizer_dir,
                self.processed_dir,
                self.generated_dir,
            ]:
                p.mkdir(parents=True, exist_ok=True)
            self.free_space_gb = self._check_available_space_gb()
            return True
        except Exception as exc:  # pragma: no cover
            msg = str(exc).lower()
            if "already" in msg or "mountpoint" in msg:
                print("Drive already mounted.")
                return True
            print(f"Drive mount failed: {exc}. Using local cache paths instead.")
            return False

    def set_checkpoint_retention(self, keep_every_n_epochs: int) -> None:
        value = int(keep_every_n_epochs)
        self.keep_every_n_epochs = 0 if value <= 0 else value

    def should_keep_checkpoint(self, epoch: int) -> bool:
        return int(epoch) % max(1, self.keep_every_n_epochs) == 0

    def _rotation_enabled(self) -> bool:
        return int(self.keep_every_n_epochs) > 0

    def sync_checkpoint(self, local_path: str, tag: str = "latest") -> bool:
        try:
            src = Path(local_path)
            if not src.exists():
                print(f"Checkpoint sync skipped: missing local file {src}")
                return False

            tmp_dst = self.checkpoints_dir / f"{tag}_tmp.safetensors"
            final_dst = self.checkpoints_dir / f"{tag}.safetensors"

            shutil.copy2(src, tmp_dst)
            os.replace(tmp_dst, final_dst)

            state_src = self._find_state_sidecar(src)
            if state_src is not None and state_src.exists():
                state_tmp = self.checkpoints_dir / f"{tag}_tmp_state.pt"
                state_dst = self.checkpoints_dir / f"{tag}_state.pt"
                shutil.copy2(state_src, state_tmp)
                os.replace(state_tmp, state_dst)

            size_mb = final_dst.stat().st_size / (1024**2)
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] Synced checkpoint '{tag}' ({size_mb:.2f} MB) -> {final_dst}")
            self.cleanup_old_checkpoints()
            return True
        except Exception as exc:
            print(f"Checkpoint sync failed for tag='{tag}': {exc}")
            return False

    def sync_checkpoint_background(self, local_path: str, tag: str) -> None:
        with self._sync_lock:
            if tag in self._pending_syncs:
                return
            self._pending_syncs.add(tag)

            t = threading.Thread(
                target=self._sync_and_clear,
                args=(local_path, tag),
                daemon=True,
            )
            self._sync_threads.append(t)
        t.start()

    def _sync_and_clear(self, local_path: str, tag: str) -> None:
        try:
            self.sync_checkpoint(local_path=local_path, tag=tag)
        finally:
            with self._sync_lock:
                self._pending_syncs.discard(tag)

    def cleanup_old_checkpoints(self) -> None:
        if not self._rotation_enabled():
            return

        for file_path in self.checkpoints_dir.glob("epoch_*.safetensors"):
            epoch_num = self._parse_epoch_from_name(file_path.name)
            if epoch_num is None:
                continue
            if not self.should_keep_checkpoint(epoch_num):
                try:
                    file_path.unlink()
                except Exception:
                    pass

        for file_path in self.checkpoints_dir.glob("epoch_*_state.pt"):
            epoch_num = self._parse_epoch_from_name(file_path.name)
            if epoch_num is None:
                continue
            if not self.should_keep_checkpoint(epoch_num):
                try:
                    file_path.unlink()
                except Exception:
                    pass

    def wait_for_sync(self) -> None:
        with self._sync_lock:
            threads = list(self._sync_threads)
            self._sync_threads.clear()

        for t in threads:
            t.join()

        print("All checkpoints synced to Drive")

    def restore_checkpoint(self, tag: str = "latest") -> Optional[str]:
        drive_ckpt = self.checkpoints_dir / f"{tag}.safetensors"
        if not drive_ckpt.exists():
            return None

        local_ckpt = self.local_checkpoints_dir / f"{tag}.safetensors"
        shutil.copy2(drive_ckpt, local_ckpt)

        self.last_restored_state_path = None
        state_candidates = [
            self.checkpoints_dir / f"{tag}_state.pt",
            self.checkpoints_dir / "latest_state.pt",
        ]
        for state_src in state_candidates:
            if state_src.exists():
                local_state = self.local_checkpoints_dir / state_src.name
                shutil.copy2(state_src, local_state)
                self.last_restored_state_path = str(local_state)
                break

        print(f"Restored checkpoint '{tag}' from {drive_ckpt} -> {local_ckpt}")
        return str(local_ckpt)

    def sync_processed_data(self) -> str:
        drive_manifest = self.processed_dir / "manifest.json"
        local_manifest = self.local_processed_dir / "manifest.json"

        drive_count = self._manifest_count(drive_manifest)
        local_count = self._manifest_count(local_manifest)

        if drive_count > 0:
            self._copy_tree(self.processed_dir, self.local_processed_dir)
            print("Processed data restored from Drive cache.")
            return "restored"

        if local_count > 0:
            self._copy_tree(self.local_processed_dir, self.processed_dir)
            print("Processed data uploaded to Drive cache.")
            return "uploaded"

        print("Processed data not found in Drive or local cache.")
        return "missing"

    def sync_tokenizer(self) -> Optional[str]:
        drive_tok = self.tokenizer_dir / "tokenizer.json"
        local_tok = self.local_tokenizer_path

        if drive_tok.exists():
            shutil.copy2(drive_tok, local_tok)
            print(f"Tokenizer restored from Drive: {drive_tok}")
            return str(local_tok)

        if local_tok.exists():
            drive_tok.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_tok, drive_tok)
            print(f"Tokenizer uploaded to Drive: {drive_tok}")
            return str(local_tok)

        print("Tokenizer not found in Drive or local cache.")
        return None

    def sync_log(self, log_data: Dict[str, Any]) -> None:
        drive_log = self.logs_dir / "training_log.json"
        local_log = Path(__file__).resolve().parent / "training_log.json"

        history = self._load_json(drive_log, {"sessions": []})
        sessions = history.get("sessions")
        if not isinstance(sessions, list):
            history["sessions"] = []
            sessions = history["sessions"]
        sessions.append(log_data)

        self._atomic_write_json(drive_log, history)
        self._atomic_write_json(local_log, history)

    def get_training_history(self) -> Dict[str, Any]:
        history_path = self.logs_dir / "training_log.json"
        history = self._load_json(history_path, {"sessions": []})
        sessions = history.get("sessions", [])
        if not isinstance(sessions, list):
            sessions = []

        total_sessions = len(sessions)
        total_epochs = sum(int(s.get("epochs_completed", 0) or 0) for s in sessions)
        total_seconds = sum(self._session_duration_seconds(s) for s in sessions)
        best_val = None
        current_epoch = 0

        for s in sessions:
            val = s.get("final_val_loss")
            if isinstance(val, (int, float)):
                if best_val is None or float(val) < float(best_val):
                    best_val = float(val)
            end_epoch = s.get("end_epoch")
            if isinstance(end_epoch, int):
                current_epoch = max(current_epoch, end_epoch)

        best_str = f"{best_val:.4f}" if best_val is not None else "n/a"
        print(
            "Training history summary: "
            f"sessions={total_sessions}, epochs={total_epochs}, "
            f"time={total_seconds / 3600.0:.2f}h, best_val={best_str}, "
            f"current_epoch={current_epoch}"
        )
        return history

    def write_heartbeat(self, payload: Dict[str, Any]) -> None:
        heartbeat_path = self.logs_dir / "heartbeat_latest.json"
        self._atomic_write_json(heartbeat_path, payload)

    def _check_available_space_gb(self) -> float:
        check_path = self.drive_root
        if self.in_colab:
            colab_drive = Path("/content/drive/MyDrive")
            if colab_drive.exists():
                check_path = colab_drive

        try:
            _total, _used, free = shutil.disk_usage(str(check_path))
            free_gb = free / (1024**3)
            if free_gb < 5.0:
                warnings.warn(
                    f"Low Drive space: {free_gb:.2f} GB free. "
                    "Consider cleaning old checkpoints."
                )
            return free_gb
        except Exception as exc:
            warnings.warn(f"Unable to read Drive space for {check_path}: {exc}")
            return -1.0

    @staticmethod
    def _is_colab_runtime() -> bool:
        try:
            import google.colab  # type: ignore # noqa: F401

            return True
        except Exception:
            return False

    @staticmethod
    def _copy_tree(src: Path, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    @staticmethod
    def _find_state_sidecar(model_path: Path) -> Optional[Path]:
        if model_path.name.endswith("_model.safetensors"):
            candidate = model_path.with_name(
                model_path.name.replace("_model.safetensors", "_state.pt")
            )
            if candidate.exists():
                return candidate

        latest_state = model_path.parent / "latest_state.pt"
        if latest_state.exists():
            return latest_state
        return None

    @staticmethod
    def _ensure_log_file(path: Path) -> None:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"sessions": []}, indent=2), encoding="utf-8")

    @staticmethod
    def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
        if not path.exists():
            return dict(default)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return dict(default)

    @staticmethod
    def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _session_duration_seconds(session: Dict[str, Any]) -> float:
        start = session.get("start_time")
        end = session.get("end_time")
        if not isinstance(start, str) or not isinstance(end, str):
            return 0.0
        try:
            start_dt = dt.datetime.fromisoformat(start)
            end_dt = dt.datetime.fromisoformat(end)
            return max(0.0, (end_dt - start_dt).total_seconds())
        except Exception:
            return 0.0

    @staticmethod
    def _manifest_count(path: Path) -> int:
        if not path.exists():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data)
        except Exception:
            return 0
        return 0

    @staticmethod
    def _parse_epoch_from_name(filename: str) -> Optional[int]:
        stem = Path(filename).stem
        parts = stem.split("_")
        if len(parts) < 2:
            return None
        if parts[0] != "epoch":
            return None
        try:
            return int(parts[1])
        except Exception:
            return None

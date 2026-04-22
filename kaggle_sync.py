from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class _MirrorRequest:
    checkpoint_dir: Path
    epoch: int
    global_step: int
    val_loss: float
    best_val_loss: float
    save_tag: str
    best: bool
    created_at: float


def _slug_to_title(slug: str) -> str:
    parts = [part for part in re.split(r"[-_]+", slug) if part]
    if not parts:
        return "Checkpoint Mirror"
    return " ".join(part.capitalize() for part in parts)


def _find_kaggle_username() -> str:
    env_username = str(os.environ.get("KAGGLE_USERNAME", "")).strip()
    if env_username:
        return env_username

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            payload = json.loads(kaggle_json.read_text(encoding="utf-8"))
            for key in ("username", "kaggle_username", "owner"):
                value = str(payload.get(key, "")).strip()
                if value:
                    return value
        except Exception:
            pass

    return ""


def _has_kaggle_credentials() -> bool:
    if str(os.environ.get("KAGGLE_API_TOKEN", "")).strip():
        return True
    if str(os.environ.get("KAGGLE_USERNAME", "")).strip() and str(
        os.environ.get("KAGGLE_KEY", "")
    ).strip():
        return True

    kaggle_home = Path.home() / ".kaggle"
    return any(
        path.exists()
        for path in [
            kaggle_home / "kaggle.json",
            kaggle_home / "access_token",
        ]
    )


class KaggleCheckpointMirror:
    """Mirror checkpoint snapshots to a Kaggle dataset in the background."""

    def __init__(
        self,
        *,
        dataset_id: str,
        checkpoint_dir: Path,
        title: str = "",
        license_name: str = "CC0-1.0",
        public: bool = False,
        staging_root: Optional[Path] = None,
        max_attempts: int = 3,
        backoff_seconds: float = 30.0,
    ) -> None:
        self.dataset_id = self._normalize_dataset_id(dataset_id)
        self.dataset_slug = self.dataset_id.split("/")[-1] if self.dataset_id else ""
        self.dataset_title = str(title).strip() or _slug_to_title(self.dataset_slug)
        self.license_name = str(license_name).strip() or "CC0-1.0"
        self.public = bool(public)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.staging_root = (
            Path(staging_root).expanduser()
            if staging_root is not None
            else self.checkpoint_dir.parent / "_kaggle_sync"
        )
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_seconds = max(1.0, float(backoff_seconds))

        self._lock = threading.Lock()
        self._pending: Optional[_MirrorRequest] = None
        self._worker: Optional[threading.Thread] = None
        self._remote_dataset_exists: Optional[bool] = None
        self._auth_warning_emitted = False

        self.staging_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_dataset_id(value: str) -> str:
        raw = str(value or "").strip().strip("/")
        if not raw:
            return ""

        raw = raw.split("?", 1)[0].split("#", 1)[0].strip("/")
        parts = [part for part in raw.split("/") if part]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"

        owner = _find_kaggle_username()
        if owner:
            return f"{owner}/{parts[0]}"

        raise ValueError(
            "Kaggle dataset id must be owner/slug, or set KAGGLE_USERNAME so a bare slug can be resolved."
        )

    @classmethod
    def from_env(
        cls,
        *,
        checkpoint_dir: Path,
        title: str = "",
        license_name: str = "CC0-1.0",
        public: bool = False,
        staging_root: Optional[Path] = None,
        max_attempts: int = 3,
        backoff_seconds: float = 30.0,
    ) -> Optional["KaggleCheckpointMirror"]:
        raw_dataset_id = str(os.environ.get("KAGGLE_SYNC_DATASET", "")).strip()
        if not raw_dataset_id:
            return None

        title_value = str(os.environ.get("KAGGLE_SYNC_TITLE", title)).strip()
        license_value = str(os.environ.get("KAGGLE_SYNC_LICENSE", license_name)).strip()
        public_env = str(os.environ.get("KAGGLE_SYNC_PUBLIC", "")).strip().lower()
        public_value = public
        if public_env:
            public_value = public_env in {"1", "true", "yes", "on"}

        return cls(
            dataset_id=raw_dataset_id,
            checkpoint_dir=checkpoint_dir,
            title=title_value,
            license_name=license_value,
            public=public_value,
            staging_root=staging_root,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )

    def enabled(self) -> bool:
        return bool(self.dataset_id)

    def _kaggle_command(self) -> List[str]:
        kaggle_executable = shutil.which("kaggle")
        if kaggle_executable:
            return [kaggle_executable]
        return [sys.executable, "-m", "kaggle"]

    def _ensure_cli(self) -> bool:
        if shutil.which("kaggle") is not None:
            return True

        try:
            import importlib.util

            if importlib.util.find_spec("kaggle") is None:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "--disable-pip-version-check",
                        "kaggle",
                    ],
                    check=True,
                )
            return importlib.util.find_spec("kaggle") is not None
        except Exception:
            return False

    def _has_credentials(self) -> bool:
        return _has_kaggle_credentials()

    def _stage_root_for(self, request: _MirrorRequest) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(request.created_at))
        return self.staging_root / f"step_{int(request.global_step):09d}_{stamp}"

    def _dataset_metadata_payload(self) -> dict:
        return {
            "title": self.dataset_title,
            "id": self.dataset_id,
            "licenses": [{"name": self.license_name}],
        }

    def _write_stage(self, stage_dir: Path, request: _MirrorRequest) -> List[str]:
        stage_dir.mkdir(parents=True, exist_ok=True)
        copied_files: List[str] = []

        for candidate in sorted(self.checkpoint_dir.glob("*.safetensors")):
            if candidate.is_file():
                shutil.copy2(candidate, stage_dir / candidate.name)
                copied_files.append(candidate.name)

        for candidate in sorted(self.checkpoint_dir.glob("*_state.pt")):
            if candidate.is_file():
                shutil.copy2(candidate, stage_dir / candidate.name)
                copied_files.append(candidate.name)

        summary = {
            "dataset_id": self.dataset_id,
            "dataset_title": self.dataset_title,
            "save_tag": request.save_tag,
            "best": bool(request.best),
            "epoch": int(request.epoch),
            "global_step": int(request.global_step),
            "val_loss": float(request.val_loss),
            "best_val_loss": float(request.best_val_loss),
            "checkpoint_dir": str(self.checkpoint_dir.resolve()),
            "created_at": float(request.created_at),
            "source_files": copied_files,
        }
        (stage_dir / "checkpoint_sync_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        (stage_dir / "dataset-metadata.json").write_text(
            json.dumps(self._dataset_metadata_payload(), indent=2),
            encoding="utf-8",
        )
        return copied_files

    def _run_kaggle(self, args: List[str], stage_dir: Path) -> subprocess.CompletedProcess[str]:
        cmd = self._kaggle_command() + args
        env = os.environ.copy()
        return subprocess.run(
            cmd,
            cwd=str(stage_dir),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    def _dataset_exists_remote(self) -> bool:
        try:
            self._run_kaggle(["datasets", "status", self.dataset_id], self.staging_root)
            return True
        except subprocess.CalledProcessError:
            return False

    def _publish_once(self, request: _MirrorRequest) -> None:
        stage_dir = self._stage_root_for(request)
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        copied_files = self._write_stage(stage_dir, request)
        if not copied_files:
            raise FileNotFoundError(
                f"No checkpoint artifacts found under {self.checkpoint_dir.resolve()}"
            )

        message = (
            f"step={int(request.global_step):06d} epoch={int(request.epoch):03d} "
            f"val_loss={float(request.val_loss):.6f} best={float(request.best_val_loss):.6f}"
        )

        if self._remote_dataset_exists is None:
            self._remote_dataset_exists = self._dataset_exists_remote()

        try:
            if not self._remote_dataset_exists:
                args = [
                    "datasets",
                    "create",
                    "-p",
                    str(stage_dir),
                    "-q",
                    "-t",
                    "-r",
                    "skip",
                ]
                if self.public:
                    args.append("--public")
                try:
                    self._run_kaggle(args, stage_dir)
                    self._remote_dataset_exists = True
                except subprocess.CalledProcessError as exc:
                    error_text = " ".join(
                        part
                        for part in [
                            str(getattr(exc, "stdout", "") or "").strip(),
                            str(getattr(exc, "stderr", "") or "").strip(),
                            str(exc),
                        ]
                        if part
                    ).lower()
                    if "already exists" not in error_text and "already" not in error_text:
                        raise
                    self._remote_dataset_exists = True
                    self._run_kaggle(
                        [
                            "datasets",
                            "version",
                            "-p",
                            str(stage_dir),
                            "-m",
                            message,
                            "-q",
                            "-t",
                            "-r",
                            "skip",
                        ],
                        stage_dir,
                    )
            else:
                self._run_kaggle(
                    [
                        "datasets",
                        "version",
                        "-p",
                        str(stage_dir),
                        "-m",
                        message,
                        "-q",
                        "-t",
                        "-r",
                        "skip",
                    ],
                    stage_dir,
                )
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    def _retry_publish(self, request: _MirrorRequest) -> None:
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                self._publish_once(request)
                return
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    time.sleep(self.backoff_seconds * float(attempt))
        if last_error is not None:
            raise last_error

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                request = self._pending
                self._pending = None
                if request is None:
                    self._worker = None
                    return

            try:
                if not self._has_credentials():
                    if not self._auth_warning_emitted:
                        print(
                            "Kaggle checkpoint mirror is disabled because no Kaggle credentials were found. "
                            "Set KAGGLE_API_TOKEN or a kaggle.json file to enable uploads."
                        )
                        self._auth_warning_emitted = True
                    continue

                if not self._ensure_cli():
                    if not self._auth_warning_emitted:
                        print(
                            "Kaggle checkpoint mirror is disabled because the kaggle CLI could not be installed."
                        )
                        self._auth_warning_emitted = True
                    continue

                self._retry_publish(request)
                print(
                    f"Kaggle checkpoint mirror uploaded step={int(request.global_step):06d} "
                    f"epoch={int(request.epoch):03d} -> {self.dataset_id}"
                )
            except Exception as exc:
                detail = str(exc)
                if isinstance(exc, subprocess.CalledProcessError):
                    stderr = str(getattr(exc, "stderr", "") or "").strip()
                    stdout = str(getattr(exc, "stdout", "") or "").strip()
                    pieces = [piece for piece in [stderr, stdout, detail] if piece]
                    detail = " | ".join(pieces)
                print(
                    f"WARNING: Kaggle checkpoint mirror failed for step {request.global_step}: {detail}"
                )

    def schedule(
        self,
        *,
        epoch: int,
        global_step: int,
        val_loss: float,
        best_val_loss: float,
        save_tag: str = "latest",
        best: bool = False,
    ) -> None:
        if not self.enabled():
            return

        request = _MirrorRequest(
            checkpoint_dir=self.checkpoint_dir,
            epoch=int(epoch),
            global_step=int(global_step),
            val_loss=float(val_loss),
            best_val_loss=float(best_val_loss),
            save_tag=str(save_tag),
            best=bool(best),
            created_at=time.time(),
        )

        with self._lock:
            self._pending = request
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._worker_loop, daemon=True)
                self._worker.start()

    def wait(self) -> None:
        while True:
            with self._lock:
                worker = self._worker
            if worker is None:
                return
            worker.join()

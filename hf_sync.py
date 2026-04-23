from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class _MirrorRequest:
    epoch: int
    global_step: int
    val_loss: float
    best_val_loss: float
    save_tag: str
    best: bool


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _load_hf_api() -> Any:
    from huggingface_hub import HfApi

    return HfApi


def normalize_hf_repo_id(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    if "huggingface.co/" in raw:
        raw = raw.split("huggingface.co/", 1)[1]

    raw = raw.split("?", 1)[0].split("#", 1)[0].strip("/")
    parts = [part for part in raw.split("/") if part]
    if not parts:
        return ""

    if parts[0] in {"models", "datasets", "spaces"} and len(parts) >= 3:
        return "/".join(parts[1:3])
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return ""


def resolve_hf_token(*, preferred: str = "") -> str:
    token = str(preferred).strip()
    if token:
        return token

    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = str(os.environ.get(env_name, "")).strip()
        if token:
            return token

    return ""


def _entry_path_text(entry: Any) -> str:
    return str(getattr(entry, "path", "") or "").strip("/")


def resolve_latest_hf_checkpoint(
    *,
    repo_id: str,
    cache_root: Path,
    token: str = "",
    repo_type: str = "model",
) -> Optional[Path]:
    normalized_repo_id = normalize_hf_repo_id(repo_id)
    if not normalized_repo_id:
        return None

    api_factory = _load_hf_api()
    api = api_factory(token=token or None)

    try:
        tree = list(api.list_repo_tree(normalized_repo_id, repo_type=repo_type, recursive=False, token=token or None))
    except Exception:
        return None

    step_names: list[str] = []
    for entry in tree:
        entry_name = _entry_path_text(entry)
        if not entry_name.startswith("step-"):
            continue
        step_suffix = entry_name.split("step-", 1)[1]
        if step_suffix.isdigit():
            step_names.append(entry_name)

    if not step_names:
        return None

    latest_step = max(step_names, key=lambda item: int(item.split("step-", 1)[1]))
    download_root = Path(cache_root).expanduser().resolve() / "hf_resume_cache"
    download_root.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        snapshot_root = Path(
            snapshot_download(
                repo_id=normalized_repo_id,
                repo_type=repo_type,
                token=token or None,
                local_dir=str(download_root),
                local_dir_use_symlinks=False,
                allow_patterns=[f"{latest_step}/**"],
            )
        )
    except Exception:
        return None

    candidate_paths = [
        snapshot_root / latest_step / "latest_state.pt",
        snapshot_root / latest_step / "best_state.pt",
    ]
    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


class HuggingFaceCheckpointMirror:
    def __init__(
        self,
        *,
        repo_id: str,
        checkpoint_dir: Path,
        token: str = "",
        repo_type: str = "model",
        private: Optional[bool] = None,
        staging_root: Optional[Path] = None,
    ) -> None:
        normalized_repo_id = normalize_hf_repo_id(repo_id)
        if not normalized_repo_id:
            raise ValueError("repo_id must not be empty")

        self.repo_id = normalized_repo_id
        self.repo_type = str(repo_type).strip() or "model"
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.token = str(token).strip()
        self.private = private
        self.staging_root = (
            Path(staging_root).expanduser().resolve()
            if staging_root is not None
            else self.checkpoint_dir.parent / "hf_checkpoint_sync"
        )
        self.staging_root.mkdir(parents=True, exist_ok=True)

        api_factory = _load_hf_api()
        self._api = api_factory(token=self.token or None)
        self._lock = threading.Lock()
        self._pending_uploads: list[tuple[Any, Path, _MirrorRequest]] = []
        self._errors: list[Exception] = []
        self._repo_ready = False

    def _verify_remote_stage(self, stage_dir: Path) -> None:
        step_name = str(stage_dir.name).strip()
        if not step_name:
            raise ValueError("stage_dir must have a valid folder name")

        try:
            entries = list(
                self._api.list_repo_tree(
                    self.repo_id,
                    repo_type=self.repo_type,
                    path_in_repo=step_name,
                    recursive=True,
                    token=self.token or None,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to verify Hugging Face upload for {self.repo_id}/{step_name}"
            ) from exc

        names = [_entry_path_text(entry) for entry in entries]
        if not names:
            raise RuntimeError(
                f"Hugging Face upload verification failed: no files found under {self.repo_id}/{step_name}"
            )

        has_summary = any(name.endswith("checkpoint_sync_summary.json") for name in names)
        has_state = any(
            name.endswith("latest_state.pt") or name.endswith("best_state.pt")
            for name in names
        )
        if not has_summary or not has_state:
            raise RuntimeError(
                f"Hugging Face upload verification failed for {self.repo_id}/{step_name}: "
                f"summary={has_summary} state={has_state} names={names}"
            )

    @classmethod
    def from_env(
        cls,
        *,
        repo_id_env: str = "HF_SYNC_REPO_ID",
        token_envs: tuple[str, ...] = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"),
        private_env: str = "HF_SYNC_PRIVATE",
        repo_type: str = "model",
        checkpoint_dir: Path,
        staging_root: Optional[Path] = None,
    ) -> Optional["HuggingFaceCheckpointMirror"]:
        repo_id = str(os.environ.get(repo_id_env, "")).strip()
        if not repo_id:
            return None

        token = ""
        for env_name in token_envs:
            token = str(os.environ.get(env_name, "")).strip()
            if token:
                break

        private_raw = str(os.environ.get(private_env, "")).strip().lower()
        private: Optional[bool]
        if private_raw in {"1", "true", "yes", "on"}:
            private = True
        elif private_raw in {"0", "false", "no", "off"}:
            private = False
        else:
            private = None

        return cls(
            repo_id=repo_id,
            checkpoint_dir=checkpoint_dir,
            token=token,
            repo_type=repo_type,
            private=private,
            staging_root=staging_root,
        )

    def ensure_repo(self) -> None:
        if self._repo_ready:
            return

        create_kwargs: dict[str, Any] = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "exist_ok": True,
        }
        if self.token:
            create_kwargs["token"] = self.token
        if self.private is not None:
            create_kwargs["private"] = bool(self.private)

        self._api.create_repo(**create_kwargs)
        self._repo_ready = True

    def _step_folder_name(self, global_step: int) -> str:
        return f"step-{int(global_step)}"

    def _stage_request(self, request: _MirrorRequest) -> Path:
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        stage_dir = self.staging_root / self._step_folder_name(request.global_step)
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)

        copied_files: list[str] = []
        for name in ("latest.safetensors", "latest_state.pt", "best.safetensors", "best_state.pt"):
            source_path = self.checkpoint_dir / name
            if source_path.exists() and source_path.is_file():
                shutil.copy2(source_path, stage_dir / name)
                copied_files.append(name)

        if not copied_files:
            raise FileNotFoundError(f"No checkpoint files found in {self.checkpoint_dir}")

        summary_payload = {
            "created_at": _utc_now_iso(),
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "epoch": int(request.epoch),
            "global_step": int(request.global_step),
            "val_loss": float(request.val_loss),
            "best_val_loss": float(request.best_val_loss),
            "save_tag": str(request.save_tag),
            "best": bool(request.best),
            "copied_files": copied_files,
            "source_checkpoint_dir": str(self.checkpoint_dir),
        }
        _atomic_json_write(stage_dir / "checkpoint_sync_summary.json", summary_payload)
        return stage_dir

    def _cleanup_finished_uploads_locked(self) -> None:
        if not self._pending_uploads:
            return

        remaining: list[tuple[Any, Path, _MirrorRequest]] = []
        for future, stage_dir, request in self._pending_uploads:
            if future.done():
                try:
                    future.result()
                    self._verify_remote_stage(stage_dir)
                    print(f"Verified Hugging Face checkpoint upload: {self.repo_id}/{stage_dir.name}")
                except Exception as exc:
                    self._errors.append(exc)
                finally:
                    if stage_dir.exists():
                        shutil.rmtree(stage_dir, ignore_errors=True)
            else:
                remaining.append((future, stage_dir, request))
        self._pending_uploads = remaining

    def schedule(
        self,
        *,
        epoch: int,
        global_step: int,
        val_loss: float,
        best_val_loss: float,
        save_tag: str,
        best: bool,
    ) -> None:
        request = _MirrorRequest(
            epoch=int(epoch),
            global_step=int(global_step),
            val_loss=float(val_loss),
            best_val_loss=float(best_val_loss),
            save_tag=str(save_tag),
            best=bool(best),
        )

        with self._lock:
            self._cleanup_finished_uploads_locked()
            self.ensure_repo()
            stage_dir = self._stage_request(request)
            future = self._api.upload_folder(
                repo_id=self.repo_id,
                folder_path=str(stage_dir),
                path_in_repo=stage_dir.name,
                token=self.token or None,
                repo_type=self.repo_type,
                commit_message=f"Upload checkpoint {stage_dir.name}",
                run_as_future=True,
            )
            self._pending_uploads.append((future, stage_dir, request))

    def wait(self) -> None:
        with self._lock:
            pending_uploads = list(self._pending_uploads)
            self._pending_uploads = []

        errors: list[Exception] = []
        for future, stage_dir, request in pending_uploads:
            try:
                future.result()
            except Exception as exc:
                errors.append(exc)
            else:
                try:
                    self._verify_remote_stage(stage_dir)
                    print(f"Verified Hugging Face checkpoint upload: {self.repo_id}/{stage_dir.name}")
                except Exception as exc:
                    errors.append(exc)
            finally:
                if stage_dir.exists():
                    shutil.rmtree(stage_dir, ignore_errors=True)

        errors.extend(self._errors)
        self._errors = []

        if errors:
            raise RuntimeError(
                f"One or more Hugging Face checkpoint uploads failed for {self.repo_id}."
            ) from errors[0]

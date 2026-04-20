from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_SCRIPT = ROOT / "scripts" / "tokenize_godzilla_local.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tokenize_godzilla_local import load_or_create_source_index


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=False)


def run_command(cmd: List[str], cwd: Path) -> None:
    printable = " ".join(cmd)
    print(f"Running: {printable}")
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {printable}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize MIDI in chunk windows, upload each chunk to a Hugging Face dataset repo, "
            "delete local chunk data, and continue until complete."
        )
    )
    parser.add_argument("--source", type=str, required=True, help="MIDI source directory or .tar/.tar.gz archive.")
    parser.add_argument(
        "--flash-root",
        type=str,
        required=True,
        help="Working directory on flash drive where temporary chunk outputs and controller state are stored.",
    )
    parser.add_argument("--repo-id", type=str, default="Chickaboo/Pulse88-data", help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Hugging Face token. If empty, reads HF_TOKEN from environment.",
    )
    parser.add_argument(
        "--upload-prefix",
        type=str,
        default="tokenized/chunks",
        help="Path prefix inside the dataset repo where chunk folders are uploaded.",
    )
    parser.add_argument("--chunk-members", type=int, default=100000, help="Members per chunk window.")
    parser.add_argument("--start-index", type=int, default=0, help="Initial source member index (0-based).")
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="Optional exclusive source member end index (0-based). -1 means source end.",
    )
    parser.add_argument("--workers", type=int, default=0, help="Tokenizer worker count (0 means tokenizer auto mode).")
    parser.add_argument(
        "--output-shard-size",
        type=int,
        default=50000,
        help="Tokenizer output shard size for data/ subfolders.",
    )
    parser.add_argument("--min-token-length", type=int, default=192, help="Tokenizer minimum token length filter.")
    parser.add_argument("--checkpoint-every", type=int, default=2000, help="Tokenizer checkpoint frequency.")
    parser.add_argument("--progress-every", type=int, default=500, help="Tokenizer progress print frequency.")
    parser.add_argument(
        "--positions-per-bar",
        type=int,
        default=31,
        help="Deprecated compatibility flag forwarded to tokenizer script.",
    )
    parser.add_argument(
        "--include-structural-prefix",
        action="store_true",
        help="Enable Density/Voices/Register prefix tokens in tokenizer output.",
    )
    parser.add_argument("--compress-output", action="store_true", help="Write compressed .npz outputs.")
    parser.add_argument("--allow-mixed-instruments", action="store_true", help="Disable strict piano filter.")
    parser.add_argument("--private", action="store_true", help="Create dataset repo as private if it does not exist.")
    parser.add_argument("--skip-upload", action="store_true", help="Run tokenization chunks but do not upload.")
    parser.add_argument("--keep-local", action="store_true", help="Keep chunk data after upload instead of deleting.")
    parser.add_argument("--max-chunks", type=int, default=0, help="Stop after this many chunks (0 means no cap).")
    parser.add_argument("--reset-state", action="store_true", help="Reset controller state and restart from --start-index.")
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to run tokenizer subprocesses.",
    )
    parser.add_argument(
        "--upload-backend",
        type=str,
        default="auto",
        choices=["auto", "large-folder", "folder"],
        help=(
            "Upload backend. auto prefers upload_large_folder when available. "
            "folder uses upload_folder directly."
        ),
    )
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=8,
        help="Starting worker count for upload_large_folder backend.",
    )
    parser.add_argument(
        "--upload-attempts",
        type=int,
        default=8,
        help="Maximum upload attempts for each chunk before giving up.",
    )
    parser.add_argument(
        "--upload-backoff-seconds",
        type=float,
        default=3.0,
        help="Base backoff (seconds) between upload retries (multiplied by attempt number).",
    )
    parser.add_argument(
        "--min-upload-workers",
        type=int,
        default=2,
        help="Minimum worker count when retrying large-folder uploads with worker backoff.",
    )
    parser.add_argument(
        "--no-folder-fallback",
        action="store_true",
        help="Disable fallback to upload_folder if upload_large_folder repeatedly fails.",
    )
    return parser.parse_args()


def is_retryable_upload_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    retryable_markers = [
        "http error 500",
        "http error 502",
        "http error 503",
        "http error 504",
        "timed out",
        "timeout",
        "connection",
        "temporarily unavailable",
        "gateway",
        "max retries exceeded",
        "retry",
        "rate limit",
        "429",
    ]
    if any(marker in msg for marker in retryable_markers):
        return True
    if any(marker in name for marker in ["timeout", "connection", "network", "http"]):
        return True
    return False


def resolve_hf_token(args: argparse.Namespace) -> str:
    token = str(args.hf_token).strip() or str(os.environ.get("HF_TOKEN", "")).strip()
    if not token and not bool(args.skip_upload):
        raise RuntimeError("No Hugging Face token provided. Set --hf-token or HF_TOKEN.")
    return token


def build_or_load_controller_index(source: Path, controller_index_path: Path, rebuild: bool) -> Dict[str, Any]:
    source_index = load_or_create_source_index(
        source=source,
        index_path=controller_index_path,
        rebuild=bool(rebuild),
    )
    return {
        "source_type": str(source_index.source_type),
        "source_path": str(source_index.source_path),
        "total_members": int(len(source_index.members)),
    }


def load_state(state_path: Path) -> Dict[str, Any]:
    default_state = {
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "source_path": "",
        "repo_id": "",
        "upload_prefix": "",
        "next_index": 0,
        "total_members": 0,
        "uploaded_chunks": [],
    }
    payload = safe_json_read(state_path, default_state)
    if not isinstance(payload, dict):
        return dict(default_state)
    for key, value in default_state.items():
        payload.setdefault(key, value)
    if not isinstance(payload.get("uploaded_chunks"), list):
        payload["uploaded_chunks"] = []
    return payload


def ensure_state_consistency(
    state: Dict[str, Any],
    *,
    source_path: str,
    repo_id: str,
    upload_prefix: str,
    total_members: int,
    reset_state: bool,
    start_index: int,
) -> Dict[str, Any]:
    if bool(reset_state):
        state = {
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "source_path": str(source_path),
            "repo_id": str(repo_id),
            "upload_prefix": str(upload_prefix),
            "next_index": int(max(0, start_index)),
            "total_members": int(total_members),
            "uploaded_chunks": [],
        }
        return state

    prior_source = str(state.get("source_path", "")).strip()
    prior_repo = str(state.get("repo_id", "")).strip()
    prior_prefix = str(state.get("upload_prefix", "")).strip()
    if prior_source and prior_source != str(source_path):
        raise RuntimeError("Controller state source_path mismatch. Use --reset-state to restart.")
    if prior_repo and prior_repo != str(repo_id):
        raise RuntimeError("Controller state repo_id mismatch. Use --reset-state to restart.")
    if prior_prefix and prior_prefix != str(upload_prefix):
        raise RuntimeError("Controller state upload_prefix mismatch. Use --reset-state to restart.")

    state["source_path"] = str(source_path)
    state["repo_id"] = str(repo_id)
    state["upload_prefix"] = str(upload_prefix)
    state["total_members"] = int(total_members)
    state["next_index"] = int(max(int(state.get("next_index", 0)), max(0, int(start_index))))
    state["updated_at"] = utc_now_iso()
    return state


def upload_chunk_folder(
    *,
    repo_id: str,
    token: str,
    local_chunk_dir: Path,
    path_in_repo: str,
    upload_backend: str,
    upload_workers: int,
    upload_attempts: int,
    upload_backoff_seconds: float,
    min_upload_workers: int,
    allow_folder_fallback: bool,
    staging_root: Path,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    normalized_repo_path = str(path_in_repo).replace("\\", "/").strip("/")
    backend = str(upload_backend).strip().lower()
    supports_large = hasattr(api, "upload_large_folder")
    use_large = backend == "large-folder" or (backend == "auto" and supports_large)

    if token:
        os.environ["HF_TOKEN"] = str(token)

    # Optional Rust transfer accelerator if installed.
    try:
        import hf_transfer  # type: ignore  # noqa: F401

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except Exception:
        pass

    if use_large and supports_large:
        workers = max(1, int(upload_workers))
        min_workers = max(1, int(min_upload_workers))
        attempts = max(1, int(upload_attempts))
        backoff_seconds = max(0.0, float(upload_backoff_seconds))
        stage_dir = staging_root / f"{local_chunk_dir.name}_{int(time.time())}"
        try:
            if normalized_repo_path:
                staged_target = stage_dir / normalized_repo_path
            else:
                staged_target = stage_dir / local_chunk_dir.name

            staged_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(local_chunk_dir, staged_target)

            for attempt in range(1, attempts + 1):
                attempt_workers = max(min_workers, workers // (2 ** (attempt - 1)))
                try:
                    print(
                        "Uploading via upload_large_folder "
                        f"(attempt {attempt}/{attempts}, workers={attempt_workers}) "
                        f"from staged dir: {stage_dir}"
                    )
                    api.upload_large_folder(
                        repo_id=repo_id,
                        repo_type="dataset",
                        folder_path=str(stage_dir),
                        num_workers=attempt_workers,
                        print_report=True,
                        print_report_every=30,
                    )
                    return
                except Exception as exc:
                    retryable = is_retryable_upload_error(exc)
                    print(
                        f"upload_large_folder attempt {attempt}/{attempts} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    is_last_attempt = attempt >= attempts
                    if (not retryable) or is_last_attempt:
                        if not allow_folder_fallback:
                            raise
                        print("Switching to upload_folder fallback for this chunk.")
                        break
                    sleep_seconds = backoff_seconds * attempt
                    if sleep_seconds > 0:
                        print(f"Retrying upload_large_folder in {sleep_seconds:.1f}s...")
                        time.sleep(sleep_seconds)
        finally:
            if stage_dir.exists():
                remove_tree(stage_dir)

    if use_large and supports_large and allow_folder_fallback:
        print("Uploading via upload_folder fallback.")

    commit_message = f"Add tokenized chunk {normalized_repo_path or local_chunk_dir.name}"
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(local_chunk_dir),
        path_in_repo=(normalized_repo_path if normalized_repo_path else None),
        token=token,
        commit_message=commit_message,
    )


def upload_controller_state(*, repo_id: str, token: str, state_path: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        path_or_fileobj=str(state_path),
        path_in_repo=path_in_repo,
        commit_message="Update tokenization controller state",
    )


def ensure_repo_exists(*, repo_id: str, token: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        private=bool(private),
        exist_ok=True,
    )


def main() -> None:
    args = parse_args()

    source = Path(args.source).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    flash_root = Path(args.flash_root).expanduser()
    flash_root.mkdir(parents=True, exist_ok=True)

    controller_dir = flash_root / "_controller"
    controller_dir.mkdir(parents=True, exist_ok=True)
    controller_state_path = controller_dir / "state.json"
    controller_index_path = controller_dir / "source_index.json"

    token = resolve_hf_token(args)

    index_info = build_or_load_controller_index(
        source=source,
        controller_index_path=controller_index_path,
        rebuild=bool(args.reset_state),
    )
    total_members = int(index_info["total_members"])
    if total_members <= 0:
        raise RuntimeError("Source index is empty. No MIDI members found.")

    state = load_state(controller_state_path)
    state = ensure_state_consistency(
        state,
        source_path=str(index_info["source_path"]),
        repo_id=str(args.repo_id),
        upload_prefix=str(args.upload_prefix).strip("/"),
        total_members=total_members,
        reset_state=bool(args.reset_state),
        start_index=int(args.start_index),
    )

    safe_json_write(controller_state_path, state)

    end_index_exclusive = int(total_members if int(args.end_index) < 0 else min(total_members, max(0, int(args.end_index))))
    current_start = int(max(0, int(state.get("next_index", 0))))

    print("Batch tokenization + upload session")
    print(f"  source: {index_info['source_path']}")
    print(f"  source_type: {index_info['source_type']}")
    print(f"  total_members: {total_members:,}")
    print(f"  start_index: {current_start:,}")
    print(f"  end_index_exclusive: {end_index_exclusive:,}")
    print(f"  chunk_members: {int(args.chunk_members):,}")
    print(f"  output_shard_size: {int(args.output_shard_size):,}")
    print(f"  positions_per_bar: {int(max(4, min(31, int(args.positions_per_bar))))}")
    print(f"  include_structural_prefix: {bool(args.include_structural_prefix)}")
    print(f"  flash_root: {flash_root}")
    print(f"  repo_id: {args.repo_id}")
    print(f"  upload_prefix: {str(args.upload_prefix).strip('/')}")
    print(f"  upload_enabled: {not bool(args.skip_upload)}")
    print(f"  delete_after_upload: {not bool(args.keep_local)}")
    print(f"  upload_backend: {str(args.upload_backend).strip().lower()}")
    print(f"  upload_workers: {max(1, int(args.upload_workers))}")
    print(f"  upload_attempts: {max(1, int(args.upload_attempts))}")
    print(f"  upload_backoff_seconds: {max(0.0, float(args.upload_backoff_seconds))}")
    print(f"  min_upload_workers: {max(1, int(args.min_upload_workers))}")
    print(f"  folder_fallback_enabled: {not bool(args.no_folder_fallback)}")

    if current_start >= end_index_exclusive:
        print("Nothing to do: start index is already at/after end index.")
        return

    if not bool(args.skip_upload):
        ensure_repo_exists(repo_id=str(args.repo_id), token=token, private=bool(args.private))

    chunks_done = 0
    while current_start < end_index_exclusive:
        if int(args.max_chunks) > 0 and chunks_done >= int(args.max_chunks):
            print(f"Reached --max-chunks={int(args.max_chunks)}. Stopping.")
            break

        chunk_members = max(1, int(args.chunk_members))
        chunk_end = int(min(end_index_exclusive, current_start + chunk_members))
        chunk_name = f"chunk_{current_start:07d}_{chunk_end - 1:07d}"
        chunk_output_root = flash_root / "work" / chunk_name

        resume_chunk = bool(chunk_output_root.exists())
        if resume_chunk:
            print(f"Resuming existing chunk folder: {chunk_output_root}")
        else:
            (chunk_output_root / "metadata").mkdir(parents=True, exist_ok=True)
            shutil.copy2(controller_index_path, chunk_output_root / "metadata" / "source_index.json")

        tokenize_cmd = [
            str(args.python_executable),
            str(TOKENIZER_SCRIPT),
            "--source",
            str(source),
            "--output-root",
            str(chunk_output_root),
            "--start-index",
            str(current_start),
            "--end-index",
            str(chunk_end),
            "--workers",
            str(int(args.workers)),
            "--output-shard-size",
            str(int(args.output_shard_size)),
            "--min-token-length",
            str(int(args.min_token_length)),
            "--checkpoint-every",
            str(int(args.checkpoint_every)),
            "--progress-every",
            str(int(args.progress_every)),
            "--positions-per-bar",
            str(int(max(4, min(31, int(args.positions_per_bar))))),
        ]
        if not resume_chunk:
            tokenize_cmd.append("--start-over")
        if bool(args.include_structural_prefix):
            tokenize_cmd.append("--include-structural-prefix")
        if bool(args.compress_output):
            tokenize_cmd.append("--compress-output")
        if bool(args.allow_mixed_instruments):
            tokenize_cmd.append("--allow-mixed-instruments")

        t0 = time.time()
        run_command(tokenize_cmd, cwd=ROOT)
        elapsed = time.time() - t0

        summary_path = chunk_output_root / "metadata" / "summary.json"
        summary = safe_json_read(summary_path, default={})
        accepted = int(summary.get("accepted", 0))
        skipped = int(summary.get("skipped", 0))
        manifest_entries = int(summary.get("manifest_entries", 0))

        print(
            f"Chunk finished: {chunk_name} | accepted={accepted:,} skipped={skipped:,} "
            f"manifest_entries={manifest_entries:,} elapsed_minutes={elapsed / 60.0:.2f}"
        )

        upload_prefix = str(args.upload_prefix).strip("/")
        hf_chunk_path = f"{upload_prefix}/{chunk_name}" if upload_prefix else chunk_name
        if not bool(args.skip_upload):
            upload_chunk_folder(
                repo_id=str(args.repo_id),
                token=token,
                local_chunk_dir=chunk_output_root,
                path_in_repo=hf_chunk_path,
                upload_backend=str(args.upload_backend),
                upload_workers=int(args.upload_workers),
                upload_attempts=int(args.upload_attempts),
                upload_backoff_seconds=float(args.upload_backoff_seconds),
                min_upload_workers=int(args.min_upload_workers),
                allow_folder_fallback=not bool(args.no_folder_fallback),
                staging_root=flash_root / "_upload_stage",
            )

        if not bool(args.keep_local):
            remove_tree(chunk_output_root)

        chunk_record = {
            "chunk_name": chunk_name,
            "start_index": int(current_start),
            "end_index_exclusive": int(chunk_end),
            "accepted": int(accepted),
            "skipped": int(skipped),
            "manifest_entries": int(manifest_entries),
            "uploaded": bool(not args.skip_upload),
            "hf_path": hf_chunk_path,
            "finished_at": utc_now_iso(),
        }
        state.setdefault("uploaded_chunks", [])
        state["uploaded_chunks"].append(chunk_record)
        state["next_index"] = int(chunk_end)
        state["updated_at"] = utc_now_iso()
        safe_json_write(controller_state_path, state)

        if not bool(args.skip_upload):
            upload_controller_state(
                repo_id=str(args.repo_id),
                token=token,
                state_path=controller_state_path,
                path_in_repo="tokenized/controller_state.json",
            )

        current_start = int(chunk_end)
        chunks_done += 1

    print("Batch pipeline complete.")
    print(f"  next_index: {int(state.get('next_index', current_start)):,}")
    print(f"  chunks_recorded: {len(state.get('uploaded_chunks', [])):,}")
    print(f"  controller_state: {controller_state_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from drive_sync import DriveSync


def _write_empty_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"sessions": []}, indent=2), encoding="utf-8")
    tmp.replace(path)


def _clear_dir_contents(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Clear training caches/logs for local development."
    )
    parser.add_argument(
        "--scope",
        choices=["local", "drive", "both"],
        default="local",
        help="Which cache to clear (default: local).",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Confirm destructive actions."
    )
    args = parser.parse_args()

    ds = DriveSync()

    # Always clear project-level training_log.json
    project_log = Path(__file__).resolve().parents[2] / "training_log.json"
    print(f"Resetting project log: {project_log}")
    _write_empty_log(project_log)

    if args.scope in ("local", "both"):
        print("Clearing local session cache:")
        print(f" - {ds.local_root}")
        _clear_dir_contents(ds.local_checkpoints_dir)
        _write_empty_log(ds.local_logs_dir / "training_log.json")
        _clear_dir_contents(ds.local_processed_dir)
        _clear_dir_contents(ds.local_tokenizer_dir)
        _clear_dir_contents(ds.local_generated_dir)

    if args.scope in ("drive", "both"):
        if not args.yes:
            print(
                "Drive-level cache clear is destructive. Re-run with --yes to confirm."
            )
            return
        print("Clearing drive cache (using DriveSync paths):")
        print(f" - {ds.drive_root}")
        # Reset drive log
        _write_empty_log(ds.logs_dir / "training_log.json")
        # Remove drive cache contents
        _clear_dir_contents(ds.checkpoints_dir)
        _clear_dir_contents(ds.processed_dir)
        _clear_dir_contents(ds.tokenizer_dir)
        _clear_dir_contents(ds.generated_dir)

    print("Cache clear complete.")


if __name__ == "__main__":
    main()

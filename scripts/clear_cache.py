from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from drive_sync import DriveSync
from utils.logging_utils import get_project_logger


LOGGER = get_project_logger()


def _write_empty_log(path: Path) -> None:
    """Write empty `{"sessions": []}` log payload atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"sessions": []}, indent=2), encoding="utf-8")
    tmp.replace(path)


def _clear_dir_contents(path: Path) -> None:
    """Delete all files/subdirectories under a directory."""

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


def main() -> None:
    """CLI entrypoint for clearing local/drive caches."""

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
    LOGGER.info("Resetting project log: %s", project_log)
    _write_empty_log(project_log)

    if args.scope in ("local", "both"):
        LOGGER.info("Clearing local session cache:")
        LOGGER.info(" - %s", ds.local_root)
        _clear_dir_contents(ds.local_checkpoints_dir)
        _write_empty_log(ds.local_logs_dir / "training_log.json")
        _clear_dir_contents(ds.local_processed_dir)
        _clear_dir_contents(ds.local_tokenizer_dir)
        _clear_dir_contents(ds.local_generated_dir)

    if args.scope in ("drive", "both"):
        if not args.yes:
            LOGGER.info(
                "Drive-level cache clear is destructive. Re-run with --yes to confirm."
            )
            return
        LOGGER.info("Clearing drive cache (using DriveSync paths):")
        LOGGER.info(" - %s", ds.drive_root)
        # Reset drive log
        _write_empty_log(ds.logs_dir / "training_log.json")
        # Remove drive cache contents
        _clear_dir_contents(ds.checkpoints_dir)
        _clear_dir_contents(ds.processed_dir)
        _clear_dir_contents(ds.tokenizer_dir)
        _clear_dir_contents(ds.generated_dir)

    LOGGER.info("Cache clear complete.")


if __name__ == "__main__":
    main()

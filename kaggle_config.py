"""
Kaggle-specific path configuration.
Handles the mapping between Kaggle's fixed input paths and
the project's expected directory structure.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_kaggle_dataset_path(possible_names: list[str]) -> Path:
    """
    Find a Kaggle input dataset by trying multiple possible mount names.
    Kaggle slugifies dataset names, so the actual path may vary.
    """
    input_root = Path("/kaggle/input")
    for name in possible_names:
        candidate = input_root / name
        if candidate.exists():
            return candidate

    if input_root.exists():
        available = sorted([p.name for p in input_root.iterdir()])
        raise FileNotFoundError(
            f"None of {possible_names} found in /kaggle/input/.\n"
            f"Available datasets: {available}\n"
            f"Check your dataset slug names in Kaggle settings."
        )

    raise FileNotFoundError("/kaggle/input/ does not exist - not running on Kaggle?")


def _safe_iterdir(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except Exception:
        return []


def get_kaggle_paths() -> dict:
    """
    Returns all relevant paths for Kaggle environment.
    Searches for the MAESTRO dataset and project code under
    the standard Kaggle input mount points.
    """
    maestro_path = find_kaggle_dataset_path(
        [
            "maestro-v3",
            "maestro-v300",
            "maestro",
            "maestro-v3-0-0",
            "maestrov3",
            "piano-maestro",
        ]
    )

    maestro_root = maestro_path
    candidates = [maestro_path, *_safe_iterdir(maestro_path)]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            has_year_dirs = any(
                c.is_dir() and c.name.isdigit() for c in candidate.iterdir()
            )
        except Exception:
            has_year_dirs = False
        if has_year_dirs:
            maestro_root = candidate
            break

    project_path = find_kaggle_dataset_path(
        [
            "piano-midi-model",
            "pianomidimodel",
            "piano-midi-model-1",
            "piano-model",
        ]
    )

    working_dir = Path("/kaggle/working")

    return {
        "maestro_root": str(maestro_root),
        "project_root": str(project_path),
        "working_dir": str(working_dir),
        "checkpoint_dir": str(working_dir / "checkpoints"),
        "processed_dir": str(working_dir / "processed"),
        "tokenizer_path": str(working_dir / "tokenizer" / "tokenizer.json"),
        "generated_dir": str(working_dir / "generated"),
        "log_path": str(working_dir / "training_log.json"),
    }


def setup_kaggle_environment() -> dict:
    """
    Full Kaggle environment setup.
    Creates working directories, adds project to Python path,
    returns configured paths.
    """
    import sys

    paths = get_kaggle_paths()

    if paths["project_root"] not in sys.path:
        sys.path.insert(0, paths["project_root"])

    for dir_key in ["checkpoint_dir", "processed_dir", "generated_dir"]:
        Path(paths[dir_key]).mkdir(parents=True, exist_ok=True)
    Path(paths["tokenizer_path"]).parent.mkdir(parents=True, exist_ok=True)

    print("Kaggle environment configured:")
    print(f"  MAESTRO root:    {paths['maestro_root']}")
    print(f"  Project root:    {paths['project_root']}")
    print(f"  Working dir:     {paths['working_dir']}")
    print(f"  Checkpoint dir:  {paths['checkpoint_dir']}")

    return paths

"""
Kaggle-specific path configuration.
Handles the mapping between Kaggle's fixed input paths and
the project's expected directory structure.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root() -> Path:
    """
    Find the Itty Bitty Piano project root by searching /kaggle/input recursively.
    Looks for a file that definitively identifies the project root.
    """
    marker_files = [
        "piano_kaggle_session.py",
        "kaggle_config.py",
        "scale_config.py",
        "session.py",
    ]

    for root, _dirs, files in os.walk("/kaggle/input"):
        for marker in marker_files:
            if marker in files:
                found = Path(root)
                print(f"Project root found at: {found}")
                return found

    print("ERROR: Could not find project root. Contents of /kaggle/input:")
    for root, _dirs, files in os.walk("/kaggle/input"):
        depth = root.replace("/kaggle/input", "").count(os.sep)
        if depth <= 2:
            indent = "  " * depth
            print(f"{indent}{os.path.basename(root)}/")
            if depth <= 1:
                for file_name in files[:5]:
                    print(f"{indent}  {file_name}")

    raise FileNotFoundError(
        "Project files not found under /kaggle/input. "
        "Make sure you added the Itty Bitty Piano GitHub repo as a dataset."
    )


def find_maestro_root() -> Path:
    """
    Find the MAESTRO dataset root by searching /kaggle/input.
    Identifies MAESTRO by the presence of its CSV metadata file.
    """
    for root, _dirs, files in os.walk("/kaggle/input"):
        for file_name in files:
            if file_name == "maestro-v3.0.0.csv" or file_name == "maestro-v2.0.0.csv":
                found = Path(root)
                print(f"MAESTRO found at: {found}")
                return found

    print("ERROR: MAESTRO dataset not found. Contents of /kaggle/input:")
    for root, _dirs, _files in os.walk("/kaggle/input"):
        depth = root.replace("/kaggle/input", "").count(os.sep)
        if depth <= 2:
            indent = "  " * depth
            print(f"{indent}{os.path.basename(root)}/")

    raise FileNotFoundError(
        "MAESTRO CSV not found under /kaggle/input. "
        "Make sure you added the MAESTRO dataset to this notebook."
    )


def get_kaggle_paths() -> dict:
    """
    Returns all relevant paths for Kaggle environment.
    Searches for the MAESTRO dataset and project code under
    Kaggle input mount points.
    """
    project_root = find_project_root()
    maestro_root = find_maestro_root()
    working_dir = Path("/kaggle/working")

    return {
        "maestro_root": str(maestro_root),
        "project_root": str(project_root),
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

    print("Kaggle environment:")
    print(f"  Project root:   {paths['project_root']}")
    print(f"  MAESTRO root:   {paths['maestro_root']}")
    print(f"  Working dir:    {paths['working_dir']}")
    print(f"  Checkpoints:    {paths['checkpoint_dir']}")

    return paths

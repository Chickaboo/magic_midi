"""Kaggle-specific path configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional

from utils.logging_utils import get_project_logger


LOGGER = get_project_logger()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def _print_input_tree(max_depth: int = 2, max_files_per_dir: int = 5) -> None:
    """Log a shallow view of `/kaggle/input` for troubleshooting."""

    for root, _dirs, files in os.walk(str(KAGGLE_INPUT_ROOT)):
        depth = root.replace(str(KAGGLE_INPUT_ROOT), "").count(os.sep)
        if depth > max_depth:
            continue
        indent = "  " * depth
        LOGGER.info("%s%s/", indent, os.path.basename(root))
        if depth <= 1:
            for file_name in files[:max_files_per_dir]:
                LOGGER.info("%s  %s", indent, file_name)


def find_kaggle_dataset_path(candidates: Iterable[str]) -> Path:
    """Find a Kaggle dataset directory by matching candidate names."""

    normalized_candidates = [
        str(c).strip().lower() for c in candidates if str(c).strip()
    ]
    if not normalized_candidates:
        raise ValueError("candidates must contain at least one non-empty dataset name")

    for entry in KAGGLE_INPUT_ROOT.iterdir():
        if not entry.is_dir():
            continue
        lower_name = entry.name.lower()
        if any(candidate in lower_name for candidate in normalized_candidates):
            LOGGER.info("Dataset '%s' resolved to %s", entry.name, entry)
            return entry

    raise FileNotFoundError(
        "Dataset not found under /kaggle/input. Checked names: "
        f"{', '.join(normalized_candidates)}"
    )


def find_project_root() -> Path:
    """Find project root by locating known marker files under `/kaggle/input`."""

    marker_files = [
        "piano_kaggle_session.py",
        "kaggle_config.py",
        "scale_config.py",
        "session.py",
    ]

    for root, _dirs, files in os.walk(str(KAGGLE_INPUT_ROOT)):
        for marker in marker_files:
            if marker in files:
                found = Path(root)
                LOGGER.info("Project root found at: %s", found)
                return found

    LOGGER.error("Could not find project root. Contents of /kaggle/input:")
    _print_input_tree(max_depth=2, max_files_per_dir=5)
    raise FileNotFoundError(
        "Project files not found under /kaggle/input. "
        "Add the Itty Bitty Piano repository as a Kaggle dataset."
    )


def find_maestro_root() -> Path:
    """Find MAESTRO dataset root by scanning for its metadata CSV."""

    for root, _dirs, files in os.walk(str(KAGGLE_INPUT_ROOT)):
        for file_name in files:
            if file_name in {"maestro-v3.0.0.csv", "maestro-v2.0.0.csv"}:
                found = Path(root)
                LOGGER.info("MAESTRO found at: %s", found)
                return found

    LOGGER.error("MAESTRO dataset not found. Contents of /kaggle/input:")
    _print_input_tree(max_depth=2, max_files_per_dir=0)
    raise FileNotFoundError(
        "MAESTRO CSV not found under /kaggle/input. "
        "Add the MAESTRO dataset to this notebook."
    )


def find_giant_midi_root() -> Optional[Path]:
    """Find GiantMIDI dataset if present, otherwise return None."""

    try:
        return find_kaggle_dataset_path(
            [
                "giant-midi-piano",
                "giantmidi",
                "giant-midi",
                "giantmidi-piano",
            ]
        )
    except FileNotFoundError:
        return None


def find_piano_e_root() -> Optional[Path]:
    """Find Piano-e-Competition dataset if present, otherwise return None."""

    try:
        return find_kaggle_dataset_path(
            [
                "piano-e",
                "pianoe",
                "piano-e-competition",
                "piano-e-competition-midi",
            ]
        )
    except FileNotFoundError:
        return None


def find_aria_midi_root() -> Optional[Path]:
    """Find Aria-MIDI dataset if present, otherwise return None."""

    try:
        return find_kaggle_dataset_path(
            [
                "aria-midi",
                "aria_midi",
                "aria midi",
                "aria",
            ]
        )
    except FileNotFoundError:
        return None


def find_adl_piano_root() -> Optional[Path]:
    """Find ADL Piano MIDI dataset if present, otherwise return None."""

    try:
        return find_kaggle_dataset_path(
            [
                "adl-piano",
                "adl_piano",
                "adl piano",
                "adl",
            ]
        )
    except FileNotFoundError:
        return None


def get_kaggle_paths() -> Dict[str, str]:
    """Return resolved Kaggle paths for training and outputs."""

    project_root = find_project_root()
    maestro_root = find_maestro_root()
    giant_midi_root = find_giant_midi_root()
    aria_midi_root = find_aria_midi_root()
    adl_piano_root = find_adl_piano_root()
    piano_e_root = find_piano_e_root()
    working_dir = Path("/kaggle/working")

    return {
        "maestro_root": str(maestro_root),
        "giant_midi_root": str(giant_midi_root) if giant_midi_root else "",
        "aria_midi_root": str(aria_midi_root) if aria_midi_root else "",
        "adl_piano_root": str(adl_piano_root) if adl_piano_root else "",
        "piano_e_root": str(piano_e_root) if piano_e_root else "",
        "project_root": str(project_root),
        "working_dir": str(working_dir),
        "checkpoint_dir": str(working_dir / "checkpoints"),
        "processed_dir": str(working_dir / "processed"),
        "tokenizer_path": str(working_dir / "tokenizer" / "tokenizer.json"),
        "generated_dir": str(working_dir / "generated"),
        "log_path": str(working_dir / "training_log.json"),
    }


def setup_kaggle_environment() -> Dict[str, str]:
    """Create runtime directories and add project root to Python path."""

    import sys

    paths = get_kaggle_paths()
    if paths["project_root"] not in sys.path:
        sys.path.insert(0, paths["project_root"])

    for dir_key in ["checkpoint_dir", "processed_dir", "generated_dir"]:
        Path(paths[dir_key]).mkdir(parents=True, exist_ok=True)
    Path(paths["tokenizer_path"]).parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Kaggle environment:")
    LOGGER.info("  Project root:   %s", paths["project_root"])
    LOGGER.info("  MAESTRO root:   %s", paths["maestro_root"])
    LOGGER.info(
        "  GiantMIDI root: %s",
        paths["giant_midi_root"] if paths["giant_midi_root"] else "not found",
    )
    LOGGER.info(
        "  Aria-MIDI root: %s",
        paths["aria_midi_root"] if paths["aria_midi_root"] else "not found",
    )
    LOGGER.info(
        "  ADL Piano root: %s",
        paths["adl_piano_root"] if paths["adl_piano_root"] else "not found",
    )
    LOGGER.info(
        "  Piano-e root:   %s",
        paths["piano_e_root"] if paths["piano_e_root"] else "not found",
    )
    LOGGER.info("  Working dir:    %s", paths["working_dir"])
    LOGGER.info("  Checkpoints:    %s", paths["checkpoint_dir"])

    return paths

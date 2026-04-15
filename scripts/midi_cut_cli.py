from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import pretty_midi
except Exception as exc:  # pragma: no cover - dependency bootstrap
    raise RuntimeError(
        "pretty_midi is required for the MIDI cut CLI. Install it with: pip install pretty_midi"
    ) from exc


EVENT_SIZE = 4
DEFAULT_TOKEN_LIMIT = 512
DEFAULT_MIN_GAP_SECONDS = 0.25
DEFAULT_ONSET_TOLERANCE = 1e-4


@dataclass(frozen=True)
class NoteEvent:
    instrument_index: int
    note_index: int
    onset: float
    end: float
    pitch: int
    velocity: int


@dataclass(frozen=True)
class NoteGroup:
    onset: float
    end: float
    note_count: int
    events: tuple[NoteEvent, ...]


@dataclass(frozen=True)
class CutCandidate:
    kept_notes: int
    kept_groups: int
    token_count: int
    cut_time: float
    next_gap_seconds: float | None
    nearest_downbeat_seconds: float | None
    nearest_beat_seconds: float | None
    score: float


def _is_midi_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in {".mid", ".midi"}


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return path.name


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix or ".mid"
    parent = path.parent
    for index in range(2, 1000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find a free output name near {path}")


def _scan_midi_files(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
        if not _is_midi_file(source):
            raise ValueError(f"Expected a .mid or .midi file, got: {source}")
        return [source.resolve()]

    if not source.is_dir():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    if recursive:
        candidates = [
            path for path in source.rglob("*") if path.is_file() and _is_midi_file(path)
        ]
    else:
        candidates = [
            path for path in source.iterdir() if path.is_file() and _is_midi_file(path)
        ]

    unique = sorted({path.resolve() for path in candidates}, key=lambda p: str(p).lower())
    return unique


def _prompt_for_file(files: Sequence[Path], root: Path) -> Path:
    if not files:
        raise FileNotFoundError("No MIDI files were found to choose from.")
    if len(files) == 1:
        chosen = files[0]
        print(f"Selected MIDI file: {_display_path(chosen, root)}")
        return chosen

    visible = list(files)
    while True:
        print("\nAvailable MIDI files:")
        for index, path in enumerate(visible, start=1):
            print(f"  {index:3d}. {_display_path(path, root)}")

        prompt = input(
            "Select a file number, type a search term to filter, or press Enter for 1: "
        ).strip()

        if not prompt:
            return visible[0]
        if prompt.lower() in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if prompt.isdigit():
            choice = int(prompt)
            if 1 <= choice <= len(visible):
                return visible[choice - 1]
            print(f"Choice must be between 1 and {len(visible)}.")
            continue

        query = prompt.lower()
        filtered = [
            path
            for path in files
            if query in _display_path(path, root).lower()
            or query in path.name.lower()
        ]
        if not filtered:
            print(f"No MIDI files matched '{prompt}'.")
            continue
        visible = filtered


def _extract_note_events(midi: pretty_midi.PrettyMIDI) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    for instrument_index, instrument in enumerate(midi.instruments):
        if instrument.is_drum:
            continue
        for note_index, note in enumerate(instrument.notes):
            pitch = int(note.pitch)
            if pitch < 21 or pitch > 108:
                continue
            onset = float(max(0.0, note.start))
            end = float(max(onset + 1e-4, note.end))
            velocity = int(max(0, min(127, int(note.velocity))))
            events.append(
                NoteEvent(
                    instrument_index=instrument_index,
                    note_index=note_index,
                    onset=onset,
                    end=end,
                    pitch=pitch,
                    velocity=velocity,
                )
            )

    events.sort(key=lambda ev: (ev.onset, ev.pitch, ev.end, ev.velocity, ev.instrument_index, ev.note_index))
    return events


def _group_note_events(events: Sequence[NoteEvent], onset_tolerance: float) -> list[NoteGroup]:
    if not events:
        return []

    groups: list[list[NoteEvent]] = []
    current_group: list[NoteEvent] = [events[0]]
    current_onset = float(events[0].onset)

    for event in events[1:]:
        if abs(float(event.onset) - current_onset) <= float(onset_tolerance):
            current_group.append(event)
            continue

        groups.append(current_group)
        current_group = [event]
        current_onset = float(event.onset)

    groups.append(current_group)

    grouped: list[NoteGroup] = []
    for group in groups:
        onset = float(group[0].onset)
        end = float(max(event.end for event in group))
        grouped.append(
            NoteGroup(
                onset=onset,
                end=end,
                note_count=len(group),
                events=tuple(group),
            )
        )
    return grouped


def _nearest_distance(value: float, anchors: Sequence[float]) -> float | None:
    if not anchors:
        return None
    return float(min(abs(float(value) - float(anchor)) for anchor in anchors))


def _resolve_downbeats_and_beats(midi: pretty_midi.PrettyMIDI) -> tuple[list[float], list[float]]:
    beats = [float(value) for value in midi.get_beats()]
    downbeats = [float(value) for value in midi.get_downbeats()]
    if not downbeats and len(beats) >= 4:
        downbeats = beats[::4]
    return downbeats, beats


def _score_candidate(
    candidate_notes: int,
    target_notes: int,
    cut_time: float,
    next_gap_seconds: float | None,
    downbeats: Sequence[float],
    beats: Sequence[float],
    min_gap_seconds: float,
) -> CutCandidate:
    if candidate_notes <= target_notes:
        token_penalty = float(target_notes - candidate_notes) * 2.0
    else:
        token_penalty = float(candidate_notes - target_notes) * 3.0

    if next_gap_seconds is None:
        gap_penalty = -0.5
    else:
        if next_gap_seconds >= min_gap_seconds:
            gap_penalty = 0.0
        elif next_gap_seconds >= 0.0:
            gap_penalty = float(min_gap_seconds - next_gap_seconds) * 80.0
        else:
            gap_penalty = float(min_gap_seconds + abs(next_gap_seconds)) * 160.0

    downbeat_distance = _nearest_distance(cut_time, downbeats)
    beat_distance = _nearest_distance(cut_time, beats)
    downbeat_penalty = float(downbeat_distance or 0.0) * 8.0
    beat_penalty = float(beat_distance or 0.0) * 1.5

    score = token_penalty + gap_penalty + downbeat_penalty + beat_penalty
    return CutCandidate(
        kept_notes=int(candidate_notes),
        kept_groups=0,
        token_count=int(candidate_notes * EVENT_SIZE),
        cut_time=float(cut_time),
        next_gap_seconds=None if next_gap_seconds is None else float(next_gap_seconds),
        nearest_downbeat_seconds=downbeat_distance,
        nearest_beat_seconds=beat_distance,
        score=float(score),
    )


def _choose_cut_candidate(
    groups: Sequence[NoteGroup],
    target_notes: int,
    min_gap_seconds: float,
    strict_limit: bool,
    downbeats: Sequence[float],
    beats: Sequence[float],
) -> CutCandidate:
    if not groups:
        raise RuntimeError("No playable note events were found in the MIDI file.")

    target_notes = max(1, int(target_notes))
    candidates: list[CutCandidate] = []

    running_notes = 0
    for group_index, group in enumerate(groups):
        running_notes += int(group.note_count)
        cut_time = float(group.end)
        next_gap_seconds = None
        if group_index + 1 < len(groups):
            next_gap_seconds = float(groups[group_index + 1].onset - cut_time)

        candidate = _score_candidate(
            candidate_notes=running_notes,
            target_notes=target_notes,
            cut_time=cut_time,
            next_gap_seconds=next_gap_seconds,
            downbeats=downbeats,
            beats=beats,
            min_gap_seconds=min_gap_seconds,
        )
        candidates.append(
            CutCandidate(
                kept_notes=candidate.kept_notes,
                kept_groups=group_index + 1,
                token_count=candidate.token_count,
                cut_time=candidate.cut_time,
                next_gap_seconds=candidate.next_gap_seconds,
                nearest_downbeat_seconds=candidate.nearest_downbeat_seconds,
                nearest_beat_seconds=candidate.nearest_beat_seconds,
                score=candidate.score,
            )
        )

    if strict_limit:
        under_limit = [candidate for candidate in candidates if candidate.kept_notes <= target_notes]
        if under_limit:
            candidates = under_limit
        else:
            print(
                "Warning: no chord-safe boundary stayed under the requested token budget; "
                "allowing the smallest overshoot instead."
            )

    return min(
        candidates,
        key=lambda cand: (
            cand.score,
            abs(cand.kept_notes - target_notes),
            -cand.kept_notes,
        ),
    )


def _build_kept_event_set(groups: Sequence[NoteGroup], kept_groups: int) -> set[tuple[int, int]]:
    kept: set[tuple[int, int]] = set()
    for group in groups[:kept_groups]:
        for event in group.events:
            kept.add((int(event.instrument_index), int(event.note_index)))
    return kept


def _write_cut_midi(
    midi: pretty_midi.PrettyMIDI,
    groups: Sequence[NoteGroup],
    candidate: CutCandidate,
    output_path: Path,
    onset_tolerance: float,
) -> Path:
    keep_set = _build_kept_event_set(groups, candidate.kept_groups)
    kept_instruments = []

    for instrument_index, instrument in enumerate(midi.instruments):
        kept_notes = []
        for note_index, note in enumerate(instrument.notes):
            if (int(instrument_index), int(note_index)) in keep_set:
                kept_notes.append(note)

        if kept_notes:
            instrument.notes = kept_notes
            instrument.pitch_bends = [bend for bend in instrument.pitch_bends if float(bend.time) <= candidate.cut_time + onset_tolerance]
            instrument.control_changes = [change for change in instrument.control_changes if float(change.time) <= candidate.cut_time + onset_tolerance]
            kept_instruments.append(instrument)

    if not kept_instruments:
        raise RuntimeError("The selected cut removed every instrument.")

    midi.instruments = kept_instruments
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    return output_path


def _resolve_output_path(source_file: Path, output_arg: str, token_count: int) -> Path:
    default_name = f"{source_file.stem}_cut_{int(token_count)}tok.mid"
    if not output_arg:
        return _unique_path(source_file.with_name(default_name))

    output = Path(output_arg).expanduser()
    if output.suffix.lower() in {".mid", ".midi"}:
        return _unique_path(output)

    return _unique_path(output / default_name)


def _resolve_report_path(report_arg: str, output_path: Path) -> Path:
    if not report_arg:
        return output_path.with_suffix(".json")

    report = Path(report_arg).expanduser()
    if report.suffix.lower() == ".json":
        return _unique_path(report)
    return _unique_path(report / f"{output_path.stem}.json")


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return int(default)
        try:
            return int(raw)
        except ValueError:
            print("Please enter a whole number.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pick a MIDI file, then cut it on a musically safe boundary near a token budget. "
            "The tool follows the repo's 4-token event-quad contract."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=".",
        help="MIDI file or folder to scan. Defaults to the current directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subfolders when the source is a directory.",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=DEFAULT_TOKEN_LIMIT,
        help="Target token budget to cut near (default: 512).",
    )
    parser.add_argument(
        "--min-gap-seconds",
        type=float,
        default=DEFAULT_MIN_GAP_SECONDS,
        help="Prefer a cut that leaves at least this much silence before the next onset group.",
    )
    parser.add_argument(
        "--onset-tolerance",
        type=float,
        default=DEFAULT_ONSET_TOLERANCE,
        help="Treat notes whose onsets are within this tolerance as one chord group.",
    )
    parser.add_argument(
        "--strict-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer staying at or under the token limit when a legal chord-safe cut exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected cut but do not write output files.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output MIDI file or directory. Defaults beside the selected source file.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional JSON report file or directory. Defaults beside the output MIDI.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not prompt when multiple MIDI files are found; choose the first one.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    source = Path(args.source).expanduser()
    display_root = source if source.is_dir() else source.parent
    midi_files = _scan_midi_files(source, recursive=bool(args.recursive))
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found under: {source}")

    if len(midi_files) == 1:
        selected_file = midi_files[0]
        print(f"Selected MIDI file: {_display_path(selected_file, display_root)}")
    else:
        if args.no_prompt:
            selected_file = midi_files[0]
            print(
                "Multiple MIDI files were found, but --no-prompt was supplied; "
                f"using the first file: {_display_path(selected_file, display_root)}"
            )
        else:
            selected_file = _prompt_for_file(midi_files, display_root)

    midi = pretty_midi.PrettyMIDI(str(selected_file))
    events = _extract_note_events(midi)
    if not events:
        raise RuntimeError(f"No piano note events were found in {selected_file}.")

    groups = _group_note_events(events, float(args.onset_tolerance))
    if not groups:
        raise RuntimeError(f"Could not build note groups for {selected_file}.")

    downbeats, beats = _resolve_downbeats_and_beats(midi)
    target_notes = max(1, int(args.token_limit) // EVENT_SIZE)
    if int(args.token_limit) < EVENT_SIZE:
        print(
            f"Warning: token limit {int(args.token_limit)} is below one event ({EVENT_SIZE} tokens). "
            f"Using one note event instead."
        )

    candidate = _choose_cut_candidate(
        groups=groups,
        target_notes=target_notes,
        min_gap_seconds=float(args.min_gap_seconds),
        strict_limit=bool(args.strict_limit),
        downbeats=downbeats,
        beats=beats,
    )

    keep_events = _build_kept_event_set(groups, candidate.kept_groups)
    kept_event_count = len(keep_events)
    kept_token_count = kept_event_count * EVENT_SIZE
    total_event_count = len(events)
    total_token_count = total_event_count * EVENT_SIZE
    duration_seconds = float(midi.get_end_time())

    output_path = _resolve_output_path(selected_file, args.output, kept_token_count)
    report_path = _resolve_report_path(args.report, output_path)

    print("\nSelected cut:")
    print(f"  source file: {selected_file}")
    print(f"  total note events: {total_event_count}")
    print(f"  total tokens: {total_token_count}")
    print(f"  target token limit: {int(args.token_limit)}")
    print(f"  kept note events: {kept_event_count}")
    print(f"  kept tokens: {kept_token_count}")
    print(f"  kept chord groups: {candidate.kept_groups}")
    print(f"  cut time: {candidate.cut_time:.3f} s")
    if candidate.next_gap_seconds is None:
        print("  next group gap: n/a (this is the final available boundary)")
    else:
        print(f"  next group gap: {candidate.next_gap_seconds:.3f} s")
    if candidate.nearest_downbeat_seconds is None:
        print("  nearest downbeat: n/a")
    else:
        print(f"  nearest downbeat: {candidate.nearest_downbeat_seconds:.3f} s")
    if candidate.nearest_beat_seconds is None:
        print("  nearest beat: n/a")
    else:
        print(f"  nearest beat: {candidate.nearest_beat_seconds:.3f} s")
    print(f"  score: {candidate.score:.3f}")
    print(f"  output MIDI: {output_path}")
    print(f"  report JSON: {report_path}")

    if candidate.token_count > int(args.token_limit):
        print(
            "Warning: the chosen cut is above the requested token limit to avoid splitting a chord boundary."
        )

    report = {
        "source_path": str(selected_file),
        "output_path": str(output_path),
        "report_path": str(report_path),
        "token_limit": int(args.token_limit),
        "event_size": int(EVENT_SIZE),
        "target_note_events": int(target_notes),
        "selected_note_events": int(kept_event_count),
        "selected_tokens": int(kept_token_count),
        "selected_chord_groups": int(candidate.kept_groups),
        "cut_time_seconds": float(candidate.cut_time),
        "next_gap_seconds": None if candidate.next_gap_seconds is None else float(candidate.next_gap_seconds),
        "nearest_downbeat_seconds": None if candidate.nearest_downbeat_seconds is None else float(candidate.nearest_downbeat_seconds),
        "nearest_beat_seconds": None if candidate.nearest_beat_seconds is None else float(candidate.nearest_beat_seconds),
        "score": float(candidate.score),
        "strict_limit": bool(args.strict_limit),
        "min_gap_seconds": float(args.min_gap_seconds),
        "onset_tolerance_seconds": float(args.onset_tolerance),
        "recursive": bool(args.recursive),
        "dry_run": bool(args.dry_run),
        "total_note_events": int(total_event_count),
        "total_tokens": int(total_token_count),
        "duration_seconds": float(duration_seconds),
    }

    if args.dry_run:
        print("\nDry run requested; no files were written.")
        print(json.dumps(report, indent=2))
        return

    written_midi = _write_cut_midi(
        midi=midi,
        groups=groups,
        candidate=CutCandidate(
            kept_notes=kept_event_count,
            kept_groups=candidate.kept_groups,
            token_count=kept_token_count,
            cut_time=candidate.cut_time,
            next_gap_seconds=candidate.next_gap_seconds,
            nearest_downbeat_seconds=candidate.nearest_downbeat_seconds,
            nearest_beat_seconds=candidate.nearest_beat_seconds,
            score=candidate.score,
        ),
        output_path=output_path,
        onset_tolerance=float(args.onset_tolerance),
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote cut MIDI: {written_midi}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from config import DataConfig, TrainConfig
    from data.dataset import PianoDataset
    from data.tokenizer_custom import CustomDeltaTokenizer
    from model.variant_a import VariantAConfig, VariantAModel
    from model.variant_b import VariantBConfig, VariantBModel
    from model.variant_c import VariantCConfig, VariantCModel
    from training.trainer import Trainer
    from utils.logging_utils import get_project_logger
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from config import DataConfig, TrainConfig
    from data.dataset import PianoDataset
    from data.tokenizer_custom import CustomDeltaTokenizer
    from model.variant_a import VariantAConfig, VariantAModel
    from model.variant_b import VariantBConfig, VariantBModel
    from model.variant_c import VariantCConfig, VariantCModel
    from training.trainer import Trainer
    from utils.logging_utils import get_project_logger


LOGGER = get_project_logger()

ARCHITECTURE_LABELS: Dict[str, str] = {
    "variant_a": "gated_delta_cfc_attention_hybrid",
    "variant_b": "transformer_cfc_hybrid",
    "variant_c": "pure_attention_transformer_baseline",
}

BALANCED_SMALL_PROFILES: Dict[str, Dict[str, int]] = {
    # Keep variants in a comparable 10M-15M budget for fair A/B/C tests.
    "variant_a": {"d_model": 544, "n_layers": 4},
    "variant_b": {"d_model": 544, "n_layers": 5},
    "variant_c": {"d_model": 480, "n_layers": 4},
}


def _set_global_seed(seed: int) -> None:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _scan_midi_files(data_dir: Path) -> List[Path]:
    files = sorted(list(data_dir.rglob("*.mid")) + list(data_dir.rglob("*.midi")))
    return [p for p in files if p.is_file()]


def _build_manifest_with_custom_tokenizer(
    midi_files: List[Path],
    tokenizer: CustomDeltaTokenizer,
    processed_dir: Path,
    min_required_tokens: int,
) -> List[Dict[str, object]]:
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, object]] = []
    for midi_path in midi_files:
        try:
            token_ids, onset_times, durations = tokenizer.encode_with_time_features(
                midi_path
            )
        except Exception as exc:
            LOGGER.warning("Skipping %s (tokenization failed: %s)", midi_path, exc)
            continue

        if len(token_ids) < int(min_required_tokens):
            continue

        key = hashlib.sha1(str(midi_path.resolve()).encode("utf-8")).hexdigest()[:16]
        tokens_path = processed_dir / f"{key}.npy"
        onset_path = processed_dir / f"{key}_onset.npy"
        duration_path = processed_dir / f"{key}_duration.npy"

        np.save(tokens_path, np.asarray(token_ids, dtype=np.int64))
        np.save(onset_path, np.asarray(onset_times, dtype=np.float32))
        np.save(duration_path, np.asarray(durations, dtype=np.float32))

        manifest.append(
            {
                "piece_id": key,
                "path": str(midi_path.resolve()),
                "tokens_path": str(tokens_path.resolve()),
                "onset_times_path": str(onset_path.resolve()),
                "durations_path": str(duration_path.resolve()),
                "tokens": int(len(token_ids)),
                "length": int(len(token_ids)),
                "source": "custom",
                "storage_format": "npy",
            }
        )

    return manifest


def _resolve_existing_npz_path(
    raw_path: str,
    *,
    manifest_path: Path,
    pretokenized_root: Optional[Path],
) -> Optional[Path]:
    candidate = Path(str(raw_path))
    probe_paths: List[Path] = []

    if candidate.is_absolute():
        probe_paths.append(candidate)
    else:
        if pretokenized_root is not None:
            probe_paths.append(pretokenized_root / candidate)
            probe_paths.append(pretokenized_root / candidate.name)
        probe_paths.append(manifest_path.parent / candidate)
        probe_paths.append(manifest_path.parent.parent / candidate)
        probe_paths.append(manifest_path.parent.parent / "data" / candidate.name)

    for path in probe_paths:
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def _load_pretokenized_manifest(
    *,
    manifest_path: Path,
    pretokenized_root: Optional[Path],
    max_pieces: int,
    min_required_tokens: int,
) -> List[Dict[str, object]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pre-tokenized manifest not found: {manifest_path}")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"Invalid or empty pre-tokenized manifest: {manifest_path}")

    loaded: List[Dict[str, object]] = []
    skipped_unresolved = 0
    skipped_short = 0

    for row in raw:
        if not isinstance(row, dict):
            skipped_unresolved += 1
            continue

        raw_npz = str(row.get("npz_path", "")).strip()
        if not raw_npz:
            md5 = str(row.get("md5", "")).strip()
            if md5:
                raw_npz = f"{md5}.npz"

        if not raw_npz:
            skipped_unresolved += 1
            continue

        npz_path = _resolve_existing_npz_path(
            raw_npz,
            manifest_path=manifest_path,
            pretokenized_root=pretokenized_root,
        )
        if npz_path is None:
            skipped_unresolved += 1
            continue

        length = int(row.get("length", row.get("tokens", -1)))
        if length <= 0:
            with np.load(npz_path, allow_pickle=False) as pack:
                length = int(pack["tokens"].shape[0])

        if length < int(min_required_tokens):
            skipped_short += 1
            continue

        loaded.append(
            {
                "piece_id": str(row.get("md5", npz_path.stem) or npz_path.stem),
                "path": str(row.get("source_path", "")),
                "tokens_path": str(npz_path),
                "onset_times_path": "",
                "durations_path": "",
                "tokens": int(length),
                "length": int(length),
                "source": "godzilla_piano",
                "storage_format": "npz",
            }
        )

        if max_pieces > 0 and len(loaded) >= int(max_pieces):
            break

    LOGGER.info(
        "Loaded pre-tokenized manifest: kept=%d skipped_unresolved=%d skipped_short=%d",
        len(loaded),
        skipped_unresolved,
        skipped_short,
    )
    if len(loaded) < 2:
        raise RuntimeError(
            "Need at least two eligible entries in pre-tokenized manifest after filtering."
        )
    return loaded


class NpzWindowDataset(PianoDataset):
    """Adapt PianoDataset windowing logic to .npz token packs."""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.manifest[idx]
        npz_path = Path(str(item["tokens_path"]))

        with np.load(npz_path, allow_pickle=False) as pack:
            token_seq = np.asarray(pack["tokens"], dtype=np.int64)
            if "onsets" in pack:
                onset_seq = np.asarray(pack["onsets"], dtype=np.float32)
            elif "onset_times" in pack:
                onset_seq = np.asarray(pack["onset_times"], dtype=np.float32)
            else:
                step = float(
                    max(1e-4, self.data_config.time_feature_fallback_step_seconds)
                )
                onset_seq = np.arange(token_seq.shape[0], dtype=np.float32) * step

            if "durations" in pack:
                duration_seq = np.asarray(pack["durations"], dtype=np.float32)
            else:
                step = float(
                    max(1e-4, self.data_config.time_feature_fallback_step_seconds)
                )
                duration_seq = np.full((token_seq.shape[0],), fill_value=step, dtype=np.float32)

        total_needed = (
            self.data_config.seed_length + self.data_config.continuation_length
        )
        if token_seq.shape[0] < total_needed:
            raise RuntimeError(
                f"Piece {npz_path} shorter than required window {total_needed}."
            )

        max_start = int(token_seq.shape[0] - total_needed)
        raw_start = self.rng.randint(0, max_start) if max_start > 0 else 0
        start = self._snap_to_triplet_boundary(raw_start, max_start)

        if start % 3 != 0:
            raise AssertionError(
                "Triplet boundary violation in dataset windowing: "
                f"start={start} (raw_start={raw_start}) is not divisible by 3"
            )

        seed = token_seq[start : start + self.data_config.seed_length]
        cont = token_seq[
            start + self.data_config.seed_length : start
            + self.data_config.seed_length
            + self.data_config.continuation_length
        ]

        onset = onset_seq[start : start + total_needed]
        duration = duration_seq[start : start + total_needed]

        seed_t = torch.from_numpy(seed.astype(np.int64, copy=False))
        cont_t = torch.from_numpy(cont.astype(np.int64, copy=False))
        onset_t = torch.from_numpy(onset.astype(np.float32, copy=False))
        duration_t = torch.from_numpy(duration.astype(np.float32, copy=False))
        return {
            "seed": seed_t,
            "continuation": cont_t,
            "token_ids": torch.cat([seed_t, cont_t], dim=0),
            "onset_times": onset_t,
            "durations": duration_t,
            "new_piece": torch.tensor(True),
        }


def _train_val_split(
    manifest: List[Dict[str, object]],
    seed: int,
    val_fraction: float = 0.1,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not manifest:
        raise RuntimeError("Manifest is empty after custom tokenization.")

    rng = random.Random(int(seed))
    shuffled = list(manifest)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_val = max(1, int(round(float(n_total) * float(val_fraction))))
    n_val = min(n_val, n_total - 1) if n_total > 1 else 1

    val_manifest = shuffled[:n_val]
    train_manifest = shuffled[n_val:]

    if not train_manifest:
        train_manifest = shuffled[:1]
    if not val_manifest:
        val_manifest = shuffled[-1:]

    return train_manifest, val_manifest


def _build_dataloaders(
    train_manifest: List[Dict[str, object]],
    val_manifest: List[Dict[str, object]],
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    combined = train_manifest + val_manifest
    use_npz = any(
        str(item.get("storage_format", "")).strip().lower() == "npz"
        for item in combined
    )
    dataset_cls = NpzWindowDataset if use_npz else PianoDataset

    train_ds = dataset_cls(train_manifest, data_cfg, seed=int(seed))
    val_ds = dataset_cls(val_manifest, data_cfg, seed=int(seed) + 1)

    use_cuda = train_cfg.device == "cuda" or (
        train_cfg.device == "auto" and torch.cuda.is_available()
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(int(seed))

    loader_kwargs = {
        "batch_size": int(train_cfg.batch_size),
        "num_workers": 0,
        "pin_memory": bool(use_cuda),
        "persistent_workers": False,
        "collate_fn": PianoDataset.collate_fn,
        "drop_last": False,
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=train_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def _trim_midi_to_target_end(midi: Any, target_end: float) -> Any:
    try:
        import pretty_midi
    except Exception as exc:
        raise RuntimeError(
            "pretty_midi is required to trim generated continuations."
        ) from exc

    trimmed = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for inst in midi.instruments:
        new_inst = pretty_midi.Instrument(
            program=int(inst.program),
            is_drum=bool(inst.is_drum),
            name=str(getattr(inst, "name", "")),
        )
        for note in inst.notes:
            start = float(note.start)
            end = float(note.end)
            if start >= float(target_end):
                continue
            clipped_end = min(end, float(target_end))
            if clipped_end <= start + 1e-4:
                continue
            new_inst.notes.append(
                pretty_midi.Note(
                    velocity=int(note.velocity),
                    pitch=int(note.pitch),
                    start=float(start),
                    end=float(clipped_end),
                )
            )
        if new_inst.notes:
            trimmed.instruments.append(new_inst)

    if not trimmed.instruments:
        return midi
    return trimmed


def _generate_one_continuation(
    *,
    model: Any,
    tokenizer: CustomDeltaTokenizer,
    seed_midi: Path,
    output_path: Path,
    seed_length: int,
    continuation_seconds: float = 30.0,
) -> Dict[str, float]:
    try:
        import pretty_midi
    except Exception as exc:
        raise RuntimeError("pretty_midi is required for generation export.") from exc

    token_ids, onset_times, _durations = tokenizer.encode_with_time_features(seed_midi)
    if not token_ids:
        raise RuntimeError(f"Seed MIDI tokenization produced no tokens: {seed_midi}")

    seed_n = min(int(seed_length), len(token_ids))
    seed_n = seed_n - (seed_n % 3)
    if seed_n < 3:
        raise RuntimeError(
            "Seed MIDI is too short for triplet generation; need at least one full triplet."
        )
    seed_tokens = token_ids[:seed_n]
    seed_onsets = onset_times[:seed_n]

    generated_ids = model.generate(
        seed_tokens=seed_tokens,
        max_new_tokens=2048,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        repetition_window=64,
        min_tokens_to_keep=3,
        seed_onset_times=torch.tensor(seed_onsets, dtype=torch.float32),
        step_seconds=0.1,
        token_id_to_events=tokenizer.decode_token_id_events,
    )

    decoded_midi = tokenizer.decode(generated_ids)
    seed_duration = float(pretty_midi.PrettyMIDI(str(seed_midi)).get_end_time())
    target_end = float(seed_duration + float(continuation_seconds))
    trimmed = _trim_midi_to_target_end(decoded_midi, target_end=target_end)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed.write(str(output_path))

    total_duration = float(trimmed.get_end_time())
    continuation_duration = max(0.0, total_duration - seed_duration)
    return {
        "seed_duration_seconds": seed_duration,
        "total_duration_seconds": total_duration,
        "continuation_duration_seconds": continuation_duration,
    }


def _make_train_config(
    *,
    epochs: int,
    batch_size: int,
    checkpoint_dir: Path,
    warmup_steps: int,
    seed: int,
) -> TrainConfig:
    cfg = TrainConfig(
        batch_size=int(batch_size),
        grad_accumulation_steps=1,
        learning_rate=3e-4,
        lr_schedule="cosine",
        min_lr_ratio=0.1,
        weight_decay=0.01,
        label_smoothing=0.1,
        max_epochs=int(epochs),
        warmup_steps=int(max(1, warmup_steps)),
        max_grad_norm=1.0,
        save_every_n_epochs=max(1, int(epochs)),
        keep_every_n_epochs=max(1, int(epochs)),
        max_checkpoints=3,
        checkpoint_dir=str(checkpoint_dir),
        use_wandb=False,
        seed=int(seed),
        device="auto",
        val_generation_check=False,
    )
    setattr(cfg, "_force_num_workers", 0)
    setattr(cfg, "_enable_data_parallel", False)
    return cfg


def _run_variant(
    *,
    variant_name: str,
    model: torch.nn.Module,
    tokenizer: CustomDeltaTokenizer,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    train_manifest: List[Dict[str, object]],
    val_manifest: List[Dict[str, object]],
    seed_midi: Optional[Path],
    output_midi_path: Optional[Path],
) -> Dict[str, Any]:
    _set_global_seed(int(train_cfg.seed))
    train_loader, val_loader = _build_dataloaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        seed=int(train_cfg.seed),
    )

    trainer_tokenizer: Any = tokenizer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        data_config=data_cfg,
        tokenizer=trainer_tokenizer,
    )
    history = trainer.train()

    core_model = trainer._unwrap_model()
    if seed_midi is not None and output_midi_path is not None:
        generation_meta = _generate_one_continuation(
            model=core_model,
            tokenizer=tokenizer,
            seed_midi=seed_midi,
            output_path=output_midi_path,
            seed_length=int(data_cfg.seed_length),
            continuation_seconds=30.0,
        )
        output_midi = str(output_midi_path.resolve())
    else:
        generation_meta = {
            "skipped": True,
            "reason": "No --seed_midi provided (or --skip_generation enabled).",
        }
        output_midi = ""

    result = {
        "variant": variant_name,
        "params": int(sum(p.numel() for p in core_model.parameters())),
        "train_loss": [float(v) for v in history.get("train_loss", [])],
        "val_loss": [float(v) for v in history.get("val_loss", [])],
        "perplexity": [float(v) for v in history.get("perplexity", [])],
        "checkpoint_dir": str(Path(train_cfg.checkpoint_dir).resolve()),
        "output_midi": output_midi,
        "generation": generation_meta,
    }

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _default_seed_midi(data_dir: Path) -> Path:
    midi_files = _scan_midi_files(data_dir)
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found under {data_dir.resolve()}")
    return midi_files[0]


def _resolve_num_heads(d_model: int, requested_heads: int) -> int:
    heads = max(1, min(int(requested_heads), int(d_model)))
    while heads > 1 and (int(d_model) % heads) != 0:
        heads -= 1
    return max(1, heads)


def _parse_variants_arg(raw: str) -> List[str]:
    mapping = {
        "a": "variant_a",
        "variant_a": "variant_a",
        "gdn": "variant_a",
        "gdn_cfc_attention": "variant_a",
        "b": "variant_b",
        "variant_b": "variant_b",
        "transformer_cfc": "variant_b",
        "c": "variant_c",
        "variant_c": "variant_c",
        "baseline": "variant_c",
        "pure_attention": "variant_c",
    }

    seen = set()
    resolved: List[str] = []
    for token in str(raw).split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in mapping:
            valid = ", ".join(sorted(mapping.keys()))
            raise ValueError(f"Unsupported variant token '{token}'. Valid: {valid}")
        name = mapping[key]
        if name in seen:
            continue
        seen.add(name)
        resolved.append(name)

    if not resolved:
        raise ValueError("No variants selected. Use --variants a,b,c (or subset).")
    return resolved


def _variant_backend_status(model: torch.nn.Module) -> Dict[str, bool]:
    status = {
        "gdn_using_fallback": False,
        "cfc_using_fallback": False,
    }
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name == "GatedDeltaNetBlock" and bool(
            getattr(module, "using_fallback", False)
        ):
            status["gdn_using_fallback"] = True
        if cls_name == "CfCBlock" and bool(getattr(module, "using_fallback", False)):
            status["cfc_using_fallback"] = True
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A/B/C architecture ablation on custom triplet tokenizer data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Root directory scanned recursively for .mid/.midi files.",
    )
    parser.add_argument(
        "--pretokenized_manifest",
        type=str,
        default="",
        help="Optional path to pre-tokenized manifest.json produced by local Godzilla tokenizer.",
    )
    parser.add_argument(
        "--pretokenized_root",
        type=str,
        default="",
        help="Optional root folder used to resolve relative npz paths from pre-tokenized manifest.",
    )
    parser.add_argument(
        "--seed_midi",
        type=str,
        default="",
        help="Seed MIDI path for post-training generation (defaults to first MIDI in data_dir unless --skip_generation is set).",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip post-training MIDI continuation export; useful for pretokenized-only runs.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output root for checkpoints, manifest, MIDI generations, and ablation_results.json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_length", type=int, default=256)
    parser.add_argument("--continuation_length", type=int, default=768)
    parser.add_argument(
        "--max_pieces",
        type=int,
        default=0,
        help="Optional cap on number of pieces used for training (0 means all).",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="a,b,c",
        help="Comma-separated subset of variants to run. Example: a,b,c or c.",
    )
    parser.add_argument(
        "--size_mode",
        type=str,
        choices=["balanced_small", "shared"],
        default="balanced_small",
        help="balanced_small uses per-variant 10M-15M profiles; shared uses one d_model/n_layers for all variants.",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="Model width used across selected variants when size_mode=shared.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="Number of stacked blocks used across selected variants when size_mode=shared.",
    )
    parser.add_argument(
        "--require_real_gdn",
        action="store_true",
        help="Fail run if Variant A cannot use real flash-linear-attention GDN kernels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if str(args.size_mode) == "shared":
        if int(args.d_model) <= 0:
            raise ValueError("--d_model must be > 0")
        if int(args.n_layers) <= 0:
            raise ValueError("--n_layers must be > 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_global_seed(int(args.seed))

    tokenizer = CustomDeltaTokenizer(include_special_tokens=False)
    tokenizer_path = output_dir / "custom_tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    min_required = int(args.seed_length) + int(args.continuation_length)
    max_pieces = int(max(0, args.max_pieces))
    data_source = "raw_midi"

    if str(args.pretokenized_manifest).strip():
        manifest_input_path = Path(str(args.pretokenized_manifest)).expanduser()
        pretokenized_root = (
            Path(str(args.pretokenized_root)).expanduser()
            if str(args.pretokenized_root).strip()
            else None
        )
        manifest = _load_pretokenized_manifest(
            manifest_path=manifest_input_path,
            pretokenized_root=pretokenized_root,
            max_pieces=max_pieces,
            min_required_tokens=min_required,
        )
        processed_dir = output_dir / "processed_pretokenized"
        data_source = "pretokenized_npz"
    else:
        data_dir_raw = str(args.data_dir).strip()
        if not data_dir_raw:
            raise ValueError(
                "Provide --data_dir for raw MIDI mode, or provide --pretokenized_manifest."
            )
        data_dir = Path(data_dir_raw)
        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {data_dir.resolve()}")

        midi_files = _scan_midi_files(data_dir)
        if max_pieces > 0:
            midi_files = midi_files[:max_pieces]
        if not midi_files:
            raise RuntimeError(f"No MIDI files found under {data_dir.resolve()}")

        processed_dir = output_dir / "processed_custom"
        manifest = _build_manifest_with_custom_tokenizer(
            midi_files=midi_files,
            tokenizer=tokenizer,
            processed_dir=processed_dir,
            min_required_tokens=min_required,
        )
        if len(manifest) < 2:
            raise RuntimeError(
                "Need at least two eligible pieces after tokenization to build train/val split."
            )

    manifest_path = processed_dir / "manifest.json"
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_manifest, val_manifest = _train_val_split(
        manifest=manifest,
        seed=int(args.seed),
        val_fraction=0.1,
    )

    if bool(args.skip_generation):
        seed_midi: Optional[Path] = None
    elif str(args.seed_midi).strip():
        seed_midi = Path(str(args.seed_midi)).expanduser()
    else:
        data_dir_raw = str(args.data_dir).strip()
        if not data_dir_raw:
            raise ValueError(
                "Provide --seed_midi when using --pretokenized_manifest without a raw MIDI --data_dir."
            )
        seed_midi = _default_seed_midi(Path(data_dir_raw))
    if seed_midi is not None and not seed_midi.exists():
        raise FileNotFoundError(f"seed_midi not found: {seed_midi.resolve()}")

    data_cfg = DataConfig(
        tokenizer_path=str(tokenizer_path),
        processed_path=str(processed_dir),
        vocab_size=int(tokenizer.vocab_size),
        tokenization_strategy="custom_delta",
        seed_length=int(args.seed_length),
        continuation_length=int(args.continuation_length),
        max_sequence_length=int(args.seed_length) + int(args.continuation_length),
        use_continuous_time=True,
        time_feature_fallback_step_seconds=0.1,
    )

    # Compute warmup from one deterministic loader build; reused identically per variant.
    warmup_probe_cfg = _make_train_config(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        checkpoint_dir=output_dir / "checkpoints" / "_probe",
        warmup_steps=10,
        seed=int(args.seed),
    )
    probe_train_loader, _ = _build_dataloaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        data_cfg=data_cfg,
        train_cfg=warmup_probe_cfg,
        seed=int(args.seed),
    )
    steps_per_epoch = max(
        1,
        math.ceil(
            len(probe_train_loader)
            / float(max(1, warmup_probe_cfg.grad_accumulation_steps))
        ),
    )
    total_steps = max(1, int(steps_per_epoch) * int(args.epochs))
    warmup_steps = max(1, int(0.1 * total_steps))

    LOGGER.info(
        "Ablation setup: pieces=%d train=%d val=%d steps/epoch=%d total_steps=%d warmup=%d",
        len(manifest),
        len(train_manifest),
        len(val_manifest),
        steps_per_epoch,
        total_steps,
        warmup_steps,
    )

    selected_variants = _parse_variants_arg(args.variants)
    variant_profiles: Dict[str, Dict[str, int]] = {}
    for variant_name in selected_variants:
        if str(args.size_mode) == "balanced_small":
            profile = BALANCED_SMALL_PROFILES[variant_name]
            variant_profiles[variant_name] = {
                "d_model": int(profile["d_model"]),
                "n_layers": int(profile["n_layers"]),
            }
        else:
            variant_profiles[variant_name] = {
                "d_model": int(args.d_model),
                "n_layers": int(args.n_layers),
            }

    def _build_variant(name: str) -> Tuple[torch.nn.Module, Dict[str, int]]:
        profile = variant_profiles[name]
        d_model = int(profile["d_model"])
        n_layers = int(profile["n_layers"])
        attn_heads = _resolve_num_heads(d_model=d_model, requested_heads=8)
        gdn_heads = _resolve_num_heads(d_model=d_model, requested_heads=4)
        gqa_groups = 4 if attn_heads % 4 == 0 else (2 if attn_heads % 2 == 0 else 1)
        gdn_inner_dim = max(128, d_model // 2)
        cfc_backbone_units = max(128, int(d_model * 0.75))

        if name == "variant_a":
            model = VariantAModel(
                VariantAConfig(
                    vocab_size=tokenizer.vocab_size,
                    d_model=d_model,
                    n_layers=n_layers,
                    max_sequence_length=data_cfg.max_sequence_length,
                    gdn_inner_dim=gdn_inner_dim,
                    gdn_num_heads=gdn_heads,
                    cfc_backbone_units=cfc_backbone_units,
                    gqa_num_heads=attn_heads,
                    gqa_groups=gqa_groups,
                )
            )
        elif name == "variant_b":
            model = VariantBModel(
                VariantBConfig(
                    vocab_size=tokenizer.vocab_size,
                    d_model=d_model,
                    n_layers=n_layers,
                    max_sequence_length=data_cfg.max_sequence_length,
                    num_attention_heads=attn_heads,
                    cfc_backbone_units=cfc_backbone_units,
                )
            )
        elif name == "variant_c":
            model = VariantCModel(
                VariantCConfig(
                    vocab_size=tokenizer.vocab_size,
                    d_model=d_model,
                    n_layers=n_layers,
                    max_sequence_length=data_cfg.max_sequence_length,
                    num_attention_heads=attn_heads,
                )
            )
        else:
            raise ValueError(f"Unsupported variant {name}")

        shape_meta = {
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "attention_heads": int(attn_heads),
            "gdn_heads": int(gdn_heads),
            "gqa_groups": int(gqa_groups),
        }
        return model, shape_meta

    if str(args.size_mode) == "balanced_small" and "variant_a" in selected_variants:
        baseline_params: List[int] = []
        for baseline_name in ("variant_b", "variant_c"):
            if baseline_name not in selected_variants:
                continue
            baseline_model, _ = _build_variant(baseline_name)
            baseline_params.append(int(sum(p.numel() for p in baseline_model.parameters())))
            del baseline_model

        target_params = (
            int(sum(baseline_params) / len(baseline_params))
            if baseline_params
            else 12_000_000
        )
        original_profile = dict(variant_profiles["variant_a"])

        candidates: List[Tuple[Tuple[int, int, int, int], Dict[str, int], int]] = []
        for d_model in range(416, 577, 32):
            for n_layers in (3, 4, 5):
                variant_profiles["variant_a"] = {
                    "d_model": int(d_model),
                    "n_layers": int(n_layers),
                }
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cand_model, _ = _build_variant("variant_a")
                    params = int(sum(p.numel() for p in cand_model.parameters()))
                    del cand_model
                except Exception:
                    continue

                in_budget = 10_000_000 <= params <= 15_000_000
                score = (
                    0 if in_budget else 1,
                    abs(params - int(target_params)),
                    abs(int(n_layers) - int(original_profile["n_layers"])),
                    abs(int(d_model) - int(original_profile["d_model"])),
                )
                candidates.append(
                    (
                        score,
                        {"d_model": int(d_model), "n_layers": int(n_layers)},
                        int(params),
                    )
                )

        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_score, best_profile, best_params = candidates[0]
            variant_profiles["variant_a"] = best_profile
            LOGGER.info(
                "Balanced-small Variant A auto-tuned: target=%.2fM -> chosen d_model=%d n_layers=%d (%.2fM, score=%s)",
                float(target_params) / 1e6,
                int(best_profile["d_model"]),
                int(best_profile["n_layers"]),
                float(best_params) / 1e6,
                best_score,
            )
        else:
            variant_profiles["variant_a"] = original_profile
            LOGGER.warning(
                "Could not auto-tune Variant A balanced-small profile; using default d_model=%d n_layers=%d.",
                int(original_profile["d_model"]),
                int(original_profile["n_layers"]),
            )

    variants: List[Tuple[str, torch.nn.Module, Dict[str, int], Dict[str, bool]]] = []
    param_by_variant: Dict[str, int] = {}
    for name in selected_variants:
        model, shape_meta = _build_variant(name)
        backend_status = _variant_backend_status(model)
        if name == "variant_a" and bool(args.require_real_gdn):
            if backend_status["gdn_using_fallback"]:
                raise RuntimeError(
                    "Variant A is using fallback GDN. Install flash-linear-attention or run without --require_real_gdn."
                )

        params = int(sum(p.numel() for p in model.parameters()))
        param_by_variant[name] = params
        LOGGER.info(
            "Variant %s setup: d_model=%d n_layers=%d params=%.2fM backend=%s",
            name,
            int(shape_meta["d_model"]),
            int(shape_meta["n_layers"]),
            float(params) / 1e6,
            backend_status,
        )
        variants.append((name, model, shape_meta, backend_status))

    if param_by_variant:
        min_params = min(param_by_variant.values())
        max_params = max(param_by_variant.values())
        ratio = float(max_params) / float(max(1, min_params))
        LOGGER.info(
            "Parameter comparability: min=%.2fM max=%.2fM ratio=%.3f",
            float(min_params) / 1e6,
            float(max_params) / 1e6,
            ratio,
        )

    results: Dict[str, Any] = {
        "seed": int(args.seed),
        "data_dir": str(args.data_dir),
        "data_source": data_source,
        "seed_midi": str(seed_midi.resolve()) if seed_midi is not None else "",
        "generation_enabled": seed_midi is not None,
        "tokenizer": "CustomDeltaTokenizer",
        "manifest_path": str(manifest_path.resolve()),
        "train_pieces": int(len(train_manifest)),
        "val_pieces": int(len(val_manifest)),
        "baseline_variant": "variant_c",
        "variant_definitions": ARCHITECTURE_LABELS,
        "selected_variants": selected_variants,
        "size_mode": str(args.size_mode),
        "variant_profiles": variant_profiles,
        "parameter_counts": param_by_variant,
        "train_config_shared": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "optimizer": "AdamW",
            "lr_schedule": "warmup_cosine",
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "warmup_steps": int(warmup_steps),
            "min_lr_ratio": 0.1,
            "grad_accumulation_steps": 1,
        },
        "variants": {},
    }

    for name, model, shape_meta, backend_status in variants:
        LOGGER.info("Starting %s", name)
        ckpt_dir = output_dir / "checkpoints" / name
        train_cfg = _make_train_config(
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            checkpoint_dir=ckpt_dir,
            warmup_steps=warmup_steps,
            seed=int(args.seed),
        )
        result = _run_variant(
            variant_name=name,
            model=model,
            tokenizer=tokenizer,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            seed_midi=seed_midi,
            output_midi_path=(output_dir / f"{name}.mid") if seed_midi is not None else None,
        )
        result["architecture"] = ARCHITECTURE_LABELS.get(name, name)
        result["shape"] = shape_meta
        result["backend_status"] = backend_status
        results["variants"][name] = result

    results_path = output_dir / "ablation_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved ablation results to %s", results_path.resolve())
    LOGGER.info("Saved generated outputs to %s", output_dir.resolve())


if __name__ == "__main__":
    main()

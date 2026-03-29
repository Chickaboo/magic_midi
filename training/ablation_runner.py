from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
            }
        )

    return manifest


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
    train_ds = PianoDataset(train_manifest, data_cfg, seed=int(seed))
    val_ds = PianoDataset(val_manifest, data_cfg, seed=int(seed) + 1)

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
    seed_midi: Path,
    output_midi_path: Path,
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
    generation_meta = _generate_one_continuation(
        model=core_model,
        tokenizer=tokenizer,
        seed_midi=seed_midi,
        output_path=output_midi_path,
        seed_length=int(data_cfg.seed_length),
        continuation_seconds=30.0,
    )

    result = {
        "variant": variant_name,
        "params": int(sum(p.numel() for p in core_model.parameters())),
        "train_loss": [float(v) for v in history.get("train_loss", [])],
        "val_loss": [float(v) for v in history.get("val_loss", [])],
        "perplexity": [float(v) for v in history.get("perplexity", [])],
        "checkpoint_dir": str(Path(train_cfg.checkpoint_dir).resolve()),
        "output_midi": str(output_midi_path.resolve()),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A/B/C architecture ablation on custom triplet tokenizer data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="maestro-v3.0.0",
        help="Root directory scanned recursively for .mid/.midi files.",
    )
    parser.add_argument(
        "--seed_midi",
        type=str,
        default="",
        help="Seed MIDI path for post-training generation (defaults to first MIDI in data_dir).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir.resolve()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_global_seed(int(args.seed))

    tokenizer = CustomDeltaTokenizer(include_special_tokens=False)
    tokenizer_path = output_dir / "custom_tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    midi_files = _scan_midi_files(data_dir)
    if not midi_files:
        raise RuntimeError(f"No MIDI files found under {data_dir.resolve()}")

    min_required = int(args.seed_length) + int(args.continuation_length)
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
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_manifest, val_manifest = _train_val_split(
        manifest=manifest,
        seed=int(args.seed),
        val_fraction=0.1,
    )

    seed_midi = Path(args.seed_midi) if args.seed_midi else _default_seed_midi(data_dir)
    if not seed_midi.exists():
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

    variants = [
        (
            "variant_a",
            VariantAModel(
                VariantAConfig(
                    vocab_size=tokenizer.vocab_size,
                    d_model=512,
                    n_layers=4,
                    max_sequence_length=data_cfg.max_sequence_length,
                )
            ),
        ),
        (
            "variant_b",
            VariantBModel(
                VariantBConfig(
                    vocab_size=tokenizer.vocab_size,
                    d_model=512,
                    n_layers=4,
                    max_sequence_length=data_cfg.max_sequence_length,
                )
            ),
        ),
        (
            "variant_c",
            VariantCModel(
                VariantCConfig(
                    vocab_size=tokenizer.vocab_size,
                    d_model=512,
                    n_layers=4,
                    max_sequence_length=data_cfg.max_sequence_length,
                )
            ),
        ),
    ]

    results: Dict[str, Any] = {
        "seed": int(args.seed),
        "data_dir": str(data_dir.resolve()),
        "seed_midi": str(seed_midi.resolve()),
        "tokenizer": "CustomDeltaTokenizer",
        "manifest_path": str(manifest_path.resolve()),
        "train_pieces": int(len(train_manifest)),
        "val_pieces": int(len(val_manifest)),
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

    for name, model in variants:
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
            output_midi_path=output_dir / f"{name}.mid",
        )
        results["variants"][name] = result

    results_path = output_dir / "ablation_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved ablation results to %s", results_path.resolve())
    LOGGER.info("Saved generated outputs to %s", output_dir.resolve())


if __name__ == "__main__":
    main()

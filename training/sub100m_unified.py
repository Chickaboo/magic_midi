from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from config import DataConfig, TrainConfig
from data.tokenizer_custom import CustomDeltaTokenizer
from model.variant_c import VariantCConfig, VariantCModel
from model.variant_e import VariantEConfig, VariantEModel
from training.ablation_runner import (
    _build_dataloaders,
    _load_pretokenized_manifest,
    _run_variant,
    _set_global_seed,
    _train_val_split,
    _variant_backend_status,
)


UNIFIED_40M_PROFILES: Dict[str, Dict[str, float]] = {
    "c": {
        "d_model": 512,
        "n_layers": 12,
        "num_attention_heads": 8,
        "ffn_expansion": 4,
        "target_params": 40_000_000,
    },
    "e": {
        "d_model": 768,
        "n_layers": 13,
        "attention_every_n_layers": 2,
        "gdn_inner_ratio": 0.5,
        "gdn_num_heads": 4,
        "gqa_num_heads": 8,
        "gqa_groups": 4,
        "target_params": 40_000_000,
    },
}


@dataclass
class UnifiedSub100mConfig:
    variant: str = "e"
    output_dir: str = "outputs/sub100m_unified_e_100k"
    pretokenized_manifest: str = ""
    pretokenized_root: str = ""
    npz_root: str = ""
    force_rebuild_manifest: bool = False

    max_pieces: int = 100_000
    seed: int = 42
    seed_length: int = 512
    continuation_length: int = 1536
    max_sequence_length: int = 2048

    epochs: int = 20
    batch_size: int = 1
    grad_accumulation_steps: int = 32
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    num_workers: int = 0
    log_every_n_steps: int = 20
    save_every_n_epochs: int = 5
    keep_every_n_epochs: int = 10
    max_checkpoints: int = 8

    seed_midi: str = ""
    generation_max_new_tokens: int = 8192
    generation_continuation_seconds: float = 120.0
    generation_temperature: float = 0.9
    generation_top_p: float = 0.95
    generation_top_k: int = 50
    generation_repetition_penalty: float = 1.1
    generation_repetition_window: int = 64
    generation_min_tokens_to_keep: int = 3
    generation_max_consecutive_zero_deltas: int = 12

    auto_resume: bool = True
    resume_from_checkpoint: str = ""
    resume_mode: str = "remaining"

    enable_data_parallel_c: bool = True
    enable_data_parallel_e: bool = True
    allow_gdn_data_parallel: bool = False
    allow_fallback_gdn: bool = False

    dry_run: bool = False


class Sub100mConfigError(ValueError):
    pass


def _resolve_divisible_heads(width: int, requested_heads: int) -> int:
    heads = max(1, min(int(requested_heads), int(width)))
    while heads > 1 and (int(width) % heads) != 0:
        heads -= 1
    return max(1, heads)


def discover_npz_root(candidates: Iterable[str]) -> Path:
    for raw in candidates:
        candidate = Path(str(raw)).expanduser()
        if not candidate.exists():
            continue
        hit = next(candidate.rglob("*.npz"), None)
        if hit is not None:
            return hit.parent.resolve()

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for folder in kaggle_input.iterdir():
            if not folder.is_dir():
                continue
            hit = next(folder.rglob("*.npz"), None)
            if hit is not None:
                return hit.parent.resolve()

    raise FileNotFoundError("Could not locate tokenized NPZ files.")


def build_manifest_from_npz(npz_root: Path, manifest_path: Path) -> int:
    npz_files = sorted(npz_root.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under: {npz_root}")

    manifest = []
    skipped = 0
    for npz_path in npz_files:
        try:
            with np.load(npz_path, allow_pickle=False) as pack:
                if "tokens" not in pack.files:
                    skipped += 1
                    continue
                token_len = int(np.asarray(pack["tokens"]).shape[0])
        except Exception:
            skipped += 1
            continue

        manifest.append(
            {
                "md5": npz_path.stem,
                "npz_path": str(npz_path),
                "length": token_len,
                "source_path": str(npz_path.parent),
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return int(len(manifest))


def _normalize_config(cfg: UnifiedSub100mConfig) -> UnifiedSub100mConfig:
    variant = str(cfg.variant).strip().lower()
    if variant not in {"c", "e"}:
        raise Sub100mConfigError("variant must be 'c' or 'e'")
    cfg.variant = variant

    cfg.resume_mode = str(cfg.resume_mode).strip().lower()
    if cfg.resume_mode not in {"remaining", "additional"}:
        raise Sub100mConfigError("resume_mode must be 'remaining' or 'additional'")

    if int(cfg.seed_length) <= 0 or int(cfg.continuation_length) <= 0:
        raise Sub100mConfigError("seed_length and continuation_length must be > 0")

    expected_seq = int(cfg.seed_length) + int(cfg.continuation_length)
    if int(cfg.max_sequence_length) != expected_seq:
        raise Sub100mConfigError(
            "max_sequence_length must equal seed_length + continuation_length"
        )

    cfg.batch_size = max(1, int(cfg.batch_size))
    cfg.grad_accumulation_steps = max(1, int(cfg.grad_accumulation_steps))
    cfg.max_pieces = max(0, int(cfg.max_pieces))
    cfg.num_workers = max(0, int(cfg.num_workers))
    cfg.epochs = max(1, int(cfg.epochs))

    return cfg


def _resolve_resume_checkpoint(
    *,
    checkpoint_dir: Path,
    auto_resume: bool,
    resume_from_checkpoint: str,
) -> Optional[Path]:
    def _pick_from_dir(dir_path: Path) -> Optional[Path]:
        candidates = [
            dir_path / "latest_state.pt",
            dir_path / "latest.safetensors",
            dir_path / "best_state.pt",
            dir_path / "best.safetensors",
        ]
        return next((p for p in candidates if p.exists()), None)

    raw = str(resume_from_checkpoint).strip()
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            found = _pick_from_dir(candidate)
            if found is None:
                raise FileNotFoundError(
                    f"No checkpoint files found in directory {candidate.resolve()}"
                )
            return found
        raise FileNotFoundError(f"resume checkpoint not found: {candidate.resolve()}")

    if bool(auto_resume):
        return _pick_from_dir(checkpoint_dir)
    return None


def _build_variant_model(
    *,
    variant: str,
    vocab_size: int,
    max_sequence_length: int,
) -> Tuple[Any, Dict[str, int]]:
    profile = dict(UNIFIED_40M_PROFILES[variant])

    if variant == "c":
        d_model = int(profile["d_model"])
        heads = _resolve_divisible_heads(d_model, int(profile["num_attention_heads"]))

        model = VariantCModel(
            VariantCConfig(
                vocab_size=int(vocab_size),
                d_model=d_model,
                n_layers=int(profile["n_layers"]),
                max_sequence_length=int(max_sequence_length),
                num_attention_heads=int(heads),
                ffn_expansion=int(profile["ffn_expansion"]),
            )
        )
        shape = {
            "d_model": int(d_model),
            "n_layers": int(profile["n_layers"]),
            "num_attention_heads": int(heads),
            "ffn_expansion": int(profile["ffn_expansion"]),
            "target_params": int(profile.get("target_params", 0)),
        }
        return model, shape

    d_model = int(profile["d_model"])
    gdn_inner_dim = max(128, int(round(float(d_model) * float(profile["gdn_inner_ratio"]))))
    gdn_heads = _resolve_divisible_heads(gdn_inner_dim, int(profile["gdn_num_heads"]))
    gqa_heads = _resolve_divisible_heads(d_model, int(profile["gqa_num_heads"]))

    gqa_groups = max(1, min(int(profile["gqa_groups"]), int(gqa_heads)))
    while gqa_groups > 1 and (gqa_heads % gqa_groups) != 0:
        gqa_groups -= 1

    model = VariantEModel(
        VariantEConfig(
            vocab_size=int(vocab_size),
            d_model=d_model,
            n_layers=int(profile["n_layers"]),
            max_sequence_length=int(max_sequence_length),
            gdn_inner_dim=int(gdn_inner_dim),
            gdn_num_heads=int(gdn_heads),
            gqa_num_heads=int(gqa_heads),
            gqa_groups=int(gqa_groups),
            attention_every_n_layers=int(profile["attention_every_n_layers"]),
        )
    )
    shape = {
        "d_model": int(d_model),
        "n_layers": int(profile["n_layers"]),
        "attention_every_n_layers": int(profile["attention_every_n_layers"]),
        "gdn_inner_dim": int(gdn_inner_dim),
        "gdn_num_heads": int(gdn_heads),
        "gqa_num_heads": int(gqa_heads),
        "gqa_groups": int(gqa_groups),
        "target_params": int(profile.get("target_params", 0)),
    }
    return model, shape


def _build_train_cfg(cfg: UnifiedSub100mConfig, checkpoint_dir: Path, warmup_steps: int) -> TrainConfig:
    train_cfg = TrainConfig(
        batch_size=int(cfg.batch_size),
        grad_accumulation_steps=int(cfg.grad_accumulation_steps),
        learning_rate=float(cfg.learning_rate),
        lr_schedule="cosine",
        min_lr_ratio=float(cfg.min_lr_ratio),
        weight_decay=float(cfg.weight_decay),
        label_smoothing=float(cfg.label_smoothing),
        max_epochs=int(cfg.epochs),
        warmup_steps=int(max(1, warmup_steps)),
        max_grad_norm=float(cfg.max_grad_norm),
        save_every_n_epochs=int(max(1, cfg.save_every_n_epochs)),
        keep_every_n_epochs=int(max(1, cfg.keep_every_n_epochs)),
        max_checkpoints=int(max(1, cfg.max_checkpoints)),
        checkpoint_dir=str(checkpoint_dir),
        use_wandb=False,
        seed=int(cfg.seed),
        device="auto",
        val_generation_check=False,
    )

    setattr(train_cfg, "_force_num_workers", int(max(0, cfg.num_workers)))
    setattr(train_cfg, "_log_every_n_steps", int(max(1, cfg.log_every_n_steps)))

    if cfg.variant == "c":
        setattr(train_cfg, "_enable_data_parallel", bool(cfg.enable_data_parallel_c))
    else:
        setattr(train_cfg, "_enable_data_parallel", bool(cfg.enable_data_parallel_e))
        setattr(train_cfg, "_allow_gdn_data_parallel", bool(cfg.allow_gdn_data_parallel))

    return train_cfg


def _prepare_manifest(cfg: UnifiedSub100mConfig, output_dir: Path) -> Tuple[Path, Path]:
    if str(cfg.pretokenized_manifest).strip():
        manifest_path = Path(str(cfg.pretokenized_manifest)).expanduser()
    else:
        manifest_path = output_dir / "processed_pretokenized" / "manifest.json"

    if str(cfg.pretokenized_root).strip():
        npz_root = Path(str(cfg.pretokenized_root)).expanduser()
    elif str(cfg.npz_root).strip():
        npz_root = Path(str(cfg.npz_root)).expanduser()
    else:
        raise Sub100mConfigError(
            "Either pretokenized_root, npz_root, or pretokenized_manifest must be provided."
        )

    if bool(cfg.force_rebuild_manifest) or not manifest_path.exists():
        build_manifest_from_npz(npz_root=npz_root, manifest_path=manifest_path)

    return manifest_path, npz_root


def run_unified_sub100m(cfg: UnifiedSub100mConfig) -> Dict[str, Any]:
    cfg = _normalize_config(cfg)
    output_dir = Path(str(cfg.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_global_seed(int(cfg.seed))

    tokenizer = CustomDeltaTokenizer(include_special_tokens=False)
    event_size = int(getattr(tokenizer, "event_size", 1))
    if event_size > 1:
        if int(cfg.seed_length) % event_size != 0:
            raise Sub100mConfigError(
                f"seed_length must be divisible by event_size={event_size}"
            )
        if int(cfg.continuation_length) % event_size != 0:
            raise Sub100mConfigError(
                f"continuation_length must be divisible by event_size={event_size}"
            )

    tokenizer_path = output_dir / "custom_tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    manifest_path, npz_root = _prepare_manifest(cfg, output_dir)

    manifest = _load_pretokenized_manifest(
        manifest_path=manifest_path,
        pretokenized_root=npz_root,
        max_pieces=int(max(0, cfg.max_pieces)),
        min_required_tokens=int(cfg.max_sequence_length),
    )

    processed_dir = output_dir / "processed_pretokenized"
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = processed_dir / "manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_manifest, val_manifest = _train_val_split(
        manifest=manifest,
        seed=int(cfg.seed),
        val_fraction=0.1,
    )

    data_cfg = DataConfig(
        tokenizer_path=str(tokenizer_path),
        processed_path=str(processed_dir),
        vocab_size=int(tokenizer.vocab_size),
        tokenization_strategy="custom_delta",
        seed_length=int(cfg.seed_length),
        continuation_length=int(cfg.continuation_length),
        max_sequence_length=int(cfg.max_sequence_length),
        use_continuous_time=True,
        time_feature_fallback_step_seconds=0.1,
    )

    probe_cfg = TrainConfig(
        batch_size=int(cfg.batch_size),
        grad_accumulation_steps=int(cfg.grad_accumulation_steps),
        learning_rate=float(cfg.learning_rate),
        lr_schedule="cosine",
        min_lr_ratio=float(cfg.min_lr_ratio),
        weight_decay=float(cfg.weight_decay),
        label_smoothing=float(cfg.label_smoothing),
        max_epochs=1,
        warmup_steps=1,
        max_grad_norm=float(cfg.max_grad_norm),
        save_every_n_epochs=1,
        keep_every_n_epochs=1,
        max_checkpoints=1,
        checkpoint_dir=str(output_dir / "checkpoints" / "_probe"),
        use_wandb=False,
        seed=int(cfg.seed),
        device="auto",
        val_generation_check=False,
    )
    setattr(probe_cfg, "_force_num_workers", int(max(0, cfg.num_workers)))
    setattr(probe_cfg, "_enable_data_parallel", False)

    probe_train_loader, _ = _build_dataloaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        data_cfg=data_cfg,
        train_cfg=probe_cfg,
        seed=int(cfg.seed),
    )

    steps_per_epoch = max(
        1,
        math.ceil(
            len(probe_train_loader)
            / float(max(1, int(cfg.grad_accumulation_steps)))
        ),
    )
    total_steps = max(1, int(steps_per_epoch) * int(cfg.epochs))
    warmup_steps = (
        int(max(1, cfg.warmup_steps))
        if int(cfg.warmup_steps) > 0
        else int(max(1, round(float(cfg.warmup_ratio) * float(total_steps))))
    )

    model, model_shape = _build_variant_model(
        variant=cfg.variant,
        vocab_size=int(tokenizer.vocab_size),
        max_sequence_length=int(cfg.max_sequence_length),
    )
    params = int(sum(p.numel() for p in model.parameters()))
    backend_status = _variant_backend_status(model)

    if cfg.variant == "e" and backend_status.get("gdn_using_fallback", False):
        if not bool(cfg.allow_fallback_gdn):
            raise RuntimeError(
                "Variant E is using fallback GDN. Install flash-linear-attention or set allow_fallback_gdn=True."
            )

    checkpoint_dir = output_dir / "checkpoints" / f"variant_{cfg.variant}_40m"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = _resolve_resume_checkpoint(
        checkpoint_dir=checkpoint_dir,
        auto_resume=bool(cfg.auto_resume),
        resume_from_checkpoint=str(cfg.resume_from_checkpoint),
    )

    train_cfg = _build_train_cfg(cfg=cfg, checkpoint_dir=checkpoint_dir, warmup_steps=warmup_steps)

    seed_midi_path: Optional[Path]
    if str(cfg.seed_midi).strip():
        seed_midi_path = Path(str(cfg.seed_midi)).expanduser()
        if not seed_midi_path.exists():
            raise FileNotFoundError(f"seed_midi not found: {seed_midi_path.resolve()}")
    else:
        seed_midi_path = None

    if bool(cfg.dry_run):
        result: Dict[str, Any] = {
            "dry_run": True,
            "variant": cfg.variant,
            "params": int(params),
            "resume": {
                "enabled": bool(resume_checkpoint is not None),
                "checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else "",
                "mode": str(cfg.resume_mode),
            },
        }
    else:
        generation_out = (
            output_dir / "generated" / f"variant_{cfg.variant}_40m.mid"
            if seed_midi_path is not None
            else None
        )
        result = _run_variant(
            variant_name=f"variant_{cfg.variant}",
            model=model,
            tokenizer=tokenizer,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            seed_midi=seed_midi_path,
            output_midi_path=generation_out,
            generation_max_new_tokens=int(max(1, cfg.generation_max_new_tokens)),
            generation_continuation_seconds=float(max(1.0, cfg.generation_continuation_seconds)),
            generation_temperature=float(max(0.1, cfg.generation_temperature)),
            generation_top_p=float(min(1.0, max(0.0, cfg.generation_top_p))),
            generation_top_k=int(max(1, cfg.generation_top_k)),
            generation_repetition_penalty=float(max(1.0, cfg.generation_repetition_penalty)),
            generation_repetition_window=int(max(1, cfg.generation_repetition_window)),
            generation_min_tokens_to_keep=int(max(1, cfg.generation_min_tokens_to_keep)),
            generation_max_consecutive_zero_deltas=int(max(1, cfg.generation_max_consecutive_zero_deltas)),
            resume_from_checkpoint=resume_checkpoint,
            resume_mode=str(cfg.resume_mode).lower(),
        )

    payload = {
        "profile": f"variant_{cfg.variant}_40m",
        "target_params": int(UNIFIED_40M_PROFILES[cfg.variant].get("target_params", 0)),
        "model_profile": model_shape,
        "backend_status": backend_status,
        "data": {
            "manifest_path": str(manifest_out.resolve()),
            "train_pieces": int(len(train_manifest)),
            "val_pieces": int(len(val_manifest)),
            "source_npz_root": str(npz_root.resolve()),
        },
        "training_profile": {
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "grad_accumulation_steps": int(cfg.grad_accumulation_steps),
            "learning_rate": float(cfg.learning_rate),
            "warmup_steps": int(warmup_steps),
            "steps_per_epoch": int(steps_per_epoch),
            "total_steps": int(total_steps),
            "max_pieces": int(cfg.max_pieces),
            "resume_mode": str(cfg.resume_mode),
            "resume_from_checkpoint": str(resume_checkpoint.resolve())
            if resume_checkpoint is not None
            else "",
        },
        "result": result,
    }

    result_path = output_dir / f"variant_{cfg.variant}_40m_result.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "result_path": str(result_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "params": int(params),
        "payload": payload,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified sub-100M runner for Variant C and Variant E (~40M profiles)."
    )
    parser.add_argument("--variant", choices=["c", "e"], default="e")
    parser.add_argument("--output_dir", type=str, default="outputs/sub100m_unified_e_100k")
    parser.add_argument("--pretokenized_manifest", type=str, default="")
    parser.add_argument("--pretokenized_root", type=str, default="")
    parser.add_argument("--npz_root", type=str, default="")
    parser.add_argument("--force_rebuild_manifest", action="store_true")

    parser.add_argument("--max_pieces", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_length", type=int, default=512)
    parser.add_argument("--continuation_length", type=int, default=1536)
    parser.add_argument("--max_sequence_length", type=int, default=2048)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every_n_steps", type=int, default=20)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--keep_every_n_epochs", type=int, default=10)
    parser.add_argument("--max_checkpoints", type=int, default=8)

    parser.add_argument("--seed_midi", type=str, default="")
    parser.add_argument("--generation_max_new_tokens", type=int, default=8192)
    parser.add_argument("--generation_continuation_seconds", type=float, default=120.0)
    parser.add_argument("--generation_temperature", type=float, default=0.9)
    parser.add_argument("--generation_top_p", type=float, default=0.95)
    parser.add_argument("--generation_top_k", type=int, default=50)
    parser.add_argument("--generation_repetition_penalty", type=float, default=1.1)
    parser.add_argument("--generation_repetition_window", type=int, default=64)
    parser.add_argument("--generation_min_tokens_to_keep", type=int, default=3)
    parser.add_argument("--generation_max_consecutive_zero_deltas", type=int, default=12)

    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--resume_mode", choices=["remaining", "additional"], default="remaining")

    parser.add_argument("--enable_data_parallel_c", action="store_true")
    parser.add_argument("--enable_data_parallel_e", action="store_true")
    parser.add_argument("--allow_gdn_data_parallel", action="store_true")
    parser.add_argument("--allow_fallback_gdn", action="store_true")

    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = UnifiedSub100mConfig(
        variant=str(args.variant),
        output_dir=str(args.output_dir),
        pretokenized_manifest=str(args.pretokenized_manifest),
        pretokenized_root=str(args.pretokenized_root),
        npz_root=str(args.npz_root),
        force_rebuild_manifest=bool(args.force_rebuild_manifest),
        max_pieces=int(args.max_pieces),
        seed=int(args.seed),
        seed_length=int(args.seed_length),
        continuation_length=int(args.continuation_length),
        max_sequence_length=int(args.max_sequence_length),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        grad_accumulation_steps=int(args.grad_accumulation_steps),
        learning_rate=float(args.learning_rate),
        warmup_ratio=float(args.warmup_ratio),
        warmup_steps=int(args.warmup_steps),
        min_lr_ratio=float(args.min_lr_ratio),
        weight_decay=float(args.weight_decay),
        label_smoothing=float(args.label_smoothing),
        max_grad_norm=float(args.max_grad_norm),
        num_workers=int(args.num_workers),
        log_every_n_steps=int(args.log_every_n_steps),
        save_every_n_epochs=int(args.save_every_n_epochs),
        keep_every_n_epochs=int(args.keep_every_n_epochs),
        max_checkpoints=int(args.max_checkpoints),
        seed_midi=str(args.seed_midi),
        generation_max_new_tokens=int(args.generation_max_new_tokens),
        generation_continuation_seconds=float(args.generation_continuation_seconds),
        generation_temperature=float(args.generation_temperature),
        generation_top_p=float(args.generation_top_p),
        generation_top_k=int(args.generation_top_k),
        generation_repetition_penalty=float(args.generation_repetition_penalty),
        generation_repetition_window=int(args.generation_repetition_window),
        generation_min_tokens_to_keep=int(args.generation_min_tokens_to_keep),
        generation_max_consecutive_zero_deltas=int(args.generation_max_consecutive_zero_deltas),
        auto_resume=bool(args.auto_resume),
        resume_from_checkpoint=str(args.resume_from_checkpoint),
        resume_mode=str(args.resume_mode),
        enable_data_parallel_c=bool(args.enable_data_parallel_c),
        enable_data_parallel_e=bool(args.enable_data_parallel_e),
        allow_gdn_data_parallel=bool(args.allow_gdn_data_parallel),
        allow_fallback_gdn=bool(args.allow_fallback_gdn),
        dry_run=bool(args.dry_run),
    )

    report = run_unified_sub100m(cfg)
    payload = dict(report.get("payload") or {})
    result = dict(payload.get("result") or {})

    print("Unified sub-100M run complete")
    print(f"  variant: {cfg.variant}")
    print(f"  params: {int(report['params']):,} ({float(report['params']) / 1e6:.2f}M)")
    print(f"  output_dir: {report['output_dir']}")
    print(f"  result_json: {report['result_path']}")
    if isinstance(result.get("val_loss"), list) and result["val_loss"]:
        print(f"  best_val_loss: {min(float(v) for v in result['val_loss']):.6f}")


if __name__ == "__main__":
    main()

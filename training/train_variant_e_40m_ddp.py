from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DataConfig, TrainConfig
from data.dataset import PianoDataset
from data.tokenizer import CustomDeltaTokenizer
from model.variant_e import VariantEConfig, VariantEModel
from training.ablation_runner import (
    NpzWindowDataset,
    _generate_one_continuation,
    _load_pretokenized_manifest,
    _set_global_seed,
    _train_val_split,
    _variant_backend_status,
)
from training.losses import (
    build_piece_boundary_mask,
    create_targets,
    next_token_accuracy,
    next_token_loss,
    next_token_slot_accuracies,
)
from training.ddp_common import (
    _average_from_sums,
    _is_main_process,
    _load_checkpoint,
    _log,
    _rank_info,
    _resolve_divisible_heads,
    _resolve_resume_checkpoint,
    _save_checkpoint,
)
from training.scheduler import WarmupCosineScheduler


VARIANT_E_40M_PROFILE: Dict[str, float] = {
    "d_model": 640,
    "n_layers": 13,
    "attention_every_n_layers": 2,
    "gdn_inner_ratio": 0.5,
    "gdn_num_heads": 4,
    "gqa_num_heads": 8,
    "gqa_groups": 4,
    "target_params": 40_000_000,
}


def _snapshot_epoch_bundle(
    *,
    bundle_root: Path,
    checkpoint_dir: Path,
    data_cfg: DataConfig,
    epoch: int,
    global_step: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    include_best_artifacts: bool = False,
) -> Path:
    """Copy key checkpoint artifacts into one epoch-scoped folder."""

    epoch_dir = bundle_root / f"epoch_{int(epoch):03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    copied_files: List[str] = []
    checkpoint_names: List[str] = ["latest.safetensors", "latest_state.pt"]
    if bool(include_best_artifacts):
        checkpoint_names.extend(["best.safetensors", "best_state.pt"])

    for name in checkpoint_names:
        src = checkpoint_dir / name
        if src.exists() and src.is_file():
            dst = epoch_dir / name
            shutil.copy2(src, dst)
            copied_files.append(name)

    tokenizer_path = Path(str(data_cfg.tokenizer_path))
    if tokenizer_path.exists() and tokenizer_path.is_file():
        dst = epoch_dir / tokenizer_path.name
        shutil.copy2(tokenizer_path, dst)
        copied_files.append(tokenizer_path.name)

    manifest_path = Path(str(data_cfg.processed_path)) / "manifest.json"
    if manifest_path.exists() and manifest_path.is_file():
        dst = epoch_dir / "manifest.json"
        shutil.copy2(manifest_path, dst)
        copied_files.append("manifest.json")

    summary = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "copied_files": copied_files,
    }
    (epoch_dir / "epoch_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return epoch_dir


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0

    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += int(item.stat().st_size)
        except OSError:
            continue
    return int(total)


def _prune_epoch_bundles(
    *,
    bundle_root: Path,
    max_keep: int,
    max_total_bytes: int,
    protected_dir: Path,
) -> Tuple[List[Tuple[str, int]], int, int]:
    """Prune oldest epoch bundles by count and total-size budget."""

    if not bundle_root.exists():
        return [], 0, 0

    epoch_dirs = [
        p for p in bundle_root.iterdir() if p.is_dir() and p.name.startswith("epoch_")
    ]
    epoch_dirs.sort(key=lambda p: p.name)

    size_map: Dict[Path, int] = {p: _dir_size_bytes(p) for p in epoch_dirs}
    total_bytes = int(sum(size_map.values()))
    removed: List[Tuple[str, int]] = []
    protected = protected_dir.resolve()

    while epoch_dirs:
        over_count = int(max_keep) > 0 and len(epoch_dirs) > int(max_keep)
        over_bytes = int(max_total_bytes) > 0 and total_bytes > int(max_total_bytes)
        if not (over_count or over_bytes):
            break

        candidate = next(
            (p for p in epoch_dirs if p.resolve() != protected),
            None,
        )
        if candidate is None:
            break

        removed_bytes = int(size_map.get(candidate, 0))
        shutil.rmtree(candidate, ignore_errors=True)
        removed.append((candidate.name, removed_bytes))
        epoch_dirs.remove(candidate)
        total_bytes = max(0, int(total_bytes - removed_bytes))

    return removed, int(len(epoch_dirs)), int(total_bytes)


def _maybe_run_epoch_upload_command(
    *,
    template: str,
    epoch: int,
    epoch_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
) -> None:
    """Run optional user-provided upload command template per epoch."""

    raw = str(template).strip()
    if not raw:
        return

    command = raw.format(
        epoch=int(epoch),
        epoch_dir=str(epoch_dir),
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
    )
    print(f"Epoch upload command: {command}")
    subprocess.run(command, shell=True, check=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distributed DDP trainer for Variant E 40M profile on pretokenized NPZ data."
    )
    parser.add_argument("--pretokenized_manifest", type=str, required=True)
    parser.add_argument("--pretokenized_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/sub100m_unified_e_100k")

    parser.add_argument("--max_pieces", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_length", type=int, default=512)
    parser.add_argument("--continuation_length", type=int, default=1536)
    parser.add_argument("--max_sequence_length", type=int, default=2048)

    parser.add_argument("--d_model", type=int, default=int(VARIANT_E_40M_PROFILE["d_model"]))
    parser.add_argument("--n_layers", type=int, default=int(VARIANT_E_40M_PROFILE["n_layers"]))
    parser.add_argument(
        "--attention_every_n_layers",
        type=int,
        default=int(VARIANT_E_40M_PROFILE["attention_every_n_layers"]),
    )
    parser.add_argument("--full_attention", action="store_true")
    parser.add_argument(
        "--gdn_inner_ratio",
        type=float,
        default=float(VARIANT_E_40M_PROFILE["gdn_inner_ratio"]),
    )
    parser.add_argument("--gdn_num_heads", type=int, default=int(VARIANT_E_40M_PROFILE["gdn_num_heads"]))
    parser.add_argument("--gqa_num_heads", type=int, default=int(VARIANT_E_40M_PROFILE["gqa_num_heads"]))
    parser.add_argument("--gqa_groups", type=int, default=int(VARIANT_E_40M_PROFILE["gqa_groups"]))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--max_time_seconds", type=float, default=1200.0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every_n_steps", type=int, default=20)
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=500,
        help="Save latest checkpoint every N optimizer steps (set 0 to disable mid-epoch saves).",
    )
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--epoch_bundle_root", type=str, default="")
    parser.add_argument(
        "--epoch_bundle_include_best",
        action="store_true",
        help="Include best.* checkpoint files in each epoch export bundle (uses more storage).",
    )
    parser.add_argument(
        "--max_epoch_bundles",
        type=int,
        default=2,
        help="Keep at most this many epoch bundle folders (<=0 disables count-based pruning).",
    )
    parser.add_argument(
        "--max_epoch_bundle_total_gb",
        type=float,
        default=12.0,
        help="Approximate total size cap for epoch bundles in GB (<=0 disables size-based pruning).",
    )
    parser.add_argument("--epoch_upload_cmd_template", type=str, default="")

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

    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--resume_mode", choices=["remaining", "additional"], default="remaining")
    parser.add_argument("--no_auto_resume", dest="auto_resume", action="store_false")
    parser.set_defaults(auto_resume=True)

    parser.add_argument("--allow_fallback_gdn", action="store_true")
    parser.add_argument("--disable_slot_aware_loss", action="store_true")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic kernels and strict reproducibility mode (slower).",
    )
    parser.add_argument("--use_amp", action="store_true")
    parser.set_defaults(use_amp=True)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    rank, world_size, local_rank = _rank_info()
    distributed = int(world_size) > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA, but CUDA is not available.")
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=45),
        )
        device = torch.device(f"cuda:{int(local_rank)}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if bool(args.deterministic):
            _set_global_seed(int(args.seed) + int(rank))
            _log(rank, "Deterministic mode enabled (may reduce throughput).")
        else:
            random.seed(int(args.seed) + int(rank))
            np.random.seed(int(args.seed) + int(rank))
            torch.manual_seed(int(args.seed) + int(rank))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.seed) + int(rank))
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        if int(args.seed_length) <= 0 or int(args.continuation_length) <= 0:
            raise ValueError("seed_length and continuation_length must be > 0")
        if int(args.max_sequence_length) != (int(args.seed_length) + int(args.continuation_length)):
            raise ValueError("max_sequence_length must equal seed_length + continuation_length")

        output_dir = Path(str(args.output_dir)).expanduser()
        checkpoint_dir = output_dir / "checkpoints" / "variant_e_40m_ddp"
        epoch_bundle_root = (
            Path(str(args.epoch_bundle_root)).expanduser()
            if str(args.epoch_bundle_root).strip()
            else output_dir / "epoch_exports"
        )
        epoch_bundle_max_keep = int(max(0, int(args.max_epoch_bundles)))
        epoch_bundle_total_gb = float(max(0.0, float(args.max_epoch_bundle_total_gb)))
        epoch_bundle_max_total_bytes = int(round(epoch_bundle_total_gb * (1024**3)))
        if _is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            epoch_bundle_root.mkdir(parents=True, exist_ok=True)
            _log(
                rank,
                "Epoch bundle retention: "
                f"max_keep={epoch_bundle_max_keep if epoch_bundle_max_keep > 0 else 'disabled'} "
                f"max_total_gb={epoch_bundle_total_gb if epoch_bundle_total_gb > 0 else 'disabled'} "
                f"include_best={bool(args.epoch_bundle_include_best)}",
            )
        if distributed:
            dist.barrier()

        tokenizer = CustomDeltaTokenizer(include_special_tokens=False)
        event_size = int(getattr(tokenizer, "event_size", 1))
        if event_size != 4:
            raise RuntimeError(
                f"Variant E DDP flow expects CustomDeltaTokenizer event_size=4, got {event_size}."
            )
        if int(args.seed_length) % event_size != 0 or int(args.continuation_length) % event_size != 0:
            raise RuntimeError(
                "seed_length and continuation_length must be divisible by tokenizer event_size."
            )

        d_model = int(max(64, int(args.d_model)))
        n_layers = int(max(1, int(args.n_layers)))
        attention_every_n_layers = int(max(1, int(args.attention_every_n_layers)))
        gdn_inner_dim = max(128, int(round(float(d_model) * float(args.gdn_inner_ratio))))
        gdn_heads = _resolve_divisible_heads(gdn_inner_dim, int(max(1, int(args.gdn_num_heads))))
        gqa_heads = _resolve_divisible_heads(d_model, int(max(1, int(args.gqa_num_heads))))

        gqa_groups = max(1, min(int(max(1, int(args.gqa_groups))), int(gqa_heads)))
        while gqa_groups > 1 and (gqa_heads % gqa_groups) != 0:
            gqa_groups -= 1

        model = VariantEModel(
            VariantEConfig(
                vocab_size=int(tokenizer.vocab_size),
                d_model=int(d_model),
                n_layers=int(n_layers),
                max_sequence_length=int(args.max_sequence_length),
                dropout=float(max(0.0, args.dropout)),
                attention_dropout=float(max(0.0, args.attention_dropout)),
                gdn_inner_dim=int(gdn_inner_dim),
                gdn_num_heads=int(gdn_heads),
                gqa_num_heads=int(gqa_heads),
                gqa_groups=int(gqa_groups),
                attention_every_n_layers=int(attention_every_n_layers),
                full_attention=bool(args.full_attention),
                use_continuous_time=True,
                max_time_seconds=float(max(1.0, args.max_time_seconds)),
            )
        )

        backend_status = _variant_backend_status(model)
        if backend_status.get("gdn_using_fallback", False) and not bool(args.allow_fallback_gdn):
            raise RuntimeError(
                "Variant E is using fallback GDN in this runtime. Install flash-linear-attention "
                "or pass --allow_fallback_gdn to continue."
            )

        model = model.to(device)
        if distributed:
            model = DDP(
                model,
                device_ids=[int(local_rank)],
                output_device=int(local_rank),
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

        params = int(
            sum(p.numel() for p in (model.module.parameters() if isinstance(model, DDP) else model.parameters()))
        )
        _log(rank, f"Variant-E 40M DDP model params: {params:,} ({params / 1e6:.2f}M)")
        _log(rank, f"Backend status: {backend_status}")

        data_cfg = DataConfig(
            tokenizer_path=str(output_dir / "custom_tokenizer.json"),
            processed_path=str(output_dir / "processed_pretokenized"),
            vocab_size=int(tokenizer.vocab_size),
            tokenization_strategy="custom_delta",
            seed_length=int(args.seed_length),
            continuation_length=int(args.continuation_length),
            max_sequence_length=int(args.max_sequence_length),
            use_continuous_time=True,
            time_feature_fallback_step_seconds=0.1,
        )
        if _is_main_process(rank):
            tokenizer.save(data_cfg.tokenizer_path)
        if distributed:
            dist.barrier()

        manifest_path = Path(str(args.pretokenized_manifest)).expanduser()
        pretokenized_root = (
            Path(str(args.pretokenized_root)).expanduser()
            if str(args.pretokenized_root).strip()
            else None
        )

        manifest = _load_pretokenized_manifest(
            manifest_path=manifest_path,
            pretokenized_root=pretokenized_root,
            max_pieces=int(max(0, args.max_pieces)),
            min_required_tokens=int(args.max_sequence_length),
        )
        train_manifest, val_manifest = _train_val_split(
            manifest=manifest,
            seed=int(args.seed),
            val_fraction=0.1,
        )

        processed_dir = Path(data_cfg.processed_path)
        if _is_main_process(rank):
            processed_dir.mkdir(parents=True, exist_ok=True)
            (processed_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        if distributed:
            dist.barrier()

        train_ds = NpzWindowDataset(train_manifest, data_cfg, seed=int(args.seed))
        val_ds = NpzWindowDataset(val_manifest, data_cfg, seed=int(args.seed) + 1)

        train_sampler = (
            DistributedSampler(
                train_ds,
                num_replicas=int(world_size),
                rank=int(rank),
                shuffle=True,
                seed=int(args.seed),
                drop_last=False,
            )
            if distributed
            else None
        )
        val_sampler = (
            DistributedSampler(
                val_ds,
                num_replicas=int(world_size),
                rank=int(rank),
                shuffle=False,
                seed=int(args.seed),
                drop_last=False,
            )
            if distributed
            else None
        )

        num_workers = max(0, int(args.num_workers))
        loader_common = {
            "batch_size": int(max(1, args.batch_size)),
            "num_workers": int(num_workers),
            "pin_memory": bool(device.type == "cuda"),
            "persistent_workers": bool(num_workers > 0),
            "collate_fn": PianoDataset.collate_fn,
            "drop_last": False,
        }
        if num_workers > 0:
            loader_common["prefetch_factor"] = 2

        train_loader = DataLoader(
            train_ds,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            **loader_common,
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            sampler=val_sampler,
            **loader_common,
        )

        train_cfg = TrainConfig(
            batch_size=int(max(1, args.batch_size)),
            grad_accumulation_steps=int(max(1, args.grad_accumulation_steps)),
            learning_rate=float(args.learning_rate),
            lr_schedule="cosine",
            min_lr_ratio=float(args.min_lr_ratio),
            weight_decay=float(args.weight_decay),
            label_smoothing=float(args.label_smoothing),
            max_epochs=int(max(1, args.epochs)),
            warmup_steps=1,
            max_grad_norm=float(args.max_grad_norm),
            save_every_n_steps=int(max(0, args.save_every_n_steps)),
            save_every_n_epochs=int(max(1, args.save_every_n_epochs)),
            keep_every_n_epochs=int(max(1, args.save_every_n_epochs)),
            max_checkpoints=8,
            checkpoint_dir=str(checkpoint_dir),
            use_wandb=False,
            seed=int(args.seed),
            device="cuda" if device.type == "cuda" else "cpu",
            val_generation_check=False,
            use_amp=bool(args.use_amp),
        )

        if int(train_cfg.save_every_n_steps) <= 0:
            _log(
                rank,
                "WARNING: save_every_n_steps=0 disables mid-epoch checkpoints; "
                "interruption recovery will be epoch-boundary only.",
            )

        optimizer = AdamW(
            model.parameters(),
            lr=float(train_cfg.learning_rate),
            weight_decay=float(train_cfg.weight_decay),
        )

        batches_per_epoch = int(len(train_loader))
        steps_per_epoch = max(
            1,
            math.ceil(float(batches_per_epoch) / float(max(1, int(train_cfg.grad_accumulation_steps)))),
        )
        total_steps = max(1, int(steps_per_epoch) * int(train_cfg.max_epochs))
        warmup_steps = (
            int(max(1, int(args.warmup_steps)))
            if int(args.warmup_steps) > 0
            else int(max(1, round(float(args.warmup_ratio) * float(total_steps))))
        )
        train_cfg.warmup_steps = int(warmup_steps)

        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=int(train_cfg.warmup_steps),
            total_steps=int(total_steps),
            min_lr_ratio=float(train_cfg.min_lr_ratio),
        )

        use_amp = bool(train_cfg.use_amp) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        slot_aware_loss = not bool(args.disable_slot_aware_loss)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_raw_ce": [],
            "val_token_acc": [],
            "val_token_acc_delta": [],
            "val_token_acc_pitch": [],
            "val_token_acc_duration": [],
            "val_token_acc_velocity": [],
            "perplexity": [],
            "lr": [],
        }
        global_step = 0
        best_val_loss = float("inf")
        resume_epoch = 0
        resume_batch_in_epoch = 0
        resume_epoch_complete = True
        first_epoch = 1
        first_epoch_skip_batches = 0
        final_epoch = int(train_cfg.max_epochs)

        resume_checkpoint = _resolve_resume_checkpoint(
            checkpoint_dir=checkpoint_dir,
            auto_resume=bool(args.auto_resume),
            resume_from_checkpoint=str(args.resume_from_checkpoint),
        )
        epochs_to_run = int(train_cfg.max_epochs)

        if resume_checkpoint is not None:
            resumed_state = _load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                checkpoint_path=resume_checkpoint,
                device=device,
            )
            resume_epoch = int(resumed_state.get("epoch", 0))
            history_state = resumed_state.get("history")
            if isinstance(history_state, dict):
                for key in history:
                    if isinstance(history_state.get(key), list):
                        history[key] = [float(v) for v in history_state.get(key, [])]
            global_step = int(resumed_state.get("global_step", 0))
            best_val_loss = float(resumed_state.get("best_val_loss", best_val_loss))

            resume_state = resumed_state.get("resume_state")
            if isinstance(resume_state, dict):
                resume_epoch = int(resume_state.get("epoch", resume_epoch))
                resume_batch_in_epoch = int(resume_state.get("batch_step_in_epoch", 0))
                resume_epoch_complete = bool(resume_state.get("is_epoch_complete", True))

            resume_batch_in_epoch = int(max(0, min(resume_batch_in_epoch, batches_per_epoch)))

            if bool(resume_epoch_complete):
                first_epoch = int(max(1, resume_epoch + 1))
                first_epoch_skip_batches = 0
            else:
                first_epoch = int(max(1, resume_epoch))
                first_epoch_skip_batches = int(resume_batch_in_epoch)

            if str(args.resume_mode).strip().lower() == "remaining":
                final_epoch = int(train_cfg.max_epochs)
            else:
                final_epoch = int(resume_epoch + int(train_cfg.max_epochs))
                if not bool(resume_epoch_complete):
                    final_epoch -= 1

            final_epoch = int(max(first_epoch, final_epoch))
            epochs_to_run = int(max(0, final_epoch - first_epoch + 1))

            _log(
                rank,
                "Resuming DDP run "
                f"from {resume_checkpoint.resolve()} at epoch={resume_epoch} "
                f"batch_in_epoch={resume_batch_in_epoch} "
                f"epoch_complete={resume_epoch_complete} mode={args.resume_mode} "
                f"first_epoch={first_epoch} final_epoch={final_epoch} "
                f"epochs_to_run={epochs_to_run}",
            )
        else:
            first_epoch = 1
            first_epoch_skip_batches = 0
            final_epoch = int(train_cfg.max_epochs)
            epochs_to_run = int(max(0, final_epoch - first_epoch + 1))

        epoch_iterable = list(range(int(first_epoch), int(final_epoch) + 1))
        for epoch_run_idx, epoch in enumerate(epoch_iterable, start=1):
            if distributed and isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(int(epoch))

            model.train()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            running_loss = 0.0
            running_count = 0
            skip_batches_this_epoch = (
                int(first_epoch_skip_batches)
                if int(epoch) == int(first_epoch) and int(first_epoch_skip_batches) > 0
                else 0
            )
            if skip_batches_this_epoch > 0:
                _log(
                    rank,
                    f"Resuming inside epoch {epoch:03d}: skipping {skip_batches_this_epoch}/{batches_per_epoch} batches already completed.",
                )

            for step_idx, batch in enumerate(train_loader, start=1):
                if int(step_idx) <= int(skip_batches_this_epoch):
                    continue

                seed = batch["seed"].to(device, non_blocking=True)
                continuation = batch["continuation"].to(device, non_blocking=True)
                input_ids = batch["token_ids"].to(device, non_blocking=True)
                onset_times = batch["onset_times"].to(device, non_blocking=True)
                durations = batch["durations"].to(device, non_blocking=True)

                targets = create_targets(seed, continuation).to(device, non_blocking=True)
                boundary_mask = build_piece_boundary_mask(
                    seed=seed,
                    continuation=continuation,
                    new_piece=batch["new_piece"],
                ).to(device, non_blocking=True)

                autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp) if device.type == "cuda" else nullcontext()
                with autocast_ctx:
                    logits = model(
                        token_ids=input_ids,
                        onset_times=onset_times,
                        durations=durations,
                        memory=None,
                        return_memory=False,
                        position_offset=0,
                    )
                    if isinstance(logits, tuple):
                        logits = logits[0]

                    loss = next_token_loss(
                        logits,
                        targets,
                        label_smoothing=float(train_cfg.label_smoothing),
                        piece_boundary_mask=boundary_mask,
                        slot_aware=bool(slot_aware_loss),
                        event_size=int(event_size),
                    )

                loss_value = float(loss.item())
                epoch_loss_sum += float(loss_value)
                epoch_loss_count += 1
                running_loss += float(loss_value)
                running_count += 1

                scaled_loss = loss / float(max(1, train_cfg.grad_accumulation_steps))
                if use_amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = (
                    step_idx % int(max(1, train_cfg.grad_accumulation_steps)) == 0
                    or step_idx == int(batches_per_epoch)
                )
                if should_step:
                    if use_amp:
                        scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(train_cfg.max_grad_norm),
                    )

                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    scheduler.step()
                    global_step += 1
                    history["lr"].append(float(optimizer.param_groups[0]["lr"]))

                    if (
                        _is_main_process(rank)
                        and int(max(0, train_cfg.save_every_n_steps)) > 0
                        and int(global_step) % int(max(1, train_cfg.save_every_n_steps)) == 0
                    ):
                        step_avg_loss = float(running_loss / max(1, running_count))
                        _save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            train_cfg=train_cfg,
                            data_cfg=data_cfg,
                            checkpoint_dir=checkpoint_dir,
                            epoch=int(epoch),
                            val_loss=float(step_avg_loss),
                            history=history,
                            best_val_loss=float(best_val_loss),
                            global_step=int(global_step),
                            best=False,
                            resume_state={
                                "epoch": int(epoch),
                                "batch_step_in_epoch": int(step_idx),
                                "batches_per_epoch": int(batches_per_epoch),
                                "is_epoch_complete": bool(int(step_idx) >= int(batches_per_epoch)),
                            },
                        )
                        print(
                            f"Step checkpoint saved: epoch={epoch:03d} "
                            f"step={global_step:06d} batch={int(step_idx):05d}/{int(batches_per_epoch):05d} "
                            f"loss~{step_avg_loss:.4f}"
                        )

                    if _is_main_process(rank) and int(global_step) % int(max(1, args.log_every_n_steps)) == 0 and running_count > 0:
                        avg_running = float(running_loss / max(1, running_count))
                        lr = float(optimizer.param_groups[0]["lr"])
                        print(
                            f"epoch={epoch:03d} step={global_step:06d} "
                            f"train_loss={avg_running:.4f} lr={lr:.6e}"
                        )
                        running_loss = 0.0
                        running_count = 0

            train_loss = _average_from_sums(
                total_sum=float(epoch_loss_sum),
                total_count=float(epoch_loss_count),
                device=device,
            )

            model.eval()
            val_loss_sum = 0.0
            val_loss_count = 0
            val_raw_ce_sum = 0.0
            val_token_acc_sum = 0.0
            val_slot_acc_sum: Dict[str, float] = {
                "delta": 0.0,
                "pitch": 0.0,
                "duration": 0.0,
                "velocity": 0.0,
            }
            with torch.no_grad():
                for batch in val_loader:
                    seed = batch["seed"].to(device, non_blocking=True)
                    continuation = batch["continuation"].to(device, non_blocking=True)
                    input_ids = batch["token_ids"].to(device, non_blocking=True)
                    onset_times = batch["onset_times"].to(device, non_blocking=True)
                    durations = batch["durations"].to(device, non_blocking=True)

                    targets = create_targets(seed, continuation).to(device, non_blocking=True)
                    boundary_mask = build_piece_boundary_mask(
                        seed=seed,
                        continuation=continuation,
                        new_piece=batch["new_piece"],
                    ).to(device, non_blocking=True)

                    autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp) if device.type == "cuda" else nullcontext()
                    with autocast_ctx:
                        logits = model(
                            token_ids=input_ids,
                            onset_times=onset_times,
                            durations=durations,
                            memory=None,
                            return_memory=False,
                            position_offset=0,
                        )
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        loss = next_token_loss(
                            logits,
                            targets,
                            label_smoothing=float(train_cfg.label_smoothing),
                            piece_boundary_mask=boundary_mask,
                            slot_aware=bool(slot_aware_loss),
                            event_size=int(event_size),
                        )
                        raw_ce = next_token_loss(
                            logits,
                            targets,
                            label_smoothing=0.0,
                            piece_boundary_mask=boundary_mask,
                            slot_aware=bool(slot_aware_loss),
                            event_size=int(event_size),
                        )

                    val_loss_sum += float(loss.item())
                    val_raw_ce_sum += float(raw_ce.item())
                    val_loss_count += 1

                    token_acc = next_token_accuracy(
                        logits,
                        targets,
                        piece_boundary_mask=boundary_mask,
                        slot_aware=bool(slot_aware_loss),
                        event_size=int(event_size),
                    )
                    slot_acc = next_token_slot_accuracies(
                        logits,
                        targets,
                        piece_boundary_mask=boundary_mask,
                        event_size=int(event_size),
                        slot_aware=bool(slot_aware_loss),
                    )
                    val_token_acc_sum += float(token_acc)
                    for key in val_slot_acc_sum:
                        val_slot_acc_sum[key] += float(slot_acc.get(key, 0.0))

            val_loss = _average_from_sums(
                total_sum=float(val_loss_sum),
                total_count=float(val_loss_count),
                device=device,
            )
            val_raw_ce = _average_from_sums(
                total_sum=float(val_raw_ce_sum),
                total_count=float(val_loss_count),
                device=device,
            )
            val_token_acc = _average_from_sums(
                total_sum=float(val_token_acc_sum),
                total_count=float(val_loss_count),
                device=device,
            )
            val_slot_acc = {
                key: _average_from_sums(
                    total_sum=float(total),
                    total_count=float(val_loss_count),
                    device=device,
                )
                for key, total in val_slot_acc_sum.items()
            }
            perplexity = float(math.exp(min(20.0, float(val_loss))))

            if _is_main_process(rank):
                history["train_loss"].append(float(train_loss))
                history["val_loss"].append(float(val_loss))
                history["val_raw_ce"].append(float(val_raw_ce))
                history["val_token_acc"].append(float(val_token_acc))
                history["val_token_acc_delta"].append(float(val_slot_acc.get("delta", 0.0)))
                history["val_token_acc_pitch"].append(float(val_slot_acc.get("pitch", 0.0)))
                history["val_token_acc_duration"].append(float(val_slot_acc.get("duration", 0.0)))
                history["val_token_acc_velocity"].append(float(val_slot_acc.get("velocity", 0.0)))
                history["perplexity"].append(float(perplexity))

                _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_cfg=train_cfg,
                    data_cfg=data_cfg,
                    checkpoint_dir=checkpoint_dir,
                    epoch=int(epoch),
                    val_loss=float(val_loss),
                    history=history,
                    best_val_loss=float(best_val_loss),
                    global_step=int(global_step),
                    best=False,
                    resume_state={
                        "epoch": int(epoch),
                        "batch_step_in_epoch": int(batches_per_epoch),
                        "batches_per_epoch": int(batches_per_epoch),
                        "is_epoch_complete": True,
                    },
                )

                if float(val_loss) < float(best_val_loss):
                    best_val_loss = float(val_loss)
                    _save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        train_cfg=train_cfg,
                        data_cfg=data_cfg,
                        checkpoint_dir=checkpoint_dir,
                        epoch=int(epoch),
                        val_loss=float(val_loss),
                        history=history,
                        best_val_loss=float(best_val_loss),
                        global_step=int(global_step),
                        best=True,
                        resume_state={
                            "epoch": int(epoch),
                            "batch_step_in_epoch": int(batches_per_epoch),
                            "batches_per_epoch": int(batches_per_epoch),
                            "is_epoch_complete": True,
                        },
                    )

                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                    f"| val_loss={val_loss:.4f} | raw_ce={val_raw_ce:.4f} "
                    f"| acc={val_token_acc:.4f} | ppl={perplexity:.2f}"
                )

                is_last_epoch_this_run = int(epoch_run_idx) == int(epochs_to_run)
                should_export_epoch_bundle = (
                    int(train_cfg.save_every_n_epochs) > 0
                    and (
                        int(epoch) % int(train_cfg.save_every_n_epochs) == 0
                        or bool(is_last_epoch_this_run)
                    )
                )
                if should_export_epoch_bundle:
                    epoch_dir = _snapshot_epoch_bundle(
                        bundle_root=epoch_bundle_root,
                        checkpoint_dir=checkpoint_dir,
                        data_cfg=data_cfg,
                        epoch=int(epoch),
                        global_step=int(global_step),
                        train_loss=float(train_loss),
                        val_loss=float(val_loss),
                        best_val_loss=float(best_val_loss),
                        include_best_artifacts=bool(args.epoch_bundle_include_best),
                    )
                    print(f"Epoch bundle saved: {epoch_dir}")

                    try:
                        _maybe_run_epoch_upload_command(
                            template=str(args.epoch_upload_cmd_template),
                            epoch=int(epoch),
                            epoch_dir=epoch_dir,
                            output_dir=output_dir,
                            checkpoint_dir=checkpoint_dir,
                        )
                    except Exception as exc:
                        print(f"WARNING: epoch upload command failed at epoch {epoch:03d}: {exc}")

                    removed_bundles, remaining_count, remaining_bytes = _prune_epoch_bundles(
                        bundle_root=epoch_bundle_root,
                        max_keep=int(epoch_bundle_max_keep),
                        max_total_bytes=int(epoch_bundle_max_total_bytes),
                        protected_dir=epoch_dir,
                    )
                    if removed_bundles:
                        removed_bytes = int(sum(size for _, size in removed_bundles))
                        removed_gb = float(removed_bytes) / float(1024**3)
                        remaining_gb = float(remaining_bytes) / float(1024**3)
                        names = ", ".join(name for name, _ in removed_bundles)
                        print(
                            "Pruned epoch bundles: "
                            f"{len(removed_bundles)} removed ({removed_gb:.2f} GB) -> "
                            f"remaining={remaining_count} ({remaining_gb:.2f} GB). "
                            f"Removed: {names}"
                        )

            if distributed:
                dist.barrier()

        generation_meta: Dict[str, Any] = {
            "skipped": True,
            "reason": "No --seed_midi provided.",
        }
        output_midi = ""

        if _is_main_process(rank) and str(args.seed_midi).strip():
            core_model = model.module if isinstance(model, DDP) else model
            seed_midi_path = Path(str(args.seed_midi)).expanduser()
            if not seed_midi_path.exists():
                raise FileNotFoundError(f"seed_midi not found: {seed_midi_path.resolve()}")

            out_path = output_dir / "generated" / "variant_e_40m_ddp.mid"
            generation_meta = _generate_one_continuation(
                model=core_model,
                tokenizer=tokenizer,
                seed_midi=seed_midi_path,
                output_path=out_path,
                seed_length=int(args.seed_length),
                max_new_tokens=int(max(1, args.generation_max_new_tokens)),
                continuation_seconds=float(max(1.0, args.generation_continuation_seconds)),
                temperature=float(max(0.1, args.generation_temperature)),
                top_p=float(min(1.0, max(0.0, args.generation_top_p))),
                top_k=int(max(1, args.generation_top_k)),
                repetition_penalty=float(max(1.0, args.generation_repetition_penalty)),
                repetition_window=int(max(1, args.generation_repetition_window)),
                min_tokens_to_keep=int(max(1, args.generation_min_tokens_to_keep)),
                max_consecutive_zero_deltas=int(max(1, args.generation_max_consecutive_zero_deltas)),
            )
            output_midi = str(out_path.resolve())

        if _is_main_process(rank):
            result_payload = {
                "profile": "variant_e_40m_ddp",
                "target_params": int(VARIANT_E_40M_PROFILE["target_params"]),
                "model_profile": {
                    "d_model": int(d_model),
                    "n_layers": int(n_layers),
                    "attention_every_n_layers": int(attention_every_n_layers),
                    "full_attention": bool(args.full_attention),
                    "dropout": float(args.dropout),
                    "attention_dropout": float(args.attention_dropout),
                    "max_time_seconds": float(args.max_time_seconds),
                    "gdn_inner_dim": int(gdn_inner_dim),
                    "gdn_num_heads": int(gdn_heads),
                    "gqa_num_heads": int(gqa_heads),
                    "gqa_groups": int(gqa_groups),
                },
                "backend_status": backend_status,
                "distributed": {
                    "enabled": bool(distributed),
                    "world_size": int(world_size),
                    "device": str(device),
                },
                "training_profile": {
                    "epochs": int(args.epochs),
                    "batch_size_per_gpu": int(args.batch_size),
                    "global_batch_per_step": int(args.batch_size) * int(world_size),
                    "grad_accumulation_steps": int(args.grad_accumulation_steps),
                    "effective_batch": int(args.batch_size) * int(world_size) * int(args.grad_accumulation_steps),
                    "learning_rate": float(args.learning_rate),
                    "warmup_steps": int(train_cfg.warmup_steps),
                    "steps_per_epoch": int(steps_per_epoch),
                    "total_steps": int(total_steps),
                    "save_every_n_steps": int(max(0, train_cfg.save_every_n_steps)),
                    "slot_aware_loss": bool(slot_aware_loss),
                    "epoch_bundle_root": str(epoch_bundle_root.resolve()),
                    "epoch_bundle_include_best": bool(args.epoch_bundle_include_best),
                    "max_epoch_bundles": int(epoch_bundle_max_keep),
                    "max_epoch_bundle_total_gb": float(epoch_bundle_total_gb),
                    "epoch_upload_cmd_enabled": bool(str(args.epoch_upload_cmd_template).strip()),
                    "max_pieces": int(args.max_pieces),
                },
                "data": {
                    "manifest_path": str((processed_dir / "manifest.json").resolve()),
                    "train_pieces": int(len(train_manifest)),
                    "val_pieces": int(len(val_manifest)),
                    "source_manifest": str(manifest_path.resolve()),
                },
                "result": {
                    "variant": "variant_e",
                    "params": int(params),
                    "train_loss": [float(v) for v in history.get("train_loss", [])],
                    "val_loss": [float(v) for v in history.get("val_loss", [])],
                    "val_raw_ce": [float(v) for v in history.get("val_raw_ce", [])],
                    "val_token_acc": [float(v) for v in history.get("val_token_acc", [])],
                    "val_token_acc_delta": [float(v) for v in history.get("val_token_acc_delta", [])],
                    "val_token_acc_pitch": [float(v) for v in history.get("val_token_acc_pitch", [])],
                    "val_token_acc_duration": [float(v) for v in history.get("val_token_acc_duration", [])],
                    "val_token_acc_velocity": [float(v) for v in history.get("val_token_acc_velocity", [])],
                    "perplexity": [float(v) for v in history.get("perplexity", [])],
                    "checkpoint_dir": str(checkpoint_dir.resolve()),
                    "output_midi": output_midi,
                    "generation": generation_meta,
                    "resume": {
                        "enabled": bool(resume_checkpoint is not None),
                        "checkpoint": str(resume_checkpoint.resolve()) if resume_checkpoint is not None else "",
                        "mode": str(args.resume_mode),
                        "resumed_from_epoch": int(resume_epoch),
                        "resumed_from_batch_in_epoch": int(resume_batch_in_epoch),
                        "resumed_epoch_complete": bool(resume_epoch_complete),
                        "first_epoch_this_invocation": int(first_epoch),
                        "final_epoch_this_invocation": int(final_epoch),
                        "epochs_ran_this_invocation": int(epochs_to_run),
                    },
                },
            }

            result_path = output_dir / "variant_e_40m_ddp_result.json"
            result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
            print(f"DDP run complete. Result JSON: {result_path.resolve()}")

        if distributed:
            dist.barrier()

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

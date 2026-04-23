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
    _distributed_sum,
    _is_main_process,
    _load_checkpoint,
    _log,
    _rank_info,
    _resolve_loader_workers,
    _resolve_divisible_heads,
    _resolve_resume_checkpoint,
    _save_checkpoint,
)
from training.scheduler import WarmupCosineScheduler
from hf_sync import (
    HuggingFaceCheckpointMirror,
    normalize_hf_repo_id,
    resolve_hf_token,
    resolve_latest_hf_checkpoint,
)


VARIANT_E_100M_PROFILE: Dict[str, float] = {
    "d_model": 1024,
    "n_layers": 14,
    "attention_every_n_layers": 2,
    "gdn_inner_ratio": 0.5,
    "gdn_num_heads": 4,
    "gqa_num_heads": 8,
    "gqa_groups": 4,
    "target_params": 100_000_000,
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
        description="Distributed DDP trainer for Variant E ~100M profile on pretokenized NPZ data."
    )
    parser.add_argument("--pretokenized_manifest", type=str, required=True)
    parser.add_argument("--pretokenized_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/variant_e_100m_500k")

    parser.add_argument("--max_pieces", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_length", type=int, default=512)
    parser.add_argument("--continuation_length", type=int, default=1536)
    parser.add_argument("--max_sequence_length", type=int, default=2048)
    parser.add_argument(
        "--tokenization_strategy",
        type=str,
        default="custom_delta",
        choices=["custom_delta"],
        help="Unified tokenizer strategy (fixed to custom_delta).",
    )
    parser.add_argument(
        "--event_size",
        type=int,
        default=4,
        choices=[4],
        help="Tokenizer event size (fixed to 4 for unified custom_delta).",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=374,
        help="Tokenizer vocabulary size used to initialize embeddings (unified spec default).",
    )

    parser.add_argument("--d_model", type=int, default=int(VARIANT_E_100M_PROFILE["d_model"]))
    parser.add_argument("--n_layers", type=int, default=int(VARIANT_E_100M_PROFILE["n_layers"]))
    parser.add_argument(
        "--attention_every_n_layers",
        type=int,
        default=int(VARIANT_E_100M_PROFILE["attention_every_n_layers"]),
    )
    parser.add_argument("--full_attention", action="store_true")
    parser.add_argument(
        "--gdn_inner_ratio",
        type=float,
        default=float(VARIANT_E_100M_PROFILE["gdn_inner_ratio"]),
    )
    parser.add_argument("--gdn_num_heads", type=int, default=int(VARIANT_E_100M_PROFILE["gdn_num_heads"]))
    parser.add_argument("--gqa_num_heads", type=int, default=int(VARIANT_E_100M_PROFILE["gqa_num_heads"]))
    parser.add_argument("--gqa_groups", type=int, default=int(VARIANT_E_100M_PROFILE["gqa_groups"]))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--max_time_seconds", type=float, default=1200.0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.015,
        help="Warmup fraction used when --warmup_steps=0 (recommended 0.01-0.02 for long 500k-piece runs).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Absolute warmup steps override. If >0, this takes precedence over --warmup_ratio.",
    )
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="DataLoader workers per rank. Use -1 for auto tuning (default).",
    )
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
    parser.add_argument(
        "--hf_sync_repo_id",
        type=str,
        default="",
        help="Hugging Face model repo used for checkpoint mirrors (blank disables).",
    )
    parser.add_argument(
        "--hf_sync_private",
        action="store_true",
        help="Create the Hugging Face checkpoint mirror repo as private if it needs to be created.",
    )
    parser.add_argument(
        "--hf_sync_every_n_steps",
        type=int,
        default=0,
        help="Mirror checkpoints every N optimizer steps (0 uses --save_every_n_steps).",
    )
    parser.add_argument("--kaggle_sync_dataset", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--kaggle_sync_title", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--kaggle_sync_license", type=str, default="CC0-1.0", help=argparse.SUPPRESS)
    parser.add_argument("--kaggle_sync_public", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--kaggle_sync_every_n_steps",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )

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
        ddp_device = torch.device(f"cuda:{int(local_rank)}")
        try:
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=45),
                device_id=ddp_device,
            )
        except TypeError:
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=45),
            )
        device = ddp_device
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
        checkpoint_dir = output_dir / "checkpoints" / "variant_e_100m_ddp"
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

        legacy_sync_requested = bool(str(args.kaggle_sync_dataset).strip()) or bool(
            str(os.environ.get("KAGGLE_SYNC_DATASET", "")).strip()
        )
        hf_sync_repo_id = normalize_hf_repo_id(
            str(args.hf_sync_repo_id).strip()
            or str(os.environ.get("HF_SYNC_REPO_ID", "")).strip()
            or str(args.kaggle_sync_dataset).strip()
            or str(os.environ.get("KAGGLE_SYNC_DATASET", "")).strip()
        )
        if not hf_sync_repo_id and legacy_sync_requested:
            hf_sync_repo_id = "Chickaboo/Pulse88-E-85M-Alpha"
        hf_sync_private_env = str(os.environ.get("HF_SYNC_PRIVATE", "")).strip().lower()
        hf_sync_private = bool(args.hf_sync_private)
        if hf_sync_private_env:
            hf_sync_private = hf_sync_private_env in {"1", "true", "yes", "on"}
        elif legacy_sync_requested:
            hf_sync_private = not bool(args.kaggle_sync_public)
        hf_sync_steps = int(args.hf_sync_every_n_steps)
        if hf_sync_steps <= 0:
            hf_sync_steps = int(args.kaggle_sync_every_n_steps)
        if hf_sync_steps <= 0:
            hf_sync_steps = int(max(0, args.save_every_n_steps))

        hf_sync: HuggingFaceCheckpointMirror | None = None
        if _is_main_process(rank) and hf_sync_repo_id:
            try:
                hf_token = resolve_hf_token()
                hf_sync = HuggingFaceCheckpointMirror(
                    repo_id=hf_sync_repo_id,
                    checkpoint_dir=checkpoint_dir,
                    token=hf_token,
                    private=bool(hf_sync_private),
                    staging_root=output_dir / "hf_checkpoint_sync",
                )
                hf_sync.ensure_repo()
                _log(
                    rank,
                    "Hugging Face checkpoint mirror enabled: "
                    f"repo={hf_sync.repo_id} "
                    f"steps={int(hf_sync_steps) if int(hf_sync_steps) > 0 else 'disabled'}",
                )
            except Exception as exc:
                hf_sync = None
                _log(
                    rank,
                    "WARNING: Hugging Face checkpoint mirror disabled: "
                    f"{exc}",
                )
        elif _is_main_process(rank):
            _log(
                rank,
                "Hugging Face checkpoint mirror disabled. Set HF_SYNC_REPO_ID or "
                "--hf_sync_repo_id to enable real-time step uploads.",
            )

        last_hf_sync_step = -1

        tokenizer = CustomDeltaTokenizer(include_special_tokens=False)
        configured_vocab_size = int(max(1, int(args.vocab_size)))
        configured_event_size = int(max(1, int(args.event_size)))
        configured_strategy = str(args.tokenization_strategy).strip().lower()

        vocab_size = int(tokenizer.vocab_size)
        event_size = int(tokenizer.event_size)

        if configured_strategy != "custom_delta":
            raise RuntimeError(
                "Only tokenization_strategy=custom_delta is supported by this trainer."
            )
        if configured_event_size != event_size:
            raise RuntimeError(
                f"event_size mismatch: got {configured_event_size}, expected {event_size}."
            )
        if configured_vocab_size != vocab_size:
            raise RuntimeError(
                f"vocab_size mismatch: got {configured_vocab_size}, expected {vocab_size}."
            )

        if int(event_size) > 1 and (
            int(args.seed_length) % int(event_size) != 0
            or int(args.continuation_length) % int(event_size) != 0
        ):
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
                vocab_size=int(vocab_size),
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
        _log(rank, f"Variant-E 100M DDP model params: {params:,} ({params / 1e6:.2f}M)")
        _log(rank, f"Backend status: {backend_status}")

        data_cfg = DataConfig(
            tokenizer_path=str(output_dir / "custom_tokenizer.json"),
            processed_path=str(output_dir / "processed_pretokenized"),
            vocab_size=int(vocab_size),
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

        num_workers = _resolve_loader_workers(
            requested_workers=int(args.num_workers),
            world_size=int(world_size),
        )
        _log(
            rank,
            "DataLoader workers: "
            f"requested={int(args.num_workers)} resolved={int(num_workers)}",
        )
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

        adamw_kwargs: Dict[str, Any] = {
            "lr": float(train_cfg.learning_rate),
            "weight_decay": float(train_cfg.weight_decay),
        }
        if device.type == "cuda":
            adamw_kwargs["fused"] = True
        try:
            optimizer = AdamW(model.parameters(), **adamw_kwargs)
        except TypeError:
            adamw_kwargs.pop("fused", None)
            optimizer = AdamW(model.parameters(), **adamw_kwargs)

        batches_per_epoch = int(len(train_loader))
        steps_per_epoch = max(
            1,
            math.ceil(float(batches_per_epoch) / float(max(1, int(train_cfg.grad_accumulation_steps)))),
        )
        total_steps = max(1, int(steps_per_epoch) * int(train_cfg.max_epochs))
        warmup_ratio = float(min(0.20, max(0.0, float(args.warmup_ratio))))
        if int(args.warmup_steps) > 0:
            warmup_steps = int(max(1, int(args.warmup_steps)))
        else:
            warmup_steps = int(max(1, int(float(total_steps) * warmup_ratio)))
            if int(total_steps) >= 2000:
                warmup_steps = int(max(100, warmup_steps))
        warmup_steps = int(min(warmup_steps, max(1, int(total_steps) - 1)))
        train_cfg.warmup_steps = int(warmup_steps)

        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=int(train_cfg.warmup_steps),
            total_steps=int(total_steps),
            min_lr_ratio=float(train_cfg.min_lr_ratio),
        )

        _log(
            rank,
            "Scheduler setup: "
            f"steps_per_epoch={int(steps_per_epoch)} total_steps={int(total_steps)} "
            f"warmup_steps={int(train_cfg.warmup_steps)} warmup_ratio={float(warmup_ratio):.4f}",
        )

        use_amp = bool(train_cfg.use_amp) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        slot_aware_loss = bool(event_size == 4 and not bool(args.disable_slot_aware_loss))
        if int(event_size) != 4 and not bool(args.disable_slot_aware_loss):
            _log(
                rank,
                f"Slot-aware loss disabled automatically for event_size={event_size}.",
            )

        slot_metric_names = (
            ["delta", "pitch", "duration", "velocity"]
            if int(event_size) == 4
            else [f"slot_{i}" for i in range(int(event_size))]
        )

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_loss_total": [],
            "train_tokens": [],
            "val_loss": [],
            "val_raw_ce": [],
            "val_token_acc": [],
            "perplexity": [],
            "lr": [],
        }
        for slot_name in slot_metric_names:
            history[f"val_token_acc_{slot_name}"] = []

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
        if resume_checkpoint is None and hf_sync_repo_id:
            hf_resume_checkpoint = resolve_latest_hf_checkpoint(
                repo_id=hf_sync_repo_id,
                cache_root=output_dir,
                token=resolve_hf_token(),
                repo_type="model",
            )
            if hf_resume_checkpoint is not None:
                resume_checkpoint = hf_resume_checkpoint
                _log(rank, f"Resolved Hugging Face resume checkpoint: {resume_checkpoint}")
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
            epoch_loss_total = 0.0
            epoch_token_count = 0
            running_loss = 0.0
            running_count = 0
            running_loss_total = 0.0
            running_token_count = 0
            running_slot_rescued = 0
            running_slot_checked = 0
            warned_slot_mismatch = False
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

                    loss_result = next_token_loss(
                        logits,
                        targets,
                        label_smoothing=float(train_cfg.label_smoothing),
                        piece_boundary_mask=boundary_mask,
                        slot_aware=bool(slot_aware_loss),
                        event_size=int(event_size),
                        return_stats=True,
                    )
                    loss, valid_token_count, slot_rescued_count = loss_result

                loss_value = float(loss.item())
                epoch_loss_sum += float(loss_value)
                epoch_loss_count += 1
                running_loss += float(loss_value)
                running_count += 1

                valid_token_count = int(max(0, int(valid_token_count)))
                slot_rescued_count = int(max(0, int(slot_rescued_count)))
                batch_loss_total = float(loss_value) * float(valid_token_count)
                epoch_loss_total += float(batch_loss_total)
                epoch_token_count += int(valid_token_count)
                running_loss_total += float(batch_loss_total)
                running_token_count += int(valid_token_count)
                running_slot_rescued += int(slot_rescued_count)
                running_slot_checked += int(valid_token_count)

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
                        step_avg_loss = float(running_loss_total / max(1, running_token_count))
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
                            f"loss_avg~{step_avg_loss:.4f} loss_total~{float(running_loss_total):.1f}"
                        )
                        if hf_sync is not None and int(hf_sync_steps) > 0 and int(global_step) % int(hf_sync_steps) == 0:
                            hf_sync.schedule(
                                epoch=int(epoch),
                                global_step=int(global_step),
                                val_loss=float(step_avg_loss),
                                best_val_loss=float(best_val_loss),
                                save_tag=f"step-{int(global_step)}",
                                best=False,
                            )
                            last_hf_sync_step = int(global_step)

                    if _is_main_process(rank) and int(global_step) % int(max(1, args.log_every_n_steps)) == 0 and running_count > 0:
                        avg_running = float(running_loss_total / max(1, running_token_count))
                        rescue_ratio = float(running_slot_rescued / max(1, running_slot_checked))
                        if bool(slot_aware_loss) and not bool(warned_slot_mismatch) and int(running_slot_rescued) > 0:
                            _log(
                                rank,
                                "WARNING: slot-aware mask rescued "
                                f"{running_slot_rescued}/{running_slot_checked} targets "
                                f"({rescue_ratio * 100.0:.2f}%). This indicates token-slot misalignment "
                                "or non-quad/meta tokens in training targets.",
                            )
                            warned_slot_mismatch = True
                        lr = float(optimizer.param_groups[0]["lr"])
                        print(
                            f"epoch={epoch:03d} step={global_step:06d} "
                            f"train_loss={avg_running:.4f} train_loss_total={float(running_loss_total):.1f} "
                            f"slot_rescue={rescue_ratio:.4f} lr={lr:.6e}"
                        )
                        running_loss = 0.0
                        running_count = 0
                        running_loss_total = 0.0
                        running_token_count = 0
                        running_slot_rescued = 0
                        running_slot_checked = 0

            train_loss_total = _distributed_sum(float(epoch_loss_total), device=device)
            train_token_count = _distributed_sum(float(epoch_token_count), device=device)
            if float(train_token_count) <= 0.0:
                train_loss = 0.0
            else:
                train_loss = float(train_loss_total / train_token_count)

            model.eval()
            val_loss_total = 0.0
            val_token_count = 0
            val_raw_ce_total = 0.0
            val_raw_token_count = 0
            val_loss_count = 0
            val_token_acc_sum = 0.0
            val_slot_acc_sum: Dict[str, float] = {k: 0.0 for k in slot_metric_names}
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
                        loss_result = next_token_loss(
                            logits,
                            targets,
                            label_smoothing=float(train_cfg.label_smoothing),
                            piece_boundary_mask=boundary_mask,
                            slot_aware=bool(slot_aware_loss),
                            event_size=int(event_size),
                            return_stats=True,
                        )
                        raw_result = next_token_loss(
                            logits,
                            targets,
                            label_smoothing=0.0,
                            piece_boundary_mask=boundary_mask,
                            slot_aware=bool(slot_aware_loss),
                            event_size=int(event_size),
                            return_stats=True,
                        )
                        loss, loss_tokens, _ = loss_result
                        raw_ce, raw_tokens, _ = raw_result

                    loss_tokens = int(max(0, int(loss_tokens)))
                    raw_tokens = int(max(0, int(raw_tokens)))
                    val_loss_total += float(loss.item()) * float(loss_tokens)
                    val_token_count += int(loss_tokens)
                    val_raw_ce_total += float(raw_ce.item()) * float(raw_tokens)
                    val_raw_token_count += int(raw_tokens)
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
                total_sum=float(val_loss_total),
                total_count=float(val_token_count),
                device=device,
            )
            val_raw_ce = _average_from_sums(
                total_sum=float(val_raw_ce_total),
                total_count=float(val_raw_token_count),
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
                history["train_loss_total"].append(float(train_loss_total))
                history["train_tokens"].append(float(train_token_count))
                history["val_loss"].append(float(val_loss))
                history["val_raw_ce"].append(float(val_raw_ce))
                history["val_token_acc"].append(float(val_token_acc))
                for slot_name in slot_metric_names:
                    history[f"val_token_acc_{slot_name}"].append(
                        float(val_slot_acc.get(slot_name, 0.0))
                    )
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
                    f"| train_loss_total={float(train_loss_total):.1f} "
                    f"| train_tokens={int(train_token_count)} "
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

        if _is_main_process(rank) and str(args.seed_midi).strip() and int(event_size) != 4:
            generation_meta = {
                "skipped": True,
                "reason": f"Generation helper currently supports quad event_size=4 only (got {event_size}).",
            }

        if _is_main_process(rank) and str(args.seed_midi).strip() and int(event_size) == 4:
            core_model = model.module if isinstance(model, DDP) else model
            seed_midi_path = Path(str(args.seed_midi)).expanduser()
            if not seed_midi_path.exists():
                raise FileNotFoundError(f"seed_midi not found: {seed_midi_path.resolve()}")

            out_path = output_dir / "generated" / "variant_e_100m_ddp.mid"
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
            if hf_sync is not None:
                if int(global_step) != int(last_hf_sync_step):
                    latest_val_loss = (
                        float(history["val_loss"][-1])
                        if history.get("val_loss")
                        else float("inf")
                    )
                    hf_sync.schedule(
                        epoch=int(final_epoch),
                        global_step=int(global_step),
                        val_loss=float(latest_val_loss),
                        best_val_loss=float(best_val_loss),
                        save_tag=f"step-{int(global_step)}",
                        best=False,
                    )
                hf_sync.wait()

            result_payload = {
                "profile": "variant_e_100m_ddp",
                "target_params": int(VARIANT_E_100M_PROFILE["target_params"]),
                "model_profile": {
                    "d_model": int(d_model),
                    "n_layers": int(n_layers),
                    "attention_every_n_layers": int(attention_every_n_layers),
                    "full_attention": bool(args.full_attention),
                    "dropout": float(args.dropout),
                    "attention_dropout": float(args.attention_dropout),
                    "max_time_seconds": float(args.max_time_seconds),
                    "event_size": int(event_size),
                    "vocab_size": int(vocab_size),
                    "tokenization_strategy": str(data_cfg.tokenization_strategy),
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
                    "train_loss_total": [float(v) for v in history.get("train_loss_total", [])],
                    "train_tokens": [float(v) for v in history.get("train_tokens", [])],
                    "val_loss": [float(v) for v in history.get("val_loss", [])],
                    "val_raw_ce": [float(v) for v in history.get("val_raw_ce", [])],
                    "val_token_acc": [float(v) for v in history.get("val_token_acc", [])],
                    "val_token_acc_by_slot": {
                        slot_name: [float(v) for v in history.get(f"val_token_acc_{slot_name}", [])]
                        for slot_name in slot_metric_names
                    },
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

            result_path = output_dir / "variant_e_100m_ddp_result.json"
            result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
            print(f"DDP run complete. Result JSON: {result_path.resolve()}")

        if distributed:
            dist.barrier()

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

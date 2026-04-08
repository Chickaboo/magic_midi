from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file as safetensors_save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DataConfig, TrainConfig
from data.dataset import PianoDataset
from data.tokenizer_custom import CustomDeltaTokenizer
from model.variant_e import VariantEConfig, VariantEModel
from training.ablation_runner import (
    NpzWindowDataset,
    _generate_one_continuation,
    _load_pretokenized_manifest,
    _set_global_seed,
    _train_val_split,
    _variant_backend_status,
)
from training.losses import build_piece_boundary_mask, create_targets, next_token_loss
from training.scheduler import WarmupCosineScheduler


VARIANT_E_40M_PROFILE: Dict[str, float] = {
    "d_model": 768,
    "n_layers": 13,
    "attention_every_n_layers": 2,
    "gdn_inner_ratio": 0.5,
    "gdn_num_heads": 4,
    "gqa_num_heads": 8,
    "gqa_groups": 4,
    "target_params": 40_000_000,
}


def _rank_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _is_main_process(rank: int) -> bool:
    return int(rank) == 0


def _log(rank: int, message: str) -> None:
    if _is_main_process(rank):
        print(message)


def _resolve_divisible_heads(width: int, requested_heads: int) -> int:
    heads = max(1, min(int(requested_heads), int(width)))
    while heads > 1 and (int(width) % heads) != 0:
        heads -= 1
    return max(1, heads)


def _prepare_state_for_safetensors(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    prepared: Dict[str, torch.Tensor] = {}
    seen_ptrs: set[int] = set()
    for key, tensor in state_dict.items():
        ptr = int(tensor.untyped_storage().data_ptr())
        if ptr in seen_ptrs:
            prepared[key] = tensor.clone()
        else:
            prepared[key] = tensor
            seen_ptrs.add(ptr)
    return prepared


def _distributed_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _average_from_sums(
    *,
    total_sum: float,
    total_count: float,
    device: torch.device,
) -> float:
    reduced_sum = _distributed_sum(float(total_sum), device=device)
    reduced_count = _distributed_sum(float(total_count), device=device)
    if float(reduced_count) <= 0.0:
        return 0.0
    return float(reduced_sum / reduced_count)


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


def _save_checkpoint(
    *,
    model: VariantEModel | DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    checkpoint_dir: Path,
    epoch: int,
    val_loss: float,
    history: Dict[str, List[float]],
    best_val_loss: float,
    global_step: int,
    best: bool,
) -> None:
    core_model = model.module if isinstance(model, DDP) else model

    if best:
        model_path = checkpoint_dir / "best.safetensors"
        state_path = checkpoint_dir / "best_state.pt"
    else:
        model_path = checkpoint_dir / "latest.safetensors"
        state_path = checkpoint_dir / "latest_state.pt"

    model_state = {
        key: value.detach().cpu().contiguous()
        for key, value in core_model.state_dict().items()
    }
    safetensors_save_file(
        _prepare_state_for_safetensors(model_state),
        str(model_path),
        metadata={
            "epoch": str(int(epoch)),
            "val_loss": f"{float(val_loss):.8f}",
            "train_config": json.dumps(asdict(train_cfg)),
            "data_config": json.dumps(asdict(data_cfg)),
            "model_config": json.dumps(asdict(core_model.config)),
        },
    )

    state = {
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "train_config": asdict(train_cfg),
        "data_config": asdict(data_cfg),
        "model_config": asdict(core_model.config),
        "model_weights_path": str(model_path.name),
        "history": history,
        "best_val_loss": float(best_val_loss),
        "global_step": int(global_step),
    }
    torch.save(state, state_path)


def _load_checkpoint(
    *,
    model: VariantEModel | DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    core_model = model.module if isinstance(model, DDP) else model

    if checkpoint_path.suffix != ".pt":
        raise RuntimeError(
            "For DDP resumes, use a .pt state checkpoint (latest_state.pt or best_state.pt)."
        )

    state = torch.load(checkpoint_path, map_location=device)
    model_weights_path = str(state.get("model_weights_path", "")).strip()
    candidates = []
    if model_weights_path:
        candidates.append(checkpoint_path.parent / model_weights_path)
    candidates.extend(
        [
            checkpoint_path.parent / "latest.safetensors",
            checkpoint_path.parent / "best.safetensors",
        ]
    )

    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            f"No model safetensors found next to {checkpoint_path.resolve()}"
        )

    from safetensors.torch import load_file as safetensors_load_file

    core_model.load_state_dict(safetensors_load_file(str(model_path), device="cpu"))

    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if "scaler" in state and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])

    return dict(state)


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
    parser.add_argument("--save_every_n_epochs", type=int, default=5)

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
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        _set_global_seed(int(args.seed))
        random.seed(int(args.seed) + int(rank))
        np.random.seed(int(args.seed) + int(rank))
        torch.manual_seed(int(args.seed) + int(rank))

        if int(args.seed_length) <= 0 or int(args.continuation_length) <= 0:
            raise ValueError("seed_length and continuation_length must be > 0")
        if int(args.max_sequence_length) != (int(args.seed_length) + int(args.continuation_length)):
            raise ValueError("max_sequence_length must equal seed_length + continuation_length")

        output_dir = Path(str(args.output_dir)).expanduser()
        checkpoint_dir = output_dir / "checkpoints" / "variant_e_40m_ddp"
        if _is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
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

        d_model = int(VARIANT_E_40M_PROFILE["d_model"])
        gdn_inner_dim = max(128, int(round(float(d_model) * float(VARIANT_E_40M_PROFILE["gdn_inner_ratio"]))))
        gdn_heads = _resolve_divisible_heads(gdn_inner_dim, int(VARIANT_E_40M_PROFILE["gdn_num_heads"]))
        gqa_heads = _resolve_divisible_heads(d_model, int(VARIANT_E_40M_PROFILE["gqa_num_heads"]))

        gqa_groups = max(1, min(int(VARIANT_E_40M_PROFILE["gqa_groups"]), int(gqa_heads)))
        while gqa_groups > 1 and (gqa_heads % gqa_groups) != 0:
            gqa_groups -= 1

        model = VariantEModel(
            VariantEConfig(
                vocab_size=int(tokenizer.vocab_size),
                d_model=int(d_model),
                n_layers=int(VARIANT_E_40M_PROFILE["n_layers"]),
                max_sequence_length=int(args.max_sequence_length),
                gdn_inner_dim=int(gdn_inner_dim),
                gdn_num_heads=int(gdn_heads),
                gqa_num_heads=int(gqa_heads),
                gqa_groups=int(gqa_groups),
                attention_every_n_layers=int(VARIANT_E_40M_PROFILE["attention_every_n_layers"]),
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

        optimizer = AdamW(
            model.parameters(),
            lr=float(train_cfg.learning_rate),
            weight_decay=float(train_cfg.weight_decay),
        )

        steps_per_epoch = max(
            1,
            math.ceil(len(train_loader) / float(max(1, int(train_cfg.grad_accumulation_steps)))),
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

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "perplexity": [],
            "lr": [],
        }
        global_step = 0
        best_val_loss = float("inf")

        resume_checkpoint = _resolve_resume_checkpoint(
            checkpoint_dir=checkpoint_dir,
            auto_resume=bool(args.auto_resume),
            resume_from_checkpoint=str(args.resume_from_checkpoint),
        )
        start_epoch = 0
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
            start_epoch = int(resumed_state.get("epoch", 0))
            history_state = resumed_state.get("history")
            if isinstance(history_state, dict):
                for key in history:
                    if isinstance(history_state.get(key), list):
                        history[key] = [float(v) for v in history_state.get(key, [])]
            global_step = int(resumed_state.get("global_step", 0))
            best_val_loss = float(resumed_state.get("best_val_loss", best_val_loss))

            if str(args.resume_mode).strip().lower() == "remaining":
                epochs_to_run = max(0, int(train_cfg.max_epochs) - int(start_epoch))
            else:
                epochs_to_run = int(train_cfg.max_epochs)

            _log(
                rank,
                "Resuming DDP run "
                f"from {resume_checkpoint.resolve()} at epoch={start_epoch} "
                f"mode={args.resume_mode} epochs_to_run={epochs_to_run}",
            )

        for local_epoch in range(1, int(epochs_to_run) + 1):
            epoch = int(start_epoch + local_epoch)
            if distributed and isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(int(epoch))

            model.train()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            running_loss = 0.0
            running_count = 0

            for step_idx, batch in enumerate(train_loader, start=1):
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
                    or step_idx == len(train_loader)
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
                        )

                    val_loss_sum += float(loss.item())
                    val_loss_count += 1

            val_loss = _average_from_sums(
                total_sum=float(val_loss_sum),
                total_count=float(val_loss_count),
                device=device,
            )
            perplexity = float(math.exp(min(20.0, float(val_loss))))

            if _is_main_process(rank):
                history["train_loss"].append(float(train_loss))
                history["val_loss"].append(float(val_loss))
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
                    )

                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                    f"| val_loss={val_loss:.4f} | ppl={perplexity:.2f}"
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
                    "n_layers": int(VARIANT_E_40M_PROFILE["n_layers"]),
                    "attention_every_n_layers": int(VARIANT_E_40M_PROFILE["attention_every_n_layers"]),
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
                    "perplexity": [float(v) for v in history.get("perplexity", [])],
                    "checkpoint_dir": str(checkpoint_dir.resolve()),
                    "output_midi": output_midi,
                    "generation": generation_meta,
                    "resume": {
                        "enabled": bool(resume_checkpoint is not None),
                        "checkpoint": str(resume_checkpoint.resolve()) if resume_checkpoint is not None else "",
                        "mode": str(args.resume_mode),
                        "resumed_from_epoch": int(start_epoch),
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

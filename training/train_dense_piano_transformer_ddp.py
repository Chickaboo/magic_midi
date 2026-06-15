from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
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
from data.tokenizer_remi_bpe import PianoREMIBPETokenizer
from model.dense_piano_transformer import (
    DensePianoTransformer,
    DensePianoTransformerConfig,
    FLASH_ATTENTION_AVAILABLE,
)
from training.ablation_runner import (
    NpzWindowDataset,
    _load_pretokenized_manifest,
    _set_global_seed,
    _train_val_split,
)
from training.ddp_common import (
    _average_from_sums,
    _distributed_sum,
    _is_main_process,
    _load_checkpoint,
    _log,
    _rank_info,
    _resolve_loader_workers,
    _resolve_resume_checkpoint,
    _save_checkpoint,
)
from training.losses import (
    build_piece_boundary_mask,
    create_targets,
    next_token_accuracy,
    next_token_loss,
)
from training.scheduler import WarmupCosineScheduler


DENSE_PIANO_PROFILE: Dict[str, Any] = {
    "d_model": 576,
    "n_layers": 12,
    "num_attention_heads": 9,
    "head_dim": 64,
    "max_sequence_length": 8192,
    "window_schedule": [512, 512, 512, 512, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048],
    "global_anchor_count": 256,
    "global_anchor_start_layer": 9,
    "target_params": 85_000_000,
}


def _resolve_amp_dtype(requested: str, rank: int) -> Tuple[torch.dtype, str]:
    key = str(requested).strip().lower()
    if key in {"none", "off", "fp32", "float32"}:
        return torch.float32, "float32"
    if key in {"bf16", "bfloat16"}:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        _log(
            rank,
            "Requested bfloat16 but this CUDA device does not report bf16 support; falling back to float16.",
        )
        return torch.float16, "float16"
    if key in {"fp16", "float16"}:
        return torch.float16, "float16"
    raise ValueError("--amp_dtype must be one of bfloat16, float16, or float32")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DDP trainer for DensePianoTransformer on PianoREMIBPE NPZ data."
    )
    parser.add_argument("--pretokenized_manifest", type=str, required=True)
    parser.add_argument("--pretokenized_root", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/dense_piano_remi_bpe")
    parser.add_argument("--max_pieces", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--seed_length", type=int, default=4096)
    parser.add_argument("--continuation_length", type=int, default=4096)
    parser.add_argument("--max_sequence_length", type=int, default=8192)
    parser.add_argument("--vocab_size", type=int, default=30000)

    parser.add_argument("--d_model", type=int, default=int(DENSE_PIANO_PROFILE["d_model"]))
    parser.add_argument("--n_layers", type=int, default=int(DENSE_PIANO_PROFILE["n_layers"]))
    parser.add_argument("--num_attention_heads", type=int, default=int(DENSE_PIANO_PROFILE["num_attention_heads"]))
    parser.add_argument("--head_dim", type=int, default=int(DENSE_PIANO_PROFILE["head_dim"]))
    parser.add_argument("--global_anchor_count", type=int, default=int(DENSE_PIANO_PROFILE["global_anchor_count"]))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.015)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--log_every_n_steps", type=int, default=20)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--amp_dtype", type=str, default="bfloat16")

    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--resume_mode", choices=["remaining", "additional"], default="remaining")
    parser.add_argument("--no_auto_resume", dest="auto_resume", action="store_false")
    parser.set_defaults(auto_resume=True)
    return parser


def _copy_or_create_tokenizer(
    *,
    tokenizer_path_arg: str,
    pretokenized_root: Path | None,
    output_path: Path,
    vocab_size: int,
) -> PianoREMIBPETokenizer:
    candidates: List[Path] = []
    if str(tokenizer_path_arg).strip():
        candidates.append(Path(str(tokenizer_path_arg)).expanduser())
    if pretokenized_root is not None:
        candidates.extend(
            [
                pretokenized_root / "metadata" / "piano_remi_bpe_tokenizer.json",
                pretokenized_root / "piano_remi_bpe_tokenizer.json",
            ]
        )
    source = next((p for p in candidates if p.exists() and p.is_file()), None)
    if source is not None:
        tokenizer = PianoREMIBPETokenizer.load(source)
    else:
        tokenizer = PianoREMIBPETokenizer(vocab_size=int(vocab_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def main() -> None:
    args = _build_arg_parser().parse_args()
    rank, world_size, local_rank = _rank_info()
    distributed = int(world_size) > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA, but CUDA is not available.")
        torch.cuda.set_device(int(local_rank))
        device = torch.device(f"cuda:{int(local_rank)}")
        try:
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=45),
                device_id=device,
            )
        except TypeError:
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=45))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if int(args.max_sequence_length) != int(args.seed_length) + int(args.continuation_length):
            raise ValueError("max_sequence_length must equal seed_length + continuation_length")
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

        output_dir = Path(str(args.output_dir)).expanduser()
        checkpoint_dir = output_dir / "checkpoints" / "dense_piano_transformer"
        processed_dir = output_dir / "processed_pretokenized"
        if _is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            processed_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()

        manifest_path = Path(str(args.pretokenized_manifest)).expanduser()
        pretokenized_root = (
            Path(str(args.pretokenized_root)).expanduser()
            if str(args.pretokenized_root).strip()
            else None
        )
        tokenizer_out_path = output_dir / "piano_remi_bpe_tokenizer.json"
        if _is_main_process(rank):
            _copy_or_create_tokenizer(
                tokenizer_path_arg=str(args.tokenizer_path),
                pretokenized_root=pretokenized_root,
                output_path=tokenizer_out_path,
                vocab_size=int(args.vocab_size),
            )
        if distributed:
            dist.barrier()
        tokenizer = PianoREMIBPETokenizer.load(tokenizer_out_path)
        vocab_size = int(tokenizer.vocab_size)
        if int(args.vocab_size) > int(vocab_size):
            vocab_size = int(args.vocab_size)

        data_cfg = DataConfig(
            tokenizer_path=str(output_dir / "piano_remi_bpe_tokenizer.json"),
            processed_path=str(processed_dir),
            vocab_size=int(vocab_size),
            tokenization_strategy="piano_remi_bpe",
            seed_length=int(args.seed_length),
            continuation_length=int(args.continuation_length),
            max_sequence_length=int(args.max_sequence_length),
            use_continuous_time=True,
            time_feature_fallback_step_seconds=0.125,
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
        if _is_main_process(rank):
            (processed_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
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

        model_cfg = DensePianoTransformerConfig(
            vocab_size=int(vocab_size),
            d_model=int(args.d_model),
            n_layers=int(args.n_layers),
            max_sequence_length=int(args.max_sequence_length),
            dropout=float(args.dropout),
            attention_dropout=float(args.attention_dropout),
            num_attention_heads=int(args.num_attention_heads),
            head_dim=int(args.head_dim),
            window_schedule=tuple(int(v) for v in DENSE_PIANO_PROFILE["window_schedule"]),
            max_relative_distance=2048,
            global_anchor_count=int(args.global_anchor_count),
            global_anchor_start_layer=9,
            gradient_checkpointing=not bool(args.disable_gradient_checkpointing),
            use_v2_architecture=True,
        )
        model: torch.nn.Module = DensePianoTransformer(model_cfg).to(device)
        if distributed:
            model = DDP(
                model,
                device_ids=[int(local_rank)],
                output_device=int(local_rank),
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

        core_model = model.module if isinstance(model, DDP) else model
        params = int(sum(p.numel() for p in core_model.parameters()))
        _log(rank, f"DensePianoTransformer params: {params:,} ({params / 1e6:.2f}M)")
        _log(rank, f"FlashAttention import available: {FLASH_ATTENTION_AVAILABLE}")
        _log(rank, f"Gradient checkpointing: {bool(model_cfg.gradient_checkpointing)}")

        steps_per_epoch = max(
            1,
            math.ceil(len(train_loader) / float(max(1, int(args.grad_accumulation_steps)))),
        )
        total_steps = max(1, int(steps_per_epoch) * int(args.epochs))
        warmup_steps = (
            int(max(1, args.warmup_steps))
            if int(args.warmup_steps) > 0
            else int(max(1, round(float(args.warmup_ratio) * float(total_steps))))
        )
        train_cfg = TrainConfig(
            batch_size=int(args.batch_size),
            grad_accumulation_steps=int(args.grad_accumulation_steps),
            learning_rate=float(args.learning_rate),
            lr_schedule="cosine",
            min_lr_ratio=float(args.min_lr_ratio),
            weight_decay=float(args.weight_decay),
            label_smoothing=float(args.label_smoothing),
            max_epochs=int(args.epochs),
            warmup_steps=int(warmup_steps),
            max_grad_norm=float(args.max_grad_norm),
            save_every_n_steps=int(max(0, args.save_every_n_steps)),
            save_every_n_epochs=int(max(1, args.save_every_n_epochs)),
            checkpoint_dir=str(checkpoint_dir),
            use_wandb=False,
            seed=int(args.seed),
            device=str(device),
            val_generation_check=False,
        )

        optimizer = AdamW(
            model.parameters(),
            lr=float(train_cfg.learning_rate),
            weight_decay=float(train_cfg.weight_decay),
        )
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=int(train_cfg.warmup_steps),
            total_steps=int(total_steps),
            min_lr_ratio=float(train_cfg.min_lr_ratio),
        )
        amp_dtype, amp_label = _resolve_amp_dtype(str(args.amp_dtype), rank=rank)
        use_amp = bool(device.type == "cuda" and amp_dtype != torch.float32)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and amp_dtype == torch.float16))
        _log(rank, f"AMP dtype: {amp_label} (GradScaler enabled={scaler.is_enabled()})")

        resume_checkpoint = _resolve_resume_checkpoint(
            checkpoint_dir=checkpoint_dir,
            auto_resume=bool(args.auto_resume),
            resume_from_checkpoint=str(args.resume_from_checkpoint),
        )
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_loss_total": [],
            "train_tokens": [],
            "val_loss": [],
            "val_raw_ce": [],
            "val_token_acc": [],
            "perplexity": [],
        }
        best_val_loss = float("inf")
        global_step = 0
        start_epoch = 0
        if resume_checkpoint is not None:
            state = _load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                checkpoint_path=resume_checkpoint,
                device=device,
            )
            global_step = int(state.get("global_step", 0))
            best_val_loss = float(state.get("best_val_loss", best_val_loss))
            loaded_history = state.get("history")
            if isinstance(loaded_history, dict):
                history.update(loaded_history)
            start_epoch = int(state.get("epoch", 0))
            _log(rank, f"Resumed from {resume_checkpoint.resolve()} at epoch={start_epoch}")

        epochs_to_run = int(args.epochs)
        first_epoch = start_epoch + 1
        if resume_checkpoint is not None and str(args.resume_mode) == "remaining":
            epochs_to_run = max(0, int(args.epochs) - int(start_epoch))
        final_epoch = int(first_epoch + epochs_to_run - 1)

        for epoch in range(first_epoch, final_epoch + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(int(epoch))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss_total = 0.0
            epoch_token_count = 0
            running_loss_total = 0.0
            running_token_count = 0
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

                autocast_ctx = (
                    torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with autocast_ctx:
                    logits = model(
                        token_ids=input_ids,
                        onset_times=onset_times,
                        durations=durations,
                        memory=None,
                        return_memory=False,
                    )
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    loss, loss_tokens, _ = next_token_loss(
                        logits,
                        targets,
                        label_smoothing=float(train_cfg.label_smoothing),
                        piece_boundary_mask=boundary_mask,
                        slot_aware=False,
                        event_size=1,
                        return_stats=True,
                    )

                valid_tokens = int(max(0, int(loss_tokens)))
                loss_value = float(loss.item())
                epoch_loss_total += loss_value * float(valid_tokens)
                epoch_token_count += int(valid_tokens)
                running_loss_total += loss_value * float(valid_tokens)
                running_token_count += int(valid_tokens)
                running_count += 1

                scaled_loss = loss / int(max(1, train_cfg.grad_accumulation_steps))
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = (
                    step_idx % int(train_cfg.grad_accumulation_steps) == 0
                    or step_idx == len(train_loader)
                )
                if should_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.max_grad_norm))
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    if (
                        _is_main_process(rank)
                        and int(global_step) % int(max(1, args.log_every_n_steps)) == 0
                        and running_count > 0
                    ):
                        avg = float(running_loss_total / max(1, running_token_count))
                        lr = float(optimizer.param_groups[0]["lr"])
                        print(
                            f"ts={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                            f"epoch={epoch:03d} step={global_step:06d} "
                            f"train_loss_avg={avg:.4f} train_loss_sum={running_loss_total:.1f} lr={lr:.6e}"
                        )
                        running_loss_total = 0.0
                        running_token_count = 0
                        running_count = 0

                    if (
                        _is_main_process(rank)
                        and int(train_cfg.save_every_n_steps) > 0
                        and int(global_step) % int(train_cfg.save_every_n_steps) == 0
                    ):
                        approx_loss = float(epoch_loss_total / max(1, epoch_token_count))
                        _save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            train_cfg=train_cfg,
                            data_cfg=data_cfg,
                            checkpoint_dir=checkpoint_dir,
                            epoch=int(epoch),
                            val_loss=float(approx_loss),
                            history=history,
                            best_val_loss=float(best_val_loss),
                            global_step=int(global_step),
                            best=False,
                            resume_state={
                                "epoch": int(epoch),
                                "batch_step_in_epoch": int(step_idx),
                                "batches_per_epoch": int(len(train_loader)),
                                "is_epoch_complete": False,
                            },
                        )

            train_loss_total = _distributed_sum(float(epoch_loss_total), device=device)
            train_token_count = _distributed_sum(float(epoch_token_count), device=device)
            train_loss = float(train_loss_total / max(1.0, train_token_count))

            model.eval()
            val_loss_total = 0.0
            val_token_count = 0
            val_raw_ce_total = 0.0
            val_raw_token_count = 0
            val_loss_count = 0
            val_token_acc_sum = 0.0
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
                    autocast_ctx = (
                        torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp)
                        if device.type == "cuda"
                        else nullcontext()
                    )
                    with autocast_ctx:
                        logits = model(
                            token_ids=input_ids,
                            onset_times=onset_times,
                            durations=durations,
                            memory=None,
                            return_memory=False,
                        )
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        loss, loss_tokens, _ = next_token_loss(
                            logits,
                            targets,
                            label_smoothing=float(train_cfg.label_smoothing),
                            piece_boundary_mask=boundary_mask,
                            slot_aware=False,
                            event_size=1,
                            return_stats=True,
                        )
                        raw_ce, raw_tokens, _ = next_token_loss(
                            logits,
                            targets,
                            label_smoothing=0.0,
                            piece_boundary_mask=boundary_mask,
                            slot_aware=False,
                            event_size=1,
                            return_stats=True,
                        )
                    loss_tokens = int(max(0, int(loss_tokens)))
                    raw_tokens = int(max(0, int(raw_tokens)))
                    val_loss_total += float(loss.item()) * float(loss_tokens)
                    val_token_count += int(loss_tokens)
                    val_raw_ce_total += float(raw_ce.item()) * float(raw_tokens)
                    val_raw_token_count += int(raw_tokens)
                    val_loss_count += 1
                    val_token_acc_sum += float(
                        next_token_accuracy(
                            logits,
                            targets,
                            piece_boundary_mask=boundary_mask,
                            slot_aware=False,
                            event_size=1,
                        )
                    )

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
            perplexity = float(math.exp(min(20.0, float(val_loss))))

            if _is_main_process(rank):
                history["train_loss"].append(float(train_loss))
                history["train_loss_total"].append(float(train_loss_total))
                history["train_tokens"].append(float(train_token_count))
                history["val_loss"].append(float(val_loss))
                history["val_raw_ce"].append(float(val_raw_ce))
                history["val_token_acc"].append(float(val_token_acc))
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
                        "batch_step_in_epoch": int(len(train_loader)),
                        "batches_per_epoch": int(len(train_loader)),
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
                    )

                print(
                    f"Epoch {epoch:03d} | train_loss_avg={train_loss:.4f} "
                    f"| train_tokens={int(train_token_count)} "
                    f"| val_loss={val_loss:.4f} | raw_ce={val_raw_ce:.4f} "
                    f"| acc={val_token_acc:.4f} | ppl={perplexity:.2f}"
                )

            if distributed:
                dist.barrier()

        if _is_main_process(rank):
            result_payload = {
                "profile": "dense_piano_transformer_remi_bpe",
                "target_params": int(DENSE_PIANO_PROFILE["target_params"]),
                "model_profile": {
                    "d_model": int(args.d_model),
                    "n_layers": int(args.n_layers),
                    "num_attention_heads": int(args.num_attention_heads),
                    "head_dim": int(args.head_dim),
                    "max_sequence_length": int(args.max_sequence_length),
                    "window_schedule": list(DENSE_PIANO_PROFILE["window_schedule"]),
                    "global_anchor_count": int(args.global_anchor_count),
                    "global_anchor_start_layer": 9,
                    "gradient_checkpointing": bool(model_cfg.gradient_checkpointing),
                    "flash_attention_available": bool(FLASH_ATTENTION_AVAILABLE),
                },
                "distributed": {
                    "enabled": bool(distributed),
                    "world_size": int(world_size),
                    "device": str(device),
                    "amp_dtype": str(amp_label),
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
                "tokenizer": {
                    "name": "PianoREMIBPE",
                    "vocab_size": int(vocab_size),
                    "event_size": 1,
                    "path": str(Path(data_cfg.tokenizer_path).resolve()),
                },
                "data": {
                    "manifest_path": str((processed_dir / "manifest.json").resolve()),
                    "train_pieces": int(len(train_manifest)),
                    "val_pieces": int(len(val_manifest)),
                    "source_manifest": str(manifest_path.resolve()),
                },
                "result": {
                    "variant": "dense_piano_transformer",
                    "params": int(params),
                    "history": history,
                    "checkpoint_dir": str(checkpoint_dir.resolve()),
                },
            }
            result_path = output_dir / "dense_piano_transformer_ddp_result.json"
            result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
            print(f"DDP run complete. Result JSON: {result_path.resolve()}")

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

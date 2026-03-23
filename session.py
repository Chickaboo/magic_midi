from __future__ import annotations

import datetime as dt
import json
import math
import random
import shutil
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from config import DataConfig
from data.dataset import create_dataloaders
from data.preprocess import MultiDatasetPreprocessor, preprocess_maestro
from data.tokenizer import PianoTokenizer
from drive_sync import DriveSync
from model.factory import build_model
from scale_config import get_preset
from training.trainer import Trainer
from utils.logging_utils import get_project_logger, log_model_summary
from utils.session_utils import (
    SessionWatchdog,
    estimate_time_per_epoch,
    get_gpu_info,
    print_session_banner,
)


LOGGER = get_project_logger()


def estimate_sessions_remaining(
    current_epoch: int,
    target_epochs: int,
    time_per_epoch_seconds: float,
) -> Dict[str, float]:
    """Estimate remaining sessions and wall-clock hours to target epoch."""

    remaining_epochs = max(0, int(target_epochs) - int(current_epoch))
    if remaining_epochs <= 0:
        LOGGER.info("Target epoch reached. No additional sessions estimated.")
        return {
            "remaining_epochs": 0,
            "sessions_remaining": 0.0,
            "hours_remaining": 0.0,
        }

    epochs_per_session = max(1, int((25 * 60) / max(time_per_epoch_seconds, 1e-6)))
    sessions_remaining = math.ceil(remaining_epochs / epochs_per_session)
    hours_remaining = (remaining_epochs * time_per_epoch_seconds) / 3600.0

    LOGGER.info(
        "At current pace, "
        f"~{sessions_remaining} more sessions (~{hours_remaining:.1f} hours) "
        f"to reach epoch {target_epochs}"
    )
    return {
        "remaining_epochs": float(remaining_epochs),
        "sessions_remaining": float(sessions_remaining),
        "hours_remaining": float(hours_remaining),
    }


def _get_max_epochs_target(default: int = 200) -> int:
    try:
        import __main__

        value = getattr(__main__, "MAX_EPOCHS", default)
        if isinstance(value, int) and value > 0:
            return value
    except Exception:
        pass
    return default


def _select_subset_manifest_if_needed(
    manifest_path: Path,
    max_pieces: Optional[int],
    output_path: Path,
    seed: int = 42,
) -> Path:
    if max_pieces is None:
        return manifest_path

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, list):
        raise RuntimeError(f"Invalid manifest format at {manifest_path}")

    subset = get_stratified_subset(
        manifest_entries=manifest,
        max_pieces=int(max_pieces),
        seed=int(seed),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)

    LOGGER.info(
        "Using stratified subset: %d pieces (max_pieces=%s).",
        len(subset),
        str(max_pieces),
    )
    _print_composer_distribution(subset)
    return output_path


def _print_composer_distribution(
    entries: List[Dict[str, Any]], top_k: int = 10
) -> None:
    if not entries:
        LOGGER.info("Composer distribution: empty subset")
        return

    counts: Dict[str, int] = {}
    for entry in entries:
        composer = str(entry.get("composer", "unknown") or "unknown")
        counts[composer] = counts.get(composer, 0) + 1

    total = len(entries)
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    LOGGER.info("Composer distribution (top 10):")
    for composer, n in top:
        pct = 100.0 * n / max(1, total)
        LOGGER.info("  %-24s %4d pieces (%4.1f%%)", composer[:24], n, pct)


def get_stratified_subset(
    manifest_entries: List[Dict[str, Any]],
    max_pieces: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Sample max_pieces entries with stratification by composer.
    Ensures each composer is represented proportionally.
    Falls back to random shuffle if composer metadata missing.
    """
    if max_pieces <= 0:
        return []

    if max_pieces >= len(manifest_entries):
        shuffled = list(manifest_entries)
        random.Random(seed).shuffle(shuffled)
        return shuffled

    rng = random.Random(seed)
    by_composer: Dict[str, List[Dict[str, Any]]] = {}
    for entry in manifest_entries:
        composer = str(entry.get("composer", "unknown") or "unknown")
        by_composer.setdefault(composer, []).append(entry)

    if len(by_composer) <= 1:
        shuffled = list(manifest_entries)
        rng.shuffle(shuffled)
        return shuffled[:max_pieces]

    total = len(manifest_entries)
    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    for composer, pieces in by_composer.items():
        local = list(pieces)
        rng.shuffle(local)
        n = max(1, round(len(local) / max(1, total) * max_pieces))
        n = min(n, len(local))
        selected.extend(local[:n])
        leftovers.extend(local[n:])

    rng.shuffle(selected)
    if len(selected) > max_pieces:
        selected = selected[:max_pieces]
    elif len(selected) < max_pieces:
        rng.shuffle(leftovers)
        need = max_pieces - len(selected)
        selected.extend(leftovers[:need])

    return selected[:max_pieces]


def _build_dataloaders_from_manifest(
    effective_data_cfg: DataConfig,
    train_cfg: Any,
    manifest_path: Path,
) -> Tuple[Any, Any, Any]:
    # create_dataloaders expects processed_path/manifest.json.
    temp_cfg = DataConfig(**effective_data_cfg.__dict__)
    if manifest_path.name == "manifest.json":
        temp_cfg.processed_path = str(manifest_path.parent)
        return create_dataloaders(temp_cfg, train_cfg)

    subset_root = Path(effective_data_cfg.processed_path) / "_subset_runtime"
    subset_root.mkdir(parents=True, exist_ok=True)
    subset_manifest = subset_root / "manifest.json"
    shutil.copy2(manifest_path, subset_manifest)
    temp_cfg.processed_path = str(subset_root)
    return create_dataloaders(temp_cfg, train_cfg)


def _maybe_adjust_workers_for_environment(train_cfg: Any) -> None:
    """Avoid multiprocessing worker issues in non-Colab CPU sessions."""

    if train_cfg.device == "cpu":
        setattr(train_cfg, "_force_num_workers", 0)
        return

    if not torch.cuda.is_available() and train_cfg.device == "auto":
        setattr(train_cfg, "_force_num_workers", 0)
        return


def _make_single_worker_dataloaders(
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
) -> Tuple[Any, Any, Any]:
    from torch.utils.data import DataLoader

    train_single = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=getattr(train_loader, "collate_fn", None),
        drop_last=False,
    )
    val_single = DataLoader(
        val_loader.dataset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=getattr(val_loader, "collate_fn", None),
        drop_last=False,
    )
    test_single = DataLoader(
        test_loader.dataset,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=getattr(test_loader, "collate_fn", None),
        drop_last=False,
    )
    return train_single, val_single, test_single


def _load_resume_epoch(drive_sync: DriveSync) -> int:
    state_path = drive_sync.last_restored_state_path
    if state_path is None:
        return 0
    try:
        state = torch.load(state_path, map_location="cpu")
        epoch = int(state.get("epoch", 0))
        return max(0, epoch)
    except Exception as exc:
        warnings.warn(f"Could not read resume epoch from {state_path}: {exc}")
        return 0


def _count_manifest_entries(manifest_path: Path) -> int:
    if not manifest_path.exists():
        return 0
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
    except Exception:
        return 0
    return 0


def _get_gpu_name() -> str:
    info = get_gpu_info()
    return str(info.get("gpu_name", "CPU"))


def _safe_float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_generate_tokens(
    model: torch.nn.Module, seed_tokens: list[int], max_new_tokens: int
) -> list[int]:
    gen_fn = getattr(model, "generate", None)
    if callable(gen_fn):
        result = gen_fn(
            seed_tokens=seed_tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            repetition_window=64,
            min_tokens_to_keep=3,
        )
        if isinstance(result, list):
            return [int(x) for x in result]
        raise RuntimeError("model.generate returned unsupported output type.")

    raise RuntimeError("Model does not expose a callable generate(...) method.")


def _optimizer_steps_per_epoch(
    train_loader: Any,
    grad_accumulation_steps: int,
) -> int:
    batches = len(train_loader)
    accum = max(1, int(grad_accumulation_steps))
    return max(1, math.ceil(batches / accum))


def _queue_epoch_checkpoint_sync(
    drive_sync: DriveSync,
    local_ckpt: Path,
    epoch: int,
    queued_sync_tags: Set[Tuple[int, str]],
    keep_every_n_epochs: int,
    best_ckpt: Optional[Path] = None,
) -> None:
    if not local_ckpt.exists():
        return

    tags = ["latest"]
    keep_every = max(1, int(keep_every_n_epochs))
    if epoch % keep_every == 0:
        tags.append(f"epoch_{epoch:03d}")

    for tag in tags:
        key = (int(epoch), tag)
        if key in queued_sync_tags:
            continue
        drive_sync.sync_checkpoint_background(str(local_ckpt), tag=tag)
        queued_sync_tags.add(key)

    if best_ckpt is not None and best_ckpt.exists():
        drive_sync.sync_checkpoint_background(str(best_ckpt), tag="best")


def calibrate_preset(scale_name: str) -> int:
    """Instantiate model and print measured runtime parameter count.

    This count depends on active backend (real Mamba vs fallback).
    """
    preset = get_preset(scale_name)
    model = build_model(preset["model"])
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("%s: %s params (%.2fM)", scale_name, f"{total:,}", total / 1e6)
    model.get_num_params()
    del model
    return int(total)


def _run_preprocess(data_cfg: DataConfig) -> None:
    """Run preprocessing according to current dataset configuration."""

    if bool(data_cfg.use_multi_dataset):
        processor = MultiDatasetPreprocessor(config=data_cfg)
        processor.preprocess()
        return
    preprocess_maestro(data_cfg)


def run_session(
    scale: str = "small",
    max_epochs_this_session: Optional[int] = None,
    calibration_mode: bool = False,
) -> Dict[str, Any]:
    """
    Complete training session handler.
    Call this once per Colab session. It will:
    1. Mount Drive
    2. Restore tokenizer and processed data
    3. Resume from latest checkpoint
    4. Train until disconnect or completion
    5. Sync everything to Drive
    """

    session_id = (
        dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    )
    session_start_dt = dt.datetime.now()

    if calibration_mode:
        total = calibrate_preset(scale)
        return {"scale": scale, "params": total, "calibration_mode": True}

    preset = get_preset(scale)
    model_cfg = preset["model"]
    train_cfg = preset["train"]
    data_cfg = preset["data"]

    drive_sync = DriveSync()
    drive_sync.mount()
    drive_sync.set_checkpoint_retention(
        train_cfg.keep_every_n_epochs if train_cfg.keep_every_n_epochs > 0 else 0
    )

    # Keep checkpoints inside fast local cache, then sync to Drive explicitly.
    train_cfg.checkpoint_dir = str(drive_sync.local_checkpoints_dir)

    # Path configuration for local runtime.
    data_cfg.tokenizer_path = str(drive_sync.local_tokenizer_path)
    data_cfg.processed_path = str(drive_sync.local_processed_dir)

    max_epochs_target = _get_max_epochs_target(default=200)

    # Print banner before heavy work.
    print_session_banner(scale_preset=scale, epoch=0, drive_sync=drive_sync)
    drive_sync.write_heartbeat(
        {
            "event": "session_start",
            "session_id": session_id,
            "scale": scale,
            "timestamp": session_start_dt.isoformat(),
        }
    )

    # ---------- Data restoration ----------
    tokenizer_restore_status = drive_sync.sync_tokenizer()
    processed_status = drive_sync.sync_processed_data()

    if processed_status == "missing":
        LOGGER.info("No processed data cache found. Running preprocessing once...")
        _run_preprocess(data_cfg)
        drive_sync.sync_processed_data()

    if tokenizer_restore_status is None:
        # Preprocess should have produced tokenizer locally; sync it up.
        maybe_local = Path(data_cfg.tokenizer_path)
        if maybe_local.exists():
            drive_sync.sync_tokenizer()
        else:
            LOGGER.info(
                "Tokenizer missing locally. Running preprocessing to build tokenizer..."
            )
            _run_preprocess(data_cfg)
            drive_sync.sync_processed_data()
            drive_sync.sync_tokenizer()

    manifest_path = Path(data_cfg.processed_path) / "manifest.json"
    manifest_count = _count_manifest_entries(manifest_path)
    if manifest_count == 0:
        LOGGER.info(
            "Processed manifest missing or empty. Rebuilding preprocessing cache..."
        )
        _run_preprocess(data_cfg)
        drive_sync.sync_processed_data()
        drive_sync.sync_tokenizer()
        manifest_count = _count_manifest_entries(manifest_path)

    if manifest_count == 0:
        raise RuntimeError(
            "Manifest is still empty after preprocessing. Check MAESTRO path and preprocessing settings."
        )

    effective_manifest = manifest_path
    if data_cfg.max_pieces is not None:
        subset_manifest = Path(data_cfg.processed_path) / "manifest_subset.json"
        effective_manifest = _select_subset_manifest_if_needed(
            manifest_path=manifest_path,
            max_pieces=data_cfg.max_pieces,
            output_path=subset_manifest,
            seed=train_cfg.seed,
        )

    _maybe_adjust_workers_for_environment(train_cfg)
    train_loader, val_loader, test_loader = _build_dataloaders_from_manifest(
        effective_data_cfg=data_cfg,
        train_cfg=train_cfg,
        manifest_path=effective_manifest,
    )

    # In non-Colab CPU environments, avoid worker spawn instability.
    if not drive_sync.in_colab and not torch.cuda.is_available():
        train_loader, val_loader, test_loader = _make_single_worker_dataloaders(
            train_loader,
            val_loader,
            test_loader,
        )

    train_pieces = len(getattr(train_loader, "dataset", []))
    train_batches = len(train_loader)
    optimizer_steps = _optimizer_steps_per_epoch(
        train_loader=train_loader,
        grad_accumulation_steps=train_cfg.grad_accumulation_steps,
    )
    effective_batch = int(train_cfg.batch_size) * int(train_cfg.grad_accumulation_steps)
    LOGGER.info(
        "Optimizer schedule: "
        f"train_pieces={train_pieces}, batches/epoch={train_batches}, "
        f"grad_accum={train_cfg.grad_accumulation_steps}, "
        f"optimizer_steps/epoch={optimizer_steps}, "
        f"effective_batch={effective_batch}"
    )
    if optimizer_steps < 15:
        warnings.warn(
            "Low optimizer steps/epoch (<15). Consider lower batch size or lower gradient accumulation."
        )

    tokenizer = PianoTokenizer.load(data_cfg.tokenizer_path)
    model_cfg.vocab_size = tokenizer.vocab_size
    data_cfg.vocab_size = tokenizer.vocab_size

    # ---------- Model initialization ----------
    model = build_model(model_cfg)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        data_config=data_cfg,
        tokenizer=tokenizer,
    )

    resume_ckpt = drive_sync.restore_checkpoint("latest")
    start_epoch = 0
    if resume_ckpt is not None:
        try:
            state_path = drive_sync.last_restored_state_path
            if state_path is not None and Path(state_path).exists():
                state = trainer.load_checkpoint(state_path)
            else:
                state = trainer.load_checkpoint(resume_ckpt)
            start_epoch = int(state.get("epoch", _load_resume_epoch(drive_sync)))
            LOGGER.info("Resumed from checkpoint at epoch %d.", start_epoch)
        except Exception as exc:
            warnings.warn(
                f"Checkpoint restore failed; starting from scratch. Details: {exc}"
            )
            start_epoch = 0
    else:
        LOGGER.info("Starting from scratch")

    log_model_summary(model, model_cfg)
    print_session_banner(scale_preset=scale, epoch=start_epoch, drive_sync=drive_sync)

    # ---------- Watchdog ----------
    watchdog = SessionWatchdog(drive_sync=drive_sync, trainer=trainer)
    watchdog.start()

    # ---------- Determine epoch budget ----------
    epochs_remaining = max(0, max_epochs_target - start_epoch)
    if epochs_remaining <= 0:
        LOGGER.info(
            "Target epoch %d already reached. Nothing to train.", max_epochs_target
        )
        watchdog.stop()
        drive_sync.wait_for_sync()
        return {
            "session_id": session_id,
            "epochs_completed": 0,
            "start_epoch": start_epoch,
            "end_epoch": start_epoch,
        }

    epoch_time_seconds = None
    already_trained = 0
    queued_sync_tags: Set[Tuple[int, str]] = set()
    if max_epochs_this_session is None:
        LOGGER.info("Estimating epoch duration with one warmup epoch...")
        epoch_time_seconds = max(1e-6, estimate_time_per_epoch(trainer))
        already_trained = 1

        local_ckpt = Path(train_cfg.checkpoint_dir) / "latest.safetensors"
        best_ckpt = Path(train_cfg.checkpoint_dir) / "best.safetensors"
        _queue_epoch_checkpoint_sync(
            drive_sync=drive_sync,
            local_ckpt=local_ckpt,
            epoch=start_epoch + 1,
            queued_sync_tags=queued_sync_tags,
            keep_every_n_epochs=train_cfg.keep_every_n_epochs,
            best_ckpt=best_ckpt,
        )

        available_seconds = 25 * 60
        estimated_total_epochs = max(1, int(available_seconds / epoch_time_seconds))
        estimated_total_epochs = min(50, estimated_total_epochs)

        remaining_after_warmup = max(0, epochs_remaining - 1)
        additional_epochs = min(
            max(0, estimated_total_epochs - 1), remaining_after_warmup
        )
        epochs_this_session = 1 + additional_epochs

        LOGGER.info(
            "Auto epoch budget: "
            f"time/epoch={epoch_time_seconds:.1f}s, "
            f"available={available_seconds}s, "
            f"planned_epochs={epochs_this_session}"
        )
    else:
        epochs_this_session = int(max_epochs_this_session)
        if epochs_this_session <= 0:
            epochs_this_session = 1

    epochs_this_session = min(epochs_this_session, epochs_remaining)

    # ---------- Training loop ----------
    epochs_done = 0
    try:
        epochs_done = already_trained

        for k in range(max(0, epochs_this_session - already_trained)):
            epoch_base = start_epoch + already_trained + k
            trainer.train_n_epochs(n=1, start_epoch=epoch_base)
            completed_epoch = epoch_base + 1
            epochs_done += 1

            local_ckpt = Path(train_cfg.checkpoint_dir) / "latest.safetensors"
            best_ckpt = Path(train_cfg.checkpoint_dir) / "best.safetensors"
            _queue_epoch_checkpoint_sync(
                drive_sync=drive_sync,
                local_ckpt=local_ckpt,
                epoch=completed_epoch,
                queued_sync_tags=queued_sync_tags,
                keep_every_n_epochs=train_cfg.keep_every_n_epochs,
                best_ckpt=best_ckpt,
            )

            LOGGER.info("Session sync queued for epoch %03d", completed_epoch)

    except Exception:
        warnings.warn("Training loop interrupted; attempting final sync.")
        local_ckpt = Path(train_cfg.checkpoint_dir) / "latest.safetensors"
        if local_ckpt.exists():
            drive_sync.sync_checkpoint_background(str(local_ckpt), tag="latest")
        raise
    finally:
        # ---------- Session end ----------
        watchdog.stop()
        drive_sync.wait_for_sync()

    end_epoch = start_epoch + epochs_done
    final_train_loss = (
        _safe_float_or_none(trainer.history["train_loss"][-1])
        if trainer.history.get("train_loss")
        else None
    )
    final_val_loss = (
        _safe_float_or_none(trainer.history["val_loss"][-1])
        if trainer.history.get("val_loss")
        else None
    )

    # Save one generated sample into Drive/generated.
    try:
        sample_batch = next(iter(test_loader))
        if isinstance(sample_batch, dict):
            seed_batch = sample_batch["seed"]
        else:
            seed_batch = sample_batch[0]
        seed_tokens = seed_batch[0].tolist()

        seed_mid = drive_sync.generated_dir / f"{session_id}_seed.mid"
        out_mid = drive_sync.generated_dir / f"{session_id}_generated.mid"
        tokenizer.decode(seed_tokens, seed_mid)

        generated_tokens = _safe_generate_tokens(
            model=trainer.model,
            seed_tokens=seed_tokens,
            max_new_tokens=min(512, data_cfg.continuation_length),
        )
        tokenizer.decode(generated_tokens, out_mid)
    except Exception as exc:
        warnings.warn(f"Sample generation at session end failed: {exc}")

    # Session log record.
    token_counter = 0
    try:
        token_counter = int(
            epochs_done
            * len(train_loader)
            * train_cfg.batch_size
            * (data_cfg.seed_length + data_cfg.continuation_length)
        )
    except Exception:
        token_counter = 0

    session_end_dt = dt.datetime.now()
    log_record = {
        "session_id": session_id,
        "start_time": session_start_dt.isoformat(),
        "end_time": session_end_dt.isoformat(),
        "epochs_completed": int(epochs_done),
        "start_epoch": int(start_epoch),
        "end_epoch": int(end_epoch),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "scale_preset": scale,
        "total_tokens_processed": int(token_counter),
        "colab_gpu_type": _get_gpu_name(),
    }
    drive_sync.sync_log(log_record)

    if epoch_time_seconds is None:
        # Fallback estimate from total session time and epochs.
        elapsed = max(1e-6, (session_end_dt - session_start_dt).total_seconds())
        epoch_time_seconds = elapsed / max(1, epochs_done)

    LOGGER.info(
        f"Session complete: +{epochs_done} epoch(s), "
        f"epoch {start_epoch} -> {end_epoch}, "
        f"val_loss={final_val_loss if final_val_loss is not None else 'n/a'}"
    )
    estimate_sessions_remaining(
        current_epoch=end_epoch,
        target_epochs=max_epochs_target,
        time_per_epoch_seconds=float(epoch_time_seconds),
    )

    return {
        "session_id": session_id,
        "epochs_completed": epochs_done,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "scale": scale,
    }

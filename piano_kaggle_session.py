"""Kaggle training session runner for v3 presets and multi-dataset support."""

from __future__ import annotations

import datetime as dt
import argparse
import json
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from data.dataset import create_dataloaders
from data.preprocess import MultiDatasetPreprocessor, preprocess_maestro
from data.tokenizer import PianoTokenizer
from generation.generate import GenerationConfig, generate_continuation
from kaggle_config import (
    find_adl_piano_root,
    find_aria_midi_root,
    find_giant_midi_root,
    find_maestro_root,
    setup_kaggle_environment,
)
from model.factory import build_model
from scale_config import SCALE_PRESETS, get_preset, verify_preset_params
from training.trainer import (
    CHECKPOINT_KEEP_POLICY,
    Trainer,
    rotate_kaggle_checkpoint_dir,
)
from utils import checkpoint_loading as ckpt_utils
from utils.logging_utils import get_project_logger
from utils.session_utils import SessionWatchdog, get_gpu_info


LOGGER = get_project_logger()


def _as_bool_env(name: str, default: bool) -> bool:
    """Read boolean-like environment variable with a default value."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _probe_mamba_kernel() -> Tuple[bool, str]:
    """Run a tiny mamba-ssm forward pass to confirm kernel availability."""

    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"

    try:
        import model.mamba_block as mamba_block
    except Exception as exc:
        return False, f"Could not import model.mamba_block ({exc})"

    if (not bool(getattr(mamba_block, "MAMBA_AVAILABLE", False))) or getattr(
        mamba_block, "_Mamba", None
    ) is None:
        return False, "mamba_ssm package is not available in this runtime"

    try:
        device = torch.device("cuda")
        with torch.no_grad():
            probe = mamba_block._Mamba(d_model=16, d_state=8, d_conv=4, expand=2).to(
                device
            )
            x = torch.randn(1, 8, 16, device=device)
            _ = probe(x)
            torch.cuda.synchronize()
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _configure_mamba_backend() -> None:
    """Configure mamba backend strictness based on environment flags."""

    require_mamba = _as_bool_env("PIANO_REQUIRE_MAMBA", default=True)
    allow_fallback = _as_bool_env("PIANO_ALLOW_GRU_FALLBACK", default=False)

    ok, detail = _probe_mamba_kernel()
    if ok:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            LOGGER.info(
                "Mamba kernel probe passed (GPU=%s, compute capability=%d.%d).",
                name,
                cap[0],
                cap[1],
            )
        else:
            LOGGER.info("Mamba probe passed.")
        return

    reason = detail
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            reason = f"{detail} | GPU={name} (compute capability={cap[0]}.{cap[1]})"
        except Exception:
            reason = detail

    if require_mamba and not allow_fallback:
        raise RuntimeError(
            "Mamba is required but kernel probe failed. "
            f"Details: {reason}. "
            "On Kaggle, switch Accelerator to T4 and rerun setup. "
            "If you explicitly want fallback, set PIANO_ALLOW_GRU_FALLBACK=1."
        )

    try:
        import model.mamba_block as mamba_block

        mamba_block.MAMBA_AVAILABLE = False
        mamba_block._Mamba = None
        warnings.warn(f"Mamba probe failed; using GRU fallback. Details: {reason}")
    except Exception as exc:
        warnings.warn(
            f"Mamba probe failed and fallback patch could not be applied ({exc})."
        )


def _configure_kaggle_performance(train_cfg: Any, scale: Optional[str] = None) -> None:
    """Tune device, worker, and batch defaults for Kaggle runtime."""

    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        LOGGER.info("Detected CUDA GPUs (%d): %s", gpu_count, gpu_names)

    train_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    setattr(train_cfg, "_enable_data_parallel", gpu_count > 1)

    workers_env = os.environ.get("PIANO_NUM_WORKERS")
    if workers_env is not None:
        try:
            num_workers = max(0, int(workers_env))
        except Exception:
            num_workers = 4
    else:
        cpu_count = os.cpu_count() or 4
        num_workers = max(2, min(8, cpu_count))
    setattr(train_cfg, "_force_num_workers", num_workers)

    if gpu_count > 1 and os.environ.get("PIANO_DISABLE_AUTO_BATCH", "0") != "1":
        base_batch = int(train_cfg.batch_size)
        base_accum = int(train_cfg.grad_accumulation_steps)

        default_target = {
            "small": 32,
            "medium": 16,
            "large": 8,
        }.get(str(scale).lower() if scale is not None else "", 64)

        default_max = {
            "small": 96,
            "medium": 64,
            "large": 32,
        }.get(str(scale).lower() if scale is not None else "", 128)

        scaled = max(base_batch * gpu_count * 4, int(default_target))
        max_batch = int(os.environ.get("PIANO_MAX_BATCH_SIZE", str(default_max)))
        train_cfg.batch_size = max(base_batch, min(max_batch, scaled))

        accum_env = os.environ.get("PIANO_GRAD_ACCUM_STEPS")
        if accum_env is not None:
            try:
                train_cfg.grad_accumulation_steps = max(1, int(accum_env))
            except Exception:
                train_cfg.grad_accumulation_steps = 1
        else:
            train_cfg.grad_accumulation_steps = 1 if base_accum > 1 else base_accum

    LOGGER.info(
        "Performance config: device=%s, gpus=%d, batch_size=%d, grad_accum=%d, "
        "num_workers=%s, data_parallel=%s",
        train_cfg.device,
        gpu_count,
        int(train_cfg.batch_size),
        int(train_cfg.grad_accumulation_steps),
        getattr(train_cfg, "_force_num_workers", "n/a"),
        getattr(train_cfg, "_enable_data_parallel", False),
    )


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write JSON payload to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _manifest_count(path: Path) -> int:
    """Return number of manifest entries if JSON exists."""

    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
    except Exception:
        return 0
    return 0


def _build_kaggle_dataset_paths(paths: Dict[str, str]) -> Dict[str, str]:
    """Build dataset-path mapping for optional multi-dataset preprocessing."""

    dataset_paths: Dict[str, str] = {"maestro": paths["maestro_root"]}
    giant_path = paths.get("giant_midi_root", "")
    if giant_path:
        dataset_paths["giant_midi"] = giant_path
    aria_path = paths.get("aria_midi_root", "")
    if aria_path:
        dataset_paths["aria_midi"] = aria_path
    adl_path = paths.get("adl_piano_root", "")
    if adl_path:
        dataset_paths["adl_piano"] = adl_path
    piano_e_path = paths.get("piano_e_root", "")
    if piano_e_path and "adl_piano" not in dataset_paths:
        dataset_paths["adl_piano"] = piano_e_path
    return dataset_paths


def _run_preprocessing(data_cfg: Any, paths: Dict[str, str]) -> None:
    """Run single- or multi-dataset preprocessing on Kaggle."""

    dataset_paths = _build_kaggle_dataset_paths(paths)
    if bool(getattr(data_cfg, "use_multi_dataset", False)) and len(dataset_paths) > 1:
        data_cfg.dataset_paths = dataset_paths
        processor = MultiDatasetPreprocessor(config=data_cfg)
        processor.preprocess()
        return

    preprocess_maestro(data_cfg)


def _find_resume_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find latest checkpoint file for automatic resume."""

    candidates = [
        checkpoint_dir / "latest.safetensors",
        checkpoint_dir / "latest_model.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_state_sidecar(model_path: Path) -> Optional[Path]:
    """Resolve sidecar state path for a checkpoint model file."""

    if model_path.name.endswith("_model.safetensors"):
        candidate = model_path.with_name(
            model_path.name.replace("_model.safetensors", "_state.pt")
        )
        if candidate.exists():
            return candidate

    latest_state = model_path.parent / "latest_state.pt"
    if latest_state.exists():
        return latest_state
    return None


def _build_tokenizer_search_paths(
    paths: Dict[str, str],
    checkpoint_path: Path,
) -> tuple[Path, ...]:
    """Build ordered tokenizer search paths for inference-time checkpoint loading."""

    candidates: list[Path] = []

    configured_tokenizer = str(paths.get("tokenizer_path", "")).strip()
    if configured_tokenizer:
        configured_path = Path(configured_tokenizer)
        candidates.append(configured_path.with_name("custom_tokenizer.json"))
        candidates.append(configured_path.with_name("tokenizer.json"))
        candidates.append(configured_path.expanduser())
        candidates.append(configured_path)

    candidates.append(checkpoint_path.parent / "custom_tokenizer.json")
    candidates.append(checkpoint_path.parent / "tokenizer.json")

    working_dir = str(paths.get("working_dir", "")).strip()
    if working_dir:
        tokenizer_dir = Path(working_dir) / "tokenizer"
        candidates.append(tokenizer_dir / "custom_tokenizer.json")
        candidates.append(tokenizer_dir / "tokenizer.json")

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    return tuple(deduped)


def _load_inference_assets(
    checkpoint_path: Path,
    paths: Dict[str, str],
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """Load model and tokenizer for generation with strict checkpoint validation."""

    sidecar_path = ckpt_utils.resolve_sidecar_path(checkpoint_path)
    checkpoint_metadata = ckpt_utils.load_checkpoint_metadata(
        model_path=checkpoint_path,
        sidecar_path=sidecar_path,
    )

    tokenizer_search_paths = _build_tokenizer_search_paths(paths, checkpoint_path)
    tokenizer, tokenizer_meta = ckpt_utils.load_tokenizer_for_checkpoint(
        checkpoint_metadata=checkpoint_metadata,
        search_paths=tokenizer_search_paths,
    )

    bundle = ckpt_utils.load_model_from_checkpoint(
        model_path=checkpoint_path,
        sidecar_path=sidecar_path,
        device=device,
        strict=True,
    )

    expected_vocab = int(bundle.model_config.get("vocab_size", tokenizer.vocab_size))
    if int(tokenizer.vocab_size) != expected_vocab:
        raise RuntimeError(
            "Tokenizer/model vocab mismatch while loading checkpoint "
            f"{checkpoint_path.name}: tokenizer vocab={tokenizer.vocab_size}, "
            f"model vocab={expected_vocab}."
        )

    LOGGER.info(
        "Loaded checkpoint %s with %s (missing=%d, unexpected=%d)",
        checkpoint_path,
        bundle.model_class,
        bundle.missing_keys,
        bundle.unexpected_keys,
    )
    return bundle.model, tokenizer, tokenizer_meta


def _append_training_log(log_path: Path, record: Dict[str, Any]) -> None:
    """Append one session record to Kaggle training log file."""

    history = {"sessions": []}
    if log_path.exists():
        try:
            loaded = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                history = loaded
        except Exception:
            history = {"sessions": []}

    sessions = history.get("sessions")
    if not isinstance(sessions, list):
        sessions = []
    sessions.append(record)
    history["sessions"] = sessions
    _atomic_write_json(log_path, history)


def _find_first_seed_midi(root: Path) -> Optional[Path]:
    """Return deterministic first MIDI file under a root."""

    midi_files = sorted(list(root.rglob("*.mid")) + list(root.rglob("*.midi")))
    if not midi_files:
        return None
    return midi_files[0]


def _safe_generate_tokens(
    model: torch.nn.Module,
    seed_tokens: list[int],
    max_new_tokens: int,
) -> list[int]:
    """Safely run generation and normalize output token type."""

    gen_fn = getattr(model, "generate", None)
    if not callable(gen_fn):
        raise RuntimeError("Model does not expose generate(...)")
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
    if not isinstance(result, list):
        raise RuntimeError("Model generate(...) returned unsupported output type")
    return [int(x) for x in result]


class _KaggleSyncShim:
    """Minimal sync shim used by session watchdog in Kaggle context."""

    def __init__(self, checkpoint_dir: Path, heartbeat_path: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.heartbeat_path = heartbeat_path

    def write_heartbeat(self, payload: Dict[str, Any]) -> None:
        """Persist heartbeat payload for watchdog monitoring."""

        _atomic_write_json(self.heartbeat_path, payload)

    def sync_checkpoint(self, local_path: str, tag: str = "latest") -> bool:
        """Copy checkpoint/state artifacts into Kaggle working directory."""

        src = Path(local_path)
        if not src.exists():
            return False

        dst = self.checkpoint_dir / f"{tag}.safetensors"
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)

        state_src = _find_state_sidecar(src)
        if state_src is not None and state_src.exists():
            state_dst = self.checkpoint_dir / f"{tag}_state.pt"
            state_tmp = state_dst.with_name(state_dst.name + ".tmp")
            shutil.copy2(state_src, state_tmp)
            os.replace(state_tmp, state_dst)

        rotate_kaggle_checkpoint_dir(
            self.checkpoint_dir,
            keep_every_n_epochs=int(CHECKPOINT_KEEP_POLICY["milestone_every_n"]),
            max_total_checkpoints=int(CHECKPOINT_KEEP_POLICY["max_total_checkpoints"]),
        )
        return True

    def wait_for_sync(self) -> None:
        """Compatibility no-op for synchronous Kaggle writes."""

        return


class KaggleSessionWatchdog(SessionWatchdog):
    """Kaggle watchdog with longer timeout expectations than Colab free tier."""

    def __init__(
        self,
        drive_sync: Any,
        trainer: Any,
        warning_minutes: int = 600,
        checkpoint_stall_minutes: int = 60,
    ) -> None:
        super().__init__(
            drive_sync=drive_sync,
            trainer=trainer,
            warning_minutes=warning_minutes,
        )
        self.checkpoint_stall_minutes = checkpoint_stall_minutes

    def _run(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()

            if now - self._last_heartbeat >= 300:
                self._write_heartbeat(now)
                self._last_heartbeat = now

            current_epoch = int(getattr(self.trainer, "current_epoch", 0))
            if current_epoch > self._last_checkpoint_epoch:
                self._last_checkpoint_epoch = current_epoch
                self._last_checkpoint_time = now

            minutes_elapsed = (now - self._start_time) / 60.0

            if (now - self._last_checkpoint_time) >= float(
                self.checkpoint_stall_minutes * 60
            ):
                self._emergency_save(
                    reason=f"no checkpoint for {self.checkpoint_stall_minutes} minutes"
                )
                self._last_checkpoint_time = now

            if (not self._warning_fired) and minutes_elapsed >= float(
                self.warning_minutes
            ):
                LOGGER.info(
                    "[Watchdog] Session near timeout (~%d min). "
                    "Triggering emergency checkpoint.",
                    self.warning_minutes,
                )
                self._emergency_save(reason=f"{self.warning_minutes} minute warning")
                self._warning_fired = True

            time.sleep(15)


def _print_kaggle_banner(
    scale: str,
    start_epoch: int,
    max_epochs: int,
    gpu_info: Dict[str, Any],
    paths: Dict[str, str],
    resumed: bool,
    tokenization: str,
) -> None:
    """Print user-facing Kaggle session banner."""

    print("=" * 56)
    print("  Itty Bitty Piano - Kaggle Session")
    print("=" * 56)
    print(f"  Scale:             {scale}")
    print(f"  Tokenization:      {tokenization}")
    print(f"  Resume status:     {'resumed' if resumed else 'fresh start'}")
    print(f"  Epoch progress:    {start_epoch} / {max_epochs}")
    print(
        f"  GPU:               {gpu_info['gpu_name']} "
        f"({gpu_info['total_vram_gb']:.1f} GB)"
    )
    print(f"  MAESTRO root:      {paths['maestro_root']}")
    giant = paths.get("giant_midi_root", "")
    aria = paths.get("aria_midi_root", "")
    adl = paths.get("adl_piano_root", "") or paths.get("piano_e_root", "")
    print(f"  GiantMIDI root:    {giant or 'not found (optional)'}")
    print(f"  Aria-MIDI root:    {aria or 'not found (optional)'}")
    print(f"  ADL Piano root:    {adl or 'not found (optional)'}")
    print(f"  Checkpoint dir:    {paths['checkpoint_dir']}")
    print(f"  Processed dir:     {paths['processed_dir']}")
    print(f"  Tokenizer path:    {paths['tokenizer_path']}")
    print(
        f"  CUDA devices:      {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
    )
    print("=" * 56)


def run_kaggle_session(scale: str = "small", max_epochs: int = 2000) -> Dict[str, Any]:
    """Run full Kaggle training session with resume and artifact outputs."""

    paths = setup_kaggle_environment()
    _configure_mamba_backend()
    gpu = get_gpu_info()

    preset = get_preset(scale)
    model_cfg = preset["model"]
    train_cfg = preset["train"]
    data_cfg = preset["data"]

    train_cfg.max_epochs = int(max_epochs)
    train_cfg.checkpoint_dir = paths["checkpoint_dir"]
    _configure_kaggle_performance(train_cfg, scale=scale)

    data_cfg.maestro_path = paths["maestro_root"]
    data_cfg.processed_path = paths["processed_dir"]
    data_cfg.tokenizer_path = paths["tokenizer_path"]
    data_cfg.dataset_paths = _build_kaggle_dataset_paths(paths)
    data_cfg.use_multi_dataset = len(data_cfg.dataset_paths) > 1

    checkpoint_dir = Path(paths["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(data_cfg.processed_path) / "manifest.json"
    tokenizer_path = Path(data_cfg.tokenizer_path)

    needs_preprocess = (not tokenizer_path.exists()) or (
        _manifest_count(manifest_path) == 0
    )
    if needs_preprocess:
        LOGGER.info(
            "Tokenizer/processed cache missing in /kaggle/working. Running preprocessing..."
        )
        _run_preprocessing(data_cfg, paths)
    else:
        LOGGER.info("Reusing tokenizer and processed cache from /kaggle/working.")

    tokenizer = PianoTokenizer.load(data_cfg.tokenizer_path)
    model_cfg.vocab_size = tokenizer.vocab_size
    data_cfg.vocab_size = tokenizer.vocab_size

    train_loader, val_loader, test_loader = create_dataloaders(data_cfg, train_cfg)

    model = build_model(model_cfg)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        data_config=data_cfg,
        tokenizer=tokenizer,
    )

    resume_ckpt = _find_resume_checkpoint(checkpoint_dir)
    start_epoch = 0
    resumed = False
    if resume_ckpt is not None:
        try:
            state = trainer.load_checkpoint(str(resume_ckpt))
            start_epoch = int(state.get("epoch", 0))
            resumed = True
            LOGGER.info(
                "Resumed from checkpoint: %s (epoch %d)", resume_ckpt, start_epoch
            )
        except Exception as exc:
            warnings.warn(f"Checkpoint load failed; starting from scratch ({exc})")
            start_epoch = 0

    _print_kaggle_banner(
        scale=scale,
        start_epoch=start_epoch,
        max_epochs=int(max_epochs),
        gpu_info=gpu,
        paths=paths,
        resumed=resumed,
        tokenization=str(getattr(data_cfg, "tokenization_strategy", "n/a")),
    )

    if start_epoch >= int(max_epochs):
        LOGGER.info(
            "Target max_epochs=%d already reached. Nothing to train.", max_epochs
        )
        return {
            "scale": scale,
            "start_epoch": start_epoch,
            "end_epoch": start_epoch,
            "epochs_completed": 0,
            "best_val_loss": float(trainer.best_val_loss),
        }

    heartbeat_path = Path(paths["working_dir"]) / "heartbeat_latest.json"
    sync_shim = _KaggleSyncShim(
        checkpoint_dir=checkpoint_dir,
        heartbeat_path=heartbeat_path,
    )
    watchdog = KaggleSessionWatchdog(
        drive_sync=sync_shim,
        trainer=trainer,
        warning_minutes=600,
        checkpoint_stall_minutes=60,
    )

    session_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_start = time.time()
    epochs_completed = 0

    watchdog.start()
    try:
        for epoch_base in range(start_epoch, int(max_epochs)):
            trainer.train_n_epochs(n=1, start_epoch=epoch_base)
            epochs_completed += 1
    finally:
        watchdog.stop()

    end_epoch = start_epoch + epochs_completed
    best_val_loss = float(trainer.best_val_loss)

    try:
        sample_batch = next(iter(test_loader))
        if isinstance(sample_batch, dict):
            seed_batch = sample_batch["seed"]
        else:
            seed_batch = sample_batch[0]
        seed_tokens = seed_batch[0].tolist()
        generated_tokens = _safe_generate_tokens(
            model=trainer.model,
            seed_tokens=seed_tokens,
            max_new_tokens=min(512, data_cfg.continuation_length),
        )

        generated_dir = Path(paths["generated_dir"])
        generated_dir.mkdir(parents=True, exist_ok=True)
        seed_out = generated_dir / f"{session_id}_seed.mid"
        gen_out = generated_dir / f"{session_id}_generated.mid"
        tokenizer.decode(seed_tokens, seed_out)
        tokenizer.decode(generated_tokens, gen_out)
        LOGGER.info("Saved session sample: %s", gen_out)
    except Exception as exc:
        warnings.warn(f"Session-end sample generation failed: {exc}")

    elapsed = time.time() - session_start
    log_record = {
        "session_id": session_id,
        "scale": scale,
        "start_epoch": int(start_epoch),
        "end_epoch": int(end_epoch),
        "epochs_completed": int(epochs_completed),
        "best_val_loss": best_val_loss,
        "elapsed_seconds": float(elapsed),
        "gpu": gpu.get("gpu_name", "CPU"),
        "timestamp": dt.datetime.now().isoformat(),
    }
    _append_training_log(Path(paths["log_path"]), log_record)

    LOGGER.info(
        "Kaggle session complete: +%d epoch(s), epoch %d -> %d, "
        "best_val_loss=%.4f, time=%.1f min",
        epochs_completed,
        start_epoch,
        end_epoch,
        best_val_loss,
        elapsed / 60.0,
    )
    LOGGER.info("Checkpoints saved to /kaggle/working/.")

    return {
        "session_id": session_id,
        "scale": scale,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "epochs_completed": epochs_completed,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
        "checkpoint_dir": paths["checkpoint_dir"],
    }


def calibrate_on_kaggle() -> None:
    """Print measured parameter counts for all presets on this runtime."""

    _configure_mamba_backend()

    class _TmpTrainCfg:
        """Minimal train config shim used for calibration runtime setup."""

        device = "auto"
        batch_size = 4
        grad_accumulation_steps = 1

    _configure_kaggle_performance(_TmpTrainCfg)
    LOGGER.info("Parameter calibration on this runtime:")
    LOGGER.info(
        "GPU: %s",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    )
    LOGGER.info("%s", "-" * 50)
    verify_preset_params()
    LOGGER.info("%s", "-" * 50)
    for name, preset in SCALE_PRESETS.items():
        LOGGER.info("%8s: %s", name, preset["description"])


def generate_from_best_checkpoint(scale: str = "small") -> Path:
    """Generate one sample MIDI from best (or latest) checkpoint."""

    paths = setup_kaggle_environment()
    _configure_mamba_backend()
    preset = get_preset(scale)

    data_cfg = preset["data"]
    data_cfg.maestro_path = paths["maestro_root"]
    data_cfg.processed_path = paths["processed_dir"]
    data_cfg.tokenizer_path = paths["tokenizer_path"]

    checkpoint_dir = Path(paths["checkpoint_dir"])
    candidates = [
        checkpoint_dir / "best.safetensors",
        checkpoint_dir / "best_model.safetensors",
        checkpoint_dir / "latest.safetensors",
        checkpoint_dir / "latest_model.safetensors",
    ]
    checkpoint_path = next((p for p in candidates if p.exists()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}. "
            "Expected one of best.safetensors or latest.safetensors."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, tokenizer_meta = _load_inference_assets(
        checkpoint_path=checkpoint_path,
        paths=paths,
        device=device,
    )
    data_cfg.vocab_size = int(tokenizer.vocab_size)
    data_cfg.tokenizer_path = str(
        tokenizer_meta.get("tokenizer_path", data_cfg.tokenizer_path)
    )

    seed_path = _find_first_seed_midi(Path(paths["maestro_root"]))
    if seed_path is None:
        raise FileNotFoundError(
            f"No MIDI files found under MAESTRO root {paths['maestro_root']}"
        )

    output_dir = Path(paths["generated_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{checkpoint_path.stem}_{scale}_sample.mid"

    gen_cfg = GenerationConfig(
        max_new_tokens=min(512, data_cfg.continuation_length),
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        repetition_window=64,
        min_tokens_to_keep=3,
        num_samples=1,
    )

    outputs = generate_continuation(
        model=model,
        tokenizer=tokenizer,
        seed_midi_path=seed_path,
        output_path=output_path,
        config=data_cfg,
        generation_config=gen_cfg,
    )
    LOGGER.info("Generated sample from checkpoint: %s", checkpoint_path)
    LOGGER.info("Saved to: %s", outputs[0])
    return outputs[0]


def generate_from_seed_file(
    seed_path: str | Path,
    checkpoint_path: Optional[str | Path] = None,
    scale: str = "small",
    max_new_tokens: Optional[int] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Generate a long continuation from a seed MIDI file."""

    paths = setup_kaggle_environment()
    _configure_mamba_backend()
    preset = get_preset(scale)

    data_cfg = preset["data"]
    data_cfg.maestro_path = paths["maestro_root"]
    data_cfg.processed_path = paths["processed_dir"]
    data_cfg.tokenizer_path = paths["tokenizer_path"]

    ckpt_dir = Path(paths["checkpoint_dir"])
    if checkpoint_path is None:
        candidates = [
            ckpt_dir / "best.safetensors",
            ckpt_dir / "best_model.safetensors",
            ckpt_dir / "latest.safetensors",
            ckpt_dir / "latest_model.safetensors",
        ]
        checkpoint_file = next((p for p in candidates if p.exists()), None)
    else:
        checkpoint_file = Path(checkpoint_path)

    if checkpoint_file is None or not checkpoint_file.exists():
        raise FileNotFoundError(
            f"No checkpoint found. Looked in {ckpt_dir} and optional checkpoint_path."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, tokenizer_meta = _load_inference_assets(
        checkpoint_path=checkpoint_file,
        paths=paths,
        device=device,
    )
    data_cfg.vocab_size = int(tokenizer.vocab_size)
    data_cfg.tokenizer_path = str(
        tokenizer_meta.get("tokenizer_path", data_cfg.tokenizer_path)
    )

    seed_path = Path(seed_path)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed MIDI file not found: {seed_path}")

    output_dir = Path(paths["generated_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = output_dir / f"{seed_path.stem}_continuation.mid"
    else:
        output_path = Path(output_path)

    gen_tokens = max_new_tokens or max(4096, data_cfg.continuation_length * 16)
    gen_cfg = GenerationConfig(
        max_new_tokens=int(gen_tokens),
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        repetition_window=64,
        min_tokens_to_keep=3,
        num_samples=1,
    )

    outputs = generate_continuation(
        model=model,
        tokenizer=tokenizer,
        seed_midi_path=seed_path,
        output_path=output_path,
        config=data_cfg,
        generation_config=gen_cfg,
    )
    LOGGER.info("Generated continuation from seed: %s", seed_path)
    LOGGER.info("Saved to: %s", outputs[0])
    return outputs[0]


def print_dataset_availability() -> None:
    """Print MAESTRO/GiantMIDI availability for notebook preflight checks."""

    maestro = None
    giant = None
    aria = None
    adl = None
    try:
        maestro = find_maestro_root()
    except Exception:
        maestro = None
    try:
        giant = find_giant_midi_root()
    except Exception:
        giant = None
    try:
        aria = find_aria_midi_root()
    except Exception:
        aria = None
    try:
        adl = find_adl_piano_root()
    except Exception:
        adl = None

    print(f"MAESTRO:    {'✓ found' if maestro else '✗ not found'} {maestro or ''}")
    print(
        f"GiantMIDI:  {'✓ found' if giant else '✗ not found (optional)'} {giant or ''}"
    )
    print(f"Aria-MIDI:  {'✓ found' if aria else '✗ not found (optional)'} {aria or ''}")
    print(f"ADL Piano:  {'✓ found' if adl else '✗ not found (optional)'} {adl or ''}")
    print()
    if not giant:
        print(
            "TIP: Add GiantMIDI as a Kaggle dataset for dramatically more training data."
        )
        print("     Training will proceed with MAESTRO only.")


def _dry_run_session(scale: str = "large_v2") -> None:
    """Print resolved config and model stats without starting training."""

    preset = get_preset(scale)
    model_cfg = preset["model"]
    train_cfg = preset["train"]
    data_cfg = preset["data"]
    model = build_model(model_cfg)
    params = sum(p.numel() for p in model.parameters())

    print("=" * 56)
    print("Itty Bitty Piano Kaggle Dry Run")
    print("=" * 56)
    print(f"Scale: {scale}")
    print(
        f"Architecture: {'v2' if bool(getattr(model_cfg, 'use_v2_architecture', False)) else 'v1'}"
    )
    print(f"Parameters: {params:,} ({params / 1e6:.2f}M)")
    print(f"Batch size: {int(train_cfg.batch_size)}")
    print(f"Grad accumulation: {int(train_cfg.grad_accumulation_steps)}")
    print(f"Dataset weights: {dict(data_cfg.dataset_weights)}")
    print("=" * 56)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Kaggle session script."""

    parser = argparse.ArgumentParser(
        description="Itty Bitty Piano Kaggle session helper"
    )
    parser.add_argument("--scale", type=str, default="small")
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if bool(args.dry_run):
        _dry_run_session(scale=str(args.scale))
    else:
        run_kaggle_session(scale=str(args.scale), max_epochs=int(args.max_epochs))

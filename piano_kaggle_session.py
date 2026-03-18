"""
Kaggle training session runner.
Equivalent to session.py but adapted for Kaggle's environment:
- No Google Drive - checkpoints saved to /kaggle/working/
- No inactivity disconnect - can run full 12-hour sessions
- MAESTRO loaded from /kaggle/input/ (read-only)
- Project code loaded from /kaggle/input/ (read-only)
- Outputs saved to /kaggle/working/ (downloadable after session)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file as safetensors_load_file

from data.dataset import create_dataloaders
from data.preprocess import preprocess_maestro
from data.tokenizer import PianoTokenizer
from generation.generate import GenerationConfig, generate_continuation
from kaggle_config import setup_kaggle_environment
from model.hybrid import PianoHybridModel
from scale_config import SCALE_PRESETS, get_preset
from training.trainer import Trainer
from utils.session_utils import SessionWatchdog, get_gpu_info


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _manifest_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
    except Exception:
        return 0
    return 0


def _find_resume_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    candidates = [
        checkpoint_dir / "latest_model.safetensors",
        checkpoint_dir / "latest.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_state_sidecar(model_path: Path) -> Optional[Path]:
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


def _sync_checkpoint_aliases(checkpoint_dir: Path) -> None:
    aliases = [
        (
            checkpoint_dir / "latest_model.safetensors",
            checkpoint_dir / "latest.safetensors",
        ),
        (
            checkpoint_dir / "best_model.safetensors",
            checkpoint_dir / "best.safetensors",
        ),
    ]
    for src, dst in aliases:
        if not src.exists():
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)


def _append_training_log(log_path: Path, record: Dict[str, Any]) -> None:
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


def _find_first_seed_midi(maestro_root: Path) -> Optional[Path]:
    midi_files = sorted(
        list(maestro_root.rglob("*.mid")) + list(maestro_root.rglob("*.midi"))
    )
    if not midi_files:
        return None
    return midi_files[0]


def _safe_generate_tokens(
    model: torch.nn.Module,
    seed_tokens: list[int],
    max_new_tokens: int,
) -> list[int]:
    gen_fn = getattr(model, "generate", None)
    if not callable(gen_fn):
        raise RuntimeError("Model does not expose generate(...)")
    result = gen_fn(
        seed_tokens=seed_tokens,
        max_new_tokens=max_new_tokens,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
    )
    if not isinstance(result, list):
        raise RuntimeError("Model generate(...) returned unsupported output type")
    return [int(x) for x in result]


class _KaggleSyncShim:
    def __init__(self, checkpoint_dir: Path, heartbeat_path: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.heartbeat_path = heartbeat_path

    def write_heartbeat(self, payload: Dict[str, Any]) -> None:
        _atomic_write_json(self.heartbeat_path, payload)

    def sync_checkpoint(self, local_path: str, tag: str = "latest") -> bool:
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
        return True

    def wait_for_sync(self) -> None:
        return


class KaggleSessionWatchdog(SessionWatchdog):
    def __init__(
        self,
        drive_sync,
        trainer,
        warning_minutes: int = 600,
        checkpoint_stall_minutes: int = 60,
    ) -> None:
        super().__init__(
            drive_sync=drive_sync, trainer=trainer, warning_minutes=warning_minutes
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
                print(
                    f"[Watchdog] Session near timeout (~{self.warning_minutes} min). "
                    "Triggering emergency checkpoint."
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
) -> None:
    print("=" * 56)
    print("  Piano MIDI Model - Kaggle Session")
    print("=" * 56)
    print(f"  Scale:             {scale}")
    print(f"  Resume status:     {'resumed' if resumed else 'fresh start'}")
    print(f"  Epoch progress:    {start_epoch} / {max_epochs}")
    print(
        f"  GPU:               {gpu_info['gpu_name']} "
        f"({gpu_info['total_vram_gb']:.1f} GB)"
    )
    print(f"  MAESTRO root:      {paths['maestro_root']}")
    print(f"  Checkpoint dir:    {paths['checkpoint_dir']}")
    print(f"  Processed dir:     {paths['processed_dir']}")
    print(f"  Tokenizer path:    {paths['tokenizer_path']}")
    print("=" * 56)


def run_kaggle_session(scale: str = "nano", max_epochs: int = 2000) -> Dict[str, Any]:
    paths = setup_kaggle_environment()
    gpu = get_gpu_info()

    preset = get_preset(scale)
    model_cfg = preset["model"]
    train_cfg = preset["train"]
    data_cfg = preset["data"]

    train_cfg.max_epochs = int(max_epochs)
    train_cfg.checkpoint_dir = paths["checkpoint_dir"]

    data_cfg.maestro_path = paths["maestro_root"]
    data_cfg.processed_path = paths["processed_dir"]
    data_cfg.tokenizer_path = paths["tokenizer_path"]

    checkpoint_dir = Path(paths["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(data_cfg.processed_path) / "manifest.json"
    tokenizer_path = Path(data_cfg.tokenizer_path)

    needs_preprocess = (not tokenizer_path.exists()) or (
        _manifest_count(manifest_path) == 0
    )
    if needs_preprocess:
        print(
            "Tokenizer/processed cache missing in /kaggle/working. Running preprocessing..."
        )
        preprocess_maestro(data_cfg)
    else:
        print("Reusing tokenizer and processed cache from /kaggle/working.")

    tokenizer = PianoTokenizer.load(data_cfg.tokenizer_path)
    model_cfg.vocab_size = tokenizer.vocab_size
    data_cfg.vocab_size = tokenizer.vocab_size

    train_loader, val_loader, test_loader = create_dataloaders(data_cfg, train_cfg)

    model = PianoHybridModel(model_cfg)
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
            print(f"Resumed from checkpoint: {resume_ckpt} (epoch {start_epoch})")
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
    )

    if start_epoch >= int(max_epochs):
        print(f"Target max_epochs={max_epochs} already reached. Nothing to train.")
        return {
            "scale": scale,
            "start_epoch": start_epoch,
            "end_epoch": start_epoch,
            "epochs_completed": 0,
            "best_val_loss": float(trainer.best_val_loss),
        }

    heartbeat_path = Path(paths["working_dir"]) / "heartbeat_latest.json"
    sync_shim = _KaggleSyncShim(
        checkpoint_dir=checkpoint_dir, heartbeat_path=heartbeat_path
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
            _sync_checkpoint_aliases(checkpoint_dir)
            epochs_completed += 1
    finally:
        watchdog.stop()

    _sync_checkpoint_aliases(checkpoint_dir)
    end_epoch = start_epoch + epochs_completed
    best_val_loss = float(trainer.best_val_loss)

    try:
        seed_batch, _ = next(iter(test_loader))
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
        print(f"Saved session sample: {gen_out}")
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

    print("-" * 56)
    print(
        f"Kaggle session complete: +{epochs_completed} epoch(s), "
        f"epoch {start_epoch} -> {end_epoch}, best_val_loss={best_val_loss:.4f}, "
        f"time={elapsed / 60.0:.1f} min"
    )
    print("Checkpoints saved to /kaggle/working/ - download via Kaggle output panel")
    print("-" * 56)

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
    """
    Print actual parameter counts for all presets on this Kaggle runtime.
    Run this once to verify preset calibration before a long training run.
    """
    print("Parameter calibration on this runtime:")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print("-" * 50)

    for name, preset in SCALE_PRESETS.items():
        model = PianoHybridModel(preset["model"])
        total = sum(p.numel() for p in model.parameters())
        target_desc = preset["description"]
        print(f"{name:8s}: {total:>12,} params  ({total / 1e6:.2f}M)  - {target_desc}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("-" * 50)
    print("Paste these numbers to your agent for preset calibration if needed.")


def generate_from_best_checkpoint(scale: str = "nano") -> Path:
    paths = setup_kaggle_environment()
    preset = get_preset(scale)

    model_cfg = preset["model"]
    data_cfg = preset["data"]
    data_cfg.maestro_path = paths["maestro_root"]
    data_cfg.processed_path = paths["processed_dir"]
    data_cfg.tokenizer_path = paths["tokenizer_path"]

    tokenizer_path = Path(data_cfg.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Run run_kaggle_session first."
        )
    tokenizer = PianoTokenizer.load(str(tokenizer_path))
    model_cfg.vocab_size = tokenizer.vocab_size
    data_cfg.vocab_size = tokenizer.vocab_size

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
    model = PianoHybridModel(model_cfg)
    state = safetensors_load_file(str(checkpoint_path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys while loading checkpoint: {len(unexpected)}")

    model.to(device)
    model.eval()

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
    print(f"Generated sample from checkpoint: {checkpoint_path}")
    print(f"Saved to: {outputs[0]}")
    return outputs[0]

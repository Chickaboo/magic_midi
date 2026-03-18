from __future__ import annotations

import datetime as dt
import threading
import time
import warnings
from typing import Any, Dict, Optional

import torch


def get_gpu_info() -> Dict[str, Any]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / (1024**3)
        used = torch.cuda.memory_allocated(device) / (1024**3)
        free = max(0.0, total - used)
        return {
            "gpu_name": props.name,
            "total_vram_gb": float(total),
            "used_vram_gb": float(used),
            "free_vram_gb": float(free),
        }

    return {
        "gpu_name": "CPU",
        "total_vram_gb": 0.0,
        "used_vram_gb": 0.0,
        "free_vram_gb": 0.0,
    }


def estimate_time_per_epoch(trainer) -> float:
    start = time.time()
    before = getattr(trainer, "current_epoch", 0)
    trainer.train_n_epochs(1, start_epoch=int(before))
    elapsed = time.time() - start
    return float(max(elapsed, 1e-6))


class SessionWatchdog:
    def __init__(self, drive_sync, trainer, warning_minutes: int = 5):
        self.drive_sync = drive_sync
        self.trainer = trainer
        self.warning_minutes = warning_minutes

        self._start_time = time.time()
        self._last_heartbeat = 0.0
        self._last_checkpoint_time = time.time()
        self._last_checkpoint_epoch = int(getattr(trainer, "current_epoch", 0))
        self._warning_fired = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

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

            # If no new checkpoint for 15 minutes, trigger emergency save.
            if (now - self._last_checkpoint_time) >= 900:
                self._emergency_save(reason="no checkpoint for 15 minutes")
                self._last_checkpoint_time = now

            if (not self._warning_fired) and minutes_elapsed >= 25.0:
                print(
                    "[Watchdog] Session near free-tier timeout (~25 min). "
                    "Triggering emergency checkpoint + Drive sync."
                )
                self._emergency_save(reason="25 minute warning")
                self._warning_fired = True

            # warning_minutes retained for compatibility; default loop interval 15s
            _ = self.warning_minutes
            time.sleep(15)

    def _write_heartbeat(self, now: float) -> None:
        payload = {
            "timestamp": dt.datetime.now().isoformat(),
            "current_epoch": int(getattr(self.trainer, "current_epoch", 0)),
            "global_step": int(getattr(self.trainer, "global_step", 0)),
            "best_val_loss": float(
                getattr(self.trainer, "best_val_loss", float("inf"))
            ),
            "session_minutes": float((now - self._start_time) / 60.0),
        }
        try:
            self.drive_sync.write_heartbeat(payload)
            print(
                "[Watchdog] Heartbeat: "
                f"epoch={payload['current_epoch']} step={payload['global_step']}"
            )
        except Exception as exc:
            warnings.warn(f"Watchdog heartbeat failed: {exc}")

    def _emergency_save(self, reason: str) -> None:
        try:
            epoch = int(getattr(self.trainer, "current_epoch", 0))
            latest_val = (
                float(self.trainer.history["val_loss"][-1])
                if getattr(self.trainer, "history", {}).get("val_loss")
                else float("inf")
            )
            tag = f"watchdog_{int(time.time())}"
            self.trainer.save_checkpoint(epoch=epoch, val_loss=latest_val, tag=tag)

            local_path = str(self.trainer.checkpoint_dir / "latest_model.safetensors")
            self.drive_sync.sync_checkpoint(local_path=local_path, tag="latest")
            self.drive_sync.sync_checkpoint(local_path=local_path, tag=tag)
            self.drive_sync.wait_for_sync()
            print(f"[Watchdog] Emergency checkpoint saved ({reason}).")
        except Exception as exc:
            warnings.warn(f"Watchdog emergency save failed ({reason}): {exc}")


def print_session_banner(scale_preset: str, epoch: int, drive_sync) -> None:
    try:
        from scale_config import TARGET_PARAM_COUNTS

        target = TARGET_PARAM_COUNTS.get(scale_preset)
        params_str = f"~{target / 1_000_000:.0f}M" if target else "n/a"
    except Exception:
        params_str = "n/a"

    target_epochs = None
    try:
        import __main__

        v = getattr(__main__, "MAX_EPOCHS", None)
        if isinstance(v, int) and v > 0:
            target_epochs = v
    except Exception:
        target_epochs = None

    gpu = get_gpu_info()
    history = drive_sync.get_training_history()
    sessions = history.get("sessions", []) if isinstance(history, dict) else []

    total_sessions = len(sessions)
    best_val = None
    total_seconds = 0.0
    for s in sessions:
        val = s.get("final_val_loss")
        if isinstance(val, (int, float)):
            if best_val is None or float(val) < float(best_val):
                best_val = float(val)

        st = s.get("start_time")
        et = s.get("end_time")
        try:
            if isinstance(st, str) and isinstance(et, str):
                total_seconds += max(
                    0.0,
                    (
                        dt.datetime.fromisoformat(et) - dt.datetime.fromisoformat(st)
                    ).total_seconds(),
                )
        except Exception:
            pass

    best_str = f"{best_val:.3f}" if best_val is not None else "n/a"
    free = getattr(drive_sync, "free_space_gb", -1.0)
    free_str = f"{free:.1f} GB free" if free >= 0 else "unknown"

    print("=" * 48)
    print("  Piano MIDI Model - Training Session")
    print("=" * 48)
    print(f"  Scale:         {scale_preset} ({params_str})")
    if target_epochs is not None:
        print(f"  Resuming:      epoch {epoch} / {target_epochs}")
    else:
        print(f"  Resuming:      epoch {epoch}")
    print(f"  Drive space:   {free_str}")
    print(f"  GPU:           {gpu['gpu_name']} ({gpu['total_vram_gb']:.1f} GB)")
    print(
        "  Sessions:      "
        f"{total_sessions} total, {total_seconds / 3600.0:.1f} hours trained"
    )
    print(f"  Best val loss: {best_str}")
    print("=" * 48)

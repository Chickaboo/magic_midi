from __future__ import annotations

import math
import shutil
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import DataConfig, TrainConfig
from data.tokenizer import PianoTokenizer
from training.losses import create_targets, next_token_loss
from training.scheduler import WarmupCosineScheduler


KAGGLE_WORKING_ROOT = Path("/kaggle/working")
KAGGLE_CHECKPOINT_ROOT = KAGGLE_WORKING_ROOT / "checkpoints"
KAGGLE_WARN_FREE_GB = 3.0
KAGGLE_EMERGENCY_FREE_GB = 2.0
KAGGLE_KEEP_EVERY_N_EPOCHS = 10
KAGGLE_ALWAYS_KEEP_FILES = {
    "best.safetensors",
    "latest.safetensors",
    "best_state.pt",
    "latest_state.pt",
}


def _is_kaggle_checkpoint_dir(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    normalized = str(resolved).replace("\\", "/").rstrip("/")
    target = str(KAGGLE_CHECKPOINT_ROOT).replace("\\", "/").rstrip("/")
    return normalized.startswith(target)


def _checkpoint_dir_size_bytes(checkpoint_dir: Path) -> int:
    total = 0
    for file_path in checkpoint_dir.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            total += int(file_path.stat().st_size)
        except Exception:
            continue
    return int(total)


def _parse_epoch_from_rotation_name(filename: str) -> Optional[int]:
    if filename.startswith("epoch_") and filename.endswith(".safetensors"):
        token = filename[len("epoch_") : -len(".safetensors")]
        if token.isdigit():
            return int(token)

    if filename.startswith("epoch_") and filename.endswith("_state.pt"):
        token = filename[len("epoch_") : -len("_state.pt")]
        if token.isdigit():
            return int(token)

    return None


def rotate_kaggle_checkpoint_dir(
    checkpoint_dir: Path,
    keep_every_n_epochs: int = KAGGLE_KEEP_EVERY_N_EPOCHS,
) -> Dict[str, float]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    keep_every = max(1, int(keep_every_n_epochs))

    deleted_files = 0
    kept_files = 0

    for file_path in checkpoint_dir.iterdir():
        if not file_path.is_file():
            continue

        name = file_path.name
        keep = False

        if name in KAGGLE_ALWAYS_KEEP_FILES:
            keep = True
        else:
            epoch = _parse_epoch_from_rotation_name(name)
            if epoch is not None and epoch % keep_every == 0:
                keep = True

        if keep:
            kept_files += 1
            continue

        try:
            file_path.unlink()
            deleted_files += 1
        except FileNotFoundError:
            continue
        except Exception as exc:
            warnings.warn(f"Failed to delete checkpoint artifact {file_path}: {exc}")

    total_size_bytes = _checkpoint_dir_size_bytes(checkpoint_dir)
    return {
        "deleted_files": float(deleted_files),
        "kept_files": float(kept_files),
        "total_size_bytes": float(total_size_bytes),
        "total_size_mb": float(total_size_bytes / (1024**2)),
    }


def kaggle_free_space_gb(path: Path = KAGGLE_WORKING_ROOT) -> float:
    try:
        _total, _used, free = shutil.disk_usage(str(path))
    except Exception as exc:
        warnings.warn(f"Failed to read disk usage for {path}: {exc}")
        return -1.0
    return float(free / (1024**3))


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        data_config: Optional[DataConfig] = None,
        tokenizer: Optional[PianoTokenizer] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.data_config = data_config

        self.device = self._resolve_device(config.device)
        requested_data_parallel = bool(
            getattr(self.config, "_enable_data_parallel", True)
        )
        self.device_count = (
            torch.cuda.device_count() if self.device.type == "cuda" else 1
        )
        self.use_data_parallel = (
            self.device.type == "cuda"
            and self.device_count > 1
            and requested_data_parallel
        )
        if self.use_data_parallel:
            device_ids = list(range(self.device_count))
            print(f"Using DataParallel across GPUs: {device_ids}")
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        steps_per_epoch = max(
            1, math.ceil(len(self.train_loader) / self.config.grad_accumulation_steps)
        )
        self.total_steps = steps_per_epoch * self.config.max_epochs
        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=0.1,
        )

        self.use_amp = self.device.type == "cuda"
        amp_module = getattr(torch, "amp")
        self.scaler = amp_module.GradScaler("cuda", enabled=self.use_amp)

        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / "samples").mkdir(parents=True, exist_ok=True)
        self._kaggle_rotation_enabled = _is_kaggle_checkpoint_dir(self.checkpoint_dir)
        if self._kaggle_rotation_enabled:
            print(
                "Kaggle checkpoint rotation active: "
                "keep latest/best + every 10th epoch only."
            )

        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "perplexity": [],
            "lr": [],
            "gen_health_max_final_top1": [],
            "gen_health_max_raw_top1": [],
            "gen_health_min_candidates": [],
            "gen_health_passed": [],
        }
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.current_epoch = 0
        self.fixed_seed_tokens: Optional[list[int]] = None

        self.tokenizer = tokenizer
        if self.tokenizer is None and self.data_config is not None:
            tok_path = Path(self.data_config.tokenizer_path)
            if tok_path.exists():
                try:
                    self.tokenizer = PianoTokenizer.load(str(tok_path))
                except Exception as exc:
                    warnings.warn(f"Failed to load tokenizer at {tok_path}: {exc}")

        self._maybe_init_wandb()

    def _unwrap_model(self) -> Any:
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model

    @staticmethod
    def format_perplexity(loss: float) -> str:
        if loss > 20.0:
            return "overflow"
        try:
            ppl = math.exp(loss)
        except OverflowError:
            return "overflow"
        if ppl > 1_000_000:
            return f"{ppl / 1_000_000:.1f}M"
        return f"{ppl:.2f}"

    def _run_kaggle_checkpoint_rotation(self, epoch: int, trigger: str) -> None:
        if not self._kaggle_rotation_enabled:
            return
        stats = rotate_kaggle_checkpoint_dir(
            self.checkpoint_dir,
            keep_every_n_epochs=KAGGLE_KEEP_EVERY_N_EPOCHS,
        )
        print(
            "[Kaggle checkpoint cleanup] "
            f"epoch={int(epoch):03d} trigger={trigger} "
            f"deleted={int(stats['deleted_files'])} "
            f"kept={int(stats['kept_files'])} "
            f"size={float(stats['total_size_mb']):.1f}MB"
        )

    def _kaggle_pre_epoch_disk_check(self, epoch: int) -> None:
        if not self._kaggle_rotation_enabled:
            return

        free_gb = kaggle_free_space_gb(KAGGLE_WORKING_ROOT)
        if free_gb < 0:
            return

        if free_gb < KAGGLE_WARN_FREE_GB:
            print(
                "[Kaggle disk warning] "
                f"/kaggle/working free space is low: {free_gb:.2f} GB"
            )

        if free_gb < KAGGLE_EMERGENCY_FREE_GB:
            print(
                "[Kaggle disk emergency] "
                f"epoch={int(epoch):03d} free={free_gb:.2f} GB < "
                f"{KAGGLE_EMERGENCY_FREE_GB:.1f} GB. "
                "Running synchronous checkpoint cleanup before save."
            )
            self._run_kaggle_checkpoint_rotation(epoch=epoch, trigger="pre_epoch")
            free_after = kaggle_free_space_gb(KAGGLE_WORKING_ROOT)
            if free_after >= 0:
                print(
                    "[Kaggle disk status] "
                    f"free space after emergency cleanup: {free_after:.2f} GB"
                )

    def train(self) -> Dict[str, list]:
        self._warn_if_high_memory_estimate()
        start_time = time.time()

        for epoch in range(1, self.config.max_epochs + 1):
            self.current_epoch = epoch
            self._kaggle_pre_epoch_disk_check(epoch=epoch)
            epoch_start = time.time()
            self.model.train()

            self.optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            running_count = 0
            epoch_loss_sum = 0.0
            epoch_loss_count = 0

            for step_idx, batch in enumerate(self.train_loader, start=1):
                seed, continuation = batch
                seed = seed.to(self.device, non_blocking=True)
                continuation = continuation.to(self.device, non_blocking=True)

                input_ids = torch.cat([seed, continuation], dim=1)
                targets = create_targets(seed, continuation)
                targets = targets.to(self.device, non_blocking=True)

                autocast_ctx = (
                    getattr(torch, "amp").autocast("cuda")
                    if self.use_amp
                    else nullcontext()
                )
                with autocast_ctx:
                    logits, _ = self._forward_model(input_ids)
                    loss = next_token_loss(
                        logits,
                        targets,
                        label_smoothing=self.config.label_smoothing,
                    )

                loss_value = float(loss.item())
                epoch_loss_sum += loss_value
                epoch_loss_count += 1

                scaled_loss = loss / self.config.grad_accumulation_steps
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = (
                    step_idx % self.config.grad_accumulation_steps == 0
                    or step_idx == len(self.train_loader)
                )
                if should_step:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1

                    lr = self.optimizer.param_groups[0]["lr"]
                    self.history["lr"].append(float(lr))

                running_loss += loss_value
                running_count += 1

                if (
                    self.global_step > 0
                    and self.global_step % 100 == 0
                    and running_count > 0
                ):
                    avg = running_loss / running_count
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"step={self.global_step:06d} train_loss={avg:.4f} lr={lr:.6e}"
                    )
                    running_loss = 0.0
                    running_count = 0

            train_loss = epoch_loss_sum / max(1, epoch_loss_count)
            val_loss, perplexity = self.validate(epoch=epoch)

            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["perplexity"].append(float(perplexity))

            self.save_checkpoint(epoch=epoch, val_loss=val_loss)
            if epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(
                    epoch=epoch, val_loss=val_loss, tag=f"epoch_{epoch:03d}"
                )
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch=epoch, val_loss=val_loss, best=True)

            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch:03d}/{self.config.max_epochs:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"ppl={self.format_perplexity(val_loss)} | "
                f"time={elapsed:.1f}s"
            )

            self._wandb_log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "perplexity": perplexity,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

        total_elapsed = time.time() - start_time
        print(f"Training complete in {total_elapsed / 60.0:.2f} minutes.")
        return self.history

    def train_n_epochs(self, n: int, start_epoch: int = 0) -> Dict[str, list]:
        """Train exactly n epochs starting from start_epoch.

        Epoch numbering in logs/checkpoints will be absolute: start_epoch + i.
        """

        if n <= 0:
            return self.history

        self._warn_if_high_memory_estimate()
        session_start = time.time()

        try:
            for local_epoch in range(1, n + 1):
                epoch = int(start_epoch + local_epoch)
                self.current_epoch = epoch
                self._kaggle_pre_epoch_disk_check(epoch=epoch)
                epoch_start = time.time()
                self.model.train()

                self.optimizer.zero_grad(set_to_none=True)
                running_loss = 0.0
                running_count = 0
                epoch_loss_sum = 0.0
                epoch_loss_count = 0

                for step_idx, batch in enumerate(self.train_loader, start=1):
                    seed, continuation = batch
                    seed = seed.to(self.device, non_blocking=True)
                    continuation = continuation.to(self.device, non_blocking=True)

                    input_ids = torch.cat([seed, continuation], dim=1)
                    targets = create_targets(seed, continuation)
                    targets = targets.to(self.device, non_blocking=True)

                    autocast_ctx = (
                        getattr(torch, "amp").autocast("cuda")
                        if self.use_amp
                        else nullcontext()
                    )
                    with autocast_ctx:
                        logits, _ = self._forward_model(input_ids)
                        loss = next_token_loss(
                            logits,
                            targets,
                            label_smoothing=self.config.label_smoothing,
                        )

                    loss_value = float(loss.item())
                    epoch_loss_sum += loss_value
                    epoch_loss_count += 1

                    scaled_loss = loss / self.config.grad_accumulation_steps
                    if self.use_amp:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    should_step = (
                        step_idx % self.config.grad_accumulation_steps == 0
                        or step_idx == len(self.train_loader)
                    )
                    if should_step:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)

                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )

                        if self.use_amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        self.optimizer.zero_grad(set_to_none=True)
                        self.scheduler.step()
                        self.global_step += 1

                        lr = self.optimizer.param_groups[0]["lr"]
                        self.history["lr"].append(float(lr))

                    running_loss += loss_value
                    running_count += 1

                    if (
                        self.global_step > 0
                        and self.global_step % 100 == 0
                        and running_count > 0
                    ):
                        avg = running_loss / running_count
                        lr = self.optimizer.param_groups[0]["lr"]
                        print(
                            f"step={self.global_step:06d} "
                            f"train_loss={avg:.4f} lr={lr:.6e}"
                        )
                        running_loss = 0.0
                        running_count = 0

                train_loss = epoch_loss_sum / max(1, epoch_loss_count)
                val_loss, perplexity = self.validate(epoch=epoch)

                self.history["train_loss"].append(float(train_loss))
                self.history["val_loss"].append(float(val_loss))
                self.history["perplexity"].append(float(perplexity))

                self.save_checkpoint(epoch=epoch, val_loss=val_loss)
                if epoch % self.config.save_every_n_epochs == 0:
                    self.save_checkpoint(
                        epoch=epoch,
                        val_loss=val_loss,
                        tag=f"epoch_{epoch:03d}",
                    )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch=epoch, val_loss=val_loss, best=True)

                elapsed = time.time() - epoch_start
                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"ppl={self.format_perplexity(val_loss)} | "
                    f"time={elapsed:.1f}s"
                )

                self._wandb_log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "perplexity": perplexity,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            total_elapsed = time.time() - session_start
            print(
                f"Session training complete: {n} epoch(s) in "
                f"{total_elapsed / 60.0:.2f} minutes."
            )
            return self.history

        except Exception as exc:
            warnings.warn(
                f"train_n_epochs interrupted: {exc}. Attempting emergency save."
            )
            try:
                emergency_epoch = int(getattr(self, "current_epoch", start_epoch))
                latest_val = (
                    float(self.history["val_loss"][-1])
                    if self.history["val_loss"]
                    else float("inf")
                )
                self.save_checkpoint(
                    epoch=emergency_epoch,
                    val_loss=latest_val,
                    tag=f"emergency_{int(time.time())}",
                )
            except Exception as save_exc:
                warnings.warn(f"Emergency checkpoint save failed: {save_exc}")
            raise

    @torch.no_grad()
    def validate(self, epoch: int = 0) -> Tuple[float, float]:
        self.model.eval()

        val_loss_sum = 0.0
        val_count = 0

        for batch_idx, batch in enumerate(self.val_loader):
            seed, continuation = batch
            seed = seed.to(self.device, non_blocking=True)
            continuation = continuation.to(self.device, non_blocking=True)

            if self.fixed_seed_tokens is None and seed.shape[0] > 0:
                self.fixed_seed_tokens = seed[0].detach().cpu().tolist()

            input_ids = torch.cat([seed, continuation], dim=1)
            targets = create_targets(seed, continuation).to(
                self.device, non_blocking=True
            )

            autocast_ctx = (
                getattr(torch, "amp").autocast("cuda")
                if self.use_amp
                else nullcontext()
            )
            with autocast_ctx:
                logits, _ = self._forward_model(input_ids)
                loss = next_token_loss(
                    logits,
                    targets,
                    label_smoothing=self.config.label_smoothing,
                )

            val_loss_sum += float(loss.item())
            val_count += 1

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        val_loss = val_loss_sum / max(1, val_count)
        try:
            perplexity = float(math.exp(min(20.0, val_loss)))
        except OverflowError:
            perplexity = float("inf")

        self._generate_validation_sample(epoch=epoch)
        self._run_generation_health_check(epoch=epoch)
        return val_loss, perplexity

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        best: bool = False,
        tag: Optional[str] = None,
    ) -> None:
        core_model = self._unwrap_model()
        model_state = {
            k: v.detach().cpu().contiguous() for k, v in core_model.state_dict().items()
        }
        model_state_to_save = _prepare_state_for_safetensors(model_state)

        latest_model_name = (
            "latest.safetensors"
            if self._kaggle_rotation_enabled
            else "latest_model.safetensors"
        )
        latest_model_path = self.checkpoint_dir / latest_model_name
        latest_state_path = self.checkpoint_dir / "latest_state.pt"

        safetensors_save_file(model_state_to_save, str(latest_model_path))

        state = {
            "epoch": epoch,
            "val_loss": float(val_loss),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "train_config": asdict(self.config),
            "data_config": asdict(self.data_config)
            if self.data_config is not None
            else None,
            "model_config": _safe_asdict(getattr(core_model, "config", None)),
            "model_weights_path": str(latest_model_path.name),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
        }
        torch.save(state, latest_state_path)

        if best:
            best_model_name = (
                "best.safetensors"
                if self._kaggle_rotation_enabled
                else "best_model.safetensors"
            )
            best_model_path = self.checkpoint_dir / best_model_name
            best_state_path = self.checkpoint_dir / "best_state.pt"
            safetensors_save_file(model_state_to_save, str(best_model_path))
            state_best = dict(state)
            state_best["model_weights_path"] = str(best_model_path.name)
            torch.save(state_best, best_state_path)

        if tag is not None:
            tagged_model_name = (
                f"{tag}.safetensors"
                if self._kaggle_rotation_enabled
                else f"{tag}_model.safetensors"
            )
            tagged_model = self.checkpoint_dir / tagged_model_name
            tagged_state = self.checkpoint_dir / f"{tag}_state.pt"
            safetensors_save_file(model_state_to_save, str(tagged_model))
            state_tag = dict(state)
            state_tag["model_weights_path"] = str(tagged_model.name)
            torch.save(state_tag, tagged_state)

        if self._kaggle_rotation_enabled:
            trigger = "latest"
            if best:
                trigger = "best"
            if tag is not None:
                trigger = str(tag)
            self._run_kaggle_checkpoint_rotation(epoch=epoch, trigger=trigger)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

        state: Dict[str, Any] = {}

        if ckpt_path.suffix == ".pt":
            state = torch.load(ckpt_path, map_location=self.device)
            model_weights_path = state.get("model_weights_path")
            candidates = []
            if isinstance(model_weights_path, str) and model_weights_path:
                candidates.append(ckpt_path.parent / model_weights_path)
            candidates.extend(
                [
                    ckpt_path.parent / "latest.safetensors",
                    ckpt_path.parent / "latest_model.safetensors",
                ]
            )

            model_path = next((p for p in candidates if p.exists()), None)
            if model_path is None:
                raise FileNotFoundError(
                    f"No model weights found next to checkpoint state {ckpt_path}"
                )
            model_state = safetensors_load_file(str(model_path), device="cpu")
            self._unwrap_model().load_state_dict(model_state)
        elif ckpt_path.suffix == ".safetensors":
            model_state = safetensors_load_file(str(ckpt_path), device="cpu")
            self._unwrap_model().load_state_dict(model_state)
            state_candidates = []
            if ckpt_path.name.endswith("_model.safetensors"):
                state_candidates.append(
                    ckpt_path.with_name(
                        ckpt_path.name.replace("_model.safetensors", "_state.pt")
                    )
                )
            state_candidates.append(ckpt_path.with_name(f"{ckpt_path.stem}_state.pt"))
            state_candidates.append(ckpt_path.parent / "latest_state.pt")

            for state_guess in state_candidates:
                if state_guess.exists():
                    state = torch.load(state_guess, map_location=self.device)
                    break
        else:
            raise ValueError("Checkpoint path must end with .pt or .safetensors")

        if state:
            if "optimizer" in state:
                self.optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
            if self.use_amp and "scaler" in state and state["scaler"] is not None:
                self.scaler.load_state_dict(state["scaler"])
            self.global_step = int(state.get("global_step", 0))
            self.best_val_loss = float(state.get("best_val_loss", self.best_val_loss))
            history = state.get("history")
            if isinstance(history, dict):
                self.history = history

        self.model.to(self.device)
        return state

    def _generate_validation_sample(self, epoch: int) -> None:
        if self.fixed_seed_tokens is None:
            return

        if self.data_config is None:
            return

        seed = self.fixed_seed_tokens[: self.data_config.seed_length]
        if len(seed) < self.data_config.seed_length:
            return

        try:
            core_model = self._unwrap_model()
            generate_fn = getattr(core_model, "generate", None)
            if not callable(generate_fn):
                raise RuntimeError("Model does not expose generate(...)")
            generated = generate_fn(
                seed_tokens=seed,
                max_new_tokens=self.data_config.continuation_length,
                temperature=self.config.generation_health_temperature,
                top_p=self.config.generation_health_top_p,
                top_k=self.config.generation_health_top_k,
                repetition_penalty=self.config.generation_health_repetition_penalty,
                min_tokens_to_keep=self.config.generation_health_min_tokens_to_keep,
            )
            if not isinstance(generated, (list, tuple)):
                raise RuntimeError(
                    "Model generate(...) did not return a token sequence"
                )
            generated_tokens = [int(token) for token in generated]
        except Exception as exc:
            warnings.warn(f"Validation sample generation failed: {exc}")
            return

        sample_dir = self.checkpoint_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        if self.tokenizer is not None:
            sample_path = sample_dir / f"epoch_{epoch:03d}.mid"
            try:
                self.tokenizer.decode(generated_tokens, sample_path)
            except Exception as exc:
                warnings.warn(f"Failed to decode validation sample as MIDI: {exc}")
                np.save(
                    sample_dir / f"epoch_{epoch:03d}.npy",
                    np.asarray(generated_tokens, dtype=np.int64),
                )
        else:
            np.save(
                sample_dir / f"epoch_{epoch:03d}.npy",
                np.asarray(generated_tokens, dtype=np.int64),
            )

    def _run_generation_health_check(self, epoch: int) -> None:
        if self.fixed_seed_tokens is None:
            return
        if self.data_config is None:
            return

        seed = self.fixed_seed_tokens[: self.data_config.seed_length]
        if len(seed) < self.data_config.seed_length:
            return

        core_model = self._unwrap_model()
        health_fn = getattr(core_model, "generation_health_check", None)
        if not callable(health_fn):
            warnings.warn(
                "Model does not expose generation_health_check(...); skipping health check."
            )
            return

        try:
            health = health_fn(
                seed_tokens=seed,
                steps=self.config.generation_health_steps,
                temperature=self.config.generation_health_temperature,
                top_p=self.config.generation_health_top_p,
                top_k=self.config.generation_health_top_k,
                repetition_penalty=self.config.generation_health_repetition_penalty,
                min_tokens_to_keep=self.config.generation_health_min_tokens_to_keep,
                top1_threshold=self.config.generation_health_top1_threshold,
                raise_on_failure=False,
            )
        except Exception as exc:
            warnings.warn(f"Generation health check failed to run: {exc}")
            return

        if not isinstance(health, dict):
            warnings.warn(
                "Generation health check returned non-dict payload; skipping metrics."
            )
            return

        passed = bool(health.get("passed", False))
        max_final_top1 = float(health.get("max_final_top1_prob", 0.0))
        max_raw_top1 = float(health.get("max_raw_top1_prob", 0.0))
        min_candidates = int(float(health.get("min_candidate_count", 0.0)))

        print(
            "Generation health "
            f"epoch={epoch:03d} "
            f"passed={passed} "
            f"max_final_top1={max_final_top1:.4f} "
            f"max_raw_top1={max_raw_top1:.4f} "
            f"min_candidates={min_candidates}"
        )

        self.history.setdefault("gen_health_max_final_top1", []).append(max_final_top1)
        self.history.setdefault("gen_health_max_raw_top1", []).append(max_raw_top1)
        self.history.setdefault("gen_health_min_candidates", []).append(min_candidates)
        self.history.setdefault("gen_health_passed", []).append(1.0 if passed else 0.0)

        self._wandb_log(
            {
                "epoch": epoch,
                "gen_health_passed": float(1.0 if passed else 0.0),
                "gen_health_max_final_top1": max_final_top1,
                "gen_health_max_raw_top1": max_raw_top1,
                "gen_health_min_candidates": float(min_candidates),
            }
        )

        if not passed:
            raise AssertionError(
                "Generation health check threshold exceeded: "
                f"max_final_top1={max_final_top1:.4f} > "
                f"{self.config.generation_health_top1_threshold:.4f}"
            )

    def _resolve_device(self, requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    def _warn_if_high_memory_estimate(self) -> None:
        if self.device.type != "cuda":
            return

        total_vram = torch.cuda.get_device_properties(self.device).total_memory
        total_vram_gb = total_vram / (1024**3)

        core_model = self._unwrap_model()

        param_count = sum(p.numel() for p in core_model.parameters())
        params_bytes = param_count * 4
        grads_bytes = param_count * 4
        adam_bytes = param_count * 8

        seq_len = self._infer_sequence_length()
        d_model = getattr(getattr(core_model, "config", None), "d_model", 256)
        layers = getattr(getattr(core_model, "config", None), "n_layers", 4)
        activation_bytes = (
            self.config.batch_size * seq_len * d_model * max(layers, 1) * 4 * 4
        )

        estimate = params_bytes + grads_bytes + adam_bytes + activation_bytes
        estimate_gb = estimate / (1024**3)

        print(
            f"Estimated training memory: {estimate_gb:.2f} GB "
            f"(GPU available: {total_vram_gb:.2f} GB)"
        )
        if self.use_data_parallel:
            print(
                "DataParallel active across "
                f"{self.device_count} GPUs (global batch={self.config.batch_size})."
            )
        if estimate_gb > 12.0:
            warnings.warn(
                "Estimated memory usage exceeds 12GB. Consider lower batch size or shorter context."
            )

    def _forward_model(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        # Unified forward signature used by train/validate/generation.
        return self.model(
            input_ids,
            hidden_states=None,
            position_offset=0,
        )

    def _infer_sequence_length(self) -> int:
        if self.data_config is not None:
            return self.data_config.seed_length + self.data_config.continuation_length

        try:
            seed, cont = next(iter(self.train_loader))
            return int(seed.shape[1] + cont.shape[1])
        except Exception:
            return 1024

    def _maybe_init_wandb(self) -> None:
        self._wandb = None
        if not self.config.use_wandb:
            return
        try:
            import wandb  # type: ignore[import-not-found]

            self._wandb = wandb
            self._wandb.init(project="itty-bitty-piano", config=asdict(self.config))
        except Exception as exc:
            warnings.warn(f"wandb init failed; continuing without wandb ({exc})")
            self._wandb = None

    def _wandb_log(self, payload: Dict[str, Any]) -> None:
        if self._wandb is None:
            return
        try:
            self._wandb.log(payload)
        except Exception:
            pass


def _prepare_state_for_safetensors(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Clone shared-storage tensors so all keys remain loadable with safetensors."""

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


def _safe_asdict(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    try:
        return asdict(value)
    except TypeError:
        return None

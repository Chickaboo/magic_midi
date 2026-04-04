from __future__ import annotations

import json
import math
import re
import shutil
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import DataConfig, TrainConfig
from data.tokenizer import PianoTokenizer
from training.losses import build_piece_boundary_mask, create_targets, next_token_loss
from training.scheduler import WarmupCosineScheduler
from utils.logging_utils import get_project_logger


LOGGER = get_project_logger()

KAGGLE_WORKING_ROOT = Path("/kaggle/working")
CHECKPOINT_KEEP_POLICY: Dict[str, Any] = {
    "always": [
        "best.safetensors",
        "best_state.pt",
        "latest.safetensors",
        "latest_state.pt",
    ],
    "milestone_every_n": 25,
    "max_total_checkpoints": 8,
}
KAGGLE_KEEP_EVERY_N_EPOCHS = int(CHECKPOINT_KEEP_POLICY["milestone_every_n"])
MIN_FREE_DISK_GB_BEFORE_SAVE = 3.0

_EPOCH_MODEL_RE = re.compile(r"^epoch_(\d+)\.safetensors$")


def kaggle_free_space_gb(path: Path = KAGGLE_WORKING_ROOT) -> float:
    """Return free disk space in GiB for a given path."""

    try:
        _total, _used, free = shutil.disk_usage(str(path))
    except Exception as exc:
        warnings.warn(f"Failed to read disk usage for {path}: {exc}")
        return -1.0
    return float(free / (1024**3))


def _state_sidecar_for_model(model_file: Path) -> Path:
    """Return the state sidecar path corresponding to one model file."""

    return model_file.with_name(f"{model_file.stem}_state.pt")


def _checkpoint_policy(
    *,
    keep_every_n_epochs: int,
    max_total_checkpoints: int,
) -> Dict[str, Any]:
    """Build normalized checkpoint retention policy dict."""

    keep_every = max(1, int(keep_every_n_epochs))
    max_total = max(1, int(max_total_checkpoints))
    always = list(CHECKPOINT_KEEP_POLICY["always"])
    return {
        "always": always,
        "milestone_every_n": keep_every,
        "max_total_checkpoints": max_total,
    }


def _rotate_checkpoints_impl(
    checkpoint_dir: Path,
    policy: Dict[str, Any],
    reserve_slots: int = 0,
) -> Dict[str, float]:
    """Apply checkpoint retention policy and return rotation stats."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    protected = set(str(n) for n in policy["always"])
    milestone_n = max(1, int(policy["milestone_every_n"]))
    max_total = max(1, int(policy["max_total_checkpoints"]))
    reserve = max(0, int(reserve_slots))
    max_allowed_before_save = max(0, max_total - reserve)

    all_models = list(checkpoint_dir.glob("*.safetensors"))
    epoch_models = [f for f in all_models if _EPOCH_MODEL_RE.match(f.name)]

    milestones: set[str] = set()
    for model_file in epoch_models:
        m = _EPOCH_MODEL_RE.match(model_file.name)
        if m is None:
            continue
        epoch = int(m.group(1))
        if epoch % milestone_n == 0:
            milestones.add(model_file.name)
            milestones.add(_state_sidecar_for_model(model_file).name)

    deleted_files = 0
    for model_file in epoch_models:
        if model_file.name in protected or model_file.name in milestones:
            continue
        try:
            model_file.unlink(missing_ok=True)
            deleted_files += 1
        except Exception as exc:
            warnings.warn(f"Failed to delete checkpoint file {model_file}: {exc}")
        sidecar = _state_sidecar_for_model(model_file)
        try:
            sidecar.unlink(missing_ok=True)
        except Exception as exc:
            warnings.warn(f"Failed to delete checkpoint state {sidecar}: {exc}")

    remaining = list(checkpoint_dir.glob("*.safetensors"))
    if len(remaining) > max_allowed_before_save:
        non_protected = [f for f in remaining if f.name not in protected]
        non_protected.sort(key=lambda f: f.stat().st_mtime)
        over = len(remaining) - max_allowed_before_save
        for model_file in non_protected[:over]:
            try:
                model_file.unlink(missing_ok=True)
                deleted_files += 1
            except Exception as exc:
                warnings.warn(f"Failed to delete checkpoint file {model_file}: {exc}")
            sidecar = _state_sidecar_for_model(model_file)
            try:
                sidecar.unlink(missing_ok=True)
            except Exception as exc:
                warnings.warn(f"Failed to delete checkpoint state {sidecar}: {exc}")

    remaining_models = list(checkpoint_dir.glob("*.safetensors"))
    return {
        "deleted_files": float(deleted_files),
        "remaining_models": float(len(remaining_models)),
        "max_total_checkpoints": float(max_total),
        "reserve_slots": float(reserve),
    }


def rotate_kaggle_checkpoint_dir(
    checkpoint_dir: Path,
    keep_every_n_epochs: int = KAGGLE_KEEP_EVERY_N_EPOCHS,
    max_total_checkpoints: int = int(CHECKPOINT_KEEP_POLICY["max_total_checkpoints"]),
) -> Dict[str, float]:
    """Backward-compatible checkpoint rotation entrypoint used by Kaggle helpers."""

    policy = _checkpoint_policy(
        keep_every_n_epochs=keep_every_n_epochs,
        max_total_checkpoints=max_total_checkpoints,
    )
    return _rotate_checkpoints_impl(checkpoint_dir, policy, reserve_slots=0)


class Trainer:
    """Training loop, validation, checkpointing, and resume utilities."""

    @staticmethod
    def _model_uses_real_gdn(model: torch.nn.Module) -> bool:
        """Return True when model contains real (non-fallback) GatedDeltaNet blocks."""

        for module in model.modules():
            if module.__class__.__name__ != "GatedDeltaNetBlock":
                continue
            if not bool(getattr(module, "using_fallback", False)):
                return True
        return False

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
            allow_gdn_data_parallel = bool(
                getattr(self.config, "_allow_gdn_data_parallel", False)
            )
            if self._model_uses_real_gdn(self.model) and not allow_gdn_data_parallel:
                self.use_data_parallel = False
                LOGGER.warning(
                    "Detected real GatedDeltaNet kernels; disabling DataParallel due known "
                    "Triton autotuner instability across replicas. "
                    "Using single-GPU mode. Set _allow_gdn_data_parallel=True to override."
                )
        if self.use_data_parallel:
            device_ids = list(range(self.device_count))
            LOGGER.info("Using DataParallel across GPUs: %s", device_ids)
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model.to(self.device)

        core_model = self._unwrap_model()
        model_cfg = getattr(core_model, "config", None)
        self.use_v2_model = bool(getattr(model_cfg, "use_v2_architecture", False))
        self._memory_state: Optional[torch.Tensor] = None

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        steps_per_epoch = max(
            1,
            math.ceil(len(self.train_loader) / self.config.grad_accumulation_steps),
        )
        self.total_steps = steps_per_epoch * self.config.max_epochs
        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=float(getattr(self.config, "min_lr_ratio", 0.1)),
        )

        self.use_amp = self.device.type == "cuda" and bool(
            getattr(self.config, "use_amp", True)
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.log_every_n_steps = max(
            1, int(getattr(self.config, "_log_every_n_steps", 100))
        )

        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / "samples").mkdir(parents=True, exist_ok=True)

        self._retention_policy = _checkpoint_policy(
            keep_every_n_epochs=int(getattr(self.config, "keep_every_n_epochs", 25)),
            max_total_checkpoints=int(getattr(self.config, "max_checkpoints", 8)),
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

        bind_tokenizer = getattr(self._unwrap_model(), "bind_tokenizer", None)
        if callable(bind_tokenizer) and self.tokenizer is not None:
            try:
                bind_tokenizer(self.tokenizer)
            except Exception as exc:
                warnings.warn(f"Failed to bind tokenizer to model: {exc}")

        self._maybe_init_wandb()

    def _unwrap_model(self) -> Any:
        """Return underlying module for DataParallel-wrapped models."""

        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model

    @staticmethod
    def format_perplexity(loss: float) -> str:
        """Format perplexity for compact training logs."""

        if loss > 20.0:
            return "overflow"
        try:
            ppl = math.exp(loss)
        except OverflowError:
            return "overflow"
        if ppl > 1_000_000:
            return f"{ppl / 1_000_000:.1f}M"
        return f"{ppl:.2f}"

    def _rotate_checkpoints(self, reserve_slots: int = 0) -> Dict[str, float]:
        """Apply retention policy to checkpoint directory before saving."""

        return _rotate_checkpoints_impl(
            checkpoint_dir=self.checkpoint_dir,
            policy=self._retention_policy,
            reserve_slots=reserve_slots,
        )

    def _pre_save_disk_check(self, reserve_slots: int = 0) -> None:
        """Run disk-space check and emergency rotation when free space is low."""

        free_gb = kaggle_free_space_gb(self.checkpoint_dir)
        if free_gb < 0:
            return

        if free_gb < MIN_FREE_DISK_GB_BEFORE_SAVE:
            LOGGER.warning(
                "Low disk space before checkpoint save: %.2f GB free (< %.2f GB). "
                "Running emergency rotation.",
                free_gb,
                MIN_FREE_DISK_GB_BEFORE_SAVE,
            )
            self._rotate_checkpoints(reserve_slots=reserve_slots)
            free_after = kaggle_free_space_gb(self.checkpoint_dir)
            if free_after >= 0:
                LOGGER.info("Free space after emergency rotation: %.2f GB", free_after)

    def train(self) -> Dict[str, list]:
        """Train for `config.max_epochs` epochs and return training history."""

        self._warn_if_high_memory_estimate()
        start_time = time.time()

        for epoch in range(1, self.config.max_epochs + 1):
            self._run_one_epoch(epoch=epoch, max_epochs=self.config.max_epochs)

        total_elapsed = time.time() - start_time
        LOGGER.info("Training complete in %.2f minutes.", total_elapsed / 60.0)
        return self.history

    def train_n_epochs(self, n: int, start_epoch: int = 0) -> Dict[str, list]:
        """Train exactly `n` epochs starting from `start_epoch` numbering."""

        if n <= 0:
            return self.history

        self._warn_if_high_memory_estimate()
        session_start = time.time()

        try:
            for local_epoch in range(1, n + 1):
                epoch = int(start_epoch + local_epoch)
                self._run_one_epoch(epoch=epoch, max_epochs=None)

            total_elapsed = time.time() - session_start
            LOGGER.info(
                "Session training complete: %d epoch(s) in %.2f minutes.",
                n,
                total_elapsed / 60.0,
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

    def _run_one_epoch(self, epoch: int, max_epochs: Optional[int]) -> None:
        """Execute one full train/validate/checkpoint epoch."""

        self.current_epoch = int(epoch)
        epoch_start = time.time()
        self.model.train()

        est_optimizer_steps = max(
            1,
            math.ceil(len(self.train_loader) / self.config.grad_accumulation_steps),
        )
        LOGGER.info(
            "Epoch %03d start | train_batches=%d | est_optimizer_steps=%d | log_every=%d",
            int(epoch),
            int(len(self.train_loader)),
            int(est_optimizer_steps),
            int(self.log_every_n_steps),
        )

        self.optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_count = 0
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        for step_idx, batch in enumerate(self.train_loader, start=1):
            parsed = self._parse_batch(batch)
            seed = parsed["seed"]
            continuation = parsed["continuation"]
            input_ids = parsed["token_ids"]
            onset_times = parsed["onset_times"]
            durations = parsed["durations"]
            reset_memory = bool(parsed["new_piece"].any().item())

            seed = seed.to(self.device, non_blocking=True)
            continuation = continuation.to(self.device, non_blocking=True)
            input_ids = input_ids.to(self.device, non_blocking=True)
            onset_times = onset_times.to(self.device, non_blocking=True)
            durations = durations.to(self.device, non_blocking=True)
            targets = create_targets(seed, continuation)
            targets = targets.to(self.device, non_blocking=True)
            boundary_mask = build_piece_boundary_mask(
                seed=seed,
                continuation=continuation,
                new_piece=parsed["new_piece"],
            ).to(self.device, non_blocking=True)

            autocast_ctx = torch.amp.autocast("cuda") if self.use_amp else nullcontext()
            with autocast_ctx:
                logits, _ = self._forward_model(
                    input_ids,
                    onset_times=onset_times,
                    durations=durations,
                    reset_memory=reset_memory,
                )
                loss = next_token_loss(
                    logits,
                    targets,
                    label_smoothing=self.config.label_smoothing,
                    piece_boundary_mask=boundary_mask,
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
                should_step
                and self.global_step > 0
                and self.global_step % int(self.log_every_n_steps) == 0
                and running_count > 0
            ):
                avg = running_loss / running_count
                lr = self.optimizer.param_groups[0]["lr"]
                LOGGER.info(
                    "step=%06d train_loss=%.4f lr=%.6e",
                    self.global_step,
                    avg,
                    lr,
                )
                running_loss = 0.0
                running_count = 0

        train_loss = epoch_loss_sum / max(1, epoch_loss_count)
        val_loss, perplexity = self.validate(epoch=epoch)

        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(float(val_loss))
        self.history["perplexity"].append(float(perplexity))

        self.save_checkpoint(epoch=epoch, val_loss=val_loss)
        if epoch % int(self.config.save_every_n_epochs) == 0:
            self.save_checkpoint(
                epoch=epoch,
                val_loss=val_loss,
                tag=f"epoch_{epoch:03d}",
            )
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(epoch=epoch, val_loss=val_loss, best=True)

        elapsed = time.time() - epoch_start
        if max_epochs is None:
            LOGGER.info(
                "Epoch %03d | train_loss=%.4f | val_loss=%.4f | ppl=%s | time=%.1fs",
                epoch,
                train_loss,
                val_loss,
                self.format_perplexity(val_loss),
                elapsed,
            )
        else:
            LOGGER.info(
                "Epoch %03d/%03d | train_loss=%.4f | val_loss=%.4f | ppl=%s | time=%.1fs",
                epoch,
                int(max_epochs),
                train_loss,
                val_loss,
                self.format_perplexity(val_loss),
                elapsed,
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

    @torch.no_grad()
    def validate(self, epoch: int = 0) -> Tuple[float, float]:
        """Evaluate on validation set and run generation health checks."""

        self.model.eval()

        val_loss_sum = 0.0
        val_count = 0

        for batch in self.val_loader:
            parsed = self._parse_batch(batch)
            seed = parsed["seed"]
            continuation = parsed["continuation"]
            input_ids = parsed["token_ids"]
            onset_times = parsed["onset_times"]
            durations = parsed["durations"]
            reset_memory = bool(parsed["new_piece"].any().item())

            seed = seed.to(self.device, non_blocking=True)
            continuation = continuation.to(self.device, non_blocking=True)
            input_ids = input_ids.to(self.device, non_blocking=True)
            onset_times = onset_times.to(self.device, non_blocking=True)
            durations = durations.to(self.device, non_blocking=True)

            if self.fixed_seed_tokens is None and seed.shape[0] > 0:
                self.fixed_seed_tokens = seed[0].detach().cpu().tolist()

            targets = create_targets(seed, continuation).to(
                self.device,
                non_blocking=True,
            )
            boundary_mask = build_piece_boundary_mask(
                seed=seed,
                continuation=continuation,
                new_piece=parsed["new_piece"],
            ).to(self.device, non_blocking=True)

            autocast_ctx = torch.amp.autocast("cuda") if self.use_amp else nullcontext()
            with autocast_ctx:
                logits, _ = self._forward_model(
                    input_ids,
                    onset_times=onset_times,
                    durations=durations,
                    reset_memory=reset_memory,
                )
                loss = next_token_loss(
                    logits,
                    targets,
                    label_smoothing=self.config.label_smoothing,
                    piece_boundary_mask=boundary_mask,
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
        if bool(getattr(self.config, "val_generation_check", True)):
            self._run_generation_health_check(epoch=epoch)
        return val_loss, perplexity

    def _checkpoint_target_paths(
        self,
        *,
        best: bool,
        tag: Optional[str],
    ) -> Tuple[Path, Path]:
        """Resolve checkpoint model/state output paths for a save call."""

        if best:
            model_path = self.checkpoint_dir / "best.safetensors"
            state_path = self.checkpoint_dir / "best_state.pt"
            return model_path, state_path

        if tag is not None:
            model_path = self.checkpoint_dir / f"{tag}.safetensors"
            state_path = self.checkpoint_dir / f"{tag}_state.pt"
            return model_path, state_path

        model_path = self.checkpoint_dir / "latest.safetensors"
        state_path = self.checkpoint_dir / "latest_state.pt"
        return model_path, state_path

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        best: bool = False,
        tag: Optional[str] = None,
    ) -> None:
        """Save model and training state with aggressive pre-save rotation."""

        target_model_path, target_state_path = self._checkpoint_target_paths(
            best=best,
            tag=tag,
        )
        reserve_slots = 0 if target_model_path.exists() else 1

        self._rotate_checkpoints(reserve_slots=reserve_slots)
        self._pre_save_disk_check(reserve_slots=reserve_slots)

        core_model = self._unwrap_model()
        model_state = {
            key: value.detach().cpu().contiguous()
            for key, value in core_model.state_dict().items()
        }
        model_state_to_save = _prepare_state_for_safetensors(model_state)
        model_config_payload = _safe_asdict(getattr(core_model, "config", None))
        checkpoint_metadata = _checkpoint_safetensors_metadata(
            epoch=epoch,
            val_loss=val_loss,
            train_config=asdict(self.config),
            data_config=asdict(self.data_config)
            if self.data_config is not None
            else None,
            model_config=model_config_payload,
        )

        safetensors_save_file(
            model_state_to_save,
            str(target_model_path),
            metadata=checkpoint_metadata,
        )

        state = {
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "memory_state": self._memory_state.detach().cpu()
            if isinstance(self._memory_state, torch.Tensor)
            else None,
            "train_config": asdict(self.config),
            "data_config": asdict(self.data_config)
            if self.data_config is not None
            else None,
            "model_config": _safe_asdict(getattr(core_model, "config", None)),
            "model_weights_path": str(target_model_path.name),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
        }
        torch.save(state, target_state_path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load checkpoint weights and state from `.pt` or `.safetensors` file."""

        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint path not found: {ckpt_path}. Verify checkpoint location and filename."
            )

        state: Dict[str, Any] = {}

        if ckpt_path.suffix == ".pt":
            state = torch.load(ckpt_path, map_location=self.device)
            model_weights_path = state.get("model_weights_path")
            candidates: List[Path] = []
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
                    f"No model weights found next to checkpoint state {ckpt_path}. "
                    "Ensure .safetensors model file is present."
                )
            model_state = safetensors_load_file(str(model_path), device="cpu")
            self._unwrap_model().load_state_dict(model_state)
        elif ckpt_path.suffix == ".safetensors":
            model_state = safetensors_load_file(str(ckpt_path), device="cpu")
            self._unwrap_model().load_state_dict(model_state)

            state_candidates = [
                ckpt_path.with_name(f"{ckpt_path.stem}_state.pt"),
                ckpt_path.parent / "latest_state.pt",
                ckpt_path.parent / "best_state.pt",
            ]
            if ckpt_path.name.endswith("_model.safetensors"):
                state_candidates.append(
                    ckpt_path.with_name(
                        ckpt_path.name.replace("_model.safetensors", "_state.pt")
                    )
                )

            for state_guess in state_candidates:
                if state_guess.exists():
                    state = torch.load(state_guess, map_location=self.device)
                    break
        else:
            raise ValueError(
                f"Unsupported checkpoint extension for {ckpt_path}. Use .pt or .safetensors."
            )

        if state:
            if "optimizer" in state:
                self.optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
            if self.use_amp and "scaler" in state and state["scaler"] is not None:
                self.scaler.load_state_dict(state["scaler"])
            if "memory_state" in state:
                memory_state = state.get("memory_state")
                if isinstance(memory_state, torch.Tensor):
                    self._memory_state = memory_state.to(self.device)
                else:
                    self._memory_state = memory_state
            self.global_step = int(state.get("global_step", 0))
            self.best_val_loss = float(state.get("best_val_loss", self.best_val_loss))
            history = state.get("history")
            if isinstance(history, dict):
                self.history = history

        self.model.to(self.device)
        return state

    def _generate_validation_sample(self, epoch: int) -> None:
        """Generate one validation sample for quick musical inspection."""

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
                raise RuntimeError("Model does not expose generate(...) method.")
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
                    "Model generate(...) did not return a token sequence."
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
        """Run generation confidence health checks during validation."""

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
            health_steps = int(
                getattr(
                    self.config,
                    "val_generation_steps",
                    getattr(self.config, "generation_health_steps", 20),
                )
            )
            health = health_fn(
                seed_tokens=seed,
                steps=health_steps,
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

        LOGGER.info(
            "Generation health epoch=%03d passed=%s max_final_top1=%.4f "
            "max_raw_top1=%.4f min_candidates=%d",
            epoch,
            passed,
            max_final_top1,
            max_raw_top1,
            min_candidates,
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
                f"{self.config.generation_health_top1_threshold:.4f}. "
                "Increase sampling diversity or review model calibration."
            )

    def _resolve_device(self, requested: str) -> torch.device:
        """Resolve requested device string to a valid torch device."""

        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    def _warn_if_high_memory_estimate(self) -> None:
        """Log rough memory estimate on CUDA to avoid OOM surprises."""

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

        LOGGER.info(
            "Estimated training memory: %.2f GB (GPU available: %.2f GB)",
            estimate_gb,
            total_vram_gb,
        )
        if self.use_data_parallel:
            LOGGER.info(
                "DataParallel active across %d GPUs (global batch=%d).",
                self.device_count,
                int(self.config.batch_size),
            )
        if estimate_gb > 12.0:
            warnings.warn(
                "Estimated memory usage exceeds 12GB. Consider lower batch size or shorter context."
            )

    def _parse_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Normalize dataloader batch formats to one dictionary structure."""

        if isinstance(batch, dict):
            seed = batch.get("seed")
            continuation = batch.get("continuation")
            token_ids = batch.get("token_ids")
            onset_times = batch.get("onset_times")
            durations = batch.get("durations")
            new_piece = batch.get("new_piece")
        else:
            seed = None
            continuation = None
            token_ids = None
            onset_times = None
            durations = None
            new_piece = None
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                seed = batch[0]
                continuation = batch[1]

        if not isinstance(seed, torch.Tensor) or not isinstance(
            continuation, torch.Tensor
        ):
            raise ValueError("Batch must provide tensor `seed` and `continuation`.")

        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.cat([seed, continuation], dim=1)

        if not isinstance(onset_times, torch.Tensor):
            onset_times = self._fallback_onset_times(token_ids)
        if not isinstance(durations, torch.Tensor):
            durations = self._fallback_durations(token_ids)

        if not isinstance(new_piece, torch.Tensor):
            new_piece = torch.ones((token_ids.shape[0],), dtype=torch.bool)
        else:
            new_piece = new_piece.to(dtype=torch.bool)

        return {
            "seed": seed,
            "continuation": continuation,
            "token_ids": token_ids,
            "onset_times": onset_times,
            "durations": durations,
            "new_piece": new_piece,
        }

    def _fallback_onset_times(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Create fallback onset times when dataset lacks explicit timing arrays."""

        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must be rank-2, got shape {tuple(token_ids.shape)}"
            )
        step = 0.5
        if self.data_config is not None:
            step = float(
                max(
                    1e-4,
                    getattr(
                        self.data_config, "time_feature_fallback_step_seconds", 0.5
                    ),
                )
            )
        seq_len = int(token_ids.shape[1])
        base = (
            torch.arange(seq_len, device=token_ids.device, dtype=torch.float32) * step
        )
        return base.unsqueeze(0).expand(token_ids.shape[0], -1).contiguous()

    def _fallback_durations(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Create fallback duration features when dataset lacks duration arrays."""

        step = 0.5
        if self.data_config is not None:
            step = float(
                max(
                    1e-4,
                    getattr(
                        self.data_config, "time_feature_fallback_step_seconds", 0.5
                    ),
                )
            )
        return torch.full(
            size=tuple(token_ids.shape),
            fill_value=step,
            dtype=torch.float32,
            device=token_ids.device,
        )

    def _reset_memory_state(self) -> None:
        """Reset recurrent theme memory for v2 model."""

        self._memory_state = None
        core = self._unwrap_model()
        theme_memory = getattr(core, "theme_memory", None)
        reset_fn = getattr(theme_memory, "reset", None)
        if callable(reset_fn):
            reset_value = reset_fn()
            if isinstance(reset_value, torch.Tensor):
                self._memory_state = reset_value

    def _forward_model(
        self,
        input_ids: torch.Tensor,
        onset_times: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        reset_memory: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        """Run model forward with unified signature used by train/validate."""

        if self.use_v2_model:
            if reset_memory and bool(
                getattr(self.config, "theme_memory_reset_on_piece", True)
            ):
                self._reset_memory_state()

            if onset_times is None:
                onset_times = self._fallback_onset_times(input_ids)
            if durations is None:
                durations = self._fallback_durations(input_ids)

            logits, new_memory = self.model(
                token_ids=input_ids,
                onset_times=onset_times,
                durations=durations,
                memory=self._memory_state,
                return_memory=True,
            )
            if isinstance(new_memory, torch.Tensor):
                self._memory_state = new_memory.detach()
            else:
                self._memory_state = new_memory
            return logits, new_memory

        return self.model(
            input_ids,
            hidden_states=None,
            position_offset=0,
        )

    def _infer_sequence_length(self) -> int:
        """Infer total train sequence length from config or one batch."""

        if self.data_config is not None:
            return self.data_config.seed_length + self.data_config.continuation_length

        try:
            sample_batch = next(iter(self.train_loader))
            if isinstance(sample_batch, dict):
                token_ids = sample_batch.get("token_ids")
                if isinstance(token_ids, torch.Tensor):
                    return int(token_ids.shape[1])
                seed = sample_batch.get("seed")
                cont = sample_batch.get("continuation")
                if isinstance(seed, torch.Tensor) and isinstance(cont, torch.Tensor):
                    return int(seed.shape[1] + cont.shape[1])
            elif isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
                seed, cont = sample_batch[0], sample_batch[1]
                if isinstance(seed, torch.Tensor) and isinstance(cont, torch.Tensor):
                    return int(seed.shape[1] + cont.shape[1])
        except Exception:
            return 1024

        return 1024

    def _maybe_init_wandb(self) -> None:
        """Initialize Weights & Biases if enabled."""

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
        """Log one payload to wandb when enabled."""

        if self._wandb is None:
            return
        try:
            self._wandb.log(payload)
        except Exception as exc:
            warnings.warn(f"wandb log failed ({exc})")


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
    """Safely convert dataclass value to dict or return None."""

    if value is None:
        return None
    try:
        return asdict(value)
    except TypeError:
        return None


def _checkpoint_safetensors_metadata(
    *,
    epoch: int,
    val_loss: float,
    train_config: Dict[str, Any],
    data_config: Optional[Dict[str, Any]],
    model_config: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Encode metadata payload for safetensors checkpoints."""

    metadata: Dict[str, str] = {
        "checkpoint_format": "1",
        "epoch": str(int(epoch)),
        "val_loss": str(float(val_loss)),
    }

    def _encode(name: str, payload: Optional[Dict[str, Any]]) -> None:
        if payload is None:
            return
        metadata[name] = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    _encode("train_config", train_config)
    _encode("data_config", data_config)
    _encode("model_config", model_config)
    return metadata

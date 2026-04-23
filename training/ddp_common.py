from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from torch.nn.parallel import DistributedDataParallel as DDP


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


def _resolve_rope_heads(width: int, requested_heads: int) -> int:
    heads = _resolve_divisible_heads(width, requested_heads)
    while heads > 1:
        head_dim = int(width) // int(heads)
        if head_dim % 2 == 0:
            return int(heads)
        heads -= 1
        while heads > 1 and (int(width) % heads) != 0:
            heads -= 1
    return 1


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


def _resolve_loader_workers(
    *,
    requested_workers: int,
    world_size: int,
    max_workers: int = 8,
) -> int:
    """Resolve DataLoader worker count with optional auto mode.

    Pass requested_workers < 0 to enable auto tuning based on host CPU count
    and number of distributed ranks.
    """

    requested = int(requested_workers)
    if requested >= 0:
        return int(max(0, requested))

    cpu_total = int(os.cpu_count() or 2)
    ranks = int(max(1, world_size))
    per_rank_budget = int(max(1, cpu_total // ranks))

    # Keep at least one core available for the training process/main thread.
    auto_workers = int(max(1, per_rank_budget - 1))
    return int(max(1, min(int(max_workers), auto_workers)))


def _resolve_resume_checkpoint(
    *,
    checkpoint_dir: Path,
    auto_resume: bool,
    resume_from_checkpoint: str,
) -> Optional[Path]:
    def _pick_from_dir(dir_path: Path) -> Optional[Path]:
        state_candidates = [
            dir_path / "latest_state.pt",
            dir_path / "best_state.pt",
        ]
        found_state = next((p for p in state_candidates if p.exists()), None)
        if found_state is not None:
            return found_state

        weight_candidates = [
            dir_path / "latest.safetensors",
            dir_path / "best.safetensors",
        ]
        if any(p.exists() for p in weight_candidates):
            raise FileNotFoundError(
                "Found safetensors checkpoint(s) without matching .pt state checkpoint in "
                f"{dir_path.resolve()}. Resume requires latest_state.pt or best_state.pt."
            )
        return None

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


def _to_serializable_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return dict(asdict(obj))
    return {}


def _save_checkpoint(
    *,
    model: torch.nn.Module | DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    train_cfg: Any,
    data_cfg: Any,
    checkpoint_dir: Path,
    epoch: int,
    val_loss: float,
    history: Dict[str, List[float]],
    best_val_loss: float,
    global_step: int,
    best: bool,
    resume_state: Optional[Dict[str, Any]] = None,
) -> None:
    def _atomic_replace(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(src), str(dst))

    def _atomic_safetensors_save(
        tensor_state: Dict[str, torch.Tensor],
        destination: Path,
        metadata: Dict[str, str],
    ) -> None:
        tmp = destination.parent / f".{destination.name}.tmp"
        try:
            safetensors_save_file(tensor_state, str(tmp), metadata=metadata)
            _atomic_replace(tmp, destination)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def _atomic_torch_save(payload: Dict[str, Any], destination: Path) -> None:
        tmp = destination.parent / f".{destination.name}.tmp"
        try:
            torch.save(payload, tmp)
            _atomic_replace(tmp, destination)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    core_model = model.module if isinstance(model, DDP) else model

    if best:
        model_path = checkpoint_dir / "best.safetensors"
        state_path = checkpoint_dir / "best_state.pt"
    else:
        model_path = checkpoint_dir / "latest.safetensors"
        state_path = checkpoint_dir / "latest_state.pt"

    train_config = _to_serializable_dict(train_cfg)
    data_config = _to_serializable_dict(data_cfg)
    model_config = _to_serializable_dict(getattr(core_model, "config", {}))

    model_state = {
        key: value.detach().cpu().contiguous()
        for key, value in core_model.state_dict().items()
    }
    _atomic_safetensors_save(
        _prepare_state_for_safetensors(model_state),
        model_path,
        {
            "epoch": str(int(epoch)),
            "val_loss": f"{float(val_loss):.8f}",
            "train_config": json.dumps(train_config),
            "data_config": json.dumps(data_config),
            "model_config": json.dumps(model_config),
        },
    )

    state = {
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "train_config": train_config,
        "data_config": data_config,
        "model_config": model_config,
        "model_weights_path": str(model_path.name),
        "history": history,
        "best_val_loss": float(best_val_loss),
        "global_step": int(global_step),
        "resume_state": dict(resume_state or {}),
    }
    _atomic_torch_save(state, state_path)


def _load_checkpoint(
    *,
    model: torch.nn.Module | DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
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

    core_model.load_state_dict(safetensors_load_file(str(model_path), device="cpu"))

    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if "scaler" in state and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])

    return dict(state)


__all__ = [
    "_average_from_sums",
    "_is_main_process",
    "_load_checkpoint",
    "_log",
    "_rank_info",
    "_resolve_divisible_heads",
    "_resolve_resume_checkpoint",
    "_resolve_rope_heads",
    "_save_checkpoint",
]
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import DataConfig, TrainConfig
from utils.logging_utils import get_project_logger


LOGGER = get_project_logger()


def _to_int(value: Any, fallback: int = -1) -> int:
    """Convert value to int with fallback."""

    try:
        return int(value)
    except Exception:
        return fallback


def _to_float(value: Any, fallback: float = 0.0) -> float:
    """Convert value to float with fallback."""

    try:
        return float(value)
    except Exception:
        return fallback


def _item_source(item: Dict[str, object]) -> str:
    """Extract dataset source label from a manifest item."""

    source = str(item.get("source", "maestro") or "maestro").strip()
    return source or "maestro"


def _build_per_item_weights(
    manifest: List[Dict[str, object]],
    config: DataConfig,
) -> List[float]:
    """Build per-item sampling weights from dataset_weights map."""

    if not manifest:
        return []

    weight_map = dict(config.dataset_weights or {})
    if not weight_map:
        return [1.0] * len(manifest)

    weights: List[float] = []
    for item in manifest:
        source = _item_source(item)
        w = _to_float(weight_map.get(source, 1.0), fallback=1.0)
        if w <= 0:
            raise ValueError(f"dataset_weights['{source}'] must be > 0, got {w}.")
        weights.append(float(w))
    return weights


class PianoDataset(Dataset):
    """Piece-level dataset for seed/continuation training windows."""

    def __init__(
        self,
        manifest: List[Dict[str, object]],
        data_config: DataConfig,
        seed: int = 42,
    ) -> None:
        self.manifest = list(manifest)
        self.data_config = data_config
        self.rng = random.Random(seed)
        self.event_size = self._resolve_event_size()
        self.min_required = (
            self.data_config.seed_length + self.data_config.continuation_length
        )

        if self.event_size > 1:
            if int(self.data_config.seed_length) % int(self.event_size) != 0:
                raise ValueError(
                    f"seed_length must be divisible by event size {self.event_size}, "
                    f"got {self.data_config.seed_length}"
                )
            if int(self.data_config.continuation_length) % int(self.event_size) != 0:
                raise ValueError(
                    "continuation_length must be divisible by event size "
                    f"{self.event_size}, got {self.data_config.continuation_length}"
                )

        filtered_manifest: List[Dict[str, object]] = []
        for m in self.manifest:
            length_val = m.get("length", m.get("tokens"))
            if _to_int(length_val, fallback=-1) >= self.min_required:
                filtered_manifest.append(m)

        self.manifest = filtered_manifest
        if not self.manifest:
            raise RuntimeError(
                "No valid pieces in dataset split. Ensure preprocessing produced pieces "
                f"with length >= {self.min_required}."
            )

    @classmethod
    def from_manifest_path(
        cls,
        manifest_path: Path,
        data_config: DataConfig,
        seed: int = 42,
    ) -> "PianoDataset":
        """Construct dataset from manifest JSON path."""

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        return cls(manifest=manifest, data_config=data_config, seed=seed)

    def __len__(self) -> int:
        """Return number of eligible pieces."""

        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Sample one random training window from a piece."""

        item = self.manifest[idx]
        tokens_path = Path(str(item["tokens_path"]))
        token_seq = np.load(tokens_path)

        onset_seq = self._load_time_feature_array(
            item,
            key="onset_times_path",
            length=int(token_seq.shape[0]),
            fallback_step=float(self.data_config.time_feature_fallback_step_seconds),
            is_duration=False,
        )
        duration_seq = self._load_time_feature_array(
            item,
            key="durations_path",
            length=int(token_seq.shape[0]),
            fallback_step=float(self.data_config.time_feature_fallback_step_seconds),
            is_duration=True,
        )

        total_needed = (
            self.data_config.seed_length + self.data_config.continuation_length
        )
        if token_seq.shape[0] < total_needed:
            raise RuntimeError(
                f"Piece {tokens_path} shorter than required window {total_needed}."
            )

        max_start = int(token_seq.shape[0] - total_needed)
        raw_start = self.rng.randint(0, max_start) if max_start > 0 else 0
        start = self._snap_to_event_boundary(raw_start, max_start, self.event_size)

        if self.event_size > 1 and (start % self.event_size) != 0:
            raise AssertionError(
                "Event boundary violation in dataset windowing: "
                f"start={start} (raw_start={raw_start}) is not divisible by "
                f"{self.event_size}"
            )

        seed = token_seq[start : start + self.data_config.seed_length]
        cont = token_seq[
            start + self.data_config.seed_length : start
            + self.data_config.seed_length
            + self.data_config.continuation_length
        ]

        onset = onset_seq[start : start + total_needed]
        duration = duration_seq[start : start + total_needed]

        seed_t = torch.from_numpy(seed.astype(np.int64, copy=False))
        cont_t = torch.from_numpy(cont.astype(np.int64, copy=False))
        onset_t = torch.from_numpy(onset.astype(np.float32, copy=False))
        duration_t = torch.from_numpy(duration.astype(np.float32, copy=False))
        return {
            "seed": seed_t,
            "continuation": cont_t,
            "token_ids": torch.cat([seed_t, cont_t], dim=0),
            "onset_times": onset_t,
            "durations": duration_t,
            "new_piece": torch.tensor(True),
        }

    @staticmethod
    def _snap_to_event_boundary(index: int, max_start: int, event_size: int) -> int:
        """Snap index to nearest valid event boundary within bounds."""

        if int(event_size) <= 1:
            return int(max(0, min(index, max_start)))

        idx = int(max(0, min(index, max_start)))
        span = int(event_size)
        lower = idx - (idx % span)
        upper = lower + span
        if upper > max_start:
            return lower
        if (idx - lower) <= (upper - idx):
            return lower
        return upper

    def _resolve_event_size(self) -> int:
        strategy = str(getattr(self.data_config, "tokenization_strategy", "")).lower()
        if strategy == "custom_delta":
            return 4 if int(getattr(self.data_config, "vocab_size", 0)) >= 171 else 3
        return 1

    @staticmethod
    def _load_time_feature_array(
        item: Dict[str, object],
        key: str,
        length: int,
        fallback_step: float,
        is_duration: bool,
    ) -> np.ndarray:
        """Load onset/duration feature array or build safe fallback."""

        path_value = item.get(key)
        if isinstance(path_value, str) and path_value:
            array_path = Path(path_value)
            if array_path.exists():
                try:
                    arr = np.load(array_path)
                    arr = np.asarray(arr, dtype=np.float32)
                    if arr.ndim == 1 and int(arr.shape[0]) == int(length):
                        return arr
                except Exception:
                    pass

        step = float(max(1e-4, fallback_step))
        if is_duration:
            return np.full((length,), fill_value=step, dtype=np.float32)
        return np.arange(length, dtype=np.float32) * step

    @classmethod
    def collate_fn(
        cls,
        batch: List[Dict[str, torch.Tensor]],
        pad_value: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Pad and batch variable-length seed/continuation pairs."""

        seeds = [item["seed"] for item in batch]
        continuations = [item["continuation"] for item in batch]
        token_ids_list = [item["token_ids"] for item in batch]
        onset_list = [item["onset_times"] for item in batch]
        duration_list = [item["durations"] for item in batch]
        new_piece_list = [item["new_piece"] for item in batch]

        max_seed = max(t.shape[0] for t in seeds)
        max_cont = max(t.shape[0] for t in continuations)
        max_total = max(t.shape[0] for t in token_ids_list)

        seed_batch = []
        cont_batch = []
        token_batch = []
        onset_batch = []
        duration_batch = []
        for s, c in zip(seeds, continuations):
            if s.shape[0] < max_seed:
                s = F.pad(s, (0, max_seed - s.shape[0]), value=pad_value)
            if c.shape[0] < max_cont:
                c = F.pad(c, (0, max_cont - c.shape[0]), value=pad_value)
            seed_batch.append(s)
            cont_batch.append(c)

        for token_ids, onsets, durations in zip(
            token_ids_list, onset_list, duration_list
        ):
            if token_ids.shape[0] < max_total:
                token_ids = F.pad(
                    token_ids, (0, max_total - token_ids.shape[0]), value=pad_value
                )
            if onsets.shape[0] < max_total:
                onsets = F.pad(
                    onsets, (0, max_total - onsets.shape[0]), value=float(0.0)
                )
            if durations.shape[0] < max_total:
                durations = F.pad(
                    durations, (0, max_total - durations.shape[0]), value=float(1e-4)
                )
            token_batch.append(token_ids)
            onset_batch.append(onsets)
            duration_batch.append(durations)

        return {
            "seed": torch.stack(seed_batch, dim=0),
            "continuation": torch.stack(cont_batch, dim=0),
            "token_ids": torch.stack(token_batch, dim=0),
            "onset_times": torch.stack(onset_batch, dim=0),
            "durations": torch.stack(duration_batch, dim=0),
            "new_piece": torch.stack(new_piece_list, dim=0).to(dtype=torch.bool),
        }


def create_dataloaders(
    config: DataConfig,
    train_config: TrainConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test dataloaders from processed manifest."""

    manifest_path = Path(config.processed_path) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path.resolve()}. "
            "Run data preprocessing first."
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest: List[Dict[str, object]] = json.load(f)

    if not manifest:
        raise RuntimeError(
            "Manifest is empty. Preprocessing likely failed or filtered everything."
        )

    rng = random.Random(train_config.seed)
    shuffled = list(manifest)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(0.90 * n))
    n_val = max(1, int(0.05 * n))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_manifest = shuffled[:n_train]
    val_manifest = shuffled[n_train : n_train + n_val]
    test_manifest = shuffled[n_train + n_val :]

    data_cfg_dict = asdict(config)
    train_ds = PianoDataset(
        train_manifest,
        DataConfig(**data_cfg_dict),
        seed=train_config.seed,
    )
    val_ds = PianoDataset(
        val_manifest,
        DataConfig(**data_cfg_dict),
        seed=train_config.seed + 1,
    )
    test_ds = PianoDataset(
        test_manifest,
        DataConfig(**data_cfg_dict),
        seed=train_config.seed + 2,
    )

    use_cuda = train_config.device == "cuda" or (
        train_config.device == "auto" and torch.cuda.is_available()
    )
    num_workers = int(getattr(train_config, "_force_num_workers", 2))
    if num_workers < 0:
        num_workers = 0
    persistent_workers = num_workers > 0
    base_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(use_cuda),
        "persistent_workers": persistent_workers,
        "collate_fn": PianoDataset.collate_fn,
        "drop_last": False,
    }
    if num_workers > 0:
        base_loader_kwargs["prefetch_factor"] = 2

    train_sampler = None
    train_shuffle = True
    if bool(config.use_multi_dataset):
        item_weights = _build_per_item_weights(train_manifest, config)
        train_sampler = WeightedRandomSampler(
            weights=list(item_weights),
            num_samples=len(train_manifest),
            replacement=True,
        )
        train_shuffle = False

        source_counts: Dict[str, int] = {}
        for item in train_manifest:
            source = _item_source(item)
            source_counts[source] = source_counts.get(source, 0) + 1
        LOGGER.info("Multi-dataset training enabled.")
        LOGGER.info("  source counts (train): %s", source_counts)
        LOGGER.info("  dataset weights: %s", config.dataset_weights or {})

    train_loader = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        **base_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        **base_loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        **base_loader_kwargs,
    )

    LOGGER.info(
        "Dataset split by piece: train=%d, val=%d, test=%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )

    return train_loader, val_loader, test_loader

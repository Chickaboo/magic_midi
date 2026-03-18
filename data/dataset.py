from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import DataConfig, TrainConfig


def _to_int(value: Any, fallback: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


class PianoDataset(Dataset):
    def __init__(
        self,
        manifest: List[Dict[str, object]],
        data_config: DataConfig,
        seed: int = 42,
    ) -> None:
        self.manifest = list(manifest)
        self.data_config = data_config
        self.rng = random.Random(seed)
        self.min_required = (
            self.data_config.seed_length + self.data_config.continuation_length
        )

        filtered_manifest: List[Dict[str, object]] = []
        for m in self.manifest:
            length_val = m.get("length")
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
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        return cls(manifest=manifest, data_config=data_config, seed=seed)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.manifest[idx]
        tokens_path = Path(str(item["tokens_path"]))
        token_seq = np.load(tokens_path)

        total_needed = (
            self.data_config.seed_length + self.data_config.continuation_length
        )
        if token_seq.shape[0] < total_needed:
            raise RuntimeError(
                f"Piece {tokens_path} shorter than required window {total_needed}."
            )

        max_start = int(token_seq.shape[0] - total_needed)
        start = self.rng.randint(0, max_start) if max_start > 0 else 0

        seed = token_seq[start : start + self.data_config.seed_length]
        cont = token_seq[
            start + self.data_config.seed_length : start
            + self.data_config.seed_length
            + self.data_config.continuation_length
        ]

        seed_t = torch.from_numpy(seed.astype(np.int64, copy=False))
        cont_t = torch.from_numpy(cont.astype(np.int64, copy=False))
        return seed_t, cont_t

    @classmethod
    def collate_fn(
        cls,
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
        pad_value: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seeds, continuations = zip(*batch)
        max_seed = max(t.shape[0] for t in seeds)
        max_cont = max(t.shape[0] for t in continuations)

        seed_batch = []
        cont_batch = []
        for s, c in zip(seeds, continuations):
            if s.shape[0] < max_seed:
                s = F.pad(s, (0, max_seed - s.shape[0]), value=pad_value)
            if c.shape[0] < max_cont:
                c = F.pad(c, (0, max_cont - c.shape[0]), value=pad_value)
            seed_batch.append(s)
            cont_batch.append(c)

        return torch.stack(seed_batch, dim=0), torch.stack(cont_batch, dim=0)


def create_dataloaders(
    config: DataConfig,
    train_config: TrainConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
        train_manifest, DataConfig(**data_cfg_dict), seed=train_config.seed
    )
    val_ds = PianoDataset(
        val_manifest, DataConfig(**data_cfg_dict), seed=train_config.seed + 1
    )
    test_ds = PianoDataset(
        test_manifest, DataConfig(**data_cfg_dict), seed=train_config.seed + 2
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

    train_loader = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        shuffle=True,
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

    print(
        "Dataset split by piece: "
        f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    return train_loader, val_loader, test_loader

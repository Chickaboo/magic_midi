# Training Entry Points

This folder contains production and research training entrypoints.

## Core Layout

- `ddp_common.py`: Shared DDP/runtime helpers (rank detection, checkpoint save/load, resume lookup, reduced metric aggregation).
- `trainer.py`: General trainer utilities used by non-DDP pipelines.
- `losses.py`: Next-token objectives and slot-aware metrics.
- `scheduler.py`: Warmup + cosine scheduler utilities.

## Active DDP Scripts

- `train_variant_e_40m_ddp.py`: Variant E ~40M profile.
- `train_variant_e_100m_ddp.py`: Variant E ~100M profile.
- `train_variant_f_40m_ddp.py`: Variant F ~40M profile.

## Active Unified Script

- `sub100m_unified.py`: Single-runner path for C/E/F sub-100M experiments.

## Legacy

Legacy trainers are archived in `archive/legacy_2026-04-19/training/`.

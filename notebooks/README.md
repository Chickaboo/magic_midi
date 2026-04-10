# Active Notebooks

This folder now contains the active, current notebooks for sub-100M quality validation runs.

## Current notebooks

- 01_t4_sub100m_unified_variant_ce.ipynb
  - Unified Variant C / Variant E / Variant F sub-100M workflow.
  - Backed by `training/sub100m_unified.py` for a single shared training core.
  - Targets fair ~40M architecture comparisons between C, E, and F.
  - Supports larger pilot subsets (default 100k pieces), auto-resume, and NPZ-manifest auto-build.
- 02_kaggle_40m_gdn_variant_e_100k.ipynb
  - Dedicated Kaggle workflow for strict Variant E (Gated Delta + sparse attention anchors).
  - Locks to the 40M architecture profile and 100k-piece budget by default.
  - Adds explicit architecture preflight checks (event-size alignment, real GDN backend, parameter-budget sanity) before training.
  - Uses a dual-T4 DDP path for real-GDN multi-GPU runs (with single-process fallback when only one GPU is available).
- 03_colab_bluebird_generation.ipynb
  - Self-contained Google Colab GPU notebook for Bluebird continuation runs.
  - Uses notebook-defined code, downloads only the private model files/tokenizer from Hugging Face, and uses shorter token budgets for faster turnaround.

## Legacy notebooks

Older notebooks were archived to:

- archive/legacy_2026-04-06/
- archive/legacy_2026-04-04/

You can still open those any time if you need historical references.

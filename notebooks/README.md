# Active Notebooks

This folder contains current training and generation notebooks.

## Current notebooks

- 01_t4_sub100m_unified_variant_ce.ipynb
  - Unified Variant C / Variant E / Variant F workflow for controlled sub-100M comparisons.
- 02_kaggle_40m_gdn_variant_e_100k.ipynb
  - Dedicated 40M Variant E Kaggle baseline run.
- 03_colab_midi_generation.ipynb
  - Colab generation notebook for checkpoint inference workflows.
- 05_kaggle_100m_variant_e_500k_barmeta.ipynb
  - Primary 100M Variant E notebook for 500k-piece training on dual T4.
  - Enforces strict dense backend checks and unified custom-delta tokenizer contract.
  - Drives `training/train_variant_e_100m_ddp.py` with `event_size=4` / `vocab_size=374` settings.

## Legacy notebooks

Archived notebook snapshots are available under `archive/`.

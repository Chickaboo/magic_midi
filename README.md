# Piano MIDI Model (Current Workflow)

This repository is now organized around a dense Variant E training path with one unified CustomDeltaTokenizer contract.

## Active Workflows

- Local and chunked tokenization with unified custom-delta event quads:
  - scripts/tokenize_launcher.ps1 (single shared launcher)
  - token_script_local_500k_barmeta.ps1 (preset wrapper)
  - scripts/tokenize_upload_hf_batches.py
  - scripts/tokenize_godzilla_local.py
- 100M Variant E DDP training:
  - training/train_variant_e_100m_ddp.py
  - notebooks/05_kaggle_100m_variant_e_500k_barmeta.ipynb

## Directory Guides

- data/README.md
- scripts/README.md
- training/README.md

Root tokenization scripts are now wrappers over scripts/tokenize_launcher.ps1:

- token_script_local_100k.ps1
- token_script_local_500k_barmeta.ps1
- token_script.ps1

## Token Contract (Unified)

The only supported tokenizer is `CustomDeltaTokenizer` with event quads:

- Event layout: `[delta_onset, pitch, duration, velocity_bin]`
- Structural prefix: `[Density, Voices, Register, BOS]` before event quads
- Delta token IDs: `0..127` (128 bins, floor quantization)
- Pitch token IDs: `128..215` (88 piano keys)
- Duration token IDs: `216..343` (128 bins, floor quantization)
- Velocity token IDs: `344..359` (16 bins)
- Special/meta token IDs: `PAD=360`, `BOS=361`, `EOS=362`, `Density=363..366`, `Voices=367..370`, `Register=371..373`
- `event_size=4`, `vocab_size=374`

When training from this stream:

- `--tokenization_strategy custom_delta`
- `--event_size 4`
- `--vocab_size 374`

## 500k Storage Planning (Local NPZ)

Practical planning range for compressed NPZ tokenized output:

- average pieces are short-to-medium: about 18-30 GB
- average pieces are medium-to-long: about 30-55 GB
- safety budget (recommended): at least 80 GB free on local SSD/NVMe

Controller temp usage during chunking can add transient overhead, so plan for headroom above final dataset size.

## Archive Policy

Legacy docs and old 150M trainer scripts were moved to:

- archive/legacy_2026-04-19/docs
- archive/legacy_2026-04-19/training

Use archived files for historical reference only; current operational sources are the workflow files listed above.
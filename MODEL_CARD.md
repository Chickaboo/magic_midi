# Itty Bitty Piano - Model Card (Official v2)

## Overview
- Model family: `Itty Bitty Piano`
- Version: `v2` (research architecture)
- Task: piano MIDI continuation (autoregressive next-token modeling)
- Objective: exceed same-size pure transformer baselines via music-specific inductive bias

## What Changed From v1
- Added continuous-time encoding driven by per-token onset times in seconds
- Added dual-stream processing (harmonic Mamba stream + temporal CfC stream)
- Added cross-stream attention every 3 layers (large_v2 preset)
- Added hierarchical phrase summarization before global phrase reasoning
- Added episodic theme memory for long-range thematic callbacks
- Kept v1 model intact for backward compatibility and checkpoint loading

## v2 Architecture

```text
Token IDs + Onset Times
  -> Token Embedding + ContinuousTimeEncoding
  -> DualStreamSplit
       -> Harmonic stream: Mamba blocks
       -> Temporal stream: CfC blocks
       -> CrossStreamAttention (every 3 layers)
  -> Stream merge
  -> PhraseSummarizer (token -> phrase)
  -> EpisodicThemeMemory (read/write)
  -> Sparse phrase attention (ALiBi)
  -> Phrase-to-token broadcast
  -> Final norm + tied output projection
```

## Preset and Parameter Target
- Canonical v2 release preset key: `large_v2` in `scale_config.py`
- Target class: ~100M parameters on real Mamba runtime (Kaggle T4)
- Current local smoke-test measurement (runtime): `108,261,729` parameters (~108.26M)
- Current measured parameter count for v2 preflight: `108.26M`
- Final release calibration command on Kaggle:

```python
from piano_kaggle_session import calibrate_on_kaggle
calibrate_on_kaggle()
```

## Training Data
Trained / intended training corpus (piano-only focus):
- MAESTRO v3.0.0
- GiantMIDI-Piano (curated subset)
- Aria-MIDI (deduped subset)
- ADL Piano MIDI

### Dataset Mixing
- Weighted sampling is supported via `DataConfig.dataset_weights`
- Recommended defaults:
  - `maestro: 1.5`
  - `giant_midi: 1.2`
  - `aria_midi: 1.0`
  - `adl_piano: 1.3`

Dataset rationale:
- `aria_midi=1.0` is the anchor (primary large-scale generative corpus)
- `maestro=1.5` adds high-precision performance timing signal
- `giant_midi=1.2` adds broad classical composer coverage
- `adl_piano=1.3` boosts modern/pop/TV style diversity

### Quality Filters
Applied during preprocessing:
- minimum duration (profile-specific)
- velocity variance threshold
- minimum note count
- minimum distinct pitch count
- piano dominance threshold for multi-instrument files

## Continuous-Time Feature Pipeline
- Manifest includes token-aligned paths for:
  - `onset_times_path`
  - `durations_path`
- Dataloader loads these arrays and emits batch keys:
  - `token_ids`, `onset_times`, `durations`, `new_piece`
- Trainer resets theme memory at piece boundaries when configured

## Inference
- Generation supports both v1 and v2 models through `model.factory.build_model`
- v2 generation uses onset-time progression and memory threading
- v1 generation path remains unchanged for compatibility

## Backward Compatibility
- v1 `PianoHybridModel` remains available
- Legacy checkpoint names are still discoverable in loaders
- Config payload normalization backfills v2-required fields for old checkpoints

## Known Limitations
- Local CPU/fallback runtime underestimates true Mamba-scale parameterization behavior
- Full quality and throughput depend on proper Kaggle GPU runtime + `mamba-ssm`
- Phrase segmentation is fixed-window (`tokens_per_phrase`) in this version

## Future Work
- Variable-length phrase boundaries learned from structure tokens and cadence signals
- Importance-aware memory eviction (not just recency)
- Auxiliary objectives for section/phrase boundary and harmonic forecasting

## Version History
- v1: single-stream hybrid autoregressive model (Mamba + optional CfC + sparse attention)
- v2: dual-stream architecture with continuous-time encoding, phrase summarization, and episodic memory

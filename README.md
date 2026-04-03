# Itty Bitty Piano (v1 + v2)

Itty Bitty Piano trains and serves autoregressive piano continuation models on tokenized MIDI data.

This repo now supports:
- v1/v3 lineage (`PianoHybridModel`) for backward compatibility
- official v2 research architecture (`IttyBittyPianoV2`) with continuous-time + dual-stream + phrase memory

## v2 Architecture (Official)

```text
Token IDs + onset_times(seconds)
  -> token embedding + ContinuousTimeEncoding
  -> DualStreamSplit
       -> harmonic stream (Mamba)
       -> temporal stream (CfC)
       -> CrossStreamAttention every 2 layers
  -> stream merge
  -> PhraseSummarizer
  -> EpisodicThemeMemory
  -> phrase-level sparse attention
  -> phrase-to-token broadcast
  -> final norm + tied output projection
```

Core v2 files:
- `model/hybrid_v2.py`
- `model/time_encoding.py`
- `model/dual_stream.py`
- `model/phrase_memory.py`
- `model/factory.py`

## v1/v3 Architecture

```text
Input tokens
  -> token embedding (+ optional absolute position embedding)
  -> [Mamba block -> CfC block] x N
  -> sparse MusicAttentionBlock every K layers
  -> final LayerNorm
  -> output projection (optionally weight-tied) + output logit scaling
```

Defaults in v3 presets:

- `use_cfc=True`
- `use_mamba=True`
- sparse attention cadence via `attention_every_n_layers`
- ALiBi relative bias for extrapolation

FFN remains available as fallback (`use_cfc=False`) but is no longer the default.

## CfC Configuration

CfC is implemented via `ncps.torch.CfC` with v3-compatible defaults:

- `mode="pure"`
- `batch_first=True`
- `return_sequences=True` when supported by installed `ncps`
- hidden size matches `d_model`
- backbone defaults to 2 layers and hidden size `d_model // 2`

Generation paths retain CfC hidden-state threading across autoregressive steps.

## Scale Presets

Presets live in `scale_config.py`:

- `small` (v1/v3 family)
- `medium` (v1/v3 family)
- `large` (v1/v3 family)
- `large_v2` (official v2 architecture target)

Check current runtime counts:

```bash
python scale_config.py
```

On Kaggle with real `mamba-ssm`, run:

```python
from piano_kaggle_session import calibrate_on_kaggle
calibrate_on_kaggle()
```

## Multi-Dataset Training

v3 supports combining MAESTRO with optional external datasets in one manifest.

Supported dataset sources:

1. MAESTRO v3.0.0
2. GiantMIDI-Piano
3. Aria-MIDI (deduped subset)
4. ADL Piano MIDI

Core controls in `DataConfig`:

- `use_multi_dataset`
- `dataset_paths: Dict[str, str]`
- `dataset_weights: Dict[str, float]`
- `quality_filter_velocity`
- `min_duration_seconds`
- `dataset_profiles` (per-source filter overrides)
- `min_note_count`, `min_distinct_pitches`, `piano_dominance_threshold`
- `use_continuous_time`

### Example multi-dataset config

```python
from config import DataConfig

cfg = DataConfig(
    use_multi_dataset=True,
    dataset_paths={
        "maestro": "/path/to/maestro",
        "giant_midi": "/path/to/giant_midi",
        "piano_e": "/path/to/piano_e",
    },
    dataset_weights={
        "maestro": 2.0,
        "giant_midi": 1.0,
        "piano_e": 1.5,
    },
    quality_filter_velocity=True,
    min_duration_seconds=30.0,
)
```

Preprocessing entrypoints:

- `data.preprocess.preprocess_maestro(cfg)` for MAESTRO-only compatibility
- `data.preprocess.MultiDatasetPreprocessor(cfg).preprocess()` for combined datasets

### Quality filters applied during preprocessing

- minimum token length (`min_piece_length`, default 1200)
- minimum duration (source profile)
- velocity-variance filter (`std(velocity) >= 5`) when source profile enables it
- minimum non-drum note count
- minimum distinct pitch count
- piano dominance threshold for mixed-instrument files

Manifest entries include source + time features (`onset_times_path`, `durations_path`) for continuous-time training.

### Adding a new dataset

1. Add dataset root to `DataConfig.dataset_paths` with a unique source name.
2. Set optional source weight in `DataConfig.dataset_weights`.
3. Keep MIDI files under that root (`.mid` or `.midi`, recursive scan).
4. Re-run preprocessing to rebuild `processed/manifest.json`.

No code changes are required for generic MIDI datasets.

## Checkpoint Rotation (v3)

Trainer retention policy in `training/trainer.py`:

```python
CHECKPOINT_KEEP_POLICY = {
    "always": ["best.safetensors", "best_state.pt", "latest.safetensors", "latest_state.pt"],
    "milestone_every_n": 25,
    "max_total_checkpoints": 8,
}
```

Behavior:

- save tagged checkpoints every 10 epochs (`save_every_n_epochs=10`)
- keep milestone epochs every 25 epochs
- hard-cap model checkpoint files to 8
- rotate before writes
- run emergency rotation when free space is below 3 GB

## Training and Validation Safeguards

- label smoothing in next-token cross-entropy (`label_smoothing=0.1`)
- robust sampler with top-k/top-p, repetition penalty, and minimum-candidate guarantees
- generation health check executed during validation

## Smoke Test

```bash
python smoke_test_architecture.py
```

Smoke tests include:

- attention bias/block shape checks
- CfC on/off forward checks
- CfC generation hidden-state threading check
- checkpoint rotation simulation
- low-disk emergency rotation trigger
- v2 model checks (parameter range, forward/memory, generation health, reset)
- generation health sanity checks
- preset parameter verification

Current local v2 `large_v2` measurement (fallback runtime): `102,884,834` params (~102.9M).

## Kaggle Notebook

Main notebook: `notebooks/00_kaggle_training.ipynb`

- set `SCALE = "large"` for v3 target runs
- use dataset availability check cell before training
- run `run_kaggle_session(scale=SCALE, max_epochs=MAX_EPOCHS)`

## Web Inference App

Flask app lives in `app/`.

Features:

- seed MIDI upload
- checkpoint selection from `app/models/`
- sampling controls (temperature/top-p/top-k/length)
- downloadable generated MIDI

Run locally:

```bash
cd app
./setup.sh   # or setup.bat on Windows
./run.sh     # or run.bat on Windows
```

## A/B/C Architecture Ablation

For your current Godzilla research track, use these architecture labels consistently:

1. `variant_a`: Gated Delta Net + CfC + Attention hybrid (novel)
2. `variant_b`: Transformer + CfC hybrid (novel)
3. `variant_c`: pure attention Transformer baseline (control)

Run ablations in comparable small-model mode (10M-15M range per variant):

```bash
python -m training.ablation_runner \
  --pretokenized_manifest processed/godzilla_tokenized/metadata/manifest.json \
  --pretokenized_root processed/godzilla_tokenized \
  --seed_midi /path/to/seed.mid \
  --output_dir outputs/godzilla_ablation \
  --variants a,b,c \
  --size_mode balanced_small \
  --epochs 3 \
  --batch_size 4
```

`balanced_small` uses per-variant profiles tuned for fair comparison:
- `variant_a`: ~12.29M params
- `variant_b`: ~12.23M params
- `variant_c`: ~11.64M params

If you want one shared shape across all variants, use `--size_mode shared --d_model ... --n_layers ...`.

Notes:
- `variant_c` is the explicit baseline recorded in output metadata.
- Use the same tokenizer/seed split across all variants to keep comparisons fair.

## Local Godzilla Tokenization (No Hugging Face)

Use the local tokenizer runner to produce resumable triplet `.npz` packs and manifests:

```bash
python scripts/tokenize_godzilla_local.py \
  --source /path/to/Godzilla-Piano-MIDI-Dataset-CC-BY-NC-SA.tar.gz \
  --output-root processed/godzilla_tokenized \
  --checkpoint-every 1000 \
  --progress-every 200
```

Key outputs:
- `processed/godzilla_tokenized/data/*.npz`
- `processed/godzilla_tokenized/metadata/manifest.json`
- `processed/godzilla_tokenized/metadata/checkpoint.json`

The script supports resume by default; rerunning with the same source/output continues from the checkpoint.

Before long training runs, generate a readiness audit report:

```bash
python tools/audit_ablation_readiness.py \
  --variants a,b,c \
  --size_mode balanced_small \
  --pretokenized_manifest processed/godzilla_tokenized/metadata/manifest.json \
  --pretokenized_root processed/godzilla_tokenized \
  --output outputs/ablation_audit_report.md
```

## Archived Components

The Hugging Face tokenizer Space implementation has been moved out of active workflow and archived under `archive/hf_tokenizer_space`.

## Notes

- CPU/local runtimes use the torch fallback implementation when `mamba-ssm` is unavailable.
- Kaggle/Colab with compatible CUDA + `mamba-ssm` enables real Mamba kernels.

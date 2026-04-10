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

Active notebooks:

- `notebooks/01_t4_sub100m_unified_variant_ce.ipynb`
- `notebooks/03_colab_bluebird_generation.ipynb`

The unified sub-100M notebook can auto-build a fresh manifest from NPZ files when the dataset does not include `manifest.json`.

Legacy notebooks were moved to:

- `notebooks/archive/legacy_2026-04-06/`
- `notebooks/archive/legacy_2026-04-04/`

## Web Inference App

Flask app lives in `app/`.

Features:

- seed MIDI upload
- checkpoint selection from `app/models/`
- CPU-first local generation from the checkpoint/tokenizer pair you place into `app/models/` and `app/tokenizer/`
- sampling controls (temperature/top-p/top-k/length)
- downloadable generated MIDI
- rendered seed/output audio previews
- pianoroll comparison image for the generated continuation

App asset drop-in files:

- `app/models/latest.safetensors`
- `app/models/latest_state.pt`
- `app/tokenizer/custom_tokenizer.json`

Run locally:

```bash
cd app
./setup.sh   # or setup.bat on Windows
./run.sh     # or run.bat on Windows
```

## A/B/C/D/E/F Architecture Ablation

For your current Godzilla research track, use these architecture labels consistently:

1. `variant_a`: Gated Delta Net + CfC + Attention hybrid (novel)
2. `variant_b`: Transformer + CfC hybrid (novel)
3. `variant_c`: pure attention Transformer baseline (control)
4. `variant_d`: pure CfC recurrent baseline (no attention)
5. `variant_e`: Gated Delta Net + sparse attention (no CfC)
6. `variant_f`: event-hierarchical tri-path hybrid (harmonic + temporal + structural)

Run ablations in comparable small-model mode (10M-15M range per variant):

```bash
python -m training.ablation_runner \
  --pretokenized_manifest processed/godzilla_tokenized/metadata/manifest.json \
  --pretokenized_root processed/godzilla_tokenized \
  --skip_generation \
  --output_dir outputs/godzilla_ablation \
  --variants a,b,c,d,e,f \
  --size_mode balanced_small \
  --epochs 3 \
  --batch_size 4
```

`balanced_small` uses per-variant profiles tuned for fair comparison in the 10M-15M range.

Notes:
- `--skip_generation` is the recommended mode for tokenized-only datasets that do not include raw `.mid` files.
- If you want continuation demos, remove `--skip_generation` and provide `--seed_midi /path/to/seed.mid`.
- `variant_a` is auto-tuned at runtime to stay comparable whether real GDN kernels are available or fallback mode is active.
- `variant_b` and `variant_c` stay near ~12M as fixed anchors for comparison.
- `variant_d` is the pure-CfC baseline (balanced profile targets ~12M for fair comparison).
- `variant_e` is auto-tuned at runtime in balanced mode and uses sparse attention anchors (every 2 layers, always final layer) without any CfC blocks.
- `variant_f` is auto-tuned at runtime in balanced mode and adds tri-path fusion with phrase memory under the same quad-event tokenizer contract.
- On multi-GPU Kaggle runs, Trainer auto-disables DataParallel when real Variant-A GDN kernels are detected to avoid Triton replica autotuner crashes.

To directly test the current champion (`variant_c`) against both GDN-family contenders (`variant_e`, `variant_f`):

```bash
python -m training.ablation_runner \
  --pretokenized_manifest processed/godzilla_tokenized/metadata/manifest.json \
  --pretokenized_root processed/godzilla_tokenized \
  --skip_generation \
  --output_dir outputs/godzilla_c_vs_ef \
  --variants c,e,f \
  --size_mode balanced_small \
  --epochs 3 \
  --batch_size 4
```

If you want one shared shape across all variants, use `--size_mode shared --d_model ... --n_layers ...`.

Notes:
- `variant_c` is the explicit baseline recorded in output metadata.
- Use the same tokenizer/seed split across all variants to keep comparisons fair.

## Local Godzilla Tokenization (No Hugging Face)

Use the local tokenizer runner to produce resumable event-quad `.npz` packs and manifests.

The tokenizer is frozen at spec version 1 and matches the exact event-quad tokenizer used by the 150M training runs and the 40M / 100k-piece sub-100M runs. Treat this as a one-time corpus pass: do not change token IDs, bin boundaries, or event size after tokenization starts.

For code integrations, use the unified tokenizer module surface:
- `from data.tokenizer import create_tokenizer, load_tokenizer`

```bash
python scripts/tokenize_godzilla_local.py \
  --source /path/to/Godzilla-Piano-MIDI-Dataset-CC-BY-NC-SA.tar.gz \
  --output-root processed/godzilla_tokenized \
  --workers 0 \
  --checkpoint-every 1000 \
  --progress-every 200
```

Example SSD-to-SSD tokenization run:

```bash
python scripts/tokenize_godzilla_local.py \
  --source "E:/datasets/godzilla-midi" \
  --output-root "F:/tokenized/godzilla_full" \
  --workers 0 \
  --checkpoint-every 1000 \
  --progress-every 200
```

Performance notes:
- Default output is uncompressed `.npz` for maximum throughput.
- Use `--compress-output` only if you need smaller files and can tolerate slower tokenization.
- If you already ran older versions against a `.tar/.tar.gz`, run once with `--start-over` to rebuild a consistent archive-order manifest.

Key outputs:
- `processed/godzilla_tokenized/data/*.npz`
- `processed/godzilla_tokenized/metadata/manifest.json`
- `processed/godzilla_tokenized/metadata/checkpoint.json`

The script supports resume by default; rerunning with the same source/output continues from the checkpoint.

### Stream Tokenization to Hugging Face in Batches

If local storage is tight, run chunked tokenization to a temporary flash-drive folder,
upload each chunk to a Hugging Face **dataset** repo, then delete the local chunk and continue.

This workflow is implemented by:

- `scripts/tokenize_upload_hf_batches.py`

Recommended pattern (Windows PowerShell):

```powershell
$env:HF_TOKEN = "<your_hf_token>"
python scripts/tokenize_upload_hf_batches.py \
  --source "C:/datasets/godzilla-midi" \
  --flash-root "F:/pulse88_tokenize_work" \
  --repo-id "Chickaboo/Pulse88-data" \
  --upload-prefix "tokenized/chunks" \
  --chunk-members 100000 \
  --workers 0 \
  --checkpoint-every 2000 \
  --progress-every 500
```

Notes:

- The controller state is stored at `F:/pulse88_tokenize_work/_controller/state.json`.
- Rerun the same command to resume from the saved `next_index`.
- Use `--reset-state` to restart from a fresh index window.
- Add `--allow-mixed-instruments` only if you intentionally want non-piano files.
- Add `--keep-local` to keep each uploaded chunk on disk instead of deleting it.

### Upload Tokenized Output to Kaggle

From your local machine, package and publish your tokenized folder as a Kaggle dataset:

```bash
kaggle datasets init -p F:/tokenized/godzilla_full
# edit F:/tokenized/godzilla_full/dataset-metadata.json
kaggle datasets create -p F:/tokenized/godzilla_full
```

For later refreshes after re-tokenization:

```bash
kaggle datasets version -p F:/tokenized/godzilla_full -m "update tokenized corpus"
```

Keep both `data/` and `metadata/` in that Kaggle dataset so manifest-relative `.npz` paths resolve correctly.

## Variant C/E/F Sub-100M Quality Validation (Kaggle)

Use the unified notebook for sub-100M architecture validation:

- `notebooks/01_t4_sub100m_unified_variant_ce.ipynb`

This notebook replaces the retired split scripts:

- `training/train_variant_c_sub100m.py`
- `training/train_variant_e_sub100m.py`
- `training/retrain_sub100m_matrix.py`

Unified ~40M profile targets:

- Variant C: `d_model=512`, `n_layers=12`, `num_attention_heads=8`, `ffn_expansion=4` (~38.94M)
- Variant E: `d_model=640`, `n_layers=13`, `attention_every_n_layers=2`, GDN inner ratio `0.5` (~40M target)
- Variant F: `d_model=544`, `n_layers=9`, tri-path ratios `0.40/0.30/0.30`, `cross_stream_every_n_layers=2` (~40M target)

These are target-budget profiles. Realized parameter counts can differ in fallback-kernel environments versus real-kernel CUDA runs.

Recommended workflow for your larger pilot (100k pieces):

1. Open `notebooks/01_t4_sub100m_unified_variant_ce.ipynb`.
2. Set `VARIANT` to `"c"`, `"e"`, or `"f"`.
3. Keep `MAX_PIECES = 100_000` (or adjust).
4. Set `RUN_TRAINING = True`.
5. Run all cells top-to-bottom.

The notebook keeps:

- automatic NPZ manifest creation
- checkpoint auto-resume (`AUTO_RESUME` + optional `RESUME_FROM_CHECKPOINT`)
- generation controls and result JSON export
- identical preprocessing/training pipeline for all three variants to keep comparisons fair

Optional CLI (same unified backend used by the notebook):

```bash
python -m training.sub100m_unified \
  --variant f \
  --npz_root /kaggle/input/<your-tokenized-dataset>/data \
  --output_dir /kaggle/working/sub100m_f_100k \
  --max_pieces 100000 \
  --epochs 20 \
  --batch_size 1 \
  --grad_accumulation_steps 32 \
  --auto_resume
```

## Variant C/E 150M Production Training (Kaggle)

Use dedicated runners for large-scale quality comparisons between the two finalists:

- `training/train_variant_c_150m.py`
- `training/train_variant_e_150m.py`

### Variant C (~149.4M, depth-fair)

```bash
python -m training.train_variant_c_150m \
  --pretokenized_manifest /kaggle/input/<your-tokenized-dataset>/metadata/manifest.json \
  --pretokenized_root /kaggle/input/<your-tokenized-dataset> \
  --output_dir /kaggle/working/variant_c_150m \
  --epochs 100 \
  --batch_size 1 \
  --grad_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --seed 42
```

Default profile in `training/train_variant_c_150m.py`:
- Variant: `variant_c`
- Shape: `d_model=784`, `n_layers=20`, `num_attention_heads=8`
- Expected params: about `149.4M` (`149,399,824`)
- Closest valid depth-range match to the original E target (`149,822,400`) with `n_layers` in `[14, 20]`
- DataParallel: enabled by default when multiple GPUs are available

### Variant E (~151.2M after vocab/context expansion)

```bash
python -m training.train_variant_e_150m \
  --pretokenized_manifest /kaggle/input/<your-tokenized-dataset>/metadata/manifest.json \
  --pretokenized_root /kaggle/input/<your-tokenized-dataset> \
  --output_dir /kaggle/working/variant_e_150m \
  --epochs 100 \
  --batch_size 1 \
  --grad_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --seed 42
```

Default profile in `training/train_variant_e_150m.py`:
- Variant: `variant_e` (no CfC)
- Shape: `d_model=1344`, `n_layers=17`
- Expected params: about `151.2M` (`151,220,160`) with `vocab_size=171` and `max_sequence_length=2048`
- Attention cadence: every 2 layers (plus final anchor by construction)
- Real GDN kernels required by default (use `--allow_fallback_gdn` only for debugging)

### Token Format (Quad)

Tokenizer contract is now event-quads:
- Event layout: `[delta_onset, pitch, duration, velocity_bin]`
- Velocity bins: `0-15` mapped to token IDs `152-167`
- Special IDs: `PAD=168`, `BOS=169`, `EOS=170`
- Vocabulary: `171`

### Kaggle Hardware Guidance

- Variant C: prefer `2xT4` to use DataParallel throughput.
- Variant E: prefer `P100` when running real GDN with default stability guard (single-GPU mode).
- Use `2xT4` for E only if you intentionally override and validate DataParallel stability (`--allow_gdn_data_parallel`).

Quickly benchmark current runtime throughput before long jobs:

```bash
python tools/benchmark_kaggle_throughput.py \
  --variant c \
  --d_model 784 \
  --n_layers 20 \
  --seq_len 2048 \
  --batch_size 1 \
  --steps 20 \
  --use_data_parallel
```

```bash
python tools/benchmark_kaggle_throughput.py \
  --variant e \
  --d_model 1344 \
  --n_layers 17 \
  --seq_len 2048 \
  --batch_size 1 \
  --steps 20
```

Compare `tokens_per_sec` across P100 and 2xT4 sessions, then choose hardware before starting 100-epoch runs.

### Context and Long-Form Generation

Both production runners default to 2K training context:
- `--seed_length 512`
- `--continuation_length 1536`
- `--max_sequence_length 2048`

You can still override context via:
- `--seed_length`
- `--continuation_length`
- `--max_sequence_length`

Examples:
- 2K context: `--seed_length 512 --continuation_length 1536`
- 4K context: `--seed_length 1024 --continuation_length 3072`

Both runners also support longer sample export:
- `--generation_max_new_tokens 8192`
- `--generation_continuation_seconds 120`

VRAM note for 2K context:
- 2K roughly doubles activation memory versus 1K.
- Keep `batch_size=1`; reduce effective batch by lowering `grad_accumulation_steps` only if you hit wall-clock limits.
- On OOM, first reduce `d_model`/`n_layers` only as a last resort, since that changes the quality comparison.

To export one sample MIDI after training, add:

```bash
--seed_midi /kaggle/input/<seed-dataset>/seed.mid
```

Before long runs, verify shape/forward with shared-size audits:

```bash
python tools/audit_ablation_readiness.py \
  --variants c \
  --size_mode shared \
  --d_model 1248 \
  --n_layers 8 \
  --pretokenized_manifest /kaggle/input/<your-tokenized-dataset>/metadata/manifest.json \
  --pretokenized_root /kaggle/input/<your-tokenized-dataset> \
  --output /kaggle/working/ablation_audit_variant_c_150m.md
```

```bash
python tools/audit_ablation_readiness.py \
  --variants e \
  --size_mode shared \
  --d_model 1344 \
  --n_layers 17 \
  --pretokenized_manifest /kaggle/input/<your-tokenized-dataset>/metadata/manifest.json \
  --pretokenized_root /kaggle/input/<your-tokenized-dataset> \
  --output /kaggle/working/ablation_audit_variant_e_150m.md
```

Before long training runs, generate a readiness audit report:

```bash
python tools/audit_ablation_readiness.py \
  --variants a,b,c,d,e,f \
  --size_mode balanced_small \
  --pretokenized_manifest processed/godzilla_tokenized/metadata/manifest.json \
  --pretokenized_root processed/godzilla_tokenized \
  --output outputs/ablation_audit_report.md
```

## Notes

- CPU/local runtimes use the torch fallback implementation when `mamba-ssm` is unavailable.
- Kaggle/Colab with compatible CUDA + `mamba-ssm` enables real Mamba kernels.

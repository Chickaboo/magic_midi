# Itty Bitty Piano v2 (Mamba + FFN + Sparse Music Attention)

This repository trains and serves an autoregressive piano continuation model for MIDI token sequences under the Itty Bitty Piano project.

v2 upgrades the architecture and generation stack to eliminate the v1 collapse mode where one token dominated generation regardless of seed.

## What Changed in v2

- Architecture: `Mamba -> FFN` blocks with sparse `MusicAttentionBlock`.
- Attention bias: ALiBi-style relative bias support for extrapolation to longer contexts.
- Generation: robust sampling pipeline with repetition penalty, minimum candidate guarantees, and confidence capping.
- Training: label smoothing (`epsilon=0.1`) in next-token cross-entropy.
- Validation: generation health check logs collapse risk every epoch.
- Tokenization: configurable strategy (`remi` or `octuple`) for better token efficiency experiments.

## Root Cause (v1 collapse)

The collapse was not caused by top-k/top-p filtering. It was caused by extreme output confidence from tied embeddings and large unscaled logits under teacher forcing.

Key findings:

- Random-weight v1 also collapsed in generation (top-1 probability saturated near 1.0).
- Collapse persisted even with CfC hidden-state threading fixes.
- Logits remained numerically large (`std ~10+`) and peaked heavily.

v2 addresses this by scaling output logits (`1/sqrt(d_model)` by default), adding label smoothing, adding repetition-aware sampling safeguards, and introducing a generation health check.

## Architecture

```text
Input tokens
  -> token embedding (+ optional absolute position embedding)
  -> [Mamba block -> FFN block] x N
  -> sparse MusicAttentionBlock every K layers (causal, ALiBi/learned bias)
  -> final LayerNorm
  -> output projection (optionally weight-tied) + output logit scaling
```

CfC is still supported for compatibility (`use_cfc=True`) but v2 presets default to `use_cfc=False`.

## Scale Presets (v2 targets)

- `small`: ~15M (primary target)
- `medium`: ~40M (stretch target)
- `large`: ~100M (future target)

Run parameter checks:

```bash
python scale_config.py
```

## Tokenization

`DataConfig.tokenization_strategy`:

- `"remi"` (default)
- `"octuple"`

Tokenizer is created accordingly in preprocessing.

## Training

Core loss:

```python
F.cross_entropy(logits, targets, ignore_index=-100, label_smoothing=0.1)
```

Generation health check (run during validation):

- generates 20 steps from a fixed validation seed,
- asserts/logs max top-1 probability (`<= 0.95` after sampling constraints),
- logs candidate set minimum and raw-vs-final confidence.

## Generation Safeguards

`PianoHybridModel.generate(...)` now includes:

- repetition penalty over recent 64 tokens,
- temperature floor at 0.1,
- top-k/top-p filtering with minimum tokens kept (`>=3`),
- confidence cap in final sampling distribution,
- warning if >60% of generated tokens are identical.

## Smoke Test

```bash
python smoke_test_architecture.py
```

This covers:

- ALiBi bias and attention block shape checks,
- forward pass for CfC-off and CfC-on modes,
- random-weight generation health check,
- v1 checkpoint generation report with new sampler,
- preset parameter verification on current runtime.

## Web Inference App

The repository now includes a rebuilt Flask web app at `app/`.

Features:

- drag/drop MIDI seed upload,
- checkpoint selector from `app/models/`,
- generation controls (temperature, length, top-p, top-k),
- direct use of model `generate()` and shared sampling logic in `model/sampling.py`,
- downloadable output MIDI,
- CPU-compatible (GRU fallback, no mamba-ssm required).

Setup / run:

```bash
cd app
./setup.sh   # or setup.bat on Windows
./run.sh     # or run.bat on Windows
```

Place files before launch:

- tokenizer: `app/tokenizer/tokenizer.json`
- checkpoints: `app/models/*.safetensors` or `app/models/*.pt`

## Kaggle / Session Helpers

- `session.py` for long-running session orchestration.
- `piano_kaggle_session.py` for Kaggle runtime setup, resume, and generation.
- checkpoints include full model/data/train configs for reproducible reconstruction.

## Notes

- On CPU/local environments, `mamba-ssm` is typically unavailable; the model uses a causal GRU fallback implementation for compatibility.
- On CUDA runtimes with `mamba-ssm` installed, real Mamba kernels are used.

---
language:
- en
tags:
- music
- midi
- piano
- gdn
- grouped-query-attention
- autoregressive
license: cc-by-nc-sa-4.0
library_name: pytorch
datasets:
- projectlosangeles/Godzilla-MIDI-Dataset
---

# Pulse88-E-40M-Alpha-Preview (Variant E)

## Overview
Pulse88-E-40M-Alpha-Preview is a decoder-only autoregressive piano MIDI continuation model. The Variant E design combines stacked Gated Delta Network blocks with sparse Grouped-Query Attention anchor layers, targeting long-context symbolic music continuation with lower attention overhead than dense-attention-only baselines.

## Model Details
- Architecture family: Variant E (GDN + sparse GQA, no CfC)
- Task: Piano MIDI continuation (next-token autoregressive generation)
- Tokenization: Custom event-quad representation (delta, pitch, duration, velocity)
- Vocabulary size: 171 tokens
- Context window: 2048 tokens (512 seed + 1536 continuation)

## Architecture (Code-Grounded)
- Model width/depth profile: d_model=640, n_layers=13
- Block structure: pre-norm residual stack of GDN -> GDN -> optional GQA
- Attention anchor schedule: every 2 layers plus final layer
- 13-layer anchor positions: 2, 4, 6, 8, 10, 12, 13
- GDN profile: inner_dim=320, heads=4
- GQA profile: query heads=8, groups=4 (effective kv_heads=2)
- Embedding/output: tied token embedding and output projection
- Logit scaling: 1/sqrt(d_model) when output_logit_scale is not explicitly set

## Parameter Count Note
This profile is treated as a 40M-class architecture in strict training preflight (accepted range: 38M-42M with real GDN kernels enabled). If fallback kernels are used instead of real flash-linear-attention GDN kernels, observed parameter counts and behavior can differ from strict-run expectations.

## Training Configuration (Kaggle Notebook Defaults)
- Max pieces: 100,000 (run may use fewer if manifest has fewer entries)
- Epochs: 20
- Target effective batch: 32
- Per-GPU batch: 2
- Learning rate: 2e-4
- Warmup: ratio-based (0.1) with dynamic step resolution
- Weight decay: 0.01
- Label smoothing: 0.1
- Max grad norm: 1.0
- Step checkpoint cadence: every 1000 optimizer steps
- Multi-GPU mode: DDP path enabled when 2+ GPUs are present and strict real GDN is active

### Reported Run Setup
The referenced run was conducted on dual T4 Kaggle hardware and resumed across two sessions.

## Generation Configuration (Notebook)
- max_new_tokens: continuation_length from checkpoint metadata
- temperature: 0.9
- top_p: 0.95
- top_k: 50
- repetition_penalty: 1.1
- repetition_window: 64
- min_tokens_to_keep: 3
- num_samples: 1
- max_consecutive_zero_deltas: 12
- save_continuation_only: true

## Data
Training uses pretokenized piano MIDI manifests from a Godzilla-based tokenized dataset workflow. The notebook default points to 100k-piece tokenized roots and warns if fewer entries are available.

## Intended Use
- Research and experimentation in symbolic piano continuation
- Evaluation of GDN + sparse attention design choices

## Out-of-Scope Use
- Human-composed authorship attribution
- Production music publishing without human review/curation
- Claims of legal clearance for generated musical works

## Limitations
- Strict architecture behavior requires real GDN kernels (flash-linear-attention)
- Fallback-kernel execution can change quality/performance and may not match strict-run expectations
- No claim of state-of-the-art across all symbolic music benchmarks
- No guarantee of style safety, plagiarism filtering, or compositional correctness

## Warranty
This model is provided for research purposes only, without warranties, guarantees, or fitness-for-purpose commitments.

## Citation and Credits
If you use this model, credit the underlying dataset source.

### Dataset Creator
- Alex Lev (Project Los Angeles / Tegridy Code)

### Dataset Citation
```bibtex
@misc{GodzillaMIDIDataset2025,
  title        = {Godzilla MIDI Dataset: Enormous, comprehensive, normalized and searchable MIDI dataset for MIR and symbolic music AI purposes},
  author       = {Alex Lev},
  publisher    = {Project Los Angeles / Tegridy Code},
  year         = {2025},
  url          = {https://huggingface.co/datasets/projectlosangeles/Godzilla-MIDI-Dataset}
}
```

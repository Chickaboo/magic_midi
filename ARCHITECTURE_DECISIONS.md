# Itty Bitty Piano v2 - Architecture Decisions

## 1) Additive v2 Instead of Replacing v1
- Decision: implement v2 as new modules (`hybrid_v2.py`, supporting blocks) while keeping `hybrid.py` intact.
- Why: preserves checkpoint/tooling compatibility and allows direct A/B evaluation.

## 2) Continuous-Time Encoding Uses Onset Seconds
- Decision: add `ContinuousTimeEncoding` with multi-timescale sinusoidal features projected to `d_model`.
- Why: musical timing is continuous; token index alone loses tempo/duration context.

## 3) Tokenizer Emits Time-Aligned Arrays
- Decision: extend tokenizer preprocessing path to produce `(token_ids, onset_times, durations)` and persist arrays per piece.
- Why: v2 requires direct onset-time supervision and duration priors at train time.

## 4) Dual-Stream Separation
- Decision: split into harmonic and temporal streams via learned projections (`DualStreamSplit`).
- Why: pitch/harmony and expressive timing are coupled but structurally distinct.

## 5) Stream Backbones
- Decision: harmonic stream uses Mamba blocks; temporal stream uses CfC blocks.
- Why: Mamba state selectivity suits harmonic motif retention; CfC continuous-time dynamics suit timing/expression.

## 6) Cross-Stream Fusion Cadence
- Decision: run cross-stream attention every `cross_stream_every_n_layers` (default 2).
- Why: periodic fusion balances specialization with necessary interaction between harmonic and temporal cues.

## 7) Phrase-Level Hierarchy
- Decision: summarize token stream into phrase representations with attentive pooling (`PhraseSummarizer`).
- Why: phrase-level reasoning extends effective structural context at lower attention cost.

## 8) Episodic Theme Memory
- Decision: add differentiable read/write memory over phrase representations (`EpisodicThemeMemory`).
- Why: enables long-range callbacks and thematic reuse beyond local context windows.

## 9) Memory Lifecycle in Trainer
- Decision: trainer tracks `_memory_state`, resets on `new_piece` when configured, and detaches between steps.
- Why: prevents cross-piece leakage and avoids unbounded graph growth.

## 10) Unified Model Factory
- Decision: introduce `model.factory.build_model(config)` for v1/v2 routing.
- Why: centralizes model selection and keeps entrypoints (session, kaggle, tools, app) consistent.

## 11) Four-Dataset Profile Defaults
- Decision: add per-source profile defaults (duration/filter policy + sample weights) for MAESTRO, GiantMIDI, Aria-MIDI, ADL.
- Why: datasets have different quality/style distributions and should not be filtered/mixed identically.

## 12) v2 Large Preset as Separate Key
- Decision: add `large_v2` preset instead of repurposing existing `large` key.
- Why: avoids breaking existing v3 runs while providing explicit official v2 configuration.

# Piano MIDI Hybrid CfC + Mamba Workspace

This repository provides a full research workspace for piano MIDI continuation with a **hybrid CfC + Mamba** architecture.

Hypothesis: adding **CfC (Closed-form Continuous-time)** dynamics on top of sequence modeling layers improves temporal and timing consistency over a pure Mamba baseline for long-form piano generation.

The core task is: given a ~10-second seed clip, generate a musically coherent continuation.

## Architecture

```text
Input MIDI -> REMI+BPE tokens
              |
              v
      Token Embedding
             |
             v
 Positional Encoding (learned)
             |
             v
   +-------------------------+
   | Mamba Block             |  <- sequence compression, O(n)
   | CfC Block               |  <- temporal dynamics
   +-------------------------+
             |
             v (every N layers)
   +-------------------------+
   | Music Attention Block   |  <- relative position bias
   | + FFN                   |  <- in-context retrieval, motif reference
   +-------------------------+
             |
             v (repeat)
       Final LayerNorm
             |
             v
 Output Projection (weight-tied to embedding)
```

The hybrid stack alternates Mamba+CfC groups with sparse music attention blocks (`attention_every_n_layers`). Attention uses learned **relative position bias**, not absolute-position attention biasing, so rhythmic distance relationships can transfer across different piece lengths.

## Scale Presets

- `nano`: ~3M parameters (pipeline validation and quick Colab checks)
- `micro`: ~8M parameters (meaningful learning in short sessions)
- `small`: ~22M parameters (full architecture, roughly one epoch per free-tier session)
- `medium`: ~60M parameters (serious training; Colab Pro/Kaggle recommended)

Use `python scale_config.py` (or `verify_preset_params()`) to print measured parameter counts for each preset.

## Attention Design Choice

- Relative position bias is used for attention because musical structure depends more on **distance** (beat/bar offsets, phrase spacing) than absolute token index.
- Sparse attention insertion (`every N layers`) keeps the model efficient while still improving in-context motif retrieval and continuation coherence.
- This follows evidence from music generation and hybrid SSM+attention systems that a modest fraction of attention layers is often enough.

References:
- Cheng-Zhi Anna Huang et al., "Music Transformer: Generating Music with Long-Term Structure," ICLR 2019.
- Joseph Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model," 2024.

## Project Layout

The workspace is organized into focused modules for tokenization, preprocessing, model variants, training, generation, and evaluation. See the folder tree in this repository for full details.

## Setup

### Local CPU (debugging + preprocessing)

1. Create an environment (Python 3.10+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place MAESTRO under `maestro-v3.0.0/` or update `DataConfig.maestro_path`.

### Google Colab (training)

1. Open the notebooks from this repo.
2. Use runtime with GPU (T4 recommended).
3. Install Colab dependencies:

```bash
pip install -r requirements_colab.txt
```

4. Mount Google Drive and set paths to `/content/drive/MyDrive/piano_model/`.

Notes:
- `mamba-ssm` requires CUDA and may fail on CPU.
- CPU mode automatically uses the GRU-based Mamba fallback.

## Notebook Order

1. `notebooks/01_data_pipeline.ipynb`
2. `notebooks/02_baseline_training.ipynb`
3. `notebooks/03_mamba_training.ipynb`
4. `notebooks/04_hybrid_training.ipynb`
5. `notebooks/05_generation_and_eval.ipynb`

## Colab Sync via GitHub (recommended)

Instead of uploading zip files, use GitHub for instant sync:

### First time setup (local):

```bash
cd piano_midi_model
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOURUSERNAME/piano_midi_model.git
git push -u origin main
```

### Add to top of notebook 00 cell 1:

```python
import os
if not os.path.exists('/content/piano_midi_model'):
    !git clone https://github.com/YOURUSERNAME/piano_midi_model.git /content/piano_midi_model
else:
    !cd /content/piano_midi_model && git pull origin main
    print("Repository updated to latest version")
```

### After agents make changes:

```bash
git add .
git commit -m "describe what changed"
git push
```

Then run notebook 00 and it pulls automatically.

## Evaluation Metrics Guide

- `pitch_class_cosine`: key consistency between seed and continuation (higher is better).
- `pitch_class_entropy`: tonal spread / chromaticity (too high can indicate unstable tonality).
- `note_density`: notes per second (checks texture continuity).
- `rhythmic_regularity` (IOI coefficient of variation): lower means more regular pulse.
- `mean_velocity_ratio`: dynamic consistency between seed and continuation.

The primary comparison is seed-vs-continuation consistency across the three model families:
- Baseline GRU
- Mamba-only (CfC ablated)
- Hybrid Mamba+CfC

## Known Limitations

- Absolute learned token embeddings are still present in the base stack and may degrade quality far beyond configured context length.
- MIDI-only objective does not directly optimize perceptual audio quality.
- Token-level next-token loss can still miss long-horizon motif structure.
- Mamba fallback is for compatibility/debugging, not architecture equivalence.

## Next Steps

- Add multi-objective losses for rhythm and motif consistency.
- Extend evaluation with structural repetition and cadence metrics.
- Add human listening tests and blind A/B ranking.

## References

- MAESTRO dataset:
  - Curtis Hawthorne et al., "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset," ICLR 2019.
- Mamba:
  - Albert Gu and Tri Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2023.
- CfC / liquid neural dynamics:
  - Ramin Hasani et al., "Closed-form Continuous-time Neural Networks," Nature Machine Intelligence, 2022.

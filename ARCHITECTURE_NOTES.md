# Ablation Architecture Notes (A/B/C)

This document records runtime-measured parameter counts and block-level breakdowns for the three ablation variants at `d_model=512`, `n_layers=4`, `vocab_size=155`, `max_sequence_length=1024`.

## How Counts Were Measured (Programmatic, Not Estimated)

Counts were obtained by instantiating each model in Python and summing `p.numel()` over parameters (including submodule-level sums for per-component attribution).

```python
import json
from model.variant_a import VariantAModel, VariantAConfig
from model.variant_b import VariantBModel, VariantBConfig
from model.variant_c import VariantCModel, VariantCConfig

def count_params(module):
    return sum(p.numel() for p in module.parameters())

models = {
    "variant_a": VariantAModel(VariantAConfig(d_model=512, n_layers=4, vocab_size=155)),
    "variant_b": VariantBModel(VariantBConfig(d_model=512, n_layers=4, vocab_size=155)),
    "variant_c": VariantCModel(VariantCConfig(d_model=512, n_layers=4, vocab_size=155)),
}

print({name: count_params(model) for name, model in models.items()})
```

Measured totals on this runtime:

- `variant_a`: `14,208,512`
- `variant_b`: `12,103,168`
- `variant_c`: `13,206,016`

## Variant A (GDN -> GDN -> CfC -> GQA)

Total params: `14,208,512`

- Embeddings: `603,648`
  - token embedding: `79,360`
  - position embedding: `524,288`
- Repeating block stack (4 blocks): `13,603,840`
  - Per block: `3,400,960`
    - GDN #1 path (LayerNorm + block): `459,776`
    - GDN #2 path (LayerNorm + block): `459,776`
    - CfC path (LayerNorm + block): `1,825,024`
    - GQA path (LayerNorm + block): `656,384`
- Final LayerNorm: `1,024`
- LM head: `0` additional params (weight tied to token embedding)

## Variant B (Attn+RoPE -> CfC)

Total params: `12,103,168`

- Embeddings: `603,648`
  - token embedding: `79,360`
  - position embedding: `524,288`
- Repeating block stack (4 blocks): `11,498,496`
  - Per block: `2,874,624`
    - Attention path (LayerNorm + MHA+RoPE): `1,049,600`
    - CfC path (LayerNorm + block): `1,825,024`
- Final LayerNorm: `1,024`
- LM head: `0` additional params (weight tied to token embedding)

## Variant C (Attn+RoPE -> FFN 4x GELU)

Total params: `13,206,016`

- Embeddings: `603,648`
  - token embedding: `79,360`
  - position embedding: `524,288`
- Repeating block stack (4 blocks): `12,601,344`
  - Per block: `3,150,336`
    - Attention path (LayerNorm + MHA+RoPE): `1,049,600`
    - FFN path (LayerNorm + 4x GELU MLP): `2,100,736`
- Final LayerNorm: `1,024`
- LM head: `0` additional params (weight tied to token embedding)

## Novel vs Standard Components

Novel/ablation-specific pieces:

- Fixed triplet-token regime with slot-constrained generation (`delta`, `pitch`, `duration` slots).
- CfC used with explicit elapsed-time (`delta onset`) timespans instead of absolute onsets.
- GDN integration in Variant A (`GatedDeltaNetBlock` wrapper).

Standard components:

- Learned token and position embeddings.
- Pre-LN residual transformer-style block structure.
- Causal multi-head attention with RoPE.
- FFN 4x GELU baseline block (Variant C).
- Tied LM head.

## FLA / GDN Fallback Note

`model/blocks/gdn_block.py` attempts to import `fla.layers.gated_deltanet.GatedDeltaNet`. If unavailable, it uses `_GatedDeltaFallback` (a gated MLP approximation) while preserving `(B,S,D)->(B,S,D)` behavior.

In this runtime, `flash-linear-attention` is unavailable, so Variant A uses the fallback path for both GDN sublayers in every block.

## Scientific Rationale

Variant C (control): provides a strong standard baseline (attention + FFN) with no recurrent continuous-time dynamics. It tests what can be achieved with conventional autoregressive modeling under identical tokenization, optimizer, and schedule.

Variant B (CfC hypothesis): isolates the effect of adding continuous-time recurrent dynamics via CfC on top of attention. If elapsed-time-aware CfC improves prediction quality over Variant C, this supports the hypothesis that explicit temporal dynamics matter beyond static FFN mixing.

Variant A (GDN + CfC hypothesis): adds two GDN sublayers before CfC+GQA to test whether delta-state gating improves local dynamics and conditioning efficiency beyond Variant B. Improvement over both B and C suggests extra value from gated delta-state processing, not just CfC alone.

## Evaluation Criteria (Evidence Thresholds)

Primary metric: validation cross-entropy (`val_loss`), with perplexity as a secondary view.

Suggested evidence thresholds versus Variant C:

- CfC evidence (Variant B): treat `>= 0.03` lower `val_loss` than Variant C (with matching or lower perplexity) as practically meaningful.
- GDN+CfC evidence (Variant A): treat `>= 0.05` lower `val_loss` than Variant C, and at least `>= 0.02` lower than Variant B, as meaningful support for the GDN contribution.
- Robustness check: require same-direction effect across at least 2 seeds/runs before claiming support.

These thresholds are intentionally conservative for short ablation runs, helping distinguish real architectural signal from run-to-run noise.

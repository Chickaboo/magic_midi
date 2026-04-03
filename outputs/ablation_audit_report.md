# A/B/C Readiness Audit

Generated: 2026-04-03T14:12:20.864390
Overall: WARN

## Environment

| Check | Status | Detail |
|---|---|---|
| Python | PASS | 3.14.0 |
| PyTorch | PASS | 2.9.1+cpu |
| CUDA | WARN | not available |

## Dependencies

| Check | Status | Detail |
|---|---|---|
| CfC runtime | PASS | import ncps ok |
| Symusic tokenizer runtime | PASS | import symusic ok |
| GatedDeltaNet kernel | WARN | flash-linear-attention unavailable; Variant A uses fallback |
| Mamba kernel | PASS | mamba_ssm unavailable (optional for A/B/C ablation) |

## Variant Smoke Checks

| Check | Status | Detail |
|---|---|---|
| variant_a forward | PASS | shape=(2, 96, 155) params=11.91M |
| variant_b forward | PASS | shape=(2, 96, 155) params=12.23M |
| variant_c forward | PASS | shape=(2, 96, 155) params=11.64M |
| A/B/C parameter comparability | PASS | min=11.64M max=12.23M ratio=1.051 |

## Tokenized Data

| Check | Status | Detail |
|---|---|---|
| Pre-tokenized manifest | WARN | not provided (audit skipped for tokenized data integrity) |

## Variant Details

| Variant | Architecture | Params (M) | d_model | n_layers | Backend Status |
|---|---|---:|---:|---:|---|
| variant_a | gated_delta_cfc_attention_hybrid | 11.91 | 480 | 5 | {'gdn_using_fallback': True, 'cfc_using_fallback': False} |
| variant_b | transformer_cfc_hybrid | 12.23 | 544 | 5 | {'gdn_using_fallback': False, 'cfc_using_fallback': False} |
| variant_c | pure_attention_transformer_baseline | 11.64 | 480 | 4 | {'gdn_using_fallback': False, 'cfc_using_fallback': False} |

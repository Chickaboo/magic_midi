# Architecture Readiness Audit

Generated: 2026-04-04T12:01:24.260264
Overall: WARN

## Environment

| Check | Status | Detail |
|---|---|---|
| Python | PASS | 3.14.0 |
| PyTorch | PASS | 2.10.0+cpu |
| CUDA | WARN | not available |

## Dependencies

| Check | Status | Detail |
|---|---|---|
| CfC runtime | WARN | import ncps failed: No module named 'ncps' |
| Symusic tokenizer runtime | PASS | import symusic ok |
| GatedDeltaNet kernel | WARN | flash-linear-attention unavailable; GDN-based variants use fallback |
| Mamba kernel | PASS | mamba_ssm unavailable (optional for architecture ablation) |

## Variant Smoke Checks

| Check | Status | Detail |
|---|---|---|
| variant_c forward | PASS | shape=(2, 96, 155) params=11.64M |
| variant_e forward | PASS | shape=(2, 96, 155) params=11.91M |
| Variant parameter comparability | PASS | min=11.64M max=11.91M ratio=1.023 |

## Tokenized Data

| Check | Status | Detail |
|---|---|---|
| Pre-tokenized manifest | WARN | not provided (audit skipped for tokenized data integrity) |

## Variant Details

| Variant | Architecture | Params (M) | d_model | n_layers | Backend Status |
|---|---|---:|---:|---:|---|
| variant_c | pure_attention_transformer_baseline | 11.64 | 480 | 4 | {'gdn_using_fallback': False, 'cfc_using_fallback': False} |
| variant_e | gated_delta_sparse_attention_no_cfc | 11.91 | 544 | 8 | {'gdn_using_fallback': True, 'cfc_using_fallback': False} |

# Model Card: latest_model.safetensors

## Project
- Name: `Itty Bitty Piano`

## Architecture
- Model class: `PianoHybridModel`
- d_model: `192`
- layers: `4`
- Mamba enabled: `True`
- CfC enabled: `True`
- FFN expansion: `4`
- Attention heads: `4`
- Attention cadence: every `2` layers
- Attention bias: `learned`
- Output logit scale: `0.072169`
- Tied embeddings: `True`
- Total parameters (measured): `4,342,024`

## Data / Tokenization
- Vocabulary size: `2000`
- Tokenization strategy: `remi`
- Seed length: `128`
- Continuation length: `256`

## Training History
- Epoch in checkpoint: `35`
- Last val loss in checkpoint: `6.175081729888916`
- Best val loss tracked: `6.175081729888916`
- Train loss entries: `35`
- Val loss entries: `35`
- Generation health entries: `0`

## Checkpoint Load Diagnostics
- Missing keys: `20`
- Unexpected keys: `32`

## Generation Preview
- Preview token count: `256`
- Preview unique token count: `194`
- First 32 tokens:

```text
104 434 1885 827 1155 1947 948 1130 659 214 1673 1211 178 305 1645 1411 1707 665 1669 415 429 1573 1855 1008 3 1163 781 708 1532 1751 1141 1365
```

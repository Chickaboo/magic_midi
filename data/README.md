# Data and Tokenizers

## Canonical Tokenizer Modules

- `tokenizer_custom.py`: `CustomDeltaTokenizer` implementation.
  - Frozen quad-event tokenizer.
  - `event_size = 4`.
  - `vocab_size = 374`.
  - Token slices: delta `0..127`, pitch `128..215`, duration `216..343`, velocity `344..359`.
  - Structural/special tokens: `PAD=360`, `BOS=361`, `EOS=362`, Density `363..366`, Voices `367..370`, Register `371..373`.
- `tokenizer.py`: Factory and compatibility wrapper.
  - `create_tokenizer(...)`
  - `load_tokenizer(...)`
  - Enforces unified `CustomDeltaTokenizer` usage in active workflows.

## Important Contract Note

There is one supported token stream in this repository:

- Unified quad-event stream (`CustomDeltaTokenizer`, `event_size=4`, `vocab_size=374`).

Do not train or preprocess with alternate tokenizer strategies.

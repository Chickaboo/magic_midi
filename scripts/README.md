# Scripts Overview

## Tokenization

- `tokenize_launcher.ps1`: Shared PowerShell launcher for both local tokenization and controller-based chunking/upload.
- `tokenize_godzilla_local.py`: Local tokenizer over MIDI source data.
- `tokenize_upload_hf_batches.py`: Chunked controller for large runs and optional Hugging Face uploads.

## Root Wrapper Scripts

The root-level scripts call `scripts/tokenize_launcher.ps1` with preset defaults:

 500k bar/meta controller run.

## Utilities

- `midi_cut_cli.py`: Interactive MIDI cut tool.
- `clear_cache.py`: Cache cleanup helper.

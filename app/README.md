# Itty Bitty Piano App

This is the web inference app for Itty Bitty Piano.

## Layout

- `app/server.py` - Flask backend.
- `app/templates/index.html` - UI.
- `app/static/*` - styles and client logic.
- `app/models/` - place checkpoint files here.
- `app/tokenizer/` - place tokenizer JSON files here.
- `app/runtime/outputs/` - generated MIDI, WAV previews, and pianoroll PNGs.

## Features

- dark dashboard UI with drag/drop seed upload,
- model selector from `app/models/`,
- CPU-first local generation using the pasted checkpoint/tokenizer pair,
- automatic custom-tokenizer detection for Kaggle `custom_delta` checkpoints,
- sliders: temperature, length, top-p, top-k,
- output MIDI download plus rendered seed/output audio previews,
- pianoroll comparison image for the generated continuation,
- generation health check + repetition warning when available.

## Asset Workflow

Copy the latest Kaggle artifacts into the app folders before starting the app:

- `app/models/latest.safetensors`
- `app/models/latest_state.pt`
- `app/tokenizer/custom_tokenizer.json`

If you use a plain MidiTok tokenizer instead, place it in `app/tokenizer/tokenizer.json`.

The app loads checkpoints/tokenizers directly from these app folders, so you can swap in fresh artifacts without rebuilding the app package.

## Setup (Linux/macOS)

```bash
cd app
./setup.sh
./run.sh
```

The setup script installs `mamba-ssm` automatically when it detects Linux + CUDA.
Otherwise the app stays on the built-in GRU fallback.

## Setup (Windows)

```bat
cd app
setup.bat
run.bat
```

On Windows, the app will use the built-in GRU fallback unless you run it on a
Linux + CUDA environment with `mamba-ssm` available.

Open `http://127.0.0.1:5000`.

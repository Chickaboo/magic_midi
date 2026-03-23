# Itty Bitty Piano App

This is the web inference app for Itty Bitty Piano.

## Layout

- `app/server.py` - Flask backend.
- `app/templates/index.html` - UI.
- `app/static/*` - styles and client logic.
- `app/models/` - place checkpoint files here.
- `app/tokenizer/tokenizer.json` - tokenizer loaded relative to app directory.
- `app/runtime/outputs/` - generated MIDI outputs.

## Features

- dark minimal interface with drag/drop seed upload,
- model selector from `app/models/`,
- sliders: temperature, length, top-p, top-k,
- generate endpoint uses shared `PianoHybridModel.generate()` + `model/sampling.py`,
- DataParallel key stripping on checkpoint load,
- generation health check + repetition warning if >60% token share,
- downloadable continuation MIDI.

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

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

## Setup (Windows)

```bat
cd app
setup.bat
run.bat
```

Open `http://127.0.0.1:5000`.

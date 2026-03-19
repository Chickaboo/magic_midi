#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR"
VENV_DIR="$APP_DIR/.venv"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Virtual environment not found. Run ./setup.sh first."
  exit 1
fi

source "$VENV_DIR/bin/activate"
python "$APP_DIR/server.py"

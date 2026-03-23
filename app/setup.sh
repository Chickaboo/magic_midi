#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR"
VENV_DIR="$APP_DIR/.venv"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$APP_DIR/requirements.txt"

MAMBA_READY="$(python - <<'PY'
import platform

try:
    import torch
    has_cuda = bool(torch.cuda.is_available())
except Exception:
    has_cuda = False

print("1" if platform.system() == "Linux" and has_cuda else "0")
PY
)"

if [ "$MAMBA_READY" = "1" ]; then
  echo "Linux + CUDA detected. Installing Mamba SSM packages..."
  python -m pip install "causal-conv1d>=1.4.0" --no-build-isolation
  python -m pip install "mamba-ssm[causal-conv1d]" --no-build-isolation
else
  echo "Skipping mamba-ssm install (requires Linux + CUDA). GRU fallback remains available."
fi

echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"

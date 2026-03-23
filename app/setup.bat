@echo off
setlocal

set APP_DIR=%~dp0
set VENV_DIR=%APP_DIR%.venv

python -m venv "%VENV_DIR%"
if errorlevel 1 exit /b 1

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

python -m pip install -r "%APP_DIR%requirements.txt"
if errorlevel 1 exit /b 1

python -c "import platform; import torch; raise SystemExit(0 if platform.system() == 'Linux' and torch.cuda.is_available() else 1)"
if not errorlevel 1 (
  echo Linux + CUDA detected. Installing Mamba SSM packages...
  python -m pip install "causal-conv1d>=1.4.0" --no-build-isolation
  if errorlevel 1 exit /b 1
  python -m pip install "mamba-ssm[causal-conv1d]" --no-build-isolation
  if errorlevel 1 exit /b 1
) else (
  echo Skipping mamba-ssm install (requires Linux + CUDA). GRU fallback remains available.
)

echo Setup complete.
endlocal

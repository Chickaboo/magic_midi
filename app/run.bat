@echo off
setlocal

set APP_DIR=%~dp0
set VENV_DIR=%APP_DIR%.venv

if not exist "%VENV_DIR%\Scripts\activate.bat" (
  echo Virtual environment not found. Run setup.bat first.
  exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

python "%APP_DIR%server.py"

endlocal

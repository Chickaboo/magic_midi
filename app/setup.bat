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

echo Setup complete.
endlocal

@echo off
REM Start Black-Scholes Backend Server (Windows)

cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start the FastAPI server
python backend_api.py


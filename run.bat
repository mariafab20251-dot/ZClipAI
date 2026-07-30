@echo off
cd /d "%~dp0"
rem Use the tool's OWN isolated venv so its heavy deps (torch, mediapipe,
rem opencv-headless, numpy 2.x) never leak into the global Python that other
rem apps (e.g. the video automation studio) depend on.
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" main.py
) else (
    echo [WARN] .venv not found - falling back to global python.
    echo        Run:  python -m venv .venv  ^&^&  .venv\Scripts\python -m pip install -r requirements.txt
    python main.py
)
if errorlevel 1 (
    echo.
    echo Script exited with error code %errorlevel%
    pause
)

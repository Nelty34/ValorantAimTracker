@echo off
REM Launcher for Valorant Aim Tracker UI
REM Check if PySimpleGUI is installed
python -m pip show PySimpleGUI >nul 2>&1
if %errorlevel% neq 0 (
    echo PySimpleGUI not found. Installing...
    python -m pip install PySimpleGUI
)

REM Run the UI
python ui.py
pause

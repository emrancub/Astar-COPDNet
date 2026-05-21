@echo off
setlocal

REM Run this BAT file from the project root folder.
REM Example:
REM   double-click it, or run from CMD in the project root.

if not exist scripts\check_real_paths.py (
    echo ERROR: Please place this BAT file in the project root folder.
    pause
    exit /b 1
)

python run_pipeline_step_by_step.py --config configs\windows_real_paths.json
pause

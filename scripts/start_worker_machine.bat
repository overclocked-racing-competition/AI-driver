@echo off
REM â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
REM  S4-F WORKER SCRIPT â€” Run on EACH additional machine
REM  Copy the entire S4-F folder to each machine first!
REM
REM  SETUP (one-time per machine):
REM    1. Copy D:\IBM_competition\SAC\S3_B\S4-F\ to same path on each machine
REM    2. Set STORAGE_URL below to point to the shared database
REM       (PostgreSQL for multi-machine, or a shared network SQLite path)
REM    3. Run this script on each worker machine
REM â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

SET PYTHON=D:\torcs\pyenv\Scripts\python.exe
REM Project root is one level above scripts\
SET SCRIPT_DIR=%~dp0..\
SET STUDY=teacher_v4
SET N_TRIALS=2000
SET N_INSTANCES=6

REM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
REM  CHANGE THIS: Point to shared database
REM  Option A â€” SQLite on a shared network drive:
REM    SET STORAGE=sqlite:///\\SERVER\share\optuna_teacher_v3.db
REM  Option B â€” PostgreSQL (recommended for true distributed):
REM    SET STORAGE=postgresql://user:password@HOST_IP:5432/racing
REM  Option C â€” Same machine SQLite (for testing):
REM    SET STORAGE=sqlite:///optuna_teacher_v3.db
REM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SET STORAGE=sqlite:///optuna_teacher_v3.db

echo.
echo â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
echo   S4-F Sub-80s Campaign â€” Worker Machine Launch
echo   Study:     %STUDY%
echo   Storage:   %STORAGE%
echo   Instances: %N_INSTANCES% parallel TORCS
echo â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
echo.

REM Set up instance directories on this machine
echo [Setup] Creating TORCS instance directories...
%PYTHON% "%SCRIPT_DIR%multi_instance_torcs.py" --mode setup --n %N_INSTANCES%
if errorlevel 1 (
    echo ERROR: Instance setup failed.
    pause
    exit /b 1
)

REM Start worker
echo.
echo [Worker] Connecting to study and running trials...
echo   Press Ctrl+C to stop cleanly.
echo.
%PYTHON% "%SCRIPT_DIR%optuna_teacher_v3.py" --mode worker ^
    --storage "%STORAGE%" ^
    --study-name "%STUDY%" ^
    --n-instances %N_INSTANCES% ^
    --n-trials %N_TRIALS% ^
    --n-laps 3 ^
    --verbose 1

pause


@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  S4-F LAUNCH SCRIPT — Run on the MAIN machine (coordinator)
REM  Start the Optuna teacher search with 6 parallel TORCS instances
REM ═══════════════════════════════════════════════════════════════════

SET PYTHON=D:\torcs\pyenv\Scripts\python.exe
REM Project root is one level above scripts\
SET SCRIPT_DIR=%~dp0..\
SET STORAGE=sqlite:///optuna_teacher_v3.db
SET STUDY=teacher_v4
SET N_TRIALS=2000
SET N_INSTANCES=6

echo.
echo ═══════════════════════════════════════════════════════════════
echo   S4-F Sub-80s Campaign — Main Machine Launch
echo   Instances: %N_INSTANCES% parallel TORCS
echo   Target: %N_TRIALS% Optuna trials
echo ═══════════════════════════════════════════════════════════════
echo.

REM Step 1: Set up TORCS instance directories (one-time)
echo [Step 1] Setting up %N_INSTANCES% TORCS instance directories...
%PYTHON% "%SCRIPT_DIR%multi_instance_torcs.py" --mode setup --n %N_INSTANCES%
if errorlevel 1 (
    echo ERROR: Instance setup failed. Check TORCS installation at D:\torcs\torcs\wtorcs.exe
    pause
    exit /b 1
)

REM Step 2: Create Optuna study (idempotent - safe to run multiple times)
echo.
echo [Step 2] Creating/loading Optuna study '%STUDY%'...
%PYTHON% "%SCRIPT_DIR%optuna_teacher_v3.py" --mode coordinator ^
    --storage "%STORAGE%" --study-name "%STUDY%" --n-trials %N_TRIALS%

REM Step 3: Start worker on this machine
echo.
echo [Step 3] Starting worker with %N_INSTANCES% parallel instances...
echo   Press Ctrl+C to stop. Results are saved automatically.
echo.
%PYTHON% "%SCRIPT_DIR%optuna_teacher_v3.py" --mode worker ^
    --storage "%STORAGE%" ^
    --study-name "%STUDY%" ^
    --n-instances %N_INSTANCES% ^
    --n-trials %N_TRIALS% ^
    --n-laps 3 ^
    --verbose 1

echo.
echo Worker stopped. Generating report...
%PYTHON% "%SCRIPT_DIR%optuna_teacher_v3.py" --mode report ^
    --storage "%STORAGE%" --study-name "%STUDY%"

echo.
echo Exporting best parameters...
%PYTHON% "%SCRIPT_DIR%optuna_teacher_v3.py" --mode export-best ^
    --storage "%STORAGE%" --study-name "%STUDY%" ^
    --output "%SCRIPT_DIR%checkpoints\best_teacher_v3.json"

pause

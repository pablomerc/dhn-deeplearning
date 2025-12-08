@echo off
REM ============================================
REM Master Batch File for Super-Resolution Experiment
REM Run this in your Anaconda/Python environment
REM ============================================

echo ============================================
echo DHN Super-Resolution Experiment
echo ============================================
echo.

REM Check Python
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please activate your conda environment first.
    echo Example: conda activate dhn
    pause
    exit /b 1
)

echo.
echo Select phase to run:
echo   1 - Quick Test (verify setup)
echo   2 - Generate Data
echo   3 - Train Models (200 epochs)
echo   4 - Extract Results
echo   5 - Generate Plots
echo   6 - Run ALL phases
echo.

set /p PHASE="Enter choice (1-6): "

if "%PHASE%"=="1" (
    echo Running quick test...
    python run_quick_test.py
    goto :end
)

if "%PHASE%"=="2" (
    echo Generating data...
    python run_full_superres_experiment.py --phase data
    goto :end
)

if "%PHASE%"=="3" (
    echo Training models (this will take several hours)...
    python run_full_superres_experiment.py --phase train --train_epochs 200
    goto :end
)

if "%PHASE%"=="4" (
    echo Running extraction...
    python run_full_superres_experiment.py --phase extract --extract_epochs 1000
    goto :end
)

if "%PHASE%"=="5" (
    echo Generating plots...
    python run_full_superres_experiment.py --phase plot
    goto :end
)

if "%PHASE%"=="6" (
    echo Running ALL phases (this will take many hours)...
    python run_full_superres_experiment.py --phase all --train_epochs 200 --extract_epochs 1000
    goto :end
)

echo Invalid choice!

:end
echo.
echo Done!
pause



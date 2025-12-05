@echo off
echo Training All Models (6 total)...
echo (This takes several hours - consider running overnight)
echo.
cd /d "%~dp0"
python run_full_superres_experiment.py --phase train --train_epochs 200
echo.
echo Training complete!
pause

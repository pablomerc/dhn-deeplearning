@echo off
echo Generating Data for All Systems...
echo (This takes ~10-20 minutes)
echo.
cd /d "%~dp0"
python run_full_superres_experiment.py --phase data
echo.
echo Data generation complete!
pause



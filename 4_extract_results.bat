@echo off
echo Running Extraction (Autodecoding)...
echo (This takes ~2-4 hours)
echo.
cd /d "%~dp0"
python run_full_superres_experiment.py --phase extract --extract_epochs 1000
echo.
echo Extraction complete!
pause

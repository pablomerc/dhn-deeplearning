@echo off
echo Generating Figure 14-Style Plots...
echo.
cd /d "%~dp0"
python run_full_superres_experiment.py --phase plot
echo.
echo Plots saved to visualization_scripts/
pause



@echo off
echo.
echo ========================================
echo  KNN IMAGE CLASSIFIER - QUICK START
echo ========================================
echo.
echo Starting the KNN Image Classifier...
echo.

REM Try to start with python3 first, then python
python3 start_app.py 2>nul
if errorlevel 1 (
    python start_app.py
)

if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    echo.
    echo Download from: https://www.python.org/downloads/
    pause
)

pause 
@echo off
setlocal

:: ============================================================================
::  Start Script for SUP Toolbox App (Batch for CMD)
:: ============================================================================

echo Starting SUP Toolbox App...
echo.

:: 1. Check if the virtual environment directory exists.
IF NOT EXIST ".\venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found in '.\venv'.
    echo Please run the 'setup.bat' script first to install dependencies.
    echo.
    pause
    exit /b 1
)

:: 2. Activate the virtual environment.
echo Activating virtual environment...
call ".\venv\Scripts\activate.bat"

:: 3. Run the Python application.
echo Launching the application...
echo You can close this window to stop the server.
echo.
python app.py

:: Pause at the end to see any potential errors if the app closes immediately.
echo.
echo Application has been shut down.
pause
endlocal
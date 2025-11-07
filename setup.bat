@echo off
setlocal

:: ============================================================================
::  Setup Script for sup-toolbox gradio app
::  This script creates a virtual environment, activates it,
::  and installs the required packages.
:: ============================================================================

echo Checking for a Python virtual environment...

:: 1. Check if the 'venv' directory exists.
IF NOT EXIST "venv" (
    echo Virtual environment not found. Creating a new one in '.\venv'...
    
    :: Try to find python.exe. This assumes python is in the PATH.
    python -m venv venv
    
    :: Check if the venv creation was successful
    IF %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create the virtual environment.
        echo Please ensure Python 3.8+ is installed and available in your PATH.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) ELSE (
    echo Found existing virtual environment.
)

echo.
echo Activating the virtual environment...

:: 2. Activate the virtual environment.
call ".\venv\Scripts\activate.bat"

:: Check if activation was successful by seeing if python.exe is now in the venv path
where python | findstr /I /C:"\venv\Scripts\python.exe" > nul
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate the virtual environment.
    pause
    exit /b 1
)

echo Environment activated. Python executable is now:
where python
echo.

:: ============================================================================
::  Install Python Packages
:: ============================================================================

echo Upgrading pip...
python.exe -m pip install --upgrade pip
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)

echo.
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo Setup complete!
echo The virtual environment is active. You can now run the CLI.
echo To deactivate, simply run 'deactivate' in this terminal.
echo ============================================================================
echo.

pause
endlocal
# ============================================================================
#  Start Script for SUP Toolbox App (PowerShell)
# ============================================================================

Write-Host "Starting SUP Toolbox App..."
Write-Host ""

# 1. Check if the virtual environment activation script exists.
$VenvPath = ".\venv\Scripts\Activate.ps1"
if (-not (Test-Path -Path $VenvPath)) {
    Write-Host "ERROR: Virtual environment not found in '.\venv'." -ForegroundColor Red
    Write-Host "Please run the 'setup.ps1' script first to install dependencies." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# 2. Activate the virtual environment.
# We use dot-sourcing to run the script in the current scope.
Write-Host "Activating virtual environment..."
. $VenvPath

# 3. Run the Python application.
Write-Host "Launching the application..." -ForegroundColor Green
Write-Host "You can press Ctrl+C in this terminal to stop the server." -ForegroundColor Yellow
Write-Host ""

python app.py

# A pause at the end is generally not needed in PowerShell as the window
# usually stays open, but we can add one for consistency if desired.
Write-Host ""
Write-Host "Application has been shut down."
Read-Host "Press Enter to close this window"
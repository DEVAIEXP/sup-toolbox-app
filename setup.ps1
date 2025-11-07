<#
.SYNOPSIS
    Setup script for the sup-toolbox gradio app project.
.DESCRIPTION
    This PowerShell script automates the setup process by:
    1. Creating a Python virtual environment if it doesn't exist.
    2. Activating the virtual environment for the current terminal session.
    3. Upgrading pip and installing all required project dependencies.
.NOTES
    Author: Eliseu Silva
    Version: 1.0
#>

# ============================================================================
#  Setup Script for sup-toolbox gradio app (PowerShell Version)
# ============================================================================

# Define a helper function to handle errors gracefully.
function Exit-OnError {
    param(
        [string]$Message
    )
    Write-Host "ERROR: $Message" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Checking for a Python virtual environment..."

# 1. Check if the 'venv' directory exists.
if (-not (Test-Path -Path ".\venv" -PathType Container)) {
    Write-Host "Virtual environment not found. Creating a new one in '.\venv'..."
    
    # Attempt to create the virtual environment using the python executable in PATH.
    python -m venv venv
    
    # Check if the last command failed.
    if ($LASTEXITCODE -ne 0) {
        Exit-OnError "Failed to create the virtual environment. Please ensure Python 3.8+ is installed and in your system's PATH."
    }
    Write-Host "Virtual environment created successfully."
} else {
    Write-Host "Found existing virtual environment."
}

Write-Host "" # Newline for readability
Write-Host "Activating the virtual environment..."

# 2. Activate the virtual environment using the PowerShell script.
# The `.` (dot source) operator runs the script in the current scope,
# which is necessary to modify the current session's environment.
try {
    . ".\venv\Scripts\Activate.ps1"
} catch {   
    Exit-OnError 'Failed to activate the virtual environment. You may need to run ''Set-ExecutionPolicy RemoteSigned -Scope CurrentUser'' first.'
}

Write-Host "Environment activated. Python executable is now:"
# Get-Command is a reliable way to show which executable is currently being used.
Get-Command python | Select-Object -First 1
Write-Host ""

# ============================================================================
#  Install Python Packages
# ============================================================================

Write-Host "Upgrading pip..."
python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Exit-OnError "Failed to upgrade pip."
}

Write-Host ""
Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Exit-OnError "Failed to install requirements from requirements.txt."
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host " The virtual environment is active in THIS terminal session." -ForegroundColor Green
Write-Host " You can now run the CLI or other project scripts." -ForegroundColor Green
Write-Host " To deactivate, simply run 'deactivate' or close this terminal." -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to continue..."
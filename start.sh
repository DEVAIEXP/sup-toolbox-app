#!/bin/bash

# ============================================================================
#  Start Script for SUP Toolbox App (Bash for Linux/macOS)
# ============================================================================

echo "Starting SUP Toolbox App..."
echo ""

# 1. Check if the virtual environment activation script exists.
VENV_PATH="./venv/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found in './venv'."
    echo "Please run the 'setup.sh' script first to install dependencies."
    exit 1
fi

# 2. Activate the virtual environment.
echo "Activating virtual environment..."
source "$VENV_PATH"

# 3. Run the Python application.
echo "Launching the application..."
echo "You can press Ctrl+C in this terminal to stop the server."
echo ""

python app.py

echo ""
echo "Application has been shut down."
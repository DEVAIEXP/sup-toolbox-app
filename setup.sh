#!/bin/bash

# ============================================================================
#  Setup Script for sup-toolbox gradio app
#  This script creates a Python virtual environment, activates it,
#  and installs the required packages for Linux and macOS.
# ============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Checking for Python 3 ---"
if ! command -v python3 &> /dev/null
then
    echo "ERROR: python3 could not be found. Please install Python 3.8+."
    exit 1
fi

echo "--- Checking for a Python virtual environment ---"

# 1. Check if the 'venv' directory exists.
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating a new one in './venv'..."
    
    # Create the virtual environment using python3
    python3 -m venv venv
    
    echo "Virtual environment created successfully."
else
    echo "Found existing virtual environment."
fi

echo ""
echo "--- Activating the virtual environment ---"

# 2. Activate the virtual environment.
source venv/bin/activate

echo "Environment activated. Python executable is now:"
which python
echo ""

# ============================================================================
#  Install Python Packages
# ============================================================================

echo "--- Upgrading pip ---"
python -m pip install --upgrade pip

echo ""
echo "--- Installing dependencies from requirements.txt ---"
pip install -r requirements.txt

echo ""
echo "============================================================================"
echo " Setup complete!"
echo " The virtual environment is active. You can now run the CLI or the Gradio app."
echo " To deactivate, simply run 'deactivate' in this terminal."
echo "============================================================================"
echo ""
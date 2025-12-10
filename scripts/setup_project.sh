#!/bin/bash
# ALC-Algo Setup Script (Linux/Mac/Azure Cloud Shell)

echo "================================================================="
echo "ALC-ALGO SETUP - INSTITUTIONAL GRADE"
echo "================================================================="

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found! Please install Python 3.10+"
    exit 1
fi
echo "Found: $(python3 --version)"

# 2. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "Virtual environment 'venv' already exists."
fi

# 3. Activate and Install Requirements
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found!"
fi

# 4. Configuration Setup
echo "Setting up configuration files..."

if [ ! -f "config/secrets.py" ]; then
    if [ -f "config/secrets.py.example" ]; then
        cp config/secrets.py.example config/secrets.py
        echo "Created config/secrets.py - PLEASE EDIT THIS FILE WITH YOUR KEYS"
    fi
fi

if [ ! -f ".env" ]; then
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "Created .env from env.example"
    fi
fi

# 5. Directory Structure
mkdir -p data/raw data/processed data/models logs
echo "Created data directories."

echo "================================================================="
echo "SETUP COMPLETE"
echo "1. Edit config/secrets.py with your API keys"
echo "2. Activate venv: source venv/bin/activate"
echo "3. Run: python main.py"
echo "================================================================="


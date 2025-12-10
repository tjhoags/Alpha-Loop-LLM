# ALC-Algo Setup Script (Windows PowerShell)
# Automates environment setup for Tom Hogan

Write-Host "================================================================="
Write-Host "ALC-ALGO SETUP - INSTITUTIONAL GRADE"
Write-Host "================================================================="

# 1. Check Python
$pythonVersion = python --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python not found! Please install Python 3.10+"
    exit 1
}
Write-Host "Found: $pythonVersion"

# 2. Create Virtual Environment
if (!(Test-Path "venv")) {
    Write-Host "Creating virtual environment 'venv'..."
    python -m venv venv
} else {
    Write-Host "Virtual environment 'venv' already exists."
}

# 3. Activate and Install Requirements
Write-Host "Installing dependencies... (This may take a moment)"
.\venv\Scripts\python -m pip install --upgrade pip
.\venv\Scripts\pip install -r requirements.txt

# 4. Configuration Setup
Write-Host "Setting up configuration files..."

if (!(Test-Path "config/secrets.py")) {
    Copy-Item "config/secrets.py.example" "config/secrets.py"
    Write-Host "Created config/secrets.py - PLEASE EDIT THIS FILE WITH YOUR KEYS"
}

if (!(Test-Path ".env")) {
    if (Test-Path "env.example") {
        Copy-Item "env.example" ".env"
        Write-Host "Created .env from env.example"
    } else {
        Write-Warning "env.example not found!"
    }
}

# 5. Directory Structure
$dirs = @("data/raw", "data/processed", "data/models", "logs")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "Created directory: $dir"
    }
}

Write-Host "================================================================="
Write-Host "SETUP COMPLETE"
Write-Host "1. Edit config/secrets.py with your API keys"
Write-Host "2. Activate venv: .\venv\Scripts\activate"
Write-Host "3. Run: python main.py"
Write-Host "================================================================="


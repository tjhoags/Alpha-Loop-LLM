# ================================================================================
# FULL THROTTLE TRAINING STARTUP SCRIPT (Windows PowerShell)
# ================================================================================
#
# WHAT THIS DOES:
#   Starts all data ingestion and training processes simultaneously:
#   1. Massive S3 hydration (5 years backfill)
#   2. Alpha Vantage Premium hydration (continuous)
#   3. Model training (continuous)
#
# HOW TO RUN:
#   Option 1: Right-click > Run with PowerShell
#   Option 2: Open PowerShell and run:
#      cd C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1
#      .\scripts\start_full_throttle_training.ps1
#
# TIME: Runs until stopped (Ctrl+C or close windows)
# ================================================================================

$ErrorActionPreference = "Stop"

# Get project directory (parent of scripts folder)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Set-Location $ProjectDir

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  FULL THROTTLE TRAINING STARTUP" -ForegroundColor Cyan
Write-Host "  Project: $ProjectDir" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check for virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Run these commands first:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv"
    Write-Host "  .\venv\Scripts\Activate.ps1"
    Write-Host "  pip install -r requirements.txt"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Create logs directory
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host "Starting FULL THROTTLE TRAINING..." -ForegroundColor Green
Write-Host ""

# Terminal 1: Massive S3 Hydration
Write-Host "[1/3] Starting Massive S3 Hydration (5 years backfill)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectDir'; .\venv\Scripts\Activate.ps1; python scripts/hydrate_massive.py"

Start-Sleep -Seconds 2

# Terminal 2: Alpha Vantage Premium Hydration
Write-Host "[2/3] Starting Alpha Vantage Premium Hydration..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectDir'; .\venv\Scripts\Activate.ps1; python scripts/hydrate_all_alpha_vantage.py"

Start-Sleep -Seconds 2

# Terminal 3: Model Training
Write-Host "[3/3] Starting Model Training..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectDir'; .\venv\Scripts\Activate.ps1; python src/ml/train_models.py"

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ALL PROCESSES STARTED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "3 PowerShell windows opened:" -ForegroundColor Yellow
Write-Host "  - Window 1: Massive S3 Hydration"
Write-Host "  - Window 2: Alpha Vantage Hydration"
Write-Host "  - Window 3: Model Training"
Write-Host ""
Write-Host "Monitor logs:" -ForegroundColor Yellow
Write-Host "  Get-Content logs\massive_ingest.log -Tail 50 -Wait"
Write-Host "  Get-Content logs\alpha_vantage_hydration.log -Tail 50 -Wait"
Write-Host "  Get-Content logs\model_training.log -Tail 50 -Wait"
Write-Host ""
Write-Host "To stop: Close the PowerShell windows" -ForegroundColor Yellow
Write-Host ""
Write-Host "TOMORROW MORNING (9:15 AM ET):" -ForegroundColor Magenta
Write-Host "  python src/trading/execution_engine.py" -ForegroundColor Magenta
Write-Host ""

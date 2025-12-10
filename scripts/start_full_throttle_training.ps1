# ================================================================================
# FULL THROTTLE TRAINING STARTUP SCRIPT (Windows PowerShell)
# ================================================================================
# Starts all data ingestion and training processes simultaneously:
# 1. Massive S3 hydration (5 years backfill)
# 2. Alpha Vantage Premium hydration (continuous)
# 3. Model training (continuous)
# ================================================================================

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Set-Location $ProjectDir

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "❌ Virtual environment not found. Run: python -m venv venv && .\venv\Scripts\Activate.ps1 && pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

# Create logs directory
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host "🚀 STARTING FULL THROTTLE TRAINING..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# Terminal 1: Massive S3 Hydration
Write-Host "📁 Starting Terminal 1: Massive S3 Hydration (5 years backfill)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectDir'; .\venv\Scripts\Activate.ps1; python scripts/hydrate_massive.py"

Start-Sleep -Seconds 2

# Terminal 2: Alpha Vantage Premium Hydration
Write-Host "📊 Starting Terminal 2: Alpha Vantage Premium Hydration..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectDir'; .\venv\Scripts\Activate.ps1; python scripts/hydrate_all_alpha_vantage.py"

Start-Sleep -Seconds 2

# Terminal 3: Model Training
Write-Host "🤖 Starting Terminal 3: Model Training..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectDir'; .\venv\Scripts\Activate.ps1; python src/ml/train_models.py"

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "✅ All processes started!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Monitor logs:" -ForegroundColor Yellow
Write-Host "  - Get-Content logs\massive_ingest.log -Tail 50 -Wait"
Write-Host "  - Get-Content logs\alpha_vantage_hydration.log -Tail 50 -Wait"
Write-Host "  - Get-Content logs\model_training.log -Tail 50 -Wait"
Write-Host ""
Write-Host "🛑 To stop: Close the PowerShell windows" -ForegroundColor Yellow
Write-Host ""
Write-Host "⏰ TOMORROW MORNING (9:15 AM ET):" -ForegroundColor Magenta
Write-Host "  python src/trading/execution_engine.py" -ForegroundColor Magenta


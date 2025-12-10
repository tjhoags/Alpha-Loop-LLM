# =============================================================================
# BULLETPROOF OVERNIGHT TRAINING SCRIPT
# =============================================================================
# This script:
# 1. Prevents computer from sleeping
# 2. Runs training outside of Cursor
# 3. Auto-restarts if training crashes
# 4. Logs everything
#
# HOW TO RUN:
# 1. Open Windows PowerShell (NOT in Cursor - press Win+X, select "Terminal")
# 2. Run: powershell -ExecutionPolicy Bypass -File "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1\scripts\overnight_training_robust.ps1"
# =============================================================================

$ErrorActionPreference = "Continue"
$PROJECT_DIR = "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
$LOG_FILE = "$PROJECT_DIR\logs\overnight_robust.log"
$MAX_RETRIES = 5
$RETRY_DELAY_SECONDS = 60

# -----------------------------------------------------------------------------
# STEP 1: PREVENT SLEEP
# -----------------------------------------------------------------------------
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PREVENTING COMPUTER SLEEP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Disable sleep via powercfg
powercfg -change -standby-timeout-ac 0
powercfg -change -hibernate-timeout-ac 0
powercfg -change -monitor-timeout-ac 0

Write-Host "[OK] Sleep disabled for AC power" -ForegroundColor Green

# Keep system awake using .NET (backup method)
Add-Type -AssemblyName System.Windows.Forms
$keepAwakeScript = {
    while ($true) {
        # Simulate key press to prevent sleep
        [System.Windows.Forms.SendKeys]::SendWait("{SCROLLLOCK}")
        Start-Sleep -Milliseconds 100
        [System.Windows.Forms.SendKeys]::SendWait("{SCROLLLOCK}")
        Start-Sleep -Seconds 240  # Every 4 minutes
    }
}

# Start keep-awake in background
$keepAwakeJob = Start-Job -ScriptBlock $keepAwakeScript
Write-Host "[OK] Keep-awake background job started" -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 2: SETUP LOGGING
# -----------------------------------------------------------------------------
function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage
    Add-Content -Path $LOG_FILE -Value $logMessage
}

# Create logs directory if needed
New-Item -ItemType Directory -Force -Path "$PROJECT_DIR\logs" | Out-Null

Write-Log "=========================================="
Write-Log "OVERNIGHT TRAINING STARTED"
Write-Log "=========================================="
Write-Log "Project: $PROJECT_DIR"
Write-Log "Log file: $LOG_FILE"

# -----------------------------------------------------------------------------
# STEP 3: ACTIVATE VENV AND RUN TRAINING WITH RETRIES
# -----------------------------------------------------------------------------
Set-Location $PROJECT_DIR

$retryCount = 0
$success = $false

while (-not $success -and $retryCount -lt $MAX_RETRIES) {
    $retryCount++
    Write-Log "----------------------------------------"
    Write-Log "TRAINING ATTEMPT $retryCount of $MAX_RETRIES"
    Write-Log "----------------------------------------"
    
    try {
        # Activate virtual environment
        Write-Log "Activating virtual environment..."
        & "$PROJECT_DIR\venv\Scripts\Activate.ps1"
        
        # Run training
        Write-Log "Starting ML training..."
        $startTime = Get-Date
        
        python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()"
        
        $exitCode = $LASTEXITCODE
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        if ($exitCode -eq 0) {
            Write-Log "TRAINING COMPLETED SUCCESSFULLY!"
            Write-Log "Duration: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s"
            $success = $true
        } else {
            Write-Log "Training exited with code: $exitCode"
            throw "Non-zero exit code"
        }
    }
    catch {
        Write-Log "ERROR: $_"
        Write-Log "Training crashed. Waiting $RETRY_DELAY_SECONDS seconds before retry..."
        Start-Sleep -Seconds $RETRY_DELAY_SECONDS
    }
}

# -----------------------------------------------------------------------------
# STEP 4: CLEANUP AND REPORT
# -----------------------------------------------------------------------------
Write-Log "=========================================="
if ($success) {
    Write-Log "OVERNIGHT TRAINING COMPLETE - SUCCESS"
} else {
    Write-Log "OVERNIGHT TRAINING FAILED AFTER $MAX_RETRIES ATTEMPTS"
}
Write-Log "=========================================="

# Stop keep-awake job
Stop-Job -Job $keepAwakeJob -ErrorAction SilentlyContinue
Remove-Job -Job $keepAwakeJob -ErrorAction SilentlyContinue

# Re-enable sleep (optional - comment out if you want to keep it disabled)
# powercfg -change -standby-timeout-ac 30
# powercfg -change -monitor-timeout-ac 15

# Show summary of trained models
Write-Log "MODELS CREATED:"
Get-ChildItem "$PROJECT_DIR\models\*.pkl" | ForEach-Object {
    Write-Log "  - $($_.Name)"
}

$modelCount = (Get-ChildItem "$PROJECT_DIR\models\*.pkl" | Measure-Object).Count
Write-Log "Total models: $modelCount"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "TRAINING COMPLETE!" -ForegroundColor Green
Write-Host "Models saved: $modelCount" -ForegroundColor Green
Write-Host "Check logs at: $LOG_FILE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")


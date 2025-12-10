# ============================================================================
# ALPHA LOOP CAPITAL - DATABASE SETUP SCRIPT
# Run this in PowerShell as Administrator
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ALPHA LOOP CAPITAL - DATABASE SETUP  " -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if PostgreSQL is installed
Write-Host "[1/5] Checking PostgreSQL installation..." -ForegroundColor Yellow

$pgPath = "C:\Program Files\PostgreSQL\16\bin"
if (Test-Path "$pgPath\psql.exe") {
    Write-Host "  ✓ PostgreSQL found at $pgPath" -ForegroundColor Green
} else {
    Write-Host "  ✗ PostgreSQL NOT FOUND!" -ForegroundColor Red
    Write-Host ""
    Write-Host "  DOWNLOAD AND INSTALL POSTGRESQL:" -ForegroundColor Yellow
    Write-Host "  1. Go to: https://www.postgresql.org/download/windows/" -ForegroundColor White
    Write-Host "  2. Download the Windows installer" -ForegroundColor White
    Write-Host "  3. Run installer, remember the password you set for 'postgres' user" -ForegroundColor White
    Write-Host "  4. Keep default port 5432" -ForegroundColor White
    Write-Host "  5. Run this script again after installing" -ForegroundColor White
    Write-Host ""
    exit 1
}

# Step 2: Add PostgreSQL to PATH
Write-Host "[2/5] Setting up PATH..." -ForegroundColor Yellow
$env:PATH = "$pgPath;$env:PATH"
Write-Host "  ✓ PostgreSQL added to PATH" -ForegroundColor Green

# Step 3: Get password
Write-Host ""
Write-Host "[3/5] Database credentials needed..." -ForegroundColor Yellow
$pgPassword = Read-Host "Enter your PostgreSQL 'postgres' user password" -AsSecureString
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($pgPassword)
$pgPasswordPlain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

# Set environment variable for psql
$env:PGPASSWORD = $pgPasswordPlain

# Step 4: Create database
Write-Host ""
Write-Host "[4/5] Creating database..." -ForegroundColor Yellow

# Check if database exists
$dbExists = & "$pgPath\psql.exe" -h localhost -U postgres -lqt 2>$null | Select-String "alc_training"

if ($dbExists) {
    Write-Host "  Database 'alc_training' already exists" -ForegroundColor Yellow
    $recreate = Read-Host "  Recreate? This will DELETE ALL DATA! (yes/no)"
    if ($recreate -eq "yes") {
        & "$pgPath\psql.exe" -h localhost -U postgres -c "DROP DATABASE alc_training;"
        & "$pgPath\psql.exe" -h localhost -U postgres -c "CREATE DATABASE alc_training;"
        Write-Host "  ✓ Database recreated" -ForegroundColor Green
    }
} else {
    & "$pgPath\psql.exe" -h localhost -U postgres -c "CREATE DATABASE alc_training;"
    Write-Host "  ✓ Database 'alc_training' created" -ForegroundColor Green
}

# Step 5: Run schema
Write-Host ""
Write-Host "[5/5] Creating tables..." -ForegroundColor Yellow

$schemaFile = "C:\Users\tom\ALCAlgo\ALC-Algo\database\alc_schema.sql"
if (Test-Path $schemaFile) {
    & "$pgPath\psql.exe" -h localhost -U postgres -d alc_training -f $schemaFile
    Write-Host "  ✓ Schema created successfully" -ForegroundColor Green
} else {
    Write-Host "  ✗ Schema file not found at: $schemaFile" -ForegroundColor Red
    Write-Host "  Creating schema file now..." -ForegroundColor Yellow
}

# Clear password from memory
$env:PGPASSWORD = ""
[System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DATABASE SETUP COMPLETE!             " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "CONNECTION INFO:" -ForegroundColor White
Write-Host "  Host: localhost" -ForegroundColor Gray
Write-Host "  Port: 5432" -ForegroundColor Gray
Write-Host "  Database: alc_training" -ForegroundColor Gray
Write-Host "  User: postgres" -ForegroundColor Gray
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "  1. Run: python data_pipeline.py --full" -ForegroundColor White
Write-Host "  2. This will pull 5 years of data for 80+ tickers" -ForegroundColor White
Write-Host ""

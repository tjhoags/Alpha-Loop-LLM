# ALC-Algo GitHub Setup Script (Windows PowerShell)

Write-Host "================================================================="
Write-Host "ALC-ALGO GITHUB SETUP"
Write-Host "================================================================="

# 1. Check Git
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git not found! Please install Git."
    exit 1
}

# 2. Check if already initialized
if (Test-Path ".git") {
    Write-Warning "Repository already initialized."
    $choice = Read-Host "Do you want to re-initialize? (y/n)"
    if ($choice -ne 'y') {
        exit 0
    }
}

# 3. Initialize
Write-Host "Initializing Git repository..."
git init -b main

# 4. Check .gitignore
if (!(Test-Path ".gitignore")) {
    Write-Warning ".gitignore not found! Creating default..."
    Add-Content .gitignore "config/secrets.py`n.env`ndata/`n__pycache__/`n*.log"
}

# 5. Add and Commit
Write-Host "Adding files..."
git add .
git commit -m "feat: Initial setup of ALC-Algo Institutional Platform"

# 6. Prompt for Remote
$remote = Read-Host "Enter GitHub Repository URL (e.g., https://github.com/User/Repo.git)"
if ($remote) {
    if (git remote | Select-String "origin") {
        git remote remove origin
    }
    git remote add origin $remote
    Write-Host "Remote 'origin' added."
    
    Write-Host "Pushing to main..."
    git push -u origin main
    
    Write-Host "Creating 'dev' branch..."
    git checkout -b dev
    git push -u origin dev
    
    Write-Host "Returning to 'dev' branch."
} else {
    Write-Host "Skipping remote setup. Run 'git remote add origin <url>' manually."
}

Write-Host "================================================================="
Write-Host "SETUP COMPLETE"
Write-Host "================================================================="


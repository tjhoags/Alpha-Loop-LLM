# ALC-Algo Complete Setup Instructions

**Author:** Tom Hogan | Alpha Loop Capital, LLC  
**Version:** 1.0.0  
**Last Updated:** December 2024

---

## Table of Contents

1. [Prerequisites Overview](#1-prerequisites-overview)
2. [Software Installation](#2-software-installation)
3. [Azure Account Setup](#3-azure-account-setup)
4. [Azure Infrastructure Deployment](#4-azure-infrastructure-deployment)
5. [Local Development Setup](#5-local-development-setup)
6. [API Keys and Secrets Configuration](#6-api-keys-and-secrets-configuration)
7. [Database and Storage Setup](#7-database-and-storage-setup)
8. [Running the Application](#8-running-the-application)
9. [Testing and Verification](#9-testing-and-verification)
10. [Data Quality Standards](#10-data-quality-standards)
11. [Troubleshooting](#11-troubleshooting)
12. [Quick Reference](#12-quick-reference)

---

## 1. Prerequisites Overview

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Storage | 50 GB SSD | 100+ GB SSD |
| Network | 10 Mbps | 50+ Mbps |

### Software Requirements
| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Core runtime |
| Git | 2.40+ | Version control |
| Docker | 24.0+ | Containerization |
| Terraform | 1.6+ | Infrastructure as Code |
| Azure CLI | 2.55+ | Azure management |
| Node.js | 18+ | (Optional) For some tools |

### Required Accounts
- **Azure Account** - With pay-as-you-go or subscription
- **GitHub Account** - For repository access
- **Interactive Brokers** - For trading (paper or live)
- **Alpha Vantage** - For market data (free tier available)

### Optional Accounts (Enhance Functionality)
- **OpenAI** - For GPT-4 analysis
- **Anthropic** - For Claude analysis
- **Coinbase** - For crypto trading
- **Slack** - For notifications

---

## 2. Software Installation

### 2.1 Open Your Terminal

**Windows (PowerShell):**
```powershell
# Press Windows + X, select "Windows Terminal" or "PowerShell"
# Or press Windows + R, type "powershell", press Enter
```

**macOS:**
```bash
# Press Cmd + Space, type "Terminal", press Enter
```

**Linux:**
```bash
# Press Ctrl + Alt + T
```

### 2.2 Install Python 3.10+

**Windows:**
```powershell
# Option 1: Download from python.org
# Go to https://www.python.org/downloads/
# Download Python 3.10 or later
# Run installer WITH "Add Python to PATH" CHECKED

# Option 2: Using winget (Windows Package Manager)
winget install Python.Python.3.10

# Verify installation
python --version
# Expected output: Python 3.10.x or higher
```

**macOS:**
```bash
# Using Homebrew (install Homebrew first if needed: https://brew.sh)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Verify installation
python3 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip -y

# Verify installation
python3 --version
```

### 2.3 Install Git

**Windows:**
```powershell
# Using winget
winget install Git.Git

# Or download from https://git-scm.com/download/win

# Verify installation
git --version
# Expected: git version 2.40.0 or higher

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**macOS:**
```bash
brew install git
git --version
```

**Linux:**
```bash
sudo apt install git -y
git --version
```

### 2.4 Install Docker Desktop

**Windows:**
```powershell
# Download Docker Desktop from https://www.docker.com/products/docker-desktop
# Run the installer
# Restart your computer after installation

# After restart, verify:
docker --version
# Expected: Docker version 24.x.x or higher

docker-compose --version
# Expected: Docker Compose version v2.x.x or higher
```

**macOS:**
```bash
# Download from https://www.docker.com/products/docker-desktop
# Or use Homebrew:
brew install --cask docker

# Open Docker Desktop application
# Verify:
docker --version
```

**Linux:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and back in, then verify:
docker --version
```

### 2.5 Install Azure CLI

**Windows:**
```powershell
# Using winget
winget install Microsoft.AzureCLI

# Or download from https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows

# Verify installation
az --version
# Expected: azure-cli 2.55.0 or higher
```

**macOS:**
```bash
brew install azure-cli
az --version
```

**Linux:**
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az --version
```

### 2.6 Install Terraform

**Windows:**
```powershell
# Using winget
winget install Hashicorp.Terraform

# Or using Chocolatey
choco install terraform

# Verify installation
terraform --version
# Expected: Terraform v1.6.0 or higher
```

**macOS:**
```bash
brew install terraform
terraform --version
```

**Linux:**
```bash
# Add HashiCorp GPG key
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg

# Add repository
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list

# Install
sudo apt update && sudo apt install terraform -y
terraform --version
```

### 2.7 Verify All Installations

Run this script to verify everything is installed:

```powershell
# Windows PowerShell
Write-Host "=== ALC-Algo Installation Verification ===" -ForegroundColor Green

# Python
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "3\.1[0-9]") {
    Write-Host "[OK] Python: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Python 3.10+ required. Got: $pythonVersion" -ForegroundColor Red
}

# Git
$gitVersion = git --version 2>&1
if ($gitVersion -match "git version") {
    Write-Host "[OK] Git: $gitVersion" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Git not found" -ForegroundColor Red
}

# Docker
$dockerVersion = docker --version 2>&1
if ($dockerVersion -match "Docker version") {
    Write-Host "[OK] Docker: $dockerVersion" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Docker not found" -ForegroundColor Red
}

# Azure CLI
$azVersion = az --version 2>&1 | Select-Object -First 1
if ($azVersion -match "azure-cli") {
    Write-Host "[OK] Azure CLI: $azVersion" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Azure CLI not found" -ForegroundColor Red
}

# Terraform
$tfVersion = terraform --version 2>&1 | Select-Object -First 1
if ($tfVersion -match "Terraform v") {
    Write-Host "[OK] Terraform: $tfVersion" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Terraform not found" -ForegroundColor Red
}

Write-Host "=== Verification Complete ===" -ForegroundColor Green
```

---

## 3. Azure Account Setup

### 3.1 Create Azure Account

1. **Go to Azure Portal:**
   ```
   https://portal.azure.com
   ```

2. **Create Account:**
   - Click "Start free" or "Sign in"
   - Use your Microsoft account or create one
   - Enter payment information (required even for free tier)
   - Complete phone verification

3. **Verify Account:**
   ```powershell
   # Login to Azure from terminal
   az login
   
   # This opens a browser - sign in with your Azure credentials
   # After login, you'll see your subscriptions listed
   
   # Set your subscription (if you have multiple)
   az account list --output table
   az account set --subscription "YOUR_SUBSCRIPTION_NAME_OR_ID"
   
   # Verify current subscription
   az account show --output table
   ```

### 3.2 Create Service Principal

A Service Principal is required for CI/CD and Terraform:

```powershell
# Create Service Principal with Contributor role
$subscriptionId = (az account show --query id -o tsv)
$spCredentials = az ad sp create-for-rbac `
    --name "sp-alc-algo-cicd" `
    --role Contributor `
    --scopes /subscriptions/$subscriptionId `
    --sdk-auth

# IMPORTANT: Save this output securely - you will need it for GitHub Secrets
Write-Host $spCredentials

# The output looks like:
# {
#   "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
#   "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
#   "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
#   "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
#   ...
# }
```

### 3.3 Create Resource Group for Terraform State

```powershell
# Create resource group for Terraform state
az group create --name rg-terraform-state --location eastus

# Create storage account for state file
# NOTE: Storage account name must be globally unique (3-24 chars, lowercase, numbers only)
$storageAccountName = "stalcalgotfstate" + (Get-Random -Maximum 9999)
az storage account create `
    --name $storageAccountName `
    --resource-group rg-terraform-state `
    --location eastus `
    --sku Standard_LRS `
    --encryption-services blob

# Get storage account key
$storageKey = az storage account keys list `
    --account-name $storageAccountName `
    --resource-group rg-terraform-state `
    --query '[0].value' -o tsv

# Create container for state
az storage container create `
    --name tfstate `
    --account-name $storageAccountName `
    --account-key $storageKey

Write-Host "Storage Account Name: $storageAccountName"
Write-Host "Storage Key: $storageKey"

# SAVE THESE VALUES - you'll need them for Terraform configuration
```

---

## 4. Azure Infrastructure Deployment

### 4.1 Clone the Repository

```powershell
# Navigate to your projects directory
cd C:\Users\tom\Projects  # Windows
# or
cd ~/projects  # macOS/Linux

# Clone the repository
git clone https://github.com/AlphaLoopCapital/ALC-Algo.git
cd ALC-Algo

# Verify you're in the correct directory
Get-Location  # Windows
# or
pwd  # macOS/Linux
```

### 4.2 Configure Terraform Variables

```powershell
# Navigate to Terraform directory
cd infra/azure/terraform

# Copy the example variables file
Copy-Item terraform.tfvars.example terraform.tfvars

# Edit the variables file
# Windows: Use notepad, VS Code, or your preferred editor
notepad terraform.tfvars
# or
code terraform.tfvars
```

**Edit `terraform.tfvars`:**
```hcl
# General Configuration
project_name = "alc-algo"
environment  = "dev"  # Start with dev, then staging, then prod
location     = "eastus"

tags = {
  Project     = "ALC-Algo"
  Owner       = "Tom Hogan"
  Environment = "dev"
  ManagedBy   = "Terraform"
}

# Resource naming (modify if needed)
resource_group_name     = "rg-alc-algo"
storage_account_name    = "stalcalgo"
key_vault_name          = "kv-alc-algo"
container_registry_name = "cralcalgo"

# ML Workspace
ml_workspace_name = "mlw-alc-algo"
ml_compute_name   = "alc-training-cluster"
ml_compute_vm_size = "Standard_DS3_v2"  # 4 cores, 14 GB RAM
ml_compute_min_nodes = 0  # Scale to zero when not in use
ml_compute_max_nodes = 2

# Container Apps
container_app_env_name = "cae-alc-algo"
container_app_name     = "ca-alc-algo"
```

### 4.3 Initialize and Apply Terraform

```powershell
# Set environment variables for backend (use values from step 3.3)
$env:ARM_CLIENT_ID = "your-client-id"
$env:ARM_CLIENT_SECRET = "your-client-secret"
$env:ARM_SUBSCRIPTION_ID = "your-subscription-id"
$env:ARM_TENANT_ID = "your-tenant-id"

# Initialize Terraform with backend configuration
terraform init `
    -backend-config="storage_account_name=YOUR_STORAGE_ACCOUNT" `
    -backend-config="container_name=tfstate" `
    -backend-config="key=alc-algo-dev.tfstate" `
    -backend-config="resource_group_name=rg-terraform-state"

# Expected output:
# Terraform has been successfully initialized!

# Validate configuration
terraform validate

# Expected output:
# Success! The configuration is valid.

# Preview changes (DO THIS FIRST - review what will be created)
terraform plan -out=tfplan

# Review the plan output - it should show resources to be created

# Apply the infrastructure (this will create real Azure resources)
# Type 'yes' when prompted
terraform apply tfplan

# Expected output after apply:
# Apply complete! Resources: X added, 0 changed, 0 destroyed.
# 
# Outputs:
# container_registry_login_server = "cralcalgodev.azurecr.io"
# key_vault_name = "kv-alc-algo-dev"
# ml_workspace_name = "mlw-alc-algo-dev"
# ...

# Save the outputs
terraform output > ../../../azure_outputs.txt
```

### 4.4 Verify Azure Resources

```powershell
# List all resources in your resource group
az resource list --resource-group rg-alc-algo-dev --output table

# Verify Key Vault
az keyvault show --name kv-alc-algo-dev --output table

# Verify Storage Account
az storage account show --name stalcalgodev --output table

# Verify Container Registry
az acr show --name cralcalgodev --output table

# Verify ML Workspace
az ml workspace show --name mlw-alc-algo-dev --resource-group rg-alc-algo-dev

# Expected: All commands should return details about the resources
```

---

## 5. Local Development Setup

### 5.1 Create Python Virtual Environment

```powershell
# Navigate back to project root
cd C:\Users\tom\.cursor\worktrees\ALC-Algo\adu
# or wherever your project is located

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows Command Prompt:
# venv\Scripts\activate.bat

# macOS/Linux:
# source venv/bin/activate

# Verify activation (you should see (venv) in your prompt)
# (venv) PS C:\Users\tom\...\ALC-Algo>
```

### 5.2 Install Python Dependencies

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Verify installation
pip list | Select-String "pandas|numpy|openai|anthropic"
# Should show installed packages

# Verify specific imports
python -c "import pandas; import numpy; print('Core packages OK')"
python -c "from src.core.agent_base import BaseAgent; print('Agent base OK')"
```

### 5.3 Project Structure Verification

```powershell
# Verify project structure
Get-ChildItem -Recurse -Directory | Select-Object FullName | Where-Object { $_.FullName -match "src\\agents" }

# Expected directories:
# src/agents/
#   compliance_agent/
#   data_agent/
#   execution_agent/
#   ghost_agent/
#   ...
```

---

## 6. API Keys and Secrets Configuration

### 6.1 Create Secrets File

```powershell
# Navigate to config directory
cd config

# Copy the example secrets file
Copy-Item secrets.py.example secrets.py

# Edit secrets.py
notepad secrets.py
# or
code secrets.py
```

**Edit `config/secrets.py`:**
```python
"""
ALC-Algo Secrets Configuration
Author: Tom Hogan | Alpha Loop Capital, LLC

IMPORTANT: Never commit this file to version control!
"""

# Path to your master environment file (if using Dropbox sync)
ENV_FILE_PATH = "C:/Users/tom/Alphaloopcapital Dropbox/master_alc_env"

# Or set keys directly (less secure, but works)
# Leave empty strings for APIs you don't have yet
DIRECT_KEYS = {
    # Market Data APIs
    "ALPHA_VANTAGE_API_KEY": "your-alpha-vantage-key",  # Get free at alphavantage.co
    
    # AI/ML APIs
    "OPENAI_API_KEY": "sk-your-openai-key",
    "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key",
    "GOOGLE_API_KEY_1": "your-google-key",
    
    # Broker APIs
    "IBKR_ACCOUNT_ID": "your-account-id",
    "IBKR_HOST": "127.0.0.1",
    "IBKR_PORT": "7497",  # 7497 for paper, 7496 for live
    
    # Optional APIs
    "COINBASE_API_KEY": "",
    "COINBASE_API_SECRET": "",
    "SLACK_WEBHOOK_URL": "",
    "NOTION_API_KEY": "",
}
```

### 6.2 Create Local Environment File

```powershell
# Create .env file for local development
cd ..  # Back to project root
Copy-Item env.example .env
notepad .env
```

**Edit `.env`:**
```bash
# ALC-Algo Local Environment Configuration
# Author: Tom Hogan | Alpha Loop Capital, LLC

# Environment
ENVIRONMENT=development

# Market Data
ALPHA_VANTAGE_API_KEY=your-key-here

# AI/ML (at least one required for full functionality)
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
GOOGLE_API_KEY_1=your-google-key

# Broker - IBKR
IBKR_ACCOUNT_ID=your-account-id
IBKR_HOST=127.0.0.1
IBKR_PORT=7497

# Azure (from terraform output)
AZURE_STORAGE_CONNECTION_STRING=your-connection-string
AZURE_KEY_VAULT_NAME=kv-alc-algo-dev

# Optional: Notifications
SLACK_WEBHOOK_URL=
SLACK_BOT_TOKEN=

# Logging
LOG_LEVEL=INFO
```

### 6.3 Upload Secrets to Azure Key Vault

```powershell
# Upload secrets to Azure Key Vault (more secure than local files)
$kvName = "kv-alc-algo-dev"

# Upload each secret
az keyvault secret set --vault-name $kvName --name "ALPHA-VANTAGE-API-KEY" --value "your-key"
az keyvault secret set --vault-name $kvName --name "OPENAI-API-KEY" --value "your-key"
az keyvault secret set --vault-name $kvName --name "ANTHROPIC-API-KEY" --value "your-key"
az keyvault secret set --vault-name $kvName --name "IBKR-ACCOUNT-ID" --value "your-account-id"

# Verify secrets were uploaded
az keyvault secret list --vault-name $kvName --output table
```

### 6.4 Obtain API Keys

**Alpha Vantage (Required - Free):**
1. Go to https://www.alphavantage.co/support/#api-key
2. Enter email, click "GET FREE API KEY"
3. Copy the key to your `.env` file

**OpenAI (Recommended):**
1. Go to https://platform.openai.com/api-keys
2. Create account or sign in
3. Click "Create new secret key"
4. Copy key to `.env` (starts with `sk-`)

**Anthropic (Recommended):**
1. Go to https://console.anthropic.com/
2. Create account or sign in
3. Go to API Keys section
4. Create new key
5. Copy key to `.env` (starts with `sk-ant-`)

**Interactive Brokers:**
1. Create IBKR account at https://www.interactivebrokers.com
2. Download and install TWS or IB Gateway
3. Enable API access in TWS settings:
   - File → Global Configuration → API → Settings
   - Enable "Active X and Socket Clients"
   - Set port (7497 for paper, 7496 for live)
   - Add 127.0.0.1 to trusted IPs

---

## 7. Database and Storage Setup

### 7.1 Local Data Directories

```powershell
# Create local data directories
New-Item -ItemType Directory -Force -Path data/raw
New-Item -ItemType Directory -Force -Path data/processed
New-Item -ItemType Directory -Force -Path data/portfolio
New-Item -ItemType Directory -Force -Path data/models
New-Item -ItemType Directory -Force -Path data/logs
New-Item -ItemType Directory -Force -Path data/cache

# Verify structure
Get-ChildItem -Path data -Recurse -Directory
```

### 7.2 Download Sample Data (Optional)

```powershell
# The system can fetch live data, but sample data helps testing
# Create a sample portfolio file

$sampleTrades = @"
Date,Symbol,Action,Quantity,Price,Fees,Notes
2024-01-15,AAPL,BUY,100,185.50,1.00,Initial position
2024-02-01,NVDA,BUY,50,650.00,1.00,AI thesis
2024-02-15,MSFT,BUY,75,400.00,1.00,Cloud growth
2024-03-01,AAPL,SELL,50,175.00,1.00,Take profits
"@

$sampleTrades | Out-File -FilePath data/portfolio/sample_trades.csv -Encoding UTF8

Write-Host "Sample trades file created at data/portfolio/sample_trades.csv"
```

---

## 8. Running the Application

### 8.1 Verify Configuration

```powershell
# Make sure venv is activated
.\venv\Scripts\Activate.ps1

# Test configuration loading
python -c "
from config.settings import settings
print('Configuration loaded successfully!')
print(f'Alpha Vantage Key: {\"SET\" if settings.alpha_vantage_api_key else \"NOT SET\"}')
print(f'OpenAI Key: {\"SET\" if settings.openai_api_key else \"NOT SET\"}')
print(f'IBKR Host: {settings.ibkr_host}')
print(f'IBKR Port: {settings.ibkr_port}')
"
```

### 8.2 Run Agent Initialization Test

```powershell
# Test agent initialization without full workflow
python -c "
from src.core.agent_base import BaseAgent, AgentTier, AgentToughness
print('BaseAgent imported successfully!')
print(f'Available thinking modes: {len([t for t in BaseAgent.__dict__ if \"think\" in str(t)])}')
print('Core agent infrastructure: OK')
"
```

### 8.3 Run Main Application

```powershell
# Run the main application
python main.py

# Expected output:
# ======================================================================
# ALC-ALGO - ALGORITHMIC TRADING PLATFORM
# Agent Coordination Architecture (ACA)
# Author: Tom Hogan | Alpha Loop Capital, LLC
# ======================================================================
#
# Hierarchy:
#   Tier 0: HOAGS (Human Oversight - Tom Hogan)
#   Tier 1: GhostAgent (Autonomous Master Controller)
#   Tier 2: 8 Senior Agents
#   Tier 2.5: Hacker Team (BlackHat + WhiteHat)
#   Tier 3: 65+ Swarm Agents
#
# ... (agent initialization logs)
#
# DAILY WORKFLOW COMPLETE
# ======================================================================
```

### 8.4 Run Specific Commands

```powershell
# Run morning scan (when CLI is implemented)
python -m src.alc_algo.cli scan

# Analyze specific ticker
python -m src.alc_algo.cli analyze AAPL

# View portfolio
python -m src.alc_algo.cli portfolio

# Build training dataset
python -m src.alc_algo.cli dataset
```

---

## 9. Testing and Verification

### 9.1 Run Unit Tests

```powershell
# Make sure venv is activated
.\venv\Scripts\Activate.ps1

# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Expected output:
# ============================= test session starts =============================
# platform win32 -- Python 3.10.x, pytest-7.x.x
# collected X items
# 
# tests/test_config.py::test_settings_load PASSED
# tests/test_features.py::test_technical_indicators PASSED
# ...
#
# ============================= X passed in Y.YYs ==============================
```

### 9.2 Run Specific Test Categories

```powershell
# Test configuration only
pytest tests/test_config.py -v

# Test features
pytest tests/test_features.py -v

# Test backtesting
pytest tests/test_backtest.py -v

# Test with verbose output
pytest tests/ -v --tb=long

# Run tests matching a pattern
pytest tests/ -v -k "config or feature"
```

### 9.3 Verification Checklist

Run this comprehensive verification script:

```powershell
# Save as verify_setup.py and run
python verify_setup.py
```

```python
#!/usr/bin/env python
"""
ALC-Algo Setup Verification Script
Author: Tom Hogan | Alpha Loop Capital, LLC
"""

import sys
from pathlib import Path

def check(name: str, condition: bool, details: str = ""):
    status = "✓" if condition else "✗"
    color = "\033[92m" if condition else "\033[91m"
    reset = "\033[0m"
    print(f"{color}[{status}]{reset} {name}")
    if details and not condition:
        print(f"    └─ {details}")
    return condition

def main():
    print("\n" + "=" * 60)
    print("ALC-ALGO SETUP VERIFICATION")
    print("=" * 60 + "\n")
    
    results = []
    
    # 1. Python Version
    version = sys.version_info
    results.append(check(
        "Python 3.10+",
        version.major == 3 and version.minor >= 10,
        f"Found: {version.major}.{version.minor}"
    ))
    
    # 2. Core Imports
    try:
        import pandas
        import numpy
        results.append(check("Pandas & NumPy", True))
    except ImportError as e:
        results.append(check("Pandas & NumPy", False, str(e)))
    
    # 3. ML Libraries
    try:
        import openai
        import anthropic
        results.append(check("AI/ML Libraries (OpenAI, Anthropic)", True))
    except ImportError as e:
        results.append(check("AI/ML Libraries", False, str(e)))
    
    # 4. Broker Client
    try:
        from ib_insync import IB
        results.append(check("IBKR Client (ib_insync)", True))
    except ImportError as e:
        results.append(check("IBKR Client", False, str(e)))
    
    # 5. Agent Base
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.core.agent_base import BaseAgent
        results.append(check("Agent Base Class", True))
    except ImportError as e:
        results.append(check("Agent Base Class", False, str(e)))
    
    # 6. Configuration
    try:
        from config.settings import settings
        results.append(check("Configuration Module", True))
    except Exception as e:
        results.append(check("Configuration Module", False, str(e)))
    
    # 7. Data Directories
    data_dirs = ["data/raw", "data/processed", "data/portfolio", "data/logs"]
    dirs_exist = all(Path(d).exists() for d in data_dirs)
    results.append(check(
        "Data Directories",
        dirs_exist,
        "Run: mkdir -p data/{raw,processed,portfolio,logs}"
    ))
    
    # 8. API Keys
    try:
        from config.settings import settings
        has_alpha_vantage = bool(settings.alpha_vantage_api_key)
        results.append(check(
            "Alpha Vantage API Key",
            has_alpha_vantage,
            "Set ALPHA_VANTAGE_API_KEY in .env"
        ))
    except:
        results.append(check("API Keys", False, "Configuration not loaded"))
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"\033[92mALL CHECKS PASSED ({passed}/{total})\033[0m")
        print("Your ALC-Algo setup is complete!")
    else:
        print(f"\033[93m{passed}/{total} CHECKS PASSED\033[0m")
        print("Please address the failed checks above.")
    
    print("=" * 60 + "\n")
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
```

### 9.4 Integration Test

```powershell
# Test full data pipeline
python -c "
from src.data_ingestion.market_data import MarketDataFetcher

# This requires Alpha Vantage API key
fetcher = MarketDataFetcher()
data = fetcher.fetch_daily('AAPL', days=5)
print(f'Fetched {len(data)} rows of AAPL data')
print(data.head())
"
```

### 9.5 Docker Build Test

```powershell
# Build Docker image locally
docker build -t alc-algo:test -f docker/Dockerfile.dev .

# Expected output:
# [+] Building X.Xs
# ...
# Successfully tagged alc-algo:test

# Run container
docker run --rm alc-algo:test python -c "print('Container works!')"
```

---

## 10. Data Quality Standards

### 10.1 What is GOOD Data?

**Price Data Quality Indicators:**

| Metric | Good | Acceptable | Bad |
|--------|------|------------|-----|
| Missing Values | < 0.1% | < 1% | > 1% |
| Timestamp Gaps | None | < 5 per year | > 5 per year |
| Price Jumps | < 3 std devs | < 5 std devs | > 5 std devs |
| Volume Anomalies | None | < 1% | > 1% |

**Good Data Characteristics:**
```python
# Example of good data validation
def validate_price_data(df):
    """
    Returns True if data meets quality standards.
    """
    checks = {
        "no_nulls": df.isnull().sum().sum() == 0,
        "positive_prices": (df[['Open', 'High', 'Low', 'Close']] > 0).all().all(),
        "high_gte_low": (df['High'] >= df['Low']).all(),
        "close_in_range": ((df['Close'] >= df['Low']) & (df['Close'] <= df['High'])).all(),
        "positive_volume": (df['Volume'] >= 0).all(),
        "sorted_dates": df.index.is_monotonic_increasing,
    }
    
    for check_name, passed in checks.items():
        print(f"  {check_name}: {'✓' if passed else '✗'}")
    
    return all(checks.values())
```

**Good Data Example:**
```
Date,Open,High,Low,Close,Volume
2024-01-02,185.50,186.25,184.75,185.90,45234567
2024-01-03,186.00,187.50,185.50,187.25,52345678
2024-01-04,187.00,188.25,186.50,187.80,48234567
```

### 10.2 What is BAD Data?

**Red Flags:**

| Issue | Example | Impact |
|-------|---------|--------|
| Missing Dates | Gap from Jan 2 to Jan 5 (missing Jan 3-4) | Breaks time series analysis |
| Null Values | Open=NaN | Calculation errors |
| Negative Prices | Close=-45.00 | Invalid, corrupt data |
| Zero Volume | Volume=0 for trading day | Suspicious, check source |
| Price > High | Close=200, High=150 | Data corruption |
| Duplicate Timestamps | Two rows for same date | Double counting |
| Future Dates | Date > today | Data error |
| Extreme Jumps | 50% price change overnight | May need adjustment |

**Bad Data Example:**
```
Date,Open,High,Low,Close,Volume
2024-01-02,185.50,186.25,184.75,185.90,45234567
2024-01-02,185.50,186.25,184.75,185.90,45234567  # DUPLICATE
2024-01-03,NaN,187.50,185.50,187.25,52345678      # MISSING OPEN
2024-01-04,187.00,186.00,185.50,187.80,48234567   # HIGH < OPEN
2024-01-05,-5.00,188.25,186.50,187.80,48234567    # NEGATIVE PRICE
2024-01-08,187.00,188.25,186.50,187.80,0          # ZERO VOLUME
```

### 10.3 Data Validation Script

Save and run this script to validate your data:

```python
#!/usr/bin/env python
"""
ALC-Algo Data Quality Validator
Author: Tom Hogan | Alpha Loop Capital, LLC
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


class DataQualityValidator:
    """Validates market data quality."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = []
        
    def validate_all(self) -> dict:
        """Run all validation checks."""
        results = {
            'timestamp_check': self._check_timestamps(),
            'price_check': self._check_prices(),
            'volume_check': self._check_volumes(),
            'range_check': self._check_ranges(),
            'duplicates_check': self._check_duplicates(),
            'anomaly_check': self._check_anomalies(),
        }
        
        results['overall_quality'] = self._calculate_quality_score(results)
        results['issues'] = self.issues
        
        return results
    
    def _check_timestamps(self) -> bool:
        """Check timestamp integrity."""
        # Check for sorted dates
        if not self.df.index.is_monotonic_increasing:
            self.issues.append("Timestamps not sorted chronologically")
            return False
        
        # Check for future dates
        if self.df.index.max() > pd.Timestamp.now():
            self.issues.append("Data contains future dates")
            return False
        
        # Check for gaps (excluding weekends/holidays)
        date_diffs = self.df.index.to_series().diff()
        large_gaps = date_diffs[date_diffs > timedelta(days=5)]
        if len(large_gaps) > 0:
            self.issues.append(f"Found {len(large_gaps)} gaps > 5 days")
        
        return True
    
    def _check_prices(self) -> bool:
        """Check price validity."""
        price_cols = ['Open', 'High', 'Low', 'Close']
        existing_cols = [c for c in price_cols if c in self.df.columns]
        
        # Check for nulls
        null_count = self.df[existing_cols].isnull().sum().sum()
        if null_count > 0:
            self.issues.append(f"Found {null_count} null price values")
            return False
        
        # Check for negative prices
        neg_count = (self.df[existing_cols] < 0).sum().sum()
        if neg_count > 0:
            self.issues.append(f"Found {neg_count} negative prices")
            return False
        
        # Check for zero prices
        zero_count = (self.df[existing_cols] == 0).sum().sum()
        if zero_count > 0:
            self.issues.append(f"Found {zero_count} zero prices")
        
        return True
    
    def _check_volumes(self) -> bool:
        """Check volume validity."""
        if 'Volume' not in self.df.columns:
            return True
        
        # Check for negative volume
        neg_vol = (self.df['Volume'] < 0).sum()
        if neg_vol > 0:
            self.issues.append(f"Found {neg_vol} negative volume values")
            return False
        
        # Check for excessive zero volume days
        zero_vol_pct = (self.df['Volume'] == 0).sum() / len(self.df) * 100
        if zero_vol_pct > 5:
            self.issues.append(f"Warning: {zero_vol_pct:.1f}% zero volume days")
        
        return True
    
    def _check_ranges(self) -> bool:
        """Check OHLC relationship validity."""
        if not all(c in self.df.columns for c in ['Open', 'High', 'Low', 'Close']):
            return True
        
        # High must be >= all other prices
        high_violations = (
            (self.df['High'] < self.df['Open']) |
            (self.df['High'] < self.df['Close']) |
            (self.df['High'] < self.df['Low'])
        ).sum()
        
        # Low must be <= all other prices
        low_violations = (
            (self.df['Low'] > self.df['Open']) |
            (self.df['Low'] > self.df['Close']) |
            (self.df['Low'] > self.df['High'])
        ).sum()
        
        total_violations = high_violations + low_violations
        if total_violations > 0:
            self.issues.append(f"Found {total_violations} OHLC range violations")
            return False
        
        return True
    
    def _check_duplicates(self) -> bool:
        """Check for duplicate rows."""
        dup_count = self.df.index.duplicated().sum()
        if dup_count > 0:
            self.issues.append(f"Found {dup_count} duplicate timestamps")
            return False
        return True
    
    def _check_anomalies(self) -> bool:
        """Check for statistical anomalies."""
        if 'Close' not in self.df.columns:
            return True
        
        # Calculate daily returns
        returns = self.df['Close'].pct_change().dropna()
        
        # Flag extreme returns (> 5 std devs)
        std = returns.std()
        extreme = (abs(returns) > 5 * std).sum()
        
        if extreme > 0:
            self.issues.append(f"Warning: {extreme} extreme price moves (>5 std devs)")
        
        return True
    
    def _calculate_quality_score(self, results: dict) -> str:
        """Calculate overall quality score."""
        checks = [v for k, v in results.items() if k != 'issues' and k != 'overall_quality']
        passed = sum(1 for c in checks if c)
        total = len(checks)
        
        ratio = passed / total
        if ratio == 1.0:
            return "EXCELLENT"
        elif ratio >= 0.8:
            return "GOOD"
        elif ratio >= 0.6:
            return "ACCEPTABLE"
        else:
            return "POOR"


def validate_file(filepath: str):
    """Validate a data file."""
    print(f"\nValidating: {filepath}")
    print("-" * 50)
    
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        validator = DataQualityValidator(df)
        results = validator.validate_all()
        
        print(f"Records: {len(df)}")
        print(f"Date Range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {list(df.columns)}")
        print()
        
        for check, passed in results.items():
            if check not in ['issues', 'overall_quality']:
                status = '✓' if passed else '✗'
                print(f"  [{status}] {check}")
        
        print()
        print(f"Quality Score: {results['overall_quality']}")
        
        if results['issues']:
            print("\nIssues Found:")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        return results['overall_quality'] in ['EXCELLENT', 'GOOD']
        
    except Exception as e:
        print(f"Error validating file: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        validate_file(filepath)
    else:
        print("Usage: python validate_data.py <filepath>")
        print("Example: python validate_data.py data/raw/AAPL_daily.csv")
```

### 10.4 Recommended Data Sources

| Source | Type | Quality | Cost | Rate Limit |
|--------|------|---------|------|------------|
| Alpha Vantage | EOD, Intraday | Good | Free tier available | 5 calls/min |
| Yahoo Finance | EOD | Good | Free | Unofficial |
| FRED | Economic | Excellent | Free | 120 calls/min |
| IBKR | Real-time | Excellent | Broker fee | With account |
| Polygon.io | Real-time | Excellent | $29+/month | Varies |

---

## 11. Troubleshooting

### Common Issues and Solutions

**Issue: "Module not found" error**
```powershell
# Solution 1: Ensure venv is activated
.\venv\Scripts\Activate.ps1

# Solution 2: Reinstall dependencies
pip install -r requirements.txt

# Solution 3: Verify Python path
python -c "import sys; print(sys.path)"
```

**Issue: "API key not found"**
```powershell
# Check if .env file exists
Get-Content .env

# Check if key is loaded
python -c "from config.settings import settings; print(settings.alpha_vantage_api_key)"

# Set key directly for testing
$env:ALPHA_VANTAGE_API_KEY="your-key-here"
```

**Issue: IBKR connection failed**
```powershell
# 1. Ensure TWS/Gateway is running
# 2. Check API settings in TWS:
#    - File → Global Configuration → API → Settings
#    - Enable "Active X and Socket Clients"
#    - Port: 7497 (paper) or 7496 (live)
#    - Add 127.0.0.1 to trusted IPs

# 3. Test connection
python -c "
from ib_insync import IB
ib = IB()
try:
    ib.connect('127.0.0.1', 7497, clientId=1)
    print('Connected!')
    ib.disconnect()
except Exception as e:
    print(f'Failed: {e}')
"
```

**Issue: Terraform state lock**
```powershell
# If Terraform shows state lock error
terraform force-unlock <LOCK_ID>
```

**Issue: Docker build fails**
```powershell
# Clear Docker cache
docker system prune -f
docker builder prune -f

# Rebuild
docker build --no-cache -t alc-algo:test -f docker/Dockerfile.dev .
```

**Issue: Tests fail with import errors**
```powershell
# Add project root to Python path
$env:PYTHONPATH = (Get-Location).Path
pytest tests/ -v
```

---

## 12. Quick Reference

### Essential Commands

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Run application
python main.py

# Run tests
pytest tests/ -v

# Build Docker
docker build -t alc-algo:latest -f docker/Dockerfile.prod .

# Deploy infrastructure
cd infra/azure/terraform
terraform plan
terraform apply

# Check Azure resources
az resource list --resource-group rg-alc-algo-dev --output table
```

### File Locations

| File | Purpose |
|------|---------|
| `main.py` | Application entry point |
| `config/settings.py` | Configuration loading |
| `config/secrets.py` | Local secrets (don't commit!) |
| `.env` | Environment variables |
| `requirements.txt` | Python dependencies |
| `infra/azure/terraform/` | Infrastructure code |
| `tests/` | Test files |
| `data/` | Local data storage |

### Support Contacts

- **Technical Issues:** Review logs in `data/logs/`
- **Azure Issues:** Check Azure Portal for resource status
- **API Issues:** Verify keys in `.env` and quota limits

---

**Author:** Tom Hogan | Alpha Loop Capital, LLC  
**Document Version:** 1.0.0

*Built tough as hell. No limits. No excuses.*


# Azure Complete Setup Guide - FULL Step-by-Step Instructions

**Author:** Tom Hogan | **Organization:** Alpha Loop Capital, LLC  
**Created by:** SetupAgent  
**Date:** December 9, 2025

---

## 🎯 Overview

This guide provides **COMPLETE, step-by-step instructions** for setting up Azure infrastructure for the ALC-Algo trading platform. Follow each step exactly as described.

**Requirements:**
- You have an Azure account (you mentioned you have one already ✅)
- Windows 10/11 machine (your Lenovo workstation)
- macOS (your MacBook Pro)
- Internet connection

---

## 📋 Table of Contents

1. [Install Required Tools](#step-1-install-required-tools)
2. [Login to Azure](#step-2-login-to-azure)
3. [Create Service Principal](#step-3-create-service-principal)
4. [Configure Environment Variables](#step-4-configure-environment-variables)
5. [Deploy Infrastructure with Terraform](#step-5-deploy-infrastructure-with-terraform)
6. [Upload API Keys to Key Vault](#step-6-upload-api-keys-to-key-vault)
7. [Configure Multi-Machine Training](#step-7-configure-multi-machine-training)
8. [Verify Setup](#step-8-verify-setup)

---

## Step 1: Install Required Tools

### On Windows (Your Lenovo Workstation)

#### 1.1 Open PowerShell as Administrator

1. Press `Windows Key + X`
2. Click **"Windows Terminal (Admin)"** or **"PowerShell (Admin)"**
3. Click **"Yes"** when prompted for admin permissions

#### 1.2 Install Azure CLI

Copy and paste this command:

```powershell
winget install Microsoft.AzureCLI
```

**Alternative method** (if winget doesn't work):
1. Go to: https://aka.ms/installazurecliwindows
2. Download the MSI installer
3. Run the installer, click Next through all screens
4. When finished, close and reopen PowerShell

#### 1.3 Install Terraform

```powershell
winget install HashiCorp.Terraform
```

**Alternative method:**
1. Go to: https://www.terraform.io/downloads
2. Download the Windows AMD64 version
3. Extract the zip to `C:\terraform`
4. Add to PATH:
   ```powershell
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\terraform", "User")
   ```
5. Close and reopen PowerShell

#### 1.4 Verify Installations

```powershell
az --version
terraform --version
```

You should see version numbers for both. If not, restart your computer and try again.

---

### On macOS (Your MacBook Pro)

#### 1.1 Open Terminal

1. Press `Command + Space` to open Spotlight
2. Type **"Terminal"** and press Enter

#### 1.2 Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the prompts. When finished, run:
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

#### 1.3 Install Azure CLI and Terraform

```bash
brew install azure-cli terraform
```

#### 1.4 Verify Installations

```bash
az --version
terraform --version
```

---

## Step 2: Login to Azure

### On Windows (PowerShell)

```powershell
# Navigate to your project
cd "C:\Users\tom\.cursor\worktrees\ALC-Algo\coj"

# Login to Azure (this will open a browser)
az login
```

**What happens:**
1. A browser window will open
2. Sign in with your Azure account
3. The browser will say "You have logged into Microsoft Azure"
4. Close the browser and return to PowerShell

#### 2.1 Set the correct subscription

```powershell
# List all subscriptions
az account list --output table

# Set the subscription you want to use (replace with your subscription ID)
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Verify
az account show --query "{Name:name, ID:id}" --output table
```

**Note:** Write down your Subscription ID - you'll need it later.

---

### On macOS (Terminal)

```bash
# Navigate to your project
cd ~/ALC-Algo  # or wherever you cloned the repo

# Login to Azure
az login

# Follow the same subscription steps as Windows
az account list --output table
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

---

## Step 3: Create Service Principal

A Service Principal is like a "robot user" that allows your code to access Azure resources.

### On Windows (PowerShell)

```powershell
# Navigate to project directory
cd "C:\Users\tom\.cursor\worktrees\ALC-Algo\coj"

# Get your subscription ID
$SUBSCRIPTION_ID = (az account show --query id -o tsv)
Write-Host "Subscription ID: $SUBSCRIPTION_ID"

# Create Service Principal
az ad sp create-for-rbac `
    --name "sp-alc-algo-cicd" `
    --role Contributor `
    --scopes "/subscriptions/$SUBSCRIPTION_ID" `
    --sdk-auth | Out-File -FilePath "azure_credentials.json"
```

**IMPORTANT:** This creates a file called `azure_credentials.json`. The file contains:
- `clientId` - Your App ID
- `clientSecret` - Your password (NEVER share this)
- `tenantId` - Your Azure tenant
- `subscriptionId` - Your subscription

**Open and save these values:**
```powershell
Get-Content azure_credentials.json
```

Copy these values somewhere safe (password manager, encrypted file, etc.)

---

## Step 4: Configure Environment Variables

### 4.1 Create Azure Config File

On Windows PowerShell:

```powershell
cd "C:\Users\tom\.cursor\worktrees\ALC-Algo\coj"

# Create the azure config file
@"
# =============================================================================
# ALC-Algo Azure Configuration
# Author: Tom Hogan | Alpha Loop Capital, LLC
# =============================================================================

azure:
  # Subscription and tenant (from az account show)
  subscription_id: "YOUR_SUBSCRIPTION_ID"
  tenant_id: "YOUR_TENANT_ID"
  
  # Resource Group
  resource_group: "rg-alc-algo-dev"
  location: "eastus"
  
  # Storage Account (for agent memory/state)
  storage:
    account_name: "stalcalgodev"
    container_name: "agent-memory"
    
  # Key Vault (for API secrets)
  key_vault:
    name: "kv-alc-algo-dev"
    
  # Machine Learning
  ml_workspace:
    name: "mlw-alc-algo-dev"
    compute_cluster: "alc-training-cluster"

# Multi-machine training configuration
training:
  # Enable multi-machine sync (Lenovo + MacBook)
  multi_machine_sync: true
  
  # Sync interval in seconds
  sync_interval: 300
  
  # State merge strategy: "latest", "merge", "weighted"
  merge_strategy: "merge"
"@ | Out-File -FilePath "config\azure_config.yaml" -Encoding utf8
```

### 4.2 Update secrets.py

```powershell
# Create secrets.py from example
Copy-Item "config\secrets.py.example" "config\secrets.py"

# Edit secrets.py
notepad "config\secrets.py"
```

In the file, update `ENV_FILE_PATH` to point to your Dropbox master_alc_env file:
```python
ENV_FILE_PATH = "C:/Users/tom/Alphaloopcapital Dropbox/master_alc_env"
```

### 4.3 Add Azure Credentials to your master_alc_env file

Open your `master_alc_env` file in Dropbox and add these lines:

```bash
# Azure Credentials (from azure_credentials.json)
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_TENANT_ID=your_tenant_id
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret

# Azure Storage
AZURE_STORAGE_ACCOUNT=stalcalgodev
AZURE_STORAGE_KEY=your_storage_key_here
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=stalcalgodev;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net

# Azure Key Vault
AZURE_KEY_VAULT_NAME=kv-alc-algo-dev
```

---

## Step 5: Deploy Infrastructure with Terraform

### 5.1 Create Terraform State Storage (One-time setup)

```powershell
cd "C:\Users\tom\.cursor\worktrees\ALC-Algo\coj"

# Set variables
$LOCATION = "eastus"
$STATE_RG = "rg-terraform-state"
$STATE_STORAGE = "stalcalgotfstate"

# Create resource group for Terraform state
az group create --name $STATE_RG --location $LOCATION

# Create storage account for Terraform state
az storage account create `
    --name $STATE_STORAGE `
    --resource-group $STATE_RG `
    --location $LOCATION `
    --sku Standard_LRS `
    --encryption-services blob

# Get storage account key
$STORAGE_KEY = (az storage account keys list `
    --account-name $STATE_STORAGE `
    --resource-group $STATE_RG `
    --query '[0].value' -o tsv)

# Create container for state
az storage container create `
    --name tfstate `
    --account-name $STATE_STORAGE `
    --account-key $STORAGE_KEY

Write-Host "Terraform state storage created successfully!"
Write-Host "Storage Account: $STATE_STORAGE"
Write-Host "Container: tfstate"
```

### 5.2 Initialize and Apply Terraform

```powershell
cd "C:\Users\tom\.cursor\worktrees\ALC-Algo\coj\infra\azure\terraform"

# Copy the example tfvars
Copy-Item terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars if needed
notepad terraform.tfvars

# Initialize Terraform
terraform init

# Preview what will be created
terraform plan

# Apply the configuration (type 'yes' when prompted)
terraform apply
```

**This creates:**
- ✅ Resource Group
- ✅ Storage Account with containers (data, models, logs, backups)
- ✅ Key Vault for secrets
- ✅ Container Registry for Docker images
- ✅ Azure Machine Learning Workspace
- ✅ ML Compute Cluster (scales to zero when idle)
- ✅ Application Insights for monitoring
- ✅ Log Analytics Workspace

### 5.3 Save Terraform Outputs

```powershell
# Get the outputs
terraform output -json | Out-File -FilePath "..\..\..\azure_outputs.json"

# View key outputs
terraform output storage_account_name
terraform output key_vault_name
terraform output ml_workspace_name
```

---

## Step 6: Upload API Keys to Key Vault

Now let's store your API keys securely in Azure Key Vault.

### 6.1 Get Key Vault Name

```powershell
# From Terraform output
$KEY_VAULT = "kv-alc-algo-dev"

# Verify Key Vault exists
az keyvault show --name $KEY_VAULT
```

### 6.2 Upload Your API Keys

```powershell
# OpenAI
az keyvault secret set --vault-name $KEY_VAULT --name "OPENAI-API-KEY" --value "sk-your-key-here"

# Anthropic
az keyvault secret set --vault-name $KEY_VAULT --name "ANTHROPIC-API-KEY" --value "sk-ant-your-key-here"

# Google (multiple keys)
az keyvault secret set --vault-name $KEY_VAULT --name "GOOGLE-API-KEY-1" --value "your-key-here"
az keyvault secret set --vault-name $KEY_VAULT --name "GOOGLE-API-KEY-2" --value "your-key-here"
az keyvault secret set --vault-name $KEY_VAULT --name "GOOGLE-API-KEY-3" --value "your-key-here"

# Perplexity
az keyvault secret set --vault-name $KEY_VAULT --name "PERPLEXITY-API-KEY" --value "pplx-your-key-here"

# Alpha Vantage
az keyvault secret set --vault-name $KEY_VAULT --name "ALPHA-VANTAGE-API-KEY" --value "your-key-here"

# IBKR
az keyvault secret set --vault-name $KEY_VAULT --name "IBKR-ACCOUNT-ID" --value "your-account-id"

# Slack
az keyvault secret set --vault-name $KEY_VAULT --name "SLACK-WEBHOOK-URL" --value "https://hooks.slack.com/..."

# Notion
az keyvault secret set --vault-name $KEY_VAULT --name "NOTION-API-KEY" --value "your-key-here"
```

### 6.3 Verify Secrets

```powershell
# List all secrets (names only, not values)
az keyvault secret list --vault-name $KEY_VAULT --query "[].name" -o table
```

---

## Step 7: Configure Multi-Machine Training

This is **CRITICAL** - allows your Lenovo and MacBook to train simultaneously without overwriting each other's work.

### 7.1 Get Storage Account Connection String

```powershell
# Get connection string
$STORAGE_ACCOUNT = "stalcalgodev"
$RESOURCE_GROUP = "rg-alc-algo-dev"

$CONNECTION_STRING = (az storage account show-connection-string `
    --name $STORAGE_ACCOUNT `
    --resource-group $RESOURCE_GROUP `
    --query connectionString -o tsv)

Write-Host "Connection String:"
Write-Host $CONNECTION_STRING

# SAVE THIS - add to your master_alc_env file
```

### 7.2 Create Agent Memory Container

```powershell
az storage container create `
    --name "agent-memory" `
    --account-name $STORAGE_ACCOUNT `
    --connection-string $CONNECTION_STRING
```

### 7.3 Update Your master_alc_env File

Add these to your `master_alc_env` file in Dropbox:

```bash
# Azure Storage for Multi-Machine Training
AZURE_STORAGE_ACCOUNT=stalcalgodev
AZURE_STORAGE_CONNECTION_STRING=<paste the connection string here>
```

### 7.4 Verify Multi-Machine Training Setup

The codebase already supports multi-machine training! Look at `src/core/agent_base.py`:

- Line ~430-460: `_save_persistent_state()` saves with machine-specific filename
- Line ~375-425: `_load_persistent_state()` merges data from ALL machines

**How it works:**
1. **Lenovo** saves state as: `AgentName_LENOVO_WORKSTATION_state.pkl`
2. **MacBook** saves state as: `AgentName_Toms_MacBook_Pro_state.pkl`
3. Both machines **merge** learning outcomes, mistake patterns, and belief histories
4. No data is ever overwritten - everything is combined

---

## Step 8: Verify Setup

### 8.1 Test Azure Connection (Windows)

```powershell
cd "C:\Users\tom\.cursor\worktrees\ALC-Algo\coj"

# Activate virtual environment
.\venv\Scripts\Activate

# Run verification script
python scripts\verify_azure.py
```

### 8.2 Test Storage Access

```powershell
# Test upload
echo "test" | az storage blob upload `
    --account-name "stalcalgodev" `
    --container-name "agent-memory" `
    --name "test.txt" `
    --data "test"

# Test download
az storage blob download `
    --account-name "stalcalgodev" `
    --container-name "agent-memory" `
    --name "test.txt" `
    --file "test_download.txt"

# Cleanup
az storage blob delete `
    --account-name "stalcalgodev" `
    --container-name "agent-memory" `
    --name "test.txt"
```

### 8.3 Test Agent Memory Sync

```python
# test_multi_machine.py
from src.utils.azure_storage import azure_storage

# Test connection
blobs = azure_storage.list_blobs("agent-memory")
print(f"Found {len(blobs)} agent state files")

# Test write
test_data = {"test": "data", "machine": "lenovo"}
azure_storage.save_object("agent-memory", "test_state.pkl", test_data)

# Test read
loaded = azure_storage.load_object("agent-memory", "test_state.pkl")
print(f"Loaded: {loaded}")
```

---

## 🎉 Setup Complete!

You now have:
- ✅ Azure CLI and Terraform installed
- ✅ Service Principal for CI/CD
- ✅ Full Azure infrastructure deployed
- ✅ API keys secured in Key Vault
- ✅ Multi-machine training configured
- ✅ Both Lenovo and MacBook can train simultaneously

---

## Quick Reference Commands

### Windows PowerShell

```powershell
# Navigate to project
cd "C:\Users\tom\.cursor\worktrees\ALC-Algo\coj"

# Activate virtual environment
.\venv\Scripts\Activate

# Check Azure login
az account show

# List Key Vault secrets
az keyvault secret list --vault-name "kv-alc-algo-dev" --query "[].name" -o table

# Check storage
az storage blob list --account-name "stalcalgodev" --container-name "agent-memory" -o table
```

### macOS Terminal

```bash
# Navigate to project
cd ~/ALC-Algo

# Activate virtual environment
source venv/bin/activate

# Same az commands work on macOS
```

---

## Troubleshooting

### "az: command not found"
- Restart your terminal/PowerShell
- Make sure Azure CLI is in your PATH
- Windows: Check `C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2`

### "terraform: command not found"
- Restart your terminal/PowerShell
- Make sure Terraform is in your PATH

### "Key Vault access denied"
```powershell
# Add yourself to Key Vault access policy
$USER_ID = (az ad signed-in-user show --query id -o tsv)
az keyvault set-policy --name "kv-alc-algo-dev" --object-id $USER_ID --secret-permissions get list set delete
```

### Storage connection issues
- Check that AZURE_STORAGE_CONNECTION_STRING is set correctly
- Try running `az login` again

---

## Next Steps

1. **Tonight:** Run `python main.py` to initialize agents
2. **Tonight:** Import your historical trades
3. **This Week:** Start paper trading validation
4. **30 Days:** Review performance, then consider live trading

---

*Setup Guide by SetupAgent*  
*Alpha Loop Capital, LLC*  
*Tom Hogan, Founder*

**"By end of 2026, they will know us."**


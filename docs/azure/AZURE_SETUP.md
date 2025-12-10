# Azure Setup Guide - ALC-Algo

**Institutional-Grade Infrastructure Setup**

This guide covers the complete setup of the Azure infrastructure required for ALC-Algo, from creating an account to deploying the agents.

**Author:** Tom Hogan | Alpha Loop Capital, LLC

---

## Prerequisites

Before starting, ensure you have:
1.  **Azure Account**: [Create Free Account](https://azure.microsoft.com/free/)
2.  **Azure CLI**: [Install Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
3.  **Terraform**: [Install Terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli) (Recommended for Infrastructure as Code)
4.  **Python 3.10+**: Installed locally.

---

## Step 1: Initial Configuration (The "First Code")

Open your terminal (PowerShell or Bash) and log in to Azure.

```bash
# 1. Login to Azure
az login

# 2. Set your subscription (if you have multiple)
az account set --subscription "Your-Subscription-Name-or-ID"

# 3. Create a Resource Group
# This isolates all ALC resources.
az group create --name ALC-Algo-RG --location eastus
```

---

## Step 2: Critical Infrastructure Setup

ALC-Algo relies on several key Azure services.

### 1. Azure Key Vault (Security)
Stores API keys and secrets securely. NEVER store keys in code.

```bash
# Create Key Vault
az keyvault create --name "alc-vault-unique-id" --resource-group ALC-Algo-RG --location eastus

# Add Secrets (Repeat for all keys in secrets.py.example)
az keyvault secret set --vault-name "alc-vault-unique-id" --name "OpenAI-Key" --value "your_key_here"
az keyvault secret set --vault-name "alc-vault-unique-id" --name "IBKR-Account-Paper" --value "your_account_id"
```

### 2. Azure Machine Learning (Compute)
Used for training models and running heavy agents.

```bash
# Create Machine Learning Workspace
az ml workspace create --name "alc-ml-workspace" --resource-group ALC-Algo-RG --location eastus
```

### 3. Azure Container Registry (Deployment)
Stores the Docker images for your agents.

```bash
# Create ACR
az acr create --name "alcregistryunique" --resource-group ALC-Algo-RG --sku Basic --admin-enabled true
```

---

## Step 3: Software & Data Requirements

### Software Environment
Ensure your local or cloud VM has the following:

1.  **Python Libraries**:
    ```bash
    pip install azure-identity azure-keyvault-secrets azure-ai-ml
    pip install -r requirements.txt
    ```

2.  **Docker**: Required for containerizing agents.

### Data Needs (Good vs. Bad Data)

**Good Data Criteria:**
-   **Source**: Reputable APIs (Alpha Vantage, FMP, IBKR).
-   **Continuity**: No missing candles/timeframes.
-   **Latency**: Real-time or delayed < 15min (for paper).
-   **Cleanliness**: Adjusted for splits/dividends.

**Bad Data Indicators:**
-   **Zero Volume**: Candles with 0 volume during market hours.
-   **Spikes**: Price changes > 50% in 1 minute (usually bad ticks).
-   **Gaps**: Missing timestamps.

**Action**: Run the data validation script before trading.
```bash
python src/data/ingestion.py --validate-only
```

---

## Step 4: Deployment & Testing

### 1. Configure Environment
Create a `.env` file referencing your Azure resources.

```bash
# .env
AZURE_KEY_VAULT_URL="https://alc-vault-unique-id.vault.azure.net/"
AZURE_TENANT_ID="your-tenant-id"
AZURE_CLIENT_ID="your-client-id"
AZURE_CLIENT_SECRET="your-client-secret"
```

### 2. Run the System (Terminal)

```bash
# Navigate to project root
cd C:\Users\tom\ALC-Algo

# Activate virtual environment (if used)
# .\venv\Scripts\activate

# Run Main System
python main.py
```

### 3. Verify Output
1.  **Logs**: Check console for "INITIALIZING ALC-ALGO AGENT ECOSYSTEM".
2.  **Paper Mode**: Ensure "PAPER MODE" and port **7497** are active.
3.  **Safety**: Verify "30% MoS Check" appears in logs.

---

## Step 5: Clean Up (Cost Management)

When not trading/testing, deallocate costly resources (VMs).

```bash
# Stop VM (example)
az vm deallocate --resource-group ALC-Algo-RG --name "alc-trading-vm"
```

**Note**: Key Vault and Storage are cheap/free to keep running.

---

**Troubleshooting**
-   **Auth Errors**: Run `az login` again. Check `AZURE_CLIENT_ID` permissions in Key Vault (Access Policies).
-   **Quota Errors**: Request quota increase for VM cores in Azure Portal if running large ensembles.

**Attribution**: Tom Hogan | Alpha Loop Capital, LLC


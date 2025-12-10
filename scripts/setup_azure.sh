#!/bin/bash
# Azure Infrastructure Setup for ALC-Algo
# Run this script to create all Azure resources needed for production
# Author: Tom Hogan | Alpha Loop Capital, LLC
# Date: 2025-12-09

set -e  # Exit on error

echo "========================================"
echo "ALC-ALGO AZURE INFRASTRUCTURE SETUP"
echo "========================================"
echo ""

# Configuration
RESOURCE_GROUP="alc-algo-prod"
LOCATION="eastus"
APP_NAME="alc-algo"
ENVIRONMENT="production"

echo "Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  App Name: $APP_NAME"
echo "  Environment: $ENVIRONMENT"
echo ""

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "ERROR: Azure CLI not installed"
    echo "Install from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Login check
echo "[1/8] Checking Azure login..."
if ! az account show &> /dev/null; then
    echo "Not logged in. Running 'az login'..."
    az login
else
    echo "Already logged in to Azure"
fi

# Get subscription info
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)
echo "Using subscription: $SUBSCRIPTION_NAME ($SUBSCRIPTION_ID)"
echo ""

# Create Resource Group
echo "[2/8] Creating Resource Group..."
if az group show --name $RESOURCE_GROUP &> /dev/null; then
    echo "Resource group $RESOURCE_GROUP already exists"
else
    az group create \
        --name $RESOURCE_GROUP \
        --location $LOCATION \
        --tags Environment=$ENVIRONMENT Project=ALC-Algo
    echo "Created resource group: $RESOURCE_GROUP"
fi
echo ""

# Create Application Insights
echo "[3/8] Creating Application Insights..."
APPINSIGHTS_NAME="${APP_NAME}-insights"

if az monitor app-insights component show \
    --app $APPINSIGHTS_NAME \
    --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo "Application Insights $APPINSIGHTS_NAME already exists"
else
    az monitor app-insights component create \
        --app $APPINSIGHTS_NAME \
        --location $LOCATION \
        --resource-group $RESOURCE_GROUP \
        --application-type web \
        --retention-time 90 \
        --tags Environment=$ENVIRONMENT
    echo "Created Application Insights: $APPINSIGHTS_NAME"
fi

# Get connection string
APPINSIGHTS_CONNECTION_STRING=$(az monitor app-insights component show \
    --app $APPINSIGHTS_NAME \
    --resource-group $RESOURCE_GROUP \
    --query connectionString -o tsv)

echo "Application Insights Connection String:"
echo "$APPINSIGHTS_CONNECTION_STRING"
echo ""

# Create Log Analytics Workspace
echo "[4/8] Creating Log Analytics Workspace..."
WORKSPACE_NAME="${APP_NAME}-logs"

if az monitor log-analytics workspace show \
    --workspace-name $WORKSPACE_NAME \
    --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo "Log Analytics workspace $WORKSPACE_NAME already exists"
else
    az monitor log-analytics workspace create \
        --workspace-name $WORKSPACE_NAME \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --retention-time 90 \
        --tags Environment=$ENVIRONMENT
    echo "Created Log Analytics workspace: $WORKSPACE_NAME"
fi

# Get workspace credentials
WORKSPACE_ID=$(az monitor log-analytics workspace show \
    --workspace-name $WORKSPACE_NAME \
    --resource-group $RESOURCE_GROUP \
    --query customerId -o tsv)

WORKSPACE_KEY=$(az monitor log-analytics workspace get-shared-keys \
    --workspace-name $WORKSPACE_NAME \
    --resource-group $RESOURCE_GROUP \
    --query primarySharedKey -o tsv)

echo "Log Analytics Workspace ID: $WORKSPACE_ID"
echo ""

# Create Key Vault
echo "[5/8] Creating Key Vault..."
KEYVAULT_NAME="${APP_NAME}-vault-$(date +%s | tail -c 5)"  # Unique name

if az keyvault show --name $KEYVAULT_NAME &> /dev/null; then
    echo "Key Vault $KEYVAULT_NAME already exists"
else
    az keyvault create \
        --name $KEYVAULT_NAME \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --enable-rbac-authorization true \
        --retention-days 90 \
        --tags Environment=$ENVIRONMENT
    echo "Created Key Vault: $KEYVAULT_NAME"
fi
echo ""

# Create Container Registry
echo "[6/8] Creating Container Registry..."
ACR_NAME="${APP_NAME}acr$(date +%s | tail -c 5)"  # Must be alphanumeric

if az acr show --name $ACR_NAME &> /dev/null; then
    echo "Container Registry $ACR_NAME already exists"
else
    az acr create \
        --name $ACR_NAME \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --sku Basic \
        --admin-enabled true \
        --tags Environment=$ENVIRONMENT
    echo "Created Container Registry: $ACR_NAME"
fi

ACR_LOGIN_SERVER=$(az acr show \
    --name $ACR_NAME \
    --resource-group $RESOURCE_GROUP \
    --query loginServer -o tsv)

echo "Container Registry: $ACR_LOGIN_SERVER"
echo ""

# Create Storage Account for data/logs
echo "[7/8] Creating Storage Account..."
STORAGE_NAME="${APP_NAME}storage$(date +%s | tail -c 5)"  # Must be lowercase alphanumeric

if az storage account show --name $STORAGE_NAME &> /dev/null; then
    echo "Storage account $STORAGE_NAME already exists"
else
    az storage account create \
        --name $STORAGE_NAME \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --sku Standard_LRS \
        --kind StorageV2 \
        --access-tier Hot \
        --tags Environment=$ENVIRONMENT
    echo "Created Storage Account: $STORAGE_NAME"
fi

STORAGE_CONNECTION=$(az storage account show-connection-string \
    --name $STORAGE_NAME \
    --resource-group $RESOURCE_GROUP \
    --query connectionString -o tsv)

echo "Storage Account: $STORAGE_NAME"
echo ""

# Store secrets in Key Vault
echo "[8/8] Storing secrets in Key Vault..."

az keyvault secret set \
    --vault-name $KEYVAULT_NAME \
    --name "AppInsightsConnectionString" \
    --value "$APPINSIGHTS_CONNECTION_STRING" \
    > /dev/null

az keyvault secret set \
    --vault-name $KEYVAULT_NAME \
    --name "LogAnalyticsWorkspaceId" \
    --value "$WORKSPACE_ID" \
    > /dev/null

az keyvault secret set \
    --vault-name $KEYVAULT_NAME \
    --name "LogAnalyticsWorkspaceKey" \
    --value "$WORKSPACE_KEY" \
    > /dev/null

az keyvault secret set \
    --vault-name $KEYVAULT_NAME \
    --name "StorageConnectionString" \
    --value "$STORAGE_CONNECTION" \
    > /dev/null

echo "Secrets stored in Key Vault"
echo ""

# Create environment file
echo "========================================"
echo "SETUP COMPLETE!"
echo "========================================"
echo ""
echo "Creating .env file with Azure credentials..."

cat > .env.azure <<EOF
# Azure Configuration
# Generated: $(date)

# Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING="$APPINSIGHTS_CONNECTION_STRING"

# Log Analytics
AZURE_LOG_ANALYTICS_WORKSPACE_ID="$WORKSPACE_ID"
AZURE_LOG_ANALYTICS_WORKSPACE_KEY="$WORKSPACE_KEY"

# Storage
AZURE_STORAGE_CONNECTION_STRING="$STORAGE_CONNECTION"
AZURE_STORAGE_ACCOUNT="$STORAGE_NAME"

# Key Vault
AZURE_KEY_VAULT_NAME="$KEYVAULT_NAME"

# Container Registry
AZURE_CONTAINER_REGISTRY="$ACR_LOGIN_SERVER"

# Resource Group
AZURE_RESOURCE_GROUP="$RESOURCE_GROUP"
AZURE_LOCATION="$LOCATION"
EOF

echo "Azure environment file created: .env.azure"
echo ""
echo "Copy these values to your .env file or source .env.azure:"
echo ""
cat .env.azure
echo ""

# Create summary
cat > azure_setup_summary.txt <<EOF
ALC-ALGO Azure Infrastructure Summary
Generated: $(date)
======================================

Resource Group: $RESOURCE_GROUP
Location: $LOCATION
Subscription: $SUBSCRIPTION_NAME

Resources Created:
------------------
1. Application Insights: $APPINSIGHTS_NAME
2. Log Analytics: $WORKSPACE_NAME
3. Key Vault: $KEYVAULT_NAME
4. Container Registry: $ACR_NAME
5. Storage Account: $STORAGE_NAME

Connection Strings (in Key Vault):
----------------------------------
- AppInsightsConnectionString
- LogAnalyticsWorkspaceId
- LogAnalyticsWorkspaceKey
- StorageConnectionString

Next Steps:
-----------
1. Source the environment file:
   source .env.azure

2. Add IBKR credentials to .env:
   IBKR_USERNAME=your_username
   IBKR_PASSWORD=your_password
   IBKR_ACCOUNT=your_account

3. Test the setup:
   python scripts/test_imports.py

4. Launch production:
   python run_production.py

Cost Estimate:
--------------
Application Insights: ~$2-5/day (5GB data)
Log Analytics: ~$1-2/day
Key Vault: ~$0.10/day
Container Registry: ~$0.17/day (Basic tier)
Storage: ~$0.02/day (minimal usage)

Total: ~$3-8/day (~$90-240/month)

EOF

echo "Setup summary saved to: azure_setup_summary.txt"
echo ""
echo "========================================"
echo "All Azure resources are ready!"
echo "Ready for 9:30am launch!"
echo "========================================"

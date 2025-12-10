## Azure Setup Guide for ALC-Algo

Complete guide to deploying ALC-Algo on Azure

**Target**: Production-ready deployment for 9:30am launch

---

## 🚀 Quick Setup (10 Minutes)

### Option 1: Automated Setup (Recommended)

**Mac/Linux**:
```bash
chmod +x scripts/setup_azure.sh
./scripts/setup_azure.sh
```

**Cross-Platform** (Python):
```bash
python scripts/setup_azure.py
```

**Windows** (PowerShell):
```powershell
# Coming soon - use Python version for now
python scripts/setup_azure.py
```

### Option 2: Manual Setup

Follow the step-by-step guide below.

---

## 📋 Prerequisites

1. **Azure Account**
   - Active Azure subscription
   - Billing enabled
   - Estimated cost: $3-8/day (~$90-240/month)

2. **Azure CLI**
   - Install: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
   - Verify: `az --version`
   - Login: `az login`

3. **Permissions**
   - Contributor role on subscription
   - Ability to create resources

---

## 🏗️ Resources Created

### 1. Application Insights
**Purpose**: Real-time monitoring and telemetry

**Configuration**:
- Name: `alc-algo-insights`
- Retention: 90 days
- Application Type: Web
- Cost: ~$2-5/day (5GB data/day)

**Features**:
- Live metrics dashboard
- Custom events (trades, signals)
- Exception tracking
- Performance monitoring

### 2. Log Analytics Workspace
**Purpose**: Centralized logging

**Configuration**:
- Name: `alc-algo-logs`
- Retention: 90 days
- Cost: ~$1-2/day

**Features**:
- Query historical logs
- Create alerts
- Export to storage

### 3. Key Vault
**Purpose**: Secure secret storage

**Configuration**:
- Name: `alc-algo-vault-xxxxx`
- RBAC enabled
- Soft delete: 90 days
- Cost: ~$0.10/day

**Secrets Stored**:
- Application Insights connection string
- Log Analytics credentials
- Storage connection string
- IBKR credentials (manual)

### 4. Container Registry
**Purpose**: Docker image storage

**Configuration**:
- Name: `alcalgoacrxxxxx`
- SKU: Basic
- Admin enabled
- Cost: ~$0.17/day

**Usage**:
- Store production Docker images
- CI/CD integration
- Version control

### 5. Storage Account
**Purpose**: Data and logs storage

**Configuration**:
- Name: `alcalgostoragexxxxx`
- SKU: Standard_LRS
- Access Tier: Hot
- Cost: ~$0.02/day (minimal usage)

**Containers**:
- `market-data`: Historical price data
- `backtest-results`: Backtest outputs
- `logs`: Application logs
- `models`: Trained ML models

---

## 🔧 Manual Setup Steps

### Step 1: Login to Azure
```bash
az login
az account set --subscription "Your Subscription Name"
```

### Step 2: Create Resource Group
```bash
RESOURCE_GROUP="alc-algo-prod"
LOCATION="eastus"

az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION \
  --tags Environment=production Project=ALC-Algo
```

### Step 3: Create Application Insights
```bash
az monitor app-insights component create \
  --app alc-algo-insights \
  --location $LOCATION \
  --resource-group $RESOURCE_GROUP \
  --application-type web \
  --retention-time 90

# Get connection string
az monitor app-insights component show \
  --app alc-algo-insights \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv
```

### Step 4: Create Log Analytics
```bash
az monitor log-analytics workspace create \
  --workspace-name alc-algo-logs \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --retention-time 90

# Get credentials
az monitor log-analytics workspace show \
  --workspace-name alc-algo-logs \
  --resource-group $RESOURCE_GROUP \
  --query customerId -o tsv

az monitor log-analytics workspace get-shared-keys \
  --workspace-name alc-algo-logs \
  --resource-group $RESOURCE_GROUP \
  --query primarySharedKey -o tsv
```

### Step 5: Create Key Vault
```bash
VAULT_NAME="alc-algo-vault-$(date +%s | tail -c 5)"

az keyvault create \
  --name $VAULT_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --enable-rbac-authorization true
```

### Step 6: Create Storage Account
```bash
STORAGE_NAME="alcalgost$(date +%s | tail -c 5)"

az storage account create \
  --name $STORAGE_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS \
  --kind StorageV2

# Get connection string
az storage account show-connection-string \
  --name $STORAGE_NAME \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv
```

---

## 🔐 Configure Environment Variables

### Create .env file
```bash
# Azure Configuration
APPLICATIONINSIGHTS_CONNECTION_STRING="<from Step 3>"
AZURE_LOG_ANALYTICS_WORKSPACE_ID="<from Step 4>"
AZURE_LOG_ANALYTICS_WORKSPACE_KEY="<from Step 4>"
AZURE_STORAGE_CONNECTION_STRING="<from Step 6>"
AZURE_KEY_VAULT_NAME="<from Step 5>"

# IBKR Configuration (add your credentials)
IBKR_USERNAME="your_username"
IBKR_PASSWORD="your_password"
IBKR_ACCOUNT="your_account"
IBKR_PORT="7497"  # 7497 for paper, 7496 for live
```

### Store secrets in Key Vault
```bash
# Store Application Insights connection string
az keyvault secret set \
  --vault-name $VAULT_NAME \
  --name AppInsightsConnectionString \
  --value "<connection_string>"

# Store IBKR credentials
az keyvault secret set \
  --vault-name $VAULT_NAME \
  --name IBKRUsername \
  --value "your_username"

az keyvault secret set \
  --vault-name $VAULT_NAME \
  --name IBKRPassword \
  --value "your_password"
```

---

## 📊 Verify Setup

### Test Application Insights
```python
from src.monitoring.azure_insights import AzureInsightsTracker

tracker = AzureInsightsTracker()
tracker.track_event("SystemStartup", {"test": "true"})
print("Application Insights working!")
```

### Check Azure Portal
1. Go to: https://portal.azure.com
2. Navigate to Resource Group: `alc-algo-prod`
3. Verify all 5 resources created
4. Open Application Insights → Live Metrics

---

## 🚀 Deploy to Azure

### Option 1: Local Deployment
Run on your machine with Azure monitoring:
```bash
python run_production.py
```

### Option 2: Azure Container Instance (Recommended)
Deploy as containerized app:
```bash
# Build Docker image
docker build -t alc-algo:latest .

# Tag for Azure Container Registry
docker tag alc-algo:latest alcalgoacr.azurecr.io/alc-algo:latest

# Push to ACR
az acr login --name alcalgoacr
docker push alcalgoacr.azurecr.io/alc-algo:latest

# Deploy to Container Instance
az container create \
  --resource-group alc-algo-prod \
  --name alc-algo-trading \
  --image alcalgoacr.azurecr.io/alc-algo:latest \
  --cpu 2 \
  --memory 4 \
  --environment-variables \
    APPLICATIONINSIGHTS_CONNECTION_STRING="<connection_string>" \
  --restart-policy Always
```

### Option 3: Azure VM
Deploy on dedicated VM:
```bash
# Create VM
az vm create \
  --resource-group alc-algo-prod \
  --name alc-algo-vm \
  --image UbuntuLTS \
  --size Standard_D2s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# SSH into VM
az vm show --resource-group alc-algo-prod --name alc-algo-vm \
  --query publicIpAddress -o tsv

ssh azureuser@<public-ip>

# Install dependencies and deploy
```

---

## 📈 Monitoring Dashboard

### Access Live Metrics
1. Go to Azure Portal → Application Insights
2. Click "Live Metrics"
3. View real-time:
   - Request rate
   - Response time
   - Failure rate
   - Server metrics

### Create Custom Dashboards
1. Application Insights → Dashboards
2. Add widgets:
   - Portfolio Value (custom metric)
   - Trade Count (custom event)
   - Sharpe Ratio (custom metric)
   - Active Positions (custom metric)

### Set Up Alerts
```bash
# Alert on high drawdown
az monitor metrics alert create \
  --name "High Drawdown Alert" \
  --resource-group alc-algo-prod \
  --scopes "/subscriptions/.../alc-algo-insights" \
  --condition "avg MaxDrawdown < -15" \
  --description "Alert when drawdown exceeds -15%"
```

---

## 💰 Cost Management

### Estimated Monthly Costs
- Application Insights: $60-150 (5-15 GB/day)
- Log Analytics: $30-60
- Key Vault: $3
- Container Registry: $5 (Basic tier)
- Storage: $1-2 (minimal usage)
- **Total: ~$100-220/month**

### Cost Optimization
1. **Sampling**: Enable 50% sampling in App Insights
2. **Retention**: Reduce to 30 days if not needed
3. **Tier**: Downgrade Container Registry if not using
4. **Storage**: Use Cool tier for archival data

### Set Budget Alerts
```bash
# Create budget with alert
az consumption budget create \
  --budget-name alc-algo-monthly \
  --amount 250 \
  --time-grain Monthly \
  --resource-group alc-algo-prod
```

---

## 🔧 Troubleshooting

### Cannot create resources
- Check subscription billing enabled
- Verify permissions (need Contributor role)
- Try different region if quota issues

### Connection string not working
- Verify environment variable set correctly
- Check Azure portal for correct value
- Test with: `az monitor app-insights component show`

### High costs
- Check Application Insights data volume
- Enable sampling to reduce ingestion
- Review retention policies
- Consider downgrading tiers

### Cannot access Key Vault
- Verify RBAC permissions
- Check firewall rules
- Use managed identity if on Azure VM

---

## 📚 Additional Resources

- [Azure CLI Reference](https://docs.microsoft.com/en-us/cli/azure/)
- [Application Insights Documentation](https://docs.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview)
- [Key Vault Best Practices](https://docs.microsoft.com/en-us/azure/key-vault/general/best-practices)
- [Container Instances Guide](https://docs.microsoft.com/en-us/azure/container-instances/)

---

## ✅ Production Checklist

Before 9:30am launch:

- [ ] All Azure resources created
- [ ] Application Insights connected
- [ ] Secrets stored in Key Vault
- [ ] Environment variables configured
- [ ] Monitoring dashboard set up
- [ ] Budget alerts configured
- [ ] Test event sent successfully
- [ ] Live metrics showing data

---

**Ready for production deployment!** 🚀

*Last Updated: 2025-12-09*
*Author: Tom Hogan | Alpha Loop Capital, LLC*

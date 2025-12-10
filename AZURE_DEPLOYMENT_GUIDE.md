# ALC-Algo Azure Deployment Guide

**Quick Start Guide to Get Azure Infrastructure Running**

---

## ✅ What You're Deploying

**Complete institutional-grade infrastructure:**
- Storage for agent memory, data, models, logs
- PostgreSQL database for trading data
- Redis cache for real-time operations
- Azure ML workspace for training
- Auto-scaling compute cluster (scales to $0 when idle)
- Application Insights for monitoring
- Key Vault for secrets

**Estimated Cost:** ~$50-$100/month + training costs (pay-per-use)

---

## 🚀 Quick Start (10 Minutes)

### Step 1: Install Prerequisites

**Windows:**
```powershell
# Install Azure CLI
winget install Microsoft.AzureCLI

# Install Terraform
winget install Hashicorp.Terraform

# Verify
az --version
terraform --version
```

**Mac:**
```bash
# Install Azure CLI
brew install azure-cli

# Install Terraform
brew install terraform

# Verify
az --version
terraform --version
```

### Step 2: Login to Azure

```bash
# Login
az login

# Set your subscription (if you have multiple)
az account list --output table
az account set --subscription "YOUR_SUBSCRIPTION_NAME"

# Verify
az account show
```

### Step 3: Get Your IP Address

```bash
# Windows
curl ifconfig.me

# Mac/Linux
curl ifconfig.me
```

Save this IP - you'll need it for database access.

### Step 4: Configure & Deploy

```bash
cd infrastructure/terraform

# Copy example config
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars
# IMPORTANT: Change these values:
# - postgres_password: Use a strong password!
# - my_ip_address: Use the IP from Step 3
```

**Edit `terraform.tfvars`:**
```hcl
environment = "dev"
location = "eastus"
postgres_password = "YOUR_SUPER_SECURE_PASSWORD_HERE"
my_ip_address = "YOUR_IP_FROM_STEP_3"
training_vm_size = "Standard_DS3_v2"
max_training_nodes = 2
enable_gpu_training = false
```

### Step 5: Deploy!

```bash
# Initialize Terraform
terraform init

# Preview what will be created
terraform plan

# Deploy (type 'yes' when prompted)
terraform apply
```

This takes **5-10 minutes** to deploy everything.

### Step 6: Get Connection Strings

```bash
# Storage
terraform output -raw storage_connection_string

# PostgreSQL
terraform output -raw postgres_connection_string

# Redis
terraform output -raw redis_host
terraform output -raw redis_key

# Application Insights
terraform output -raw app_insights_connection_string
```

### Step 7: Update .env File

Add these to your `c:\Users\tom\ALC-Algo\.env`:

```bash
# Azure Storage
AZURE_STORAGE_CONNECTION_STRING="<from step 6>"

# PostgreSQL
DATABASE_URL="<from step 6>"

# Redis
REDIS_URL="redis://:<redis_key>@<redis_host>:6380?ssl=true"

# Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING="<from step 6>"

# Azure ML
AZURE_ML_WORKSPACE_NAME="mlw-alc-algo-dev"
AZURE_ML_RESOURCE_GROUP="rg-alc-algo-dev"
AZURE_ML_SUBSCRIPTION_ID="<your subscription id>"
```

---

## 🧪 Test Your Deployment

### Test 1: Storage Connection

```python
from azure.storage.blob import BlobServiceClient
import os

conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
client = BlobServiceClient.from_connection_string(conn_str)

# List containers
for container in client.list_containers():
    print(f"✓ Container: {container.name}")

# Should show: agent-memory, data, models, logs, backups
```

### Test 2: PostgreSQL Connection

```python
import psycopg2
import os

conn_str = os.getenv("DATABASE_URL")
conn = psycopg2.connect(conn_str)
cursor = conn.cursor()

cursor.execute("SELECT version();")
version = cursor.fetchone()
print(f"✓ PostgreSQL: {version[0]}")

conn.close()
```

### Test 3: Redis Connection

```python
import redis
import os

# Parse REDIS_URL
redis_url = os.getenv("REDIS_URL")
r = redis.from_url(redis_url)

r.set('test', 'ALC-Algo Connected!')
value = r.get('test')
print(f"✓ Redis: {value.decode()}")
```

---

## 📊 View Your Resources

**Azure Portal:**
```bash
# Windows
start https://portal.azure.com

# Mac
open https://portal.azure.com
```

Navigate to:
1. Resource Groups → `rg-alc-algo-dev`
2. See all your resources

**Or via CLI:**
```bash
az resource list --resource-group rg-alc-algo-dev --output table
```

---

## 💰 Cost Management

### Current Costs (~$50/month base)
- Storage: $5/month
- PostgreSQL (B1ms): $12/month
- Redis (C0): $16/month
- Application Insights: $5/month
- ML Compute: $0/month (scales to zero!)
- Key Vault: $1/month

### Training Costs (Pay-per-use)
- Standard_DS3_v2: $0.20/hour
- Example: 2 hours/week training = ~$6/month

### Money-Saving Tips
1. ML compute scales to zero automatically (no charge when idle)
2. Use B-series databases (burstable, cheaper)
3. Set alerts for unusual spending
4. Delete dev environment when not using:
   ```bash
   terraform destroy
   ```

---

## 🔒 Security Best Practices

### 1. Protect Your Credentials

```bash
# Never commit these files!
# (Already in .gitignore)
terraform.tfvars
.env
*.key
*.pem
```

### 2. Use Key Vault for Secrets

```bash
# Store API keys in Key Vault
az keyvault secret set \
  --vault-name kv-alc-algo-dev \
  --name OPENAI-API-KEY \
  --value "your-secret-key"

# Retrieve in your code
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = SecretClient(
    vault_url="https://kv-alc-algo-dev.vault.azure.net",
    credential=credential
)

api_key = client.get_secret("OPENAI-API-KEY").value
```

### 3. Rotate Passwords Regularly

```bash
# Change database password every 90 days
# Update terraform.tfvars
# Run: terraform apply
```

---

## 🛠️ Troubleshooting

### "az: command not found"
- Install Azure CLI (see Step 1)
- Restart terminal after installation

### "terraform: command not found"
- Install Terraform (see Step 1)
- Restart terminal after installation

### "ResourceQuotaExceeded"
- You've hit Azure subscription limits
- Delete unused resources or request quota increase

### Can't Connect to PostgreSQL
1. Check your IP is whitelisted in terraform.tfvars
2. Verify `my_ip_address` is correct
3. Make sure you're using SSL: `?sslmode=require`

### "Authentication Failed"
```bash
# Re-login to Azure
az login

# Check your account
az account show
```

### High Costs
```bash
# Check what's running
az resource list --resource-group rg-alc-algo-dev --output table

# Stop ML compute (it should auto-scale to zero)
# Delete dev environment if not using
terraform destroy
```

---

## 📦 What's Next?

After deployment:

1. ✅ Test all connections (run tests above)
2. ✅ Store API keys in Key Vault
3. ✅ Run data logging pipeline (creates database tables)
4. ✅ Submit first ML training job
5. ✅ Set up monitoring dashboards
6. ✅ Configure alerts

---

## 🆘 Need Help?

**Check Documentation:**
- [infrastructure/terraform/README.md](infrastructure/terraform/README.md) - Detailed Terraform guide
- [Azure ML Docs](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Terraform Azure Docs](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)

**Common Issues:**
- Authentication: Run `az login` again
- Permissions: Ensure you have Contributor role
- Costs: Check Azure Cost Management in portal
- Connection errors: Verify firewall rules and SSL

---

## 🔄 Update or Destroy Infrastructure

### Update Infrastructure
```bash
cd infrastructure/terraform

# Edit terraform.tfvars (e.g., scale to 4 nodes)
max_training_nodes = 4

# Apply changes
terraform apply
```

### Destroy Everything
```bash
cd infrastructure/terraform

# WARNING: This deletes ALL resources and data!
terraform destroy
```

---

**Status:** ✅ Infrastructure ready for deployment

**Next:** Set up data logging and ML training pipelines

**Author:** Tom Hogan | Alpha Loop Capital, LLC
**Date:** 2025-12-09

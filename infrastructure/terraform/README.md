# ALC-Algo Azure Infrastructure

Complete Terraform configuration for deploying institutional-grade trading infrastructure to Azure.

## What Gets Deployed

### Core Infrastructure
- **Resource Group:** `rg-alc-algo-dev`
- **Storage Account:** Agent memory, data, models, logs, backups (5 containers)
- **PostgreSQL:** Primary database for trading data, time-series
- **Redis Cache:** High-speed data cache for real-time operations

### ML & Training
- **Azure ML Workspace:** Centralized ML training environment
- **ML Compute Cluster:** Auto-scaling training nodes (scales to 0 when idle)
- **Container Registry:** Docker images for training jobs

### Monitoring & Security
- **Application Insights:** Real-time monitoring, metrics, alerts
- **Log Analytics:** Centralized logging and queries
- **Key Vault:** Secure secrets management (API keys, credentials)

## Cost Estimate

**Development Environment (~$50-$100/month):**
- Storage: $5/month
- PostgreSQL (B1ms): $12/month
- Redis (C0): $16/month
- ML Compute: $0/month (scales to zero)
- Application Insights: $5/month
- Key Vault: $1/month
- **Total:** ~$50/month + training costs

**Training Costs (Pay-per-use):**
- Standard_DS3_v2: $0.20/hour (when running)
- Example: 2 hours/week = $1.60/week = $6/month

## Prerequisites

1. **Azure CLI**
   ```bash
   # Install Azure CLI
   # Windows: winget install Microsoft.AzureCLI
   # Mac: brew install azure-cli

   # Login
   az login

   # Set subscription
   az account set --subscription "YOUR_SUBSCRIPTION_ID"
   ```

2. **Terraform**
   ```bash
   # Install Terraform
   # Windows: winget install Hashicorp.Terraform
   # Mac: brew install terraform

   # Verify installation
   terraform version
   ```

3. **Get Your IP Address** (for database access)
   ```bash
   curl ifconfig.me
   ```

## Quick Start

### 1. Configure Variables

```bash
cd infrastructure/terraform

# Copy example variables
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your values
# IMPORTANT: Change postgres_password!
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Preview Changes

```bash
terraform plan
```

### 4. Deploy Infrastructure

```bash
terraform apply
```

Type `yes` when prompted.

### 5. Get Connection Strings

```bash
# Storage connection string
terraform output -raw storage_connection_string

# PostgreSQL connection string
terraform output -raw postgres_connection_string

# Redis connection string (build manually)
terraform output -raw redis_key
# Then: redis://:<key>@<redis_host>:6380?ssl=true
```

### 6. Update .env File

Add to your `.env`:

```bash
# Azure Storage
AZURE_STORAGE_CONNECTION_STRING="<from terraform output>"

# PostgreSQL
DATABASE_URL="<from terraform output>"

# Redis
REDIS_URL="redis://:<key>@<host>:6380?ssl=true"

# Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING="<from terraform output>"

# Azure ML Workspace
AZURE_ML_WORKSPACE_NAME="mlw-alc-algo-dev"
AZURE_ML_RESOURCE_GROUP="rg-alc-algo-dev"
AZURE_ML_SUBSCRIPTION_ID="<your subscription id>"
```

## Verify Deployment

### Check Resources in Azure Portal

```bash
# Open resource group in portal
az group show --name rg-alc-algo-dev --query id -o tsv | \
  xargs -I {} open "https://portal.azure.com/#@/resource{}"
```

### Test Storage Connection

```python
from azure.storage.blob import BlobServiceClient

# Get connection string from terraform output
conn_str = "YOUR_CONNECTION_STRING"

# Test connection
client = BlobServiceClient.from_connection_string(conn_str)
containers = client.list_containers()
for container in containers:
    print(f"✓ Container: {container.name}")
```

### Test PostgreSQL Connection

```python
import psycopg2

# Get connection string from terraform output
conn_str = "YOUR_CONNECTION_STRING"

# Test connection
conn = psycopg2.connect(conn_str)
cursor = conn.cursor()
cursor.execute("SELECT version();")
version = cursor.fetchone()
print(f"✓ PostgreSQL: {version[0]}")
conn.close()
```

### Test Redis Connection

```python
import redis

# Get connection info from terraform output
r = redis.Redis(
    host='<redis_host>',
    port=6380,
    password='<redis_key>',
    ssl=True
)

r.set('test', 'ALC-Algo')
value = r.get('test')
print(f"✓ Redis: {value.decode()}")
```

## Managing Infrastructure

### View Current State

```bash
terraform show
```

### Update Infrastructure

```bash
# Edit variables in terraform.tfvars
# Then apply changes
terraform apply
```

### Destroy Infrastructure

```bash
# WARNING: This will delete EVERYTHING!
# All data will be lost!
terraform destroy
```

### Scale Training Cluster

Edit `terraform.tfvars`:

```hcl
max_training_nodes = 4  # Scale up to 4 nodes
```

Then apply:

```bash
terraform apply
```

## Security Best Practices

### 1. Protect terraform.tfvars

```bash
# Add to .gitignore (already done)
echo "terraform.tfvars" >> .gitignore
```

### 2. Use Key Vault for Secrets

```bash
# Store API keys in Key Vault (via Azure Portal or CLI)
az keyvault secret set \
  --vault-name kv-alc-algo-dev \
  --name OPENAI-API-KEY \
  --value "your-api-key"
```

### 3. Rotate Database Password

```bash
# Change password in Azure Portal
# Then update terraform.tfvars
# Then: terraform apply
```

### 4. Enable Firewall Rules

By default, PostgreSQL allows Azure services + your IP only.

To add more IPs, edit `main.tf` and add:

```hcl
resource "azurerm_postgresql_flexible_server_firewall_rule" "office" {
  name             = "OfficeIP"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = "203.0.113.0"
  end_ip_address   = "203.0.113.255"
}
```

## Troubleshooting

### Terraform Init Fails

```bash
# Clear cache and reinitialize
rm -rf .terraform .terraform.lock.hcl
terraform init
```

### Resource Already Exists

```bash
# Import existing resource
terraform import azurerm_resource_group.main /subscriptions/{id}/resourceGroups/rg-alc-algo-dev
```

### Permission Denied

```bash
# Ensure you're logged in
az login

# Check your account
az account show

# Verify you have Contributor role
az role assignment list --assignee YOUR_EMAIL
```

### Can't Connect to PostgreSQL

1. Check firewall rules in Azure Portal
2. Verify your IP is whitelisted
3. Ensure SSL is enabled: `?sslmode=require`

### Cost Concerns

**To minimize costs:**

1. Scale ML cluster to 0 when not training (automatic)
2. Use B-series for databases (burstable, cheap)
3. Use Basic tier for Redis (not Standard)
4. Delete dev environment when not in use:
   ```bash
   terraform destroy
   ```

## Next Steps

After deployment:

1. ✅ Update `.env` with connection strings
2. ✅ Test connections (Storage, PostgreSQL, Redis)
3. ✅ Store API keys in Key Vault
4. ✅ Run data logging pipeline (creates database tables)
5. ✅ Submit first ML training job
6. ✅ Set up monitoring dashboards

## Support

**Documentation:**
- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Terraform Azure Provider](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)

**Issues:**
- Check `terraform plan` output
- Review Azure Portal for errors
- Check Application Insights for runtime errors

---

**Author:** Tom Hogan | Alpha Loop Capital, LLC
**Last Updated:** 2025-12-09

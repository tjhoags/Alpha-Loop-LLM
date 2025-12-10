# ALC-Algo Azure Deployment Guide

**Author:** Tom Hogan | **Organization:** Alpha Loop Capital, LLC  
**Status:** Phase 2 - Infrastructure Ready

---

## Overview

ALC-Algo is designed for deployment on Microsoft Azure with enterprise-grade security and scalability. This guide covers the complete deployment process.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AZURE RESOURCE GROUP                         │
│                     (alc-algo-production)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  Azure Key Vault │    │  Azure SQL DB   │    │ Blob Storage │ │
│  │   (Secrets)      │    │   (Data)        │    │  (Models)    │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                      │                     │         │
│           └──────────────────────┼─────────────────────┘         │
│                                  │                               │
│                    ┌─────────────▼─────────────┐                 │
│                    │   Azure Container Apps    │                 │
│                    │   (Agent Ecosystem)       │                 │
│                    │                           │                 │
│                    │  ┌─────────────────────┐  │                 │
│                    │  │ GhostAgent (Master) │  │                 │
│                    │  │ Senior Agents (15)  │  │                 │
│                    │  │ Swarm Agents (35+)  │  │                 │
│                    │  └─────────────────────┘  │                 │
│                    └───────────────────────────┘                 │
│                                  │                               │
│                    ┌─────────────▼─────────────┐                 │
│                    │   Application Insights   │                  │
│                    │   (Monitoring & Logs)    │                  │
│                    └───────────────────────────┘                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Azure Resources Required

| Resource | SKU | Purpose |
|----------|-----|---------|
| Resource Group | N/A | Container for all resources |
| Key Vault | Standard | Secrets management |
| Container Registry | Basic | Docker image storage |
| Container Apps | Consumption | Agent execution |
| Azure SQL | Basic | Data storage |
| Blob Storage | Standard | Model/file storage |
| Application Insights | Standard | Monitoring |

### Local Requirements

- Azure CLI installed and configured
- Docker Desktop (for local testing)
- Terraform (for infrastructure as code)
- Python 3.10+

---

## Deployment Steps

### Step 1: Azure Authentication

```bash
# Login to Azure
az login

# Set subscription
az account set --subscription "Your-Subscription-ID"

# Verify
az account show
```

### Step 2: Create Resource Group

```bash
# Create resource group
az group create \
  --name alc-algo-production \
  --location eastus
```

### Step 3: Deploy Key Vault

```bash
# Create Key Vault
az keyvault create \
  --name alc-algo-secrets \
  --resource-group alc-algo-production \
  --location eastus

# Add secrets
az keyvault secret set --vault-name alc-algo-secrets --name "OPENAI-API-KEY" --value "your-key"
az keyvault secret set --vault-name alc-algo-secrets --name "ANTHROPIC-API-KEY" --value "your-key"
az keyvault secret set --vault-name alc-algo-secrets --name "ALPHA-VANTAGE-KEY" --value "your-key"
```

### Step 4: Deploy with Terraform

```bash
cd infra/azure/terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -out=tfplan

# Apply
terraform apply tfplan
```

### Step 5: Build and Push Docker Image

```bash
# Login to ACR
az acr login --name alcalgoregistry

# Build image
docker build -t alcalgoregistry.azurecr.io/alc-algo:latest .

# Push
docker push alcalgoregistry.azurecr.io/alc-algo:latest
```

### Step 6: Deploy Container App

```bash
# Create Container App
az containerapp create \
  --name alc-algo-agents \
  --resource-group alc-algo-production \
  --image alcalgoregistry.azurecr.io/alc-algo:latest \
  --environment alc-algo-env \
  --cpu 2 \
  --memory 4Gi
```

---

## Environment Variables

Set these in Container App configuration:

```yaml
env:
  - name: ALC_ENVIRONMENT
    value: production
  - name: AZURE_KEY_VAULT_URL
    value: https://alc-algo-secrets.vault.azure.net/
  - name: IBKR_PORT
    value: "7496"  # LIVE trading
  - name: LOG_LEVEL
    value: INFO
```

---

## Monitoring

### Application Insights Queries

```kusto
// Agent execution metrics
traces
| where message contains "Agent"
| summarize count() by bin(timestamp, 1h), customDimensions.agent_name

// Error tracking
exceptions
| where timestamp > ago(24h)
| summarize count() by outerMessage, problemId
```

### Alerts Configuration

| Alert | Condition | Action |
|-------|-----------|--------|
| Agent Failure | Error count > 5/hour | Slack notification |
| High Latency | Avg response > 5s | Email |
| Memory Warning | Usage > 80% | Scale up |

---

## Security

### Network Security

- All resources in private VNet
- No public endpoints for databases
- WAF enabled for API endpoints
- TLS 1.3 required

### Identity

- Managed Identity for all services
- RBAC for Key Vault access
- MFA required for Azure Portal

### Data Protection

- Encryption at rest (Azure managed keys)
- Encryption in transit (TLS)
- Regular backups to geo-redundant storage

---

## Scaling

### Auto-scaling Rules

```json
{
  "minReplicas": 1,
  "maxReplicas": 10,
  "rules": [
    {
      "name": "cpu-scaling",
      "type": "cpu",
      "metadata": {
        "threshold": "70",
        "scaleUpBy": 2,
        "scaleDownBy": 1
      }
    }
  ]
}
```

---

## Disaster Recovery

### Backup Strategy

| Component | Frequency | Retention |
|-----------|-----------|-----------|
| SQL Database | Daily | 30 days |
| Blob Storage | Weekly | 90 days |
| Key Vault | Continuous | N/A (built-in) |
| Container Images | On release | Last 10 versions |

### Recovery Time Objectives

- RTO: 4 hours
- RPO: 1 hour

---

## Cost Optimization

### Estimated Monthly Costs

| Resource | Configuration | Est. Cost |
|----------|--------------|-----------|
| Container Apps | 2 vCPU, 4GB | ~$150 |
| Azure SQL | Basic, 5 DTU | ~$5 |
| Key Vault | Standard | ~$3 |
| Blob Storage | 100GB | ~$2 |
| Application Insights | 5GB/month | ~$10 |
| **Total** | | **~$170/month** |

---

## Troubleshooting

### Common Issues

**Container won't start:**
```bash
az containerapp logs show --name alc-algo-agents --resource-group alc-algo-production
```

**Key Vault access denied:**
```bash
az keyvault set-policy --name alc-algo-secrets --object-id <managed-identity-id> --secret-permissions get list
```

**Database connection failed:**
```bash
az sql server firewall-rule create --resource-group alc-algo-production --server alc-algo-sql --name AllowContainerApps --start-ip-address <container-app-ip> --end-ip-address <container-app-ip>
```

---

## Next Steps

1. Complete Terraform configuration
2. Set up CI/CD pipeline with GitHub Actions
3. Configure production monitoring dashboards
4. Implement blue-green deployment strategy

---

*Azure Deployment Guide - ALC-Algo*  
*Tom Hogan | Alpha Loop Capital, LLC*


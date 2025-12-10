# =============================================================================
# ALC-Algo Azure Infrastructure
# Author: Tom Hogan | Alpha Loop Capital, LLC
# =============================================================================
# Complete Azure infrastructure for institutional-grade algorithmic trading
# Includes: Storage, ML Workspace, Key Vault, Monitoring, Training Compute

terraform {
  required_version = ">= 1.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }

  # Backend for state storage (uncomment after initial setup)
  # backend "azurerm" {
  #   resource_group_name  = "rg-alc-algo-tfstate"
  #   storage_account_name = "stalcalgotfstate"
  #   container_name       = "tfstate"
  #   key                  = "alc-algo.tfstate"
  # }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = true
      recover_soft_deleted_key_vaults = true
    }
  }
}

# =============================================================================
# Local Variables
# =============================================================================
locals {
  project_name = "alc-algo"
  environment  = var.environment
  location     = var.location

  tags = {
    Project     = "ALC-Algo"
    Environment = var.environment
    ManagedBy   = "Terraform"
    Owner       = "Tom Hogan"
    CostCenter  = "AlphaLoopCapital"
  }

  # Naming convention: {type}-{project}-{env}
  resource_group_name      = "rg-${local.project_name}-${local.environment}"
  storage_account_name     = "st${replace(local.project_name, "-", "")}${local.environment}"
  key_vault_name           = "kv-${local.project_name}-${local.environment}"
  ml_workspace_name        = "mlw-${local.project_name}-${local.environment}"
  app_insights_name        = "appi-${local.project_name}-${local.environment}"
  log_analytics_name       = "log-${local.project_name}-${local.environment}"
  container_registry_name  = "cr${replace(local.project_name, "-", "")}${local.environment}"
}

# =============================================================================
# Resource Group
# =============================================================================
resource "azurerm_resource_group" "main" {
  name     = local.resource_group_name
  location = local.location
  tags     = local.tags
}

# =============================================================================
# Storage Account - Agent Memory, Data, Models
# =============================================================================
resource "azurerm_storage_account" "main" {
  name                     = local.storage_account_name
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"  # Locally redundant (cheaper)

  # Enable for versioning and recovery
  blob_properties {
    versioning_enabled = true

    delete_retention_policy {
      days = 7
    }
  }

  tags = local.tags
}

# Storage Containers
resource "azurerm_storage_container" "agent_memory" {
  name                  = "agent-memory"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "data" {
  name                  = "data"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "logs" {
  name                  = "logs"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "backups" {
  name                  = "backups"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

# =============================================================================
# Log Analytics Workspace - Centralized Logging
# =============================================================================
resource "azurerm_log_analytics_workspace" "main" {
  name                = local.log_analytics_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = local.tags
}

# =============================================================================
# Application Insights - Real-time Monitoring
# =============================================================================
resource "azurerm_application_insights" "main" {
  name                = local.app_insights_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "other"

  tags = local.tags
}

# =============================================================================
# Key Vault - Secrets Management
# =============================================================================
data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "main" {
  name                       = local.key_vault_name
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = false  # Set true for production

  # Allow current user/service principal
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = [
      "Get",
      "List",
      "Set",
      "Delete",
      "Purge",
      "Recover"
    ]
  }

  tags = local.tags
}

# =============================================================================
# Container Registry - For ML Docker Images
# =============================================================================
resource "azurerm_container_registry" "main" {
  name                = local.container_registry_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true

  tags = local.tags
}

# =============================================================================
# Azure Machine Learning Workspace
# =============================================================================
resource "azurerm_machine_learning_workspace" "main" {
  name                    = local.ml_workspace_name
  location                = azurerm_resource_group.main.location
  resource_group_name     = azurerm_resource_group.main.name
  application_insights_id = azurerm_application_insights.main.id
  key_vault_id            = azurerm_key_vault.main.id
  storage_account_id      = azurerm_storage_account.main.id
  container_registry_id   = azurerm_container_registry.main.id

  identity {
    type = "SystemAssigned"
  }

  tags = local.tags
}

# =============================================================================
# ML Compute Cluster - Auto-scaling Training Cluster
# =============================================================================
resource "azurerm_machine_learning_compute_cluster" "training" {
  name                          = "alc-training-cluster"
  location                      = azurerm_resource_group.main.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.main.id
  vm_priority                   = "Dedicated"
  vm_size                       = var.training_vm_size

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = var.max_training_nodes
    scale_down_nodes_after_idle_duration = "PT15M"  # 15 minutes
  }

  identity {
    type = "SystemAssigned"
  }

  tags = local.tags
}

# =============================================================================
# PostgreSQL Flexible Server - Primary Database
# =============================================================================
resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "psql-${local.project_name}-${local.environment}"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "15"
  administrator_login    = "alc_admin"
  administrator_password = var.postgres_password
  storage_mb             = 32768  # 32 GB
  sku_name               = "B_Standard_B1ms"  # Burstable tier (cheap for dev)

  backup_retention_days = 7

  tags = local.tags
}

# PostgreSQL Database
resource "azurerm_postgresql_flexible_server_database" "main" {
  name      = "alc_algo"
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

# Allow Azure services to access
resource "azurerm_postgresql_flexible_server_firewall_rule" "azure_services" {
  name             = "AllowAzureServices"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

# Allow your IP (replace with your IP)
resource "azurerm_postgresql_flexible_server_firewall_rule" "my_ip" {
  count            = var.my_ip_address != "" ? 1 : 0
  name             = "MyIP"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = var.my_ip_address
  end_ip_address   = var.my_ip_address
}

# =============================================================================
# Redis Cache - High-speed Data Cache
# =============================================================================
resource "azurerm_redis_cache" "main" {
  name                = "redis-${local.project_name}-${local.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = 0  # C0 (250 MB - cheapest)
  family              = "C"
  sku_name            = "Basic"
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"

  tags = local.tags
}

# =============================================================================
# Outputs - Connection Information
# =============================================================================
output "resource_group_name" {
  value = azurerm_resource_group.main.name
}

output "storage_account_name" {
  value = azurerm_storage_account.main.name
}

output "storage_account_key" {
  value     = azurerm_storage_account.main.primary_access_key
  sensitive = true
}

output "storage_connection_string" {
  value     = azurerm_storage_account.main.primary_connection_string
  sensitive = true
}

output "key_vault_name" {
  value = azurerm_key_vault.main.name
}

output "key_vault_uri" {
  value = azurerm_key_vault.main.vault_uri
}

output "ml_workspace_name" {
  value = azurerm_machine_learning_workspace.main.name
}

output "app_insights_instrumentation_key" {
  value     = azurerm_application_insights.main.instrumentation_key
  sensitive = true
}

output "app_insights_connection_string" {
  value     = azurerm_application_insights.main.connection_string
  sensitive = true
}

output "postgres_host" {
  value = azurerm_postgresql_flexible_server.main.fqdn
}

output "postgres_connection_string" {
  value     = "postgresql://alc_admin:${var.postgres_password}@${azurerm_postgresql_flexible_server.main.fqdn}:5432/alc_algo?sslmode=require"
  sensitive = true
}

output "redis_host" {
  value = azurerm_redis_cache.main.hostname
}

output "redis_key" {
  value     = azurerm_redis_cache.main.primary_access_key
  sensitive = true
}

output "container_registry_login_server" {
  value = azurerm_container_registry.main.login_server
}

output "summary" {
  value = <<-EOT

  ========================================
  ALC-Algo Azure Infrastructure Deployed
  ========================================

  Resource Group:  ${azurerm_resource_group.main.name}
  Location:        ${azurerm_resource_group.main.location}
  Environment:     ${local.environment}

  Storage Account: ${azurerm_storage_account.main.name}
  Key Vault:       ${azurerm_key_vault.main.name}
  ML Workspace:    ${azurerm_machine_learning_workspace.main.name}
  PostgreSQL:      ${azurerm_postgresql_flexible_server.main.fqdn}
  Redis:           ${azurerm_redis_cache.main.hostname}

  Next Steps:
  1. Run: terraform output storage_connection_string
  2. Add to .env: AZURE_STORAGE_CONNECTION_STRING=<value>
  3. Run: terraform output postgres_connection_string
  4. Add to .env: DATABASE_URL=<value>
  5. Run: terraform output redis_key
  6. Add to .env: REDIS_URL=redis://:<key>@${azurerm_redis_cache.main.hostname}:6380?ssl=true

  EOT
}

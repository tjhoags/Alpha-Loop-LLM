# =============================================================================
# ALC-Algo Terraform Variables
# Author: Tom Hogan | Alpha Loop Capital, LLC
# =============================================================================

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "training_vm_size" {
  description = "VM size for ML training cluster"
  type        = string
  default     = "Standard_DS3_v2"  # 4 cores, 14 GB RAM (~$0.20/hour)

  # Options:
  # Standard_DS3_v2 - 4 cores, 14 GB RAM (general purpose)
  # Standard_NC6s_v3 - 6 cores, 112 GB RAM, 1x V100 GPU (deep learning)
}

variable "max_training_nodes" {
  description = "Maximum nodes in ML training cluster"
  type        = number
  default     = 2

  validation {
    condition     = var.max_training_nodes >= 1 && var.max_training_nodes <= 10
    error_message = "Max training nodes must be between 1 and 10."
  }
}

variable "postgres_password" {
  description = "PostgreSQL administrator password"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.postgres_password) >= 8
    error_message = "PostgreSQL password must be at least 8 characters."
  }
}

variable "my_ip_address" {
  description = "Your IP address for database access (optional)"
  type        = string
  default     = ""

  # To get your IP: curl ifconfig.me
}

variable "enable_gpu_training" {
  description = "Enable GPU compute for deep learning"
  type        = bool
  default     = false

  # Note: GPU VMs are expensive (~$3/hour)
  # Only enable if you need deep learning
}

variable "tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}

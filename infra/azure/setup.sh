#!/bin/bash
# =============================================================================
# ALC-Algo Azure Setup Script
# Author: Tom Hogan | Alpha Loop Capital, LLC
# =============================================================================
# This script sets up the complete Azure infrastructure for ALC-Algo
# Run with: chmod +x setup.sh && ./setup.sh

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================
ENVIRONMENT="${ENVIRONMENT:-dev}"
LOCATION="${LOCATION:-eastus}"
RESOURCE_GROUP="rg-alc-algo-${ENVIRONMENT}"
STORAGE_ACCOUNT="stalcalgo${ENVIRONMENT}"
KEY_VAULT="kv-alc-algo-${ENVIRONMENT}"
CONTAINER_REGISTRY="cralcalgo${ENVIRONMENT}"
ML_WORKSPACE="mlw-alc-algo-${ENVIRONMENT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Pre-flight Checks
# =============================================================================
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed. Please install: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi
    
    # Check if logged in
    if ! az account show &> /dev/null; then
        log_warn "Not logged into Azure. Running 'az login'..."
        az login
    fi
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed. Please install: https://www.terraform.io/downloads"
        exit 1
    fi
    
    log_info "Prerequisites check passed!"
}

# =============================================================================
# Create Terraform State Storage
# =============================================================================
create_terraform_state_storage() {
    log_info "Creating Terraform state storage..."
    
    STATE_RG="rg-terraform-state"
    STATE_STORAGE="stalcalgotfstate"
    
    # Create resource group for state
    if ! az group show --name "$STATE_RG" &> /dev/null; then
        az group create --name "$STATE_RG" --location "$LOCATION"
        log_info "Created resource group: $STATE_RG"
    else
        log_info "Resource group exists: $STATE_RG"
    fi
    
    # Create storage account for state
    if ! az storage account show --name "$STATE_STORAGE" &> /dev/null; then
        az storage account create \
            --name "$STATE_STORAGE" \
            --resource-group "$STATE_RG" \
            --location "$LOCATION" \
            --sku Standard_LRS \
            --encryption-services blob
        log_info "Created storage account: $STATE_STORAGE"
    else
        log_info "Storage account exists: $STATE_STORAGE"
    fi
    
    # Get storage key
    STORAGE_KEY=$(az storage account keys list \
        --account-name "$STATE_STORAGE" \
        --resource-group "$STATE_RG" \
        --query '[0].value' -o tsv)
    
    # Create container
    if ! az storage container show --name tfstate --account-name "$STATE_STORAGE" &> /dev/null; then
        az storage container create \
            --name tfstate \
            --account-name "$STATE_STORAGE" \
            --account-key "$STORAGE_KEY"
        log_info "Created storage container: tfstate"
    else
        log_info "Storage container exists: tfstate"
    fi
    
    echo ""
    log_info "Terraform state storage configured:"
    log_info "  Storage Account: $STATE_STORAGE"
    log_info "  Container: tfstate"
}

# =============================================================================
# Create Service Principal
# =============================================================================
create_service_principal() {
    log_info "Creating service principal..."
    
    SP_NAME="sp-alc-algo-cicd"
    SUBSCRIPTION_ID=$(az account show --query id -o tsv)
    
    # Check if SP exists
    if az ad sp list --display-name "$SP_NAME" --query '[0].appId' -o tsv 2>/dev/null | grep -q .; then
        log_warn "Service principal $SP_NAME already exists"
        APP_ID=$(az ad sp list --display-name "$SP_NAME" --query '[0].appId' -o tsv)
        log_info "Existing App ID: $APP_ID"
    else
        # Create new SP
        SP_OUTPUT=$(az ad sp create-for-rbac \
            --name "$SP_NAME" \
            --role Contributor \
            --scopes "/subscriptions/$SUBSCRIPTION_ID" \
            --sdk-auth)
        
        log_info "Service principal created!"
        echo ""
        log_warn "IMPORTANT: Save these credentials securely!"
        echo "$SP_OUTPUT"
        echo ""
        
        # Save to file
        echo "$SP_OUTPUT" > azure_credentials.json
        log_info "Credentials saved to azure_credentials.json"
    fi
}

# =============================================================================
# Deploy Infrastructure with Terraform
# =============================================================================
deploy_infrastructure() {
    log_info "Deploying infrastructure with Terraform..."
    
    cd terraform
    
    # Initialize Terraform
    terraform init \
        -backend-config="storage_account_name=stalcalgotfstate" \
        -backend-config="container_name=tfstate" \
        -backend-config="key=alc-algo-${ENVIRONMENT}.tfstate" \
        -backend-config="resource_group_name=rg-terraform-state"
    
    # Validate
    terraform validate
    
    # Plan
    terraform plan -var="environment=${ENVIRONMENT}" -out=tfplan
    
    # Apply
    echo ""
    read -p "Apply Terraform plan? (yes/no): " CONFIRM
    if [ "$CONFIRM" == "yes" ]; then
        terraform apply tfplan
        log_info "Infrastructure deployed successfully!"
        
        # Save outputs
        terraform output > ../../../azure_outputs.txt
        log_info "Outputs saved to azure_outputs.txt"
    else
        log_warn "Terraform apply cancelled"
    fi
    
    cd ..
}

# =============================================================================
# Upload Secrets to Key Vault
# =============================================================================
upload_secrets() {
    log_info "Uploading secrets to Key Vault..."
    
    # Check if Key Vault exists
    if ! az keyvault show --name "$KEY_VAULT" &> /dev/null; then
        log_error "Key Vault $KEY_VAULT does not exist. Run Terraform first."
        return 1
    fi
    
    # Read secrets from file if exists
    if [ -f "../../config/secrets.py" ]; then
        log_info "Reading secrets from config/secrets.py..."
        # Note: This is a simplified example. In production, use proper secret management.
        log_warn "Manual secret upload recommended for security"
    fi
    
    echo ""
    log_info "To upload secrets manually, use:"
    echo "  az keyvault secret set --vault-name $KEY_VAULT --name SECRET-NAME --value 'secret-value'"
}

# =============================================================================
# Verify Deployment
# =============================================================================
verify_deployment() {
    log_info "Verifying deployment..."
    
    echo ""
    log_info "Checking Resource Group..."
    az group show --name "$RESOURCE_GROUP" --query "name" -o tsv
    
    log_info "Checking Storage Account..."
    az storage account show --name "$STORAGE_ACCOUNT" --query "name" -o tsv 2>/dev/null || log_warn "Storage account not found"
    
    log_info "Checking Key Vault..."
    az keyvault show --name "$KEY_VAULT" --query "name" -o tsv 2>/dev/null || log_warn "Key Vault not found"
    
    log_info "Checking Container Registry..."
    az acr show --name "$CONTAINER_REGISTRY" --query "name" -o tsv 2>/dev/null || log_warn "Container Registry not found"
    
    log_info "Checking ML Workspace..."
    az ml workspace show --name "$ML_WORKSPACE" --resource-group "$RESOURCE_GROUP" --query "name" -o tsv 2>/dev/null || log_warn "ML Workspace not found"
    
    echo ""
    log_info "Deployment verification complete!"
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo ""
    echo "=========================================="
    echo "ALC-Algo Azure Setup"
    echo "Author: Tom Hogan | Alpha Loop Capital"
    echo "=========================================="
    echo ""
    
    log_info "Environment: $ENVIRONMENT"
    log_info "Location: $LOCATION"
    echo ""
    
    case "${1:-all}" in
        "prereq")
            check_prerequisites
            ;;
        "state")
            check_prerequisites
            create_terraform_state_storage
            ;;
        "sp")
            check_prerequisites
            create_service_principal
            ;;
        "deploy")
            check_prerequisites
            deploy_infrastructure
            ;;
        "secrets")
            upload_secrets
            ;;
        "verify")
            verify_deployment
            ;;
        "all")
            check_prerequisites
            create_terraform_state_storage
            create_service_principal
            deploy_infrastructure
            upload_secrets
            verify_deployment
            ;;
        *)
            echo "Usage: $0 {prereq|state|sp|deploy|secrets|verify|all}"
            echo ""
            echo "Commands:"
            echo "  prereq  - Check prerequisites"
            echo "  state   - Create Terraform state storage"
            echo "  sp      - Create service principal"
            echo "  deploy  - Deploy infrastructure with Terraform"
            echo "  secrets - Upload secrets to Key Vault"
            echo "  verify  - Verify deployment"
            echo "  all     - Run all steps"
            exit 1
            ;;
    esac
    
    echo ""
    log_info "Setup complete!"
}

main "$@"


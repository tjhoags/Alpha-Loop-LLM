#!/bin/bash
# =============================================================================
# ALC-Algo Azure Setup Script
# Author: Tom Hogan | Alpha Loop Capital, LLC
# =============================================================================
# This script automates the Azure infrastructure deployment

set -e  # Exit on error

echo "========================================"
echo "ALC-Algo Azure Infrastructure Setup"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "Checking prerequisites..."

# Check Azure CLI
if ! command -v az &> /dev/null; then
    echo -e "${RED}✗ Azure CLI not found${NC}"
    echo "Install: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi
echo -e "${GREEN}✓ Azure CLI installed${NC}"

# Check Terraform
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}✗ Terraform not found${NC}"
    echo "Install: https://www.terraform.io/downloads.html"
    exit 1
fi
echo -e "${GREEN}✓ Terraform installed${NC}"

# Check Azure login
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}! Not logged into Azure${NC}"
    echo "Running: az login"
    az login
fi
echo -e "${GREEN}✓ Logged into Azure${NC}"

# Get subscription
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)
echo "Subscription: $SUBSCRIPTION_NAME ($SUBSCRIPTION_ID)"

# Get my IP address
echo ""
echo "Getting your public IP address..."
MY_IP=$(curl -s ifconfig.me)
echo "Your IP: $MY_IP"

# Check if terraform.tfvars exists
cd terraform
if [ ! -f terraform.tfvars ]; then
    echo ""
    echo -e "${YELLOW}! terraform.tfvars not found${NC}"
    echo "Creating from example..."
    cp terraform.tfvars.example terraform.tfvars

    # Generate secure password
    POSTGRES_PASSWORD=$(openssl rand -base64 24)

    # Update terraform.tfvars
    sed -i "s|YOUR_SECURE_PASSWORD_HERE|$POSTGRES_PASSWORD|g" terraform.tfvars
    sed -i "s|YOUR_IP_HERE|$MY_IP|g" terraform.tfvars

    echo -e "${GREEN}✓ Created terraform.tfvars${NC}"
    echo ""
    echo -e "${YELLOW}IMPORTANT: Save this PostgreSQL password!${NC}"
    echo "Password: $POSTGRES_PASSWORD"
    echo ""
    echo "Press Enter to continue..."
    read
fi

# Initialize Terraform
echo ""
echo "Initializing Terraform..."
terraform init

# Validate configuration
echo ""
echo "Validating Terraform configuration..."
terraform validate

# Plan deployment
echo ""
echo "Planning deployment..."
terraform plan -out=tfplan

# Ask for confirmation
echo ""
echo -e "${YELLOW}Ready to deploy infrastructure.${NC}"
echo "This will create resources in Azure (costs apply)."
echo ""
read -p "Deploy now? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Deployment cancelled."
    exit 0
fi

# Apply deployment
echo ""
echo "Deploying infrastructure..."
terraform apply tfplan

# Get outputs
echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo ""

# Save connection strings to file
echo "Saving connection strings to .azure_outputs..."
terraform output -json > .azure_outputs

echo -e "${GREEN}✓ Infrastructure deployed${NC}"
echo ""
echo "Connection strings saved to: terraform/.azure_outputs"
echo ""
echo "Next steps:"
echo "1. Run: cat terraform/.azure_outputs"
echo "2. Copy connection strings to your .env file"
echo "3. Test connections with: python scripts/test_azure_connection.py"
echo ""

# Offer to show outputs
read -p "Show connection strings now? (yes/no): " SHOW_OUTPUTS
if [ "$SHOW_OUTPUTS" == "yes" ]; then
    echo ""
    echo "========== Storage Connection =========="
    terraform output -raw storage_connection_string
    echo ""
    echo ""
    echo "========== PostgreSQL Connection =========="
    terraform output -raw postgres_connection_string
    echo ""
    echo ""
    echo "========== Redis Info =========="
    echo "Host: $(terraform output -raw redis_host)"
    echo "Key: $(terraform output -raw redis_key)"
    echo ""
fi

echo ""
echo -e "${GREEN}Setup complete!${NC}"

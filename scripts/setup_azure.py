#!/usr/bin/env python
"""
Azure Infrastructure Setup (Cross-Platform)
Creates all Azure resources needed for production deployment

Can be run on Windows, Mac, or Linux
Requires: pip install azure-cli-core azure-mgmt-resource azure-mgmt-applicationinsights

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
RESOURCE_GROUP = "alc-algo-prod"
LOCATION = "eastus"
APP_NAME = "alc-algo"
ENVIRONMENT = "production"


def run_az_command(command: str, capture_output: bool = True) -> str:
    """Run Azure CLI command"""
    try:
        result = subprocess.run(
            f"az {command}",
            shell=True,
            check=True,
            capture_output=capture_output,
            text=True
        )
        return result.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as e:
        print(f"Error running command: az {command}")
        print(f"Error: {e.stderr if hasattr(e, 'stderr') else str(e)}")
        return None


def check_azure_cli():
    """Check if Azure CLI is installed"""
    try:
        result = subprocess.run(
            ["az", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Azure CLI not installed")
        print("Install from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
        return False


def check_azure_login():
    """Check if logged into Azure"""
    result = run_az_command("account show")
    return result is not None


def create_resource_group():
    """Create Azure Resource Group"""
    print(f"\n[2/8] Creating Resource Group: {RESOURCE_GROUP}...")

    # Check if exists
    result = run_az_command(f"group show --name {RESOURCE_GROUP}")

    if result:
        print(f"Resource group {RESOURCE_GROUP} already exists")
        return True

    # Create
    command = (
        f"group create "
        f"--name {RESOURCE_GROUP} "
        f"--location {LOCATION} "
        f"--tags Environment={ENVIRONMENT} Project=ALC-Algo"
    )

    result = run_az_command(command)
    if result:
        print(f"✓ Created resource group: {RESOURCE_GROUP}")
        return True
    return False


def create_app_insights():
    """Create Application Insights"""
    print(f"\n[3/8] Creating Application Insights...")

    name = f"{APP_NAME}-insights"

    # Check if exists
    result = run_az_command(
        f"monitor app-insights component show "
        f"--app {name} "
        f"--resource-group {RESOURCE_GROUP}"
    )

    if result:
        print(f"Application Insights {name} already exists")
    else:
        # Create
        command = (
            f"monitor app-insights component create "
            f"--app {name} "
            f"--location {LOCATION} "
            f"--resource-group {RESOURCE_GROUP} "
            f"--application-type web "
            f"--retention-time 90 "
            f"--tags Environment={ENVIRONMENT}"
        )

        result = run_az_command(command)
        if result:
            print(f"✓ Created Application Insights: {name}")

    # Get connection string
    connection_string = run_az_command(
        f"monitor app-insights component show "
        f"--app {name} "
        f"--resource-group {RESOURCE_GROUP} "
        f"--query connectionString -o tsv"
    )

    return name, connection_string


def create_log_analytics():
    """Create Log Analytics Workspace"""
    print(f"\n[4/8] Creating Log Analytics Workspace...")

    name = f"{APP_NAME}-logs"

    # Check if exists
    result = run_az_command(
        f"monitor log-analytics workspace show "
        f"--workspace-name {name} "
        f"--resource-group {RESOURCE_GROUP}"
    )

    if result:
        print(f"Log Analytics workspace {name} already exists")
    else:
        # Create
        command = (
            f"monitor log-analytics workspace create "
            f"--workspace-name {name} "
            f"--resource-group {RESOURCE_GROUP} "
            f"--location {LOCATION} "
            f"--retention-time 90 "
            f"--tags Environment={ENVIRONMENT}"
        )

        result = run_az_command(command)
        if result:
            print(f"✓ Created Log Analytics workspace: {name}")

    # Get workspace ID
    workspace_id = run_az_command(
        f"monitor log-analytics workspace show "
        f"--workspace-name {name} "
        f"--resource-group {RESOURCE_GROUP} "
        f"--query customerId -o tsv"
    )

    # Get workspace key
    workspace_key = run_az_command(
        f"monitor log-analytics workspace get-shared-keys "
        f"--workspace-name {name} "
        f"--resource-group {RESOURCE_GROUP} "
        f"--query primarySharedKey -o tsv"
    )

    return name, workspace_id, workspace_key


def create_key_vault():
    """Create Key Vault"""
    print(f"\n[5/8] Creating Key Vault...")

    # Unique name (max 24 chars)
    import time
    suffix = str(int(time.time()))[-5:]
    name = f"{APP_NAME}-vault-{suffix}"

    # Check if exists
    result = run_az_command(f"keyvault show --name {name}")

    if result:
        print(f"Key Vault {name} already exists")
    else:
        # Create
        command = (
            f"keyvault create "
            f"--name {name} "
            f"--resource-group {RESOURCE_GROUP} "
            f"--location {LOCATION} "
            f"--enable-rbac-authorization true "
            f"--retention-days 90 "
            f"--tags Environment={ENVIRONMENT}"
        )

        result = run_az_command(command)
        if result:
            print(f"✓ Created Key Vault: {name}")

    return name


def create_storage_account():
    """Create Storage Account"""
    print(f"\n[7/8] Creating Storage Account...")

    # Unique name (lowercase alphanumeric only)
    import time
    suffix = str(int(time.time()))[-5:]
    name = f"{APP_NAME.replace('-', '')}st{suffix}"

    # Check if exists
    result = run_az_command(f"storage account show --name {name}")

    if result:
        print(f"Storage account {name} already exists")
    else:
        # Create
        command = (
            f"storage account create "
            f"--name {name} "
            f"--resource-group {RESOURCE_GROUP} "
            f"--location {LOCATION} "
            f"--sku Standard_LRS "
            f"--kind StorageV2 "
            f"--access-tier Hot "
            f"--tags Environment={ENVIRONMENT}"
        )

        result = run_az_command(command)
        if result:
            print(f"✓ Created Storage Account: {name}")

    # Get connection string
    connection_string = run_az_command(
        f"storage account show-connection-string "
        f"--name {name} "
        f"--resource-group {RESOURCE_GROUP} "
        f"--query connectionString -o tsv"
    )

    return name, connection_string


def save_environment_file(credentials: dict):
    """Save Azure credentials to .env file"""
    env_content = f"""# Azure Configuration
# Generated: {datetime.now().isoformat()}

# Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING="{credentials['appinsights_connection']}"

# Log Analytics
AZURE_LOG_ANALYTICS_WORKSPACE_ID="{credentials['workspace_id']}"
AZURE_LOG_ANALYTICS_WORKSPACE_KEY="{credentials['workspace_key']}"

# Storage
AZURE_STORAGE_CONNECTION_STRING="{credentials['storage_connection']}"
AZURE_STORAGE_ACCOUNT="{credentials['storage_name']}"

# Key Vault
AZURE_KEY_VAULT_NAME="{credentials['keyvault_name']}"

# Resource Group
AZURE_RESOURCE_GROUP="{RESOURCE_GROUP}"
AZURE_LOCATION="{LOCATION}"
"""

    # Save to .env.azure
    with open(".env.azure", "w") as f:
        f.write(env_content)

    print("\n✓ Azure environment file created: .env.azure")
    print("\nAdd these to your .env file:")
    print(env_content)


def main():
    """Main setup function"""
    print("=" * 60)
    print("ALC-ALGO AZURE INFRASTRUCTURE SETUP")
    print("=" * 60)
    print()

    print("Configuration:")
    print(f"  Resource Group: {RESOURCE_GROUP}")
    print(f"  Location: {LOCATION}")
    print(f"  App Name: {APP_NAME}")
    print(f"  Environment: {ENVIRONMENT}")
    print()

    # Check Azure CLI
    print("[1/8] Checking Azure CLI...")
    if not check_azure_cli():
        sys.exit(1)

    # Check login
    if not check_azure_login():
        print("Not logged in. Please run: az login")
        sys.exit(1)

    print("✓ Logged into Azure")

    # Get subscription info
    subscription = json.loads(run_az_command("account show"))
    print(f"Using subscription: {subscription['name']} ({subscription['id']})")

    credentials = {}

    # Create resources
    try:
        # Resource Group
        create_resource_group()

        # Application Insights
        appinsights_name, appinsights_connection = create_app_insights()
        credentials['appinsights_name'] = appinsights_name
        credentials['appinsights_connection'] = appinsights_connection

        # Log Analytics
        logs_name, workspace_id, workspace_key = create_log_analytics()
        credentials['logs_name'] = logs_name
        credentials['workspace_id'] = workspace_id
        credentials['workspace_key'] = workspace_key

        # Key Vault
        keyvault_name = create_key_vault()
        credentials['keyvault_name'] = keyvault_name

        # Storage
        storage_name, storage_connection = create_storage_account()
        credentials['storage_name'] = storage_name
        credentials['storage_connection'] = storage_connection

        # Save environment file
        save_environment_file(credentials)

        print("\n" + "=" * 60)
        print("SETUP COMPLETE!")
        print("=" * 60)
        print("\nAll Azure resources are ready for production!")
        print("\nNext steps:")
        print("1. Source the environment: source .env.azure (Mac/Linux) or merge into .env")
        print("2. Add IBKR credentials to .env")
        print("3. Test: python scripts/test_imports.py")
        print("4. Launch: python run_production.py")
        print("\nReady for 9:30am launch! 🚀")

    except Exception as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

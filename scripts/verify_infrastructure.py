"""
Verify ALC-Algo Infrastructure Status
Author: Tom Hogan

Checks what infrastructure is configured and working.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_env_var(name: str, required: bool = False) -> tuple[bool, str]:
    """Check if environment variable exists and has value"""
    value = os.getenv(name)
    if value and value not in ["your_key_here", "YOUR_", ""]:
        return True, f"✅ {name}: Configured"
    elif required:
        return False, f"❌ {name}: MISSING (required)"
    else:
        return False, f"⚠️  {name}: Not configured"

def main():
    print("=" * 60)
    print("ALC-Algo Infrastructure Status Check")
    print("=" * 60)
    print()

    # Azure Storage
    print("Azure Storage:")
    status, msg = check_env_var("AZURE_STORAGE_CONNECTION_STRING", required=True)
    print(f"  {msg}")
    if not status:
        print("  → Deploy with: cd infrastructure/terraform && terraform apply")
    print()

    # Databases
    print("Databases:")
    status, msg = check_env_var("DATABASE_URL")
    print(f"  PostgreSQL: {msg}")

    status, msg = check_env_var("REDIS_URL")
    print(f"  Redis: {msg}")
    print()

    # Application Insights
    print("Monitoring:")
    status, msg = check_env_var("APPLICATIONINSIGHTS_CONNECTION_STRING")
    print(f"  {msg}")
    print()

    # Azure ML
    print("Azure ML:")
    status, msg = check_env_var("AZURE_ML_WORKSPACE_NAME")
    print(f"  Workspace: {msg}")

    status, msg = check_env_var("AZURE_ML_RESOURCE_GROUP")
    print(f"  Resource Group: {msg}")

    status, msg = check_env_var("AZURE_ML_SUBSCRIPTION_ID")
    print(f"  Subscription: {msg}")
    print()

    # AI Providers
    print("AI Providers:")
    for provider in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY_1",
                     "PERPLEXITY_API_KEY"]:
        status, msg = check_env_var(provider)
        print(f"  {msg}")
    print()

    # Data Providers
    print("Market Data:")
    for provider in ["POLYGON_API_KEY", "ALPHA_VANTAGE_API_KEY", "FRED_API_KEY"]:
        status, msg = check_env_var(provider)
        print(f"  {msg}")
    print()

    # Brokers
    print("Brokers:")
    for broker in ["COINBASE_API_KEY", "IBKR_ACCOUNT"]:
        status, msg = check_env_var(broker)
        print(f"  {msg}")
    print()

    # Vertex AI / GCP
    print("Vertex AI (GCP):")
    status, msg = check_env_var("GOOGLE_APPLICATION_CREDENTIALS")
    print(f"  Credentials: {msg}")

    status, msg = check_env_var("GCP_PROJECT_ID")
    print(f"  Project ID: {msg}")
    print()

    # Summary
    print("=" * 60)
    print("Summary:")
    print("=" * 60)

    # Check critical items
    critical = [
        "AZURE_STORAGE_CONNECTION_STRING",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]

    all_critical_configured = all(
        os.getenv(var) and os.getenv(var) not in ["your_key_here", "YOUR_", ""]
        for var in critical
    )

    if all_critical_configured:
        print("✅ Core infrastructure is configured!")
        print("   Ready to start building data pipelines")
    else:
        print("❌ Core infrastructure NOT configured")
        print()
        print("Next steps:")
        print("1. Deploy Azure infrastructure:")
        print("   cd infrastructure/terraform")
        print("   terraform init && terraform apply")
        print()
        print("2. Get connection strings:")
        print("   terraform output storage_connection_string")
        print("   terraform output postgres_connection_string")
        print()
        print("3. Add to .env file")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()

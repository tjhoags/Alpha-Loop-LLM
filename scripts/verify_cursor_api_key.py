"""
================================================================================
Verify and Reload Cursor Cloud Agents API Key
================================================================================
This script helps you verify and reload your Cursor API token "ALC 2"

Based on: https://cursor.com/docs/cloud-agent/api/endpoints
================================================================================
"""

import os
import sys
import requests
from typing import Optional

# Cursor Cloud Agents API base URL
CURSOR_API_BASE = "https://api.cursor.com"


def get_api_key() -> Optional[str]:
    """Get Cursor API key from environment or user input."""
    # Check environment variable first
    api_key = os.getenv("CURSOR_API_KEY") or os.getenv("CURSOR_CLOUD_AGENTS_API_KEY")
    
    if not api_key:
        print("\n" + "=" * 80)
        print("CURSOR API KEY NOT FOUND IN ENVIRONMENT")
        print("=" * 80)
        print("\nTo get your API key:")
        print("1. Go to: https://cursor.com/settings")
        print("2. Navigate to API Keys section")
        print("3. Find or create key named 'ALC 2'")
        print("4. Copy the API key")
        print("\nYou can:")
        print("  - Set environment variable: CURSOR_API_KEY=your_key_here")
        print("  - Or enter it below (will not be saved)")
        print()
        
        api_key = input("Enter your Cursor API key (ALC 2): ").strip()
        
        if not api_key:
            print("❌ No API key provided")
            return None
    
    return api_key


def verify_api_key(api_key: str) -> dict:
    """
    Verify API key using the /v0/me endpoint.
    
    Reference: https://cursor.com/docs/cloud-agent/api/endpoints#api-key-info
    """
    url = f"{CURSOR_API_BASE}/v0/me"
    
    try:
        response = requests.get(
            url,
            auth=(api_key, ""),  # Basic Auth: username=api_key, password=""
            timeout=10
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json(),
                "status_code": response.status_code
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "message": response.text,
                "status_code": response.status_code
            }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "status_code": None
        }


def list_agents(api_key: str, limit: int = 20) -> dict:
    """
    List cloud agents for the authenticated user.
    
    Reference: https://cursor.com/docs/cloud-agent/api/endpoints#list-agents
    """
    url = f"{CURSOR_API_BASE}/v0/agents"
    
    try:
        response = requests.get(
            url,
            auth=(api_key, ""),
            params={"limit": limit},
            timeout=10
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json(),
                "status_code": response.status_code
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "message": response.text,
                "status_code": response.status_code
            }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "status_code": None
        }


def main():
    """Main function to verify and reload Cursor API key."""
    print("=" * 80)
    print("CURSOR CLOUD AGENTS API - API KEY VERIFICATION")
    print("=" * 80)
    print("\nVerifying API key: ALC 2")
    print()
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        sys.exit(1)
    
    # Verify API key
    print("🔍 Verifying API key...")
    print(f"   Using endpoint: {CURSOR_API_BASE}/v0/me")
    print()
    
    result = verify_api_key(api_key)
    
    if result["success"]:
        data = result["data"]
        print("✅ API KEY VERIFIED SUCCESSFULLY")
        print("=" * 80)
        print(f"API Key Name: {data.get('apiKeyName', 'N/A')}")
        print(f"User Email: {data.get('userEmail', 'N/A')}")
        print(f"Created At: {data.get('createdAt', 'N/A')}")
        print("=" * 80)
        
        # Test listing agents
        print("\n🔍 Testing agent listing...")
        agents_result = list_agents(api_key, limit=5)
        
        if agents_result["success"]:
            agents_data = agents_result["data"]
            agent_count = len(agents_data.get("agents", []))
            print(f"✅ Successfully connected - Found {agent_count} agent(s)")
            
            if agent_count > 0:
                print("\nRecent agents:")
                for agent in agents_data.get("agents", [])[:3]:
                    print(f"  - {agent.get('name', 'N/A')} ({agent.get('status', 'N/A')})")
        else:
            print(f"⚠️  Could not list agents: {agents_result.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 80)
        print("✅ YOUR CURSOR API KEY 'ALC 2' IS WORKING CORRECTLY")
        print("=" * 80)
        print("\nTo use this API key in your environment:")
        print(f'  Windows PowerShell: $env:CURSOR_API_KEY="{api_key}"')
        print(f'  Windows CMD: set CURSOR_API_KEY={api_key}')
        print(f'  Linux/Mac: export CURSOR_API_KEY="{api_key}"')
        print("\nOr add to your .env file:")
        print(f'  CURSOR_API_KEY={api_key}')
        print()
        
    else:
        print("❌ API KEY VERIFICATION FAILED")
        print("=" * 80)
        print(f"Error: {result.get('error', 'Unknown error')}")
        if result.get("status_code"):
            print(f"Status Code: {result['status_code']}")
        if result.get("message"):
            print(f"Message: {result['message']}")
        print("=" * 80)
        print("\nTroubleshooting:")
        print("1. Verify your API key at: https://cursor.com/settings")
        print("2. Make sure the key name is 'ALC 2'")
        print("3. Check if the key has expired")
        print("4. Ensure you have Cloud Agents API access")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()


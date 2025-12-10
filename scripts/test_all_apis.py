"""
API Connection Test Script
Tests all configured APIs and logging functionality
Run: python scripts/test_all_apis.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path, override=True)

print("=" * 70)
print("ALPHA LOOP LLM - API CONNECTION TEST")
print(f"Timestamp: {datetime.now().isoformat()}")
print("=" * 70)

results = {}

# =============================================================================
# 1. GITHUB TEST
# =============================================================================
print("\n[1/8] TESTING GITHUB...")
try:
    import requests
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPO", "tjhoags/alpha-loop-llm")

    if not github_token:
        print("   SKIP - No GITHUB_TOKEN configured")
        results["GitHub"] = "SKIP"
    else:
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        # Test repo access
        resp = requests.get(f"https://api.github.com/repos/{github_repo}", headers=headers, timeout=10)
        if resp.status_code == 200:
            repo_data = resp.json()
            print(f"   SUCCESS - Connected to: {repo_data['full_name']}")
            print(f"   Private: {repo_data['private']}")
            results["GitHub"] = "SUCCESS"
        else:
            print(f"   FAILED - Status {resp.status_code}: {resp.text[:100]}")
            results["GitHub"] = "FAILED"
except Exception as e:
    print(f"   ERROR - {e}")
    results["GitHub"] = "ERROR"

# =============================================================================
# 2. PERPLEXITY TEST
# =============================================================================
print("\n[2/8] TESTING PERPLEXITY...")
try:
    import requests
    pplx_key = os.getenv("PERPLEXITY_API_KEY")

    if not pplx_key:
        print("   SKIP - No PERPLEXITY_API_KEY configured")
        results["Perplexity"] = "SKIP"
    else:
        headers = {
            "Authorization": f"Bearer {pplx_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "sonar",
            "messages": [{"role": "user", "content": "Say 'API test successful' in 5 words or less"}]
        }
        resp = requests.post("https://api.perplexity.ai/chat/completions",
                           headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f"   SUCCESS - Response: {content[:50]}...")
            results["Perplexity"] = "SUCCESS"
        else:
            print(f"   FAILED - Status {resp.status_code}: {resp.text[:100]}")
            results["Perplexity"] = "FAILED"
except Exception as e:
    print(f"   ERROR - {e}")
    results["Perplexity"] = "ERROR"

# =============================================================================
# 3. ANTHROPIC TEST
# =============================================================================
print("\n[3/8] TESTING ANTHROPIC...")
try:
    import requests
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not anthropic_key:
        print("   SKIP - No ANTHROPIC_API_KEY configured")
        results["Anthropic"] = "SKIP"
    else:
        headers = {
            "x-api-key": anthropic_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Say 'API test successful' in 5 words or less"}]
        }
        resp = requests.post("https://api.anthropic.com/v1/messages",
                           headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            content = data["content"][0]["text"]
            print(f"   SUCCESS - Response: {content}")
            results["Anthropic"] = "SUCCESS"
        else:
            print(f"   FAILED - Status {resp.status_code}: {resp.text[:100]}")
            results["Anthropic"] = "FAILED"
except Exception as e:
    print(f"   ERROR - {e}")
    results["Anthropic"] = "ERROR"

# =============================================================================
# 4. OPENAI TEST
# =============================================================================
print("\n[4/8] TESTING OPENAI...")
try:
    import requests
    openai_key = os.getenv("OPENAI_SECRET")

    if not openai_key:
        print("   SKIP - No OPENAI_SECRET configured")
        results["OpenAI"] = "SKIP"
    else:
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Say 'API test successful' in 5 words or less"}],
            "max_tokens": 50
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions",
                           headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f"   SUCCESS - Response: {content}")
            results["OpenAI"] = "SUCCESS"
        else:
            print(f"   FAILED - Status {resp.status_code}: {resp.text[:100]}")
            results["OpenAI"] = "FAILED"
except Exception as e:
    print(f"   ERROR - {e}")
    results["OpenAI"] = "ERROR"

# =============================================================================
# 5. POLYGON/MASSIVE TEST
# =============================================================================
print("\n[5/8] TESTING POLYGON/MASSIVE...")
try:
    import requests
    polygon_key = os.getenv("PolygonIO_API_KEY")

    if not polygon_key:
        print("   SKIP - No PolygonIO_API_KEY configured")
        results["Polygon"] = "SKIP"
    else:
        # Test with a simple ticker lookup
        resp = requests.get(
            f"https://api.polygon.io/v3/reference/tickers/AAPL?apiKey={polygon_key}",
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if "results" in data:
                print(f"   SUCCESS - Ticker: {data['results']['ticker']} ({data['results']['name']})")
                results["Polygon"] = "SUCCESS"
            else:
                print(f"   FAILED - No results in response")
                results["Polygon"] = "FAILED"
        else:
            print(f"   FAILED - Status {resp.status_code}: {resp.text[:100]}")
            results["Polygon"] = "FAILED"
except Exception as e:
    print(f"   ERROR - {e}")
    results["Polygon"] = "ERROR"

# =============================================================================
# 6. ALPHA VANTAGE TEST
# =============================================================================
print("\n[6/8] TESTING ALPHA VANTAGE...")
try:
    import requests
    av_key = os.getenv("ALPHAVANTAGE_API_KEY")

    if not av_key:
        print("   SKIP - No ALPHAVANTAGE_API_KEY configured")
        results["AlphaVantage"] = "SKIP"
    else:
        resp = requests.get(
            f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={av_key}",
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if "Global Quote" in data and data["Global Quote"]:
                quote = data["Global Quote"]
                print(f"   SUCCESS - AAPL Price: ${quote.get('05. price', 'N/A')}")
                results["AlphaVantage"] = "SUCCESS"
            elif "Note" in data:
                print(f"   RATE_LIMITED - {data['Note'][:60]}...")
                results["AlphaVantage"] = "RATE_LIMITED"
            else:
                print(f"   FAILED - Unexpected response: {str(data)[:100]}")
                results["AlphaVantage"] = "FAILED"
        else:
            print(f"   FAILED - Status {resp.status_code}")
            results["AlphaVantage"] = "FAILED"
except Exception as e:
    print(f"   ERROR - {e}")
    results["AlphaVantage"] = "ERROR"

# =============================================================================
# 7. FRED TEST
# =============================================================================
print("\n[7/8] TESTING FRED (Federal Reserve)...")
try:
    import requests
    fred_key = os.getenv("FRED_API_KEY")

    if not fred_key:
        print("   SKIP - No FRED_API_KEY configured")
        results["FRED"] = "SKIP"
    else:
        resp = requests.get(
            f"https://api.stlouisfed.org/fred/series?series_id=GDP&api_key={fred_key}&file_type=json",
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if "seriess" in data:
                series = data["seriess"][0]
                print(f"   SUCCESS - Series: {series['title']}")
                results["FRED"] = "SUCCESS"
            else:
                print(f"   FAILED - Unexpected response")
                results["FRED"] = "FAILED"
        else:
            print(f"   FAILED - Status {resp.status_code}")
            results["FRED"] = "FAILED"
except Exception as e:
    print(f"   ERROR - {e}")
    results["FRED"] = "ERROR"

# =============================================================================
# 8. LOCAL LOGGING TEST
# =============================================================================
print("\n[8/8] TESTING LOCAL LOGGING...")
try:
    from loguru import logger
    from src.config.settings import get_settings

    settings = get_settings()
    log_file = settings.logs_dir / "api_test.log"

    # Configure logger
    logger.add(log_file, rotation="10 MB", level="INFO")

    # Write test log
    logger.info("API Test Script - Logging test successful")
    logger.info(f"Data directory: {settings.data_dir}")
    logger.info(f"Logs directory: {settings.logs_dir}")

    # Verify log file exists
    if log_file.exists():
        print(f"   SUCCESS - Log file created: {log_file}")
        print(f"   Data dir: {settings.data_dir}")
        print(f"   Logs dir: {settings.logs_dir}")
        results["Logging"] = "SUCCESS"
    else:
        print(f"   FAILED - Log file not created")
        results["Logging"] = "FAILED"
except Exception as e:
    print(f"   ERROR - {e}")
    results["Logging"] = "ERROR"

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

success_count = sum(1 for v in results.values() if v == "SUCCESS")
total_count = len(results)

for api, status in results.items():
    status_icon = "[OK]" if status == "SUCCESS" else ("[WARN]" if status in ["SKIP", "RATE_LIMITED"] else "[FAIL]")
    print(f"  {status_icon} {api}: {status}")

print(f"\nResult: {success_count}/{total_count} APIs connected successfully")
print("=" * 70)

# Return exit code based on critical APIs
critical_apis = ["GitHub", "Anthropic", "OpenAI", "Polygon"]
critical_failures = [api for api in critical_apis if results.get(api) in ["FAILED", "ERROR"]]

if critical_failures:
    print(f"\nWARNING: Critical API failures: {', '.join(critical_failures)}")
    sys.exit(1)
else:
    print("\nAll critical APIs operational!")
    sys.exit(0)

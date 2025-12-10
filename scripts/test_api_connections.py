"""================================================================================
API CONNECTION TESTER
================================================================================
Tests all API connections configured in the system.
Run this to verify your .env file has valid credentials.

Usage:
    python scripts/test_api_connections.py
================================================================================
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import time
from datetime import datetime

import requests
from loguru import logger

from src.config.settings import get_settings


def test_massive_api() -> dict:
    """Test Massive.com (formerly Polygon.io) API connection."""
    settings = get_settings()
    result = {"name": "Massive.com", "status": "unknown", "message": "", "latency_ms": 0}

    if not settings.massive_api_key:
        result["status"] = "skipped"
        result["message"] = "No API key configured (PolygonIO_API_KEY)"
        return result

    try:
        start = time.time()
        url = "https://api.massive.com/v2/aggs/ticker/SPY/prev"
        params = {"apiKey": settings.massive_api_key}
        resp = requests.get(url, params=params, timeout=10)
        latency = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "OK":
                result["status"] = "success"
                result["message"] = f"Connected - SPY prev close: ${data.get('results', [{}])[0].get('c', 'N/A')}"
            else:
                result["status"] = "error"
                result["message"] = f"API returned: {data.get('status', 'Unknown')}"
        else:
            result["status"] = "error"
            result["message"] = f"HTTP {resp.status_code}"

        result["latency_ms"] = round(latency, 2)
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)

    return result


def test_massive_s3() -> dict:
    """Test Massive.com S3 flat files access."""
    settings = get_settings()
    result = {"name": "Massive.com S3", "status": "unknown", "message": "", "latency_ms": 0}

    if not settings.massive_access_key or not settings.massive_secret_key:
        result["status"] = "skipped"
        result["message"] = "No S3 credentials configured"
        return result

    try:
        import boto3
        from botocore.config import Config

        start = time.time()
        s3 = boto3.client(
            "s3",
            endpoint_url=settings.massive_endpoint_url,
            aws_access_key_id=settings.massive_access_key,
            aws_secret_access_key=settings.massive_secret_key,
            config=Config(signature_version="s3v4"),
        )
        # Just list a few files to test connection
        response = s3.list_objects_v2(Bucket="flatfiles", Prefix="us_stocks_sip/", MaxKeys=5)
        latency = (time.time() - start) * 1000

        count = len(response.get("Contents", []))
        result["status"] = "success"
        result["message"] = f"Connected - Found {count} sample files"
        result["latency_ms"] = round(latency, 2)
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)

    return result


def test_alpha_vantage_api() -> dict:
    """Test Alpha Vantage API connection."""
    settings = get_settings()
    result = {"name": "Alpha Vantage", "status": "unknown", "message": "", "latency_ms": 0}

    if not settings.alpha_vantage_api_key:
        result["status"] = "skipped"
        result["message"] = "No API key configured (ALPHAVANTAGE_API_KEY)"
        return result

    try:
        start = time.time()
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": "SPY",
            "apikey": settings.alpha_vantage_api_key,
        }
        resp = requests.get(url, params=params, timeout=15)
        latency = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            if "Global Quote" in data:
                price = data["Global Quote"].get("05. price", "N/A")
                result["status"] = "success"
                result["message"] = f"Connected - SPY price: ${price}"
            elif "Note" in data:
                result["status"] = "warning"
                result["message"] = "Rate limit reached (5 calls/min)"
            else:
                result["status"] = "error"
                result["message"] = f"Unexpected response: {list(data.keys())}"
        else:
            result["status"] = "error"
            result["message"] = f"HTTP {resp.status_code}"

        result["latency_ms"] = round(latency, 2)
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)

    return result


def test_fred_api() -> dict:
    """Test FRED (Federal Reserve) API connection."""
    settings = get_settings()
    result = {"name": "FRED", "status": "unknown", "message": "", "latency_ms": 0}

    if not settings.fred_api_key:
        result["status"] = "skipped"
        result["message"] = "No API key configured (FRED_DATA_API)"
        return result

    try:
        start = time.time()
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "DFF",  # Federal Funds Rate
            "api_key": settings.fred_api_key,
            "file_type": "json",
            "limit": 1,
            "sort_order": "desc",
        }
        resp = requests.get(url, params=params, timeout=10)
        latency = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            if "observations" in data and len(data["observations"]) > 0:
                obs = data["observations"][0]
                result["status"] = "success"
                result["message"] = f"Connected - Fed Funds Rate: {obs.get('value', 'N/A')}% ({obs.get('date', '')})"
            else:
                result["status"] = "error"
                result["message"] = "No data returned"
        else:
            result["status"] = "error"
            result["message"] = f"HTTP {resp.status_code}"

        result["latency_ms"] = round(latency, 2)
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)

    return result


def test_openai_api() -> dict:
    """Test OpenAI API connection."""
    settings = get_settings()
    result = {"name": "OpenAI", "status": "unknown", "message": "", "latency_ms": 0}

    if not settings.openai_api_key:
        result["status"] = "skipped"
        result["message"] = "No API key configured (OPENAI_SECRET)"
        return result

    try:
        start = time.time()
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        resp = requests.get(url, headers=headers, timeout=10)
        latency = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            model_count = len(data.get("data", []))
            result["status"] = "success"
            result["message"] = f"Connected - {model_count} models available"
        elif resp.status_code == 401:
            result["status"] = "error"
            result["message"] = "Invalid API key"
        else:
            result["status"] = "error"
            result["message"] = f"HTTP {resp.status_code}"

        result["latency_ms"] = round(latency, 2)
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)

    return result


def test_anthropic_api() -> dict:
    """Test Anthropic API connection."""
    settings = get_settings()
    result = {"name": "Anthropic", "status": "unknown", "message": "", "latency_ms": 0}

    if not settings.anthropic_api_key:
        result["status"] = "skipped"
        result["message"] = "No API key configured (ANTHROPIC_API_KEY)"
        return result

    try:
        start = time.time()
        # Use a minimal API call to test auth
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        # Just check if we can auth (will fail with minimal payload but that's ok)
        resp = requests.post(
            url,
            headers=headers,
            json={"model": "claude-3-haiku-20240307", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
            timeout=15,
        )
        latency = (time.time() - start) * 1000

        if resp.status_code == 200:
            result["status"] = "success"
            result["message"] = "Connected - API key valid"
        elif resp.status_code == 401:
            result["status"] = "error"
            result["message"] = "Invalid API key"
        elif resp.status_code == 400:
            # Bad request but auth worked
            result["status"] = "success"
            result["message"] = "Connected - API key valid"
        else:
            result["status"] = "warning"
            result["message"] = f"HTTP {resp.status_code} - Key may be valid"

        result["latency_ms"] = round(latency, 2)
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)

    return result


def test_perplexity_api() -> dict:
    """Test Perplexity API connection."""
    settings = get_settings()
    result = {"name": "Perplexity", "status": "unknown", "message": "", "latency_ms": 0}

    if not settings.perplexity_api_key:
        result["status"] = "skipped"
        result["message"] = "No API key configured (PERPLEXITY_API_KEY)"
        return result

    try:
        start = time.time()
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.perplexity_api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            url,
            headers=headers,
            json={
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=15,
        )
        latency = (time.time() - start) * 1000

        if resp.status_code == 200:
            result["status"] = "success"
            result["message"] = "Connected - API key valid"
        elif resp.status_code == 401:
            result["status"] = "error"
            result["message"] = "Invalid API key"
        else:
            result["status"] = "warning"
            result["message"] = f"HTTP {resp.status_code}"

        result["latency_ms"] = round(latency, 2)
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)

    return result


def test_google_api() -> dict:
    """Test Google/Gemini API connection."""
    settings = get_settings()
    result = {"name": "Google/Gemini", "status": "unknown", "message": "", "latency_ms": 0}

    if not settings.google_api_key:
        result["status"] = "skipped"
        result["message"] = "No API key configured (API_KEY)"
        return result

    try:
        start = time.time()
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={settings.google_api_key}"
        resp = requests.get(url, timeout=10)
        latency = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            model_count = len(data.get("models", []))
            result["status"] = "success"
            result["message"] = f"Connected - {model_count} models available"
        elif resp.status_code == 400:
            result["status"] = "error"
            result["message"] = "Invalid API key"
        else:
            result["status"] = "error"
            result["message"] = f"HTTP {resp.status_code}"

        result["latency_ms"] = round(latency, 2)
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)

    return result


def test_database_connection() -> dict:
    """Test database connection."""
    settings = get_settings()
    result = {"name": "Database", "status": "unknown", "message": "", "latency_ms": 0}

    try:
        start = time.time()
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.sqlalchemy_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency = (time.time() - start) * 1000

        result["status"] = "success"
        db_type = "SQLite" if settings.use_sqlite else "Azure SQL"
        result["message"] = f"Connected to {db_type}"
        result["latency_ms"] = round(latency, 2)
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)[:100]

    return result


def print_result(result: dict):
    """Print a formatted result line."""
    status = result["status"]
    name = result["name"]
    message = result["message"]
    latency = result["latency_ms"]

    # Status emoji
    if status == "success":
        emoji = "✓"
        color = "\033[92m"  # Green
    elif status == "warning":
        emoji = "⚠"
        color = "\033[93m"  # Yellow
    elif status == "skipped":
        emoji = "○"
        color = "\033[90m"  # Gray
    else:
        emoji = "✗"
        color = "\033[91m"  # Red

    reset = "\033[0m"

    latency_str = f" ({latency}ms)" if latency > 0 else ""
    print(f"{color}{emoji} {name:20s}{reset} {message}{latency_str}")


def main():
    """Run all API connection tests."""
    print("=" * 70)
    print("ALPHA LOOP CAPITAL - API CONNECTION TEST")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    settings = get_settings()
    print(f"\nLoading settings from: {settings.base_dir}")
    print(f"Log directory: {settings.logs_dir}")
    print()

    # Run all tests
    tests = [
        ("DATA APIS", [
            test_massive_api,
            test_massive_s3,
            test_alpha_vantage_api,
            test_fred_api,
        ]),
        ("AI SERVICES", [
            test_openai_api,
            test_anthropic_api,
            test_perplexity_api,
            test_google_api,
        ]),
        ("INFRASTRUCTURE", [
            test_database_connection,
        ]),
    ]

    all_results = []
    for category, test_funcs in tests:
        print(f"\n{category}")
        print("-" * 40)
        for test_func in test_funcs:
            result = test_func()
            all_results.append(result)
            print_result(result)

    # Summary
    print("\n" + "=" * 70)
    success = sum(1 for r in all_results if r["status"] == "success")
    warning = sum(1 for r in all_results if r["status"] == "warning")
    error = sum(1 for r in all_results if r["status"] == "error")
    skipped = sum(1 for r in all_results if r["status"] == "skipped")

    print(f"SUMMARY: {success} success, {warning} warnings, {error} errors, {skipped} skipped")

    if error > 0:
        print("\n⚠️  Some APIs failed - check your .env file!")
        print(f"   Expected location: {settings.base_dir / '.env'}")
        print(f"   Or OneDrive: C:\\Users\\tom\\OneDrive\\Alpha Loop LLM\\API - Dec 2025.env")
        return 1

    print("\n✓ All configured APIs are working!")
    return 0


if __name__ == "__main__":
    sys.exit(main())


"""
Comprehensive API Connection Testing Script
===========================================
Tests all API connections used by Alpha Loop Capital agents.

Author: Tom Hogan | Alpha Loop Capital, LLC
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from loguru import logger

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.config.settings import get_settings

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)
logger.add(
    project_root / "logs" / "api_test_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)


class APITester:
    """Comprehensive API connection tester."""

    def __init__(self):
        self.settings = get_settings()
        self.results: Dict[str, Dict[str, any]] = {}
        self.env_file_path = r"C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env"

    def test_all(self) -> Dict[str, Dict[str, any]]:
        """Test all API connections."""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE API CONNECTION TEST")
        logger.info("=" * 80)
        logger.info(f"Environment file: {self.env_file_path}")
        logger.info(f"File exists: {os.path.exists(self.env_file_path)}")
        logger.info("")

        # Test data APIs
        self.test_massive_api()
        self.test_alpha_vantage_api()
        self.test_coinbase_api()
        self.test_fred_api()

        # Test AI APIs
        self.test_openai_api()
        self.test_anthropic_api()
        self.test_perplexity_api()
        self.test_google_api()

        # Test database
        self.test_database_connection()

        # Print summary
        self.print_summary()

        return self.results

    def test_massive_api(self) -> Tuple[bool, str]:
        """Test Massive.com API (rebranded from Polygon.io)."""
        logger.info("Testing Massive.com API (rebranded from Polygon.io)...")
        api_key = self.settings.polygon_api_key

        if not api_key:
            self.results["massive"] = {
                "status": "FAILED",
                "error": "API key not found",
                "key_present": False,
            }
            logger.error("  ❌ Massive.com API key not found")
            return False, "API key not found"

        try:
            # Test tickers endpoint
            url = "https://api.massive.com/v3/reference/tickers"
            params = {"market": "stocks", "active": "true", "limit": 1, "apiKey": api_key}
            resp = requests.get(url, params=params, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                count = len(data.get("results", []))
                self.results["massive"] = {
                    "status": "SUCCESS",
                    "status_code": resp.status_code,
                    "response_time_ms": resp.elapsed.total_seconds() * 1000,
                    "tickers_returned": count,
                    "key_present": True,
                }
                logger.info(f"  ✅ Massive.com API: Connected (returned {count} tickers)")
                return True, "Connected"
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                self.results["massive"] = {
                    "status": "FAILED",
                    "status_code": resp.status_code,
                    "error": error_msg,
                    "key_present": True,
                }
                logger.error(f"  ❌ Massive.com API: {error_msg}")
                return False, error_msg

        except Exception as e:
            error_msg = str(e)
            self.results["massive"] = {
                "status": "FAILED",
                "error": error_msg,
                "key_present": True,
            }
            logger.error(f"  ❌ Massive.com API: {error_msg}")
            return False, error_msg

    def test_alpha_vantage_api(self) -> Tuple[bool, str]:
        """Test Alpha Vantage API."""
        logger.info("Testing Alpha Vantage API...")
        api_key = self.settings.alpha_vantage_api_key

        if not api_key:
            self.results["alpha_vantage"] = {
                "status": "FAILED",
                "error": "API key not found",
                "key_present": False,
            }
            logger.error("  ❌ Alpha Vantage API key not found")
            return False, "API key not found"

        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": "AAPL",
                "interval": "5min",
                "apikey": api_key,
            }
            resp = requests.get(url, params=params, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                if "Error Message" in data:
                    error_msg = data["Error Message"]
                    self.results["alpha_vantage"] = {
                        "status": "FAILED",
                        "error": error_msg,
                        "key_present": True,
                    }
                    logger.error(f"  ❌ Alpha Vantage API: {error_msg}")
                    return False, error_msg
                else:
                    self.results["alpha_vantage"] = {
                        "status": "SUCCESS",
                        "status_code": resp.status_code,
                        "response_time_ms": resp.elapsed.total_seconds() * 1000,
                        "key_present": True,
                    }
                    logger.info("  ✅ Alpha Vantage API: Connected")
                    return True, "Connected"
            else:
                error_msg = f"HTTP {resp.status_code}"
                self.results["alpha_vantage"] = {
                    "status": "FAILED",
                    "status_code": resp.status_code,
                    "error": error_msg,
                    "key_present": True,
                }
                logger.error(f"  ❌ Alpha Vantage API: {error_msg}")
                return False, error_msg

        except Exception as e:
            error_msg = str(e)
            self.results["alpha_vantage"] = {
                "status": "FAILED",
                "error": error_msg,
                "key_present": True,
            }
            logger.error(f"  ❌ Alpha Vantage API: {error_msg}")
            return False, error_msg

    def test_coinbase_api(self) -> Tuple[bool, str]:
        """Test Coinbase API."""
        logger.info("Testing Coinbase API...")
        api_key = self.settings.coinbase_api_key
        api_secret = self.settings.coinbase_api_secret

        if not api_key or not api_secret:
            self.results["coinbase"] = {
                "status": "FAILED",
                "error": "API key or secret not found",
                "key_present": bool(api_key),
                "secret_present": bool(api_secret),
            }
            logger.error("  ❌ Coinbase API credentials not found")
            return False, "Credentials not found"

        try:
            # Test public endpoint (no auth required)
            url = "https://api.coinbase.com/v2/exchange-rates"
            resp = requests.get(url, timeout=10)

            if resp.status_code == 200:
                self.results["coinbase"] = {
                    "status": "SUCCESS",
                    "status_code": resp.status_code,
                    "response_time_ms": resp.elapsed.total_seconds() * 1000,
                    "key_present": True,
                    "secret_present": True,
                    "note": "Public endpoint tested (auth not verified)",
                }
                logger.info("  ✅ Coinbase API: Connected (public endpoint)")
                return True, "Connected"
            else:
                error_msg = f"HTTP {resp.status_code}"
                self.results["coinbase"] = {
                    "status": "FAILED",
                    "status_code": resp.status_code,
                    "error": error_msg,
                    "key_present": True,
                    "secret_present": True,
                }
                logger.error(f"  ❌ Coinbase API: {error_msg}")
                return False, error_msg

        except Exception as e:
            error_msg = str(e)
            self.results["coinbase"] = {
                "status": "FAILED",
                "error": error_msg,
                "key_present": True,
                "secret_present": True,
            }
            logger.error(f"  ❌ Coinbase API: {error_msg}")
            return False, error_msg

    def test_fred_api(self) -> Tuple[bool, str]:
        """Test FRED API."""
        logger.info("Testing FRED API...")
        api_key = self.settings.fred_api_key

        if not api_key:
            self.results["fred"] = {
                "status": "FAILED",
                "error": "API key not found",
                "key_present": False,
            }
            logger.error("  ❌ FRED API key not found")
            return False, "API key not found"

        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "GDP",
                "api_key": api_key,
                "file_type": "json",
                "limit": 1,
            }
            resp = requests.get(url, params=params, timeout=10)

            if resp.status_code == 200:
                self.results["fred"] = {
                    "status": "SUCCESS",
                    "status_code": resp.status_code,
                    "response_time_ms": resp.elapsed.total_seconds() * 1000,
                    "key_present": True,
                }
                logger.info("  ✅ FRED API: Connected")
                return True, "Connected"
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                self.results["fred"] = {
                    "status": "FAILED",
                    "status_code": resp.status_code,
                    "error": error_msg,
                    "key_present": True,
                }
                logger.error(f"  ❌ FRED API: {error_msg}")
                return False, error_msg

        except Exception as e:
            error_msg = str(e)
            self.results["fred"] = {
                "status": "FAILED",
                "error": error_msg,
                "key_present": True,
            }
            logger.error(f"  ❌ FRED API: {error_msg}")
            return False, error_msg

    def test_openai_api(self) -> Tuple[bool, str]:
        """Test OpenAI API."""
        logger.info("Testing OpenAI API...")
        api_key = self.settings.openai_api_key

        if not api_key:
            self.results["openai"] = {
                "status": "FAILED",
                "error": "API key not found",
                "key_present": False,
            }
            logger.error("  ❌ OpenAI API key not found")
            return False, "API key not found"

        try:
            url = "https://api.openai.com/v1/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            resp = requests.get(url, headers=headers, timeout=10)

            if resp.status_code == 200:
                self.results["openai"] = {
                    "status": "SUCCESS",
                    "status_code": resp.status_code,
                    "response_time_ms": resp.elapsed.total_seconds() * 1000,
                    "key_present": True,
                }
                logger.info("  ✅ OpenAI API: Connected")
                return True, "Connected"
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                self.results["openai"] = {
                    "status": "FAILED",
                    "status_code": resp.status_code,
                    "error": error_msg,
                    "key_present": True,
                }
                logger.error(f"  ❌ OpenAI API: {error_msg}")
                return False, error_msg

        except Exception as e:
            error_msg = str(e)
            self.results["openai"] = {
                "status": "FAILED",
                "error": error_msg,
                "key_present": True,
            }
            logger.error(f"  ❌ OpenAI API: {error_msg}")
            return False, error_msg

    def test_anthropic_api(self) -> Tuple[bool, str]:
        """Test Anthropic API."""
        logger.info("Testing Anthropic API...")
        api_key = self.settings.anthropic_api_key

        if not api_key:
            self.results["anthropic"] = {
                "status": "FAILED",
                "error": "API key not found",
                "key_present": False,
            }
            logger.error("  ❌ Anthropic API key not found")
            return False, "API key not found"

        try:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            # Minimal test payload
            payload = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "test"}],
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=10)

            if resp.status_code == 200:
                self.results["anthropic"] = {
                    "status": "SUCCESS",
                    "status_code": resp.status_code,
                    "response_time_ms": resp.elapsed.total_seconds() * 1000,
                    "key_present": True,
                }
                logger.info("  ✅ Anthropic API: Connected")
                return True, "Connected"
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                self.results["anthropic"] = {
                    "status": "FAILED",
                    "status_code": resp.status_code,
                    "error": error_msg,
                    "key_present": True,
                }
                logger.error(f"  ❌ Anthropic API: {error_msg}")
                return False, error_msg

        except Exception as e:
            error_msg = str(e)
            self.results["anthropic"] = {
                "status": "FAILED",
                "error": error_msg,
                "key_present": True,
            }
            logger.error(f"  ❌ Anthropic API: {error_msg}")
            return False, error_msg

    def test_perplexity_api(self) -> Tuple[bool, str]:
        """Test Perplexity API."""
        logger.info("Testing Perplexity API...")
        api_key = self.settings.perplexity_api_key

        if not api_key:
            self.results["perplexity"] = {
                "status": "FAILED",
                "error": "API key not found",
                "key_present": False,
            }
            logger.error("  ❌ Perplexity API key not found")
            return False, "API key not found"

        try:
            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=10)

            if resp.status_code == 200:
                self.results["perplexity"] = {
                    "status": "SUCCESS",
                    "status_code": resp.status_code,
                    "response_time_ms": resp.elapsed.total_seconds() * 1000,
                    "key_present": True,
                }
                logger.info("  ✅ Perplexity API: Connected")
                return True, "Connected"
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                self.results["perplexity"] = {
                    "status": "FAILED",
                    "status_code": resp.status_code,
                    "error": error_msg,
                    "key_present": True,
                }
                logger.error(f"  ❌ Perplexity API: {error_msg}")
                return False, error_msg

        except Exception as e:
            error_msg = str(e)
            self.results["perplexity"] = {
                "status": "FAILED",
                "error": error_msg,
                "key_present": True,
            }
            logger.error(f"  ❌ Perplexity API: {error_msg}")
            return False, error_msg

    def test_google_api(self) -> Tuple[bool, str]:
        """Test Google API."""
        logger.info("Testing Google API...")
        api_key = self.settings.google_api_key

        if not api_key:
            self.results["google"] = {
                "status": "FAILED",
                "error": "API key not found",
                "key_present": False,
            }
            logger.error("  ❌ Google API key not found")
            return False, "API key not found"

        try:
            # Test with a simple endpoint
            url = "https://generativelanguage.googleapis.com/v1/models"
            params = {"key": api_key}
            resp = requests.get(url, params=params, timeout=10)

            if resp.status_code == 200:
                self.results["google"] = {
                    "status": "SUCCESS",
                    "status_code": resp.status_code,
                    "response_time_ms": resp.elapsed.total_seconds() * 1000,
                    "key_present": True,
                }
                logger.info("  ✅ Google API: Connected")
                return True, "Connected"
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                self.results["google"] = {
                    "status": "FAILED",
                    "status_code": resp.status_code,
                    "error": error_msg,
                    "key_present": True,
                }
                logger.error(f"  ❌ Google API: {error_msg}")
                return False, error_msg

        except Exception as e:
            error_msg = str(e)
            self.results["google"] = {
                "status": "FAILED",
                "error": error_msg,
                "key_present": True,
            }
            logger.error(f"  ❌ Google API: {error_msg}")
            return False, error_msg

    def test_database_connection(self) -> Tuple[bool, str]:
        """Test database connection."""
        logger.info("Testing Database Connection...")
        try:
            from src.database.connection import healthcheck

            if healthcheck():
                self.results["database"] = {
                    "status": "SUCCESS",
                    "key_present": True,
                }
                logger.info("  ✅ Database: Connected")
                return True, "Connected"
            else:
                self.results["database"] = {
                    "status": "FAILED",
                    "error": "Health check failed",
                    "key_present": True,
                }
                logger.error("  ❌ Database: Health check failed")
                return False, "Health check failed"

        except Exception as e:
            error_msg = str(e)
            self.results["database"] = {
                "status": "FAILED",
                "error": error_msg,
                "key_present": False,
            }
            logger.error(f"  ❌ Database: {error_msg}")
            return False, error_msg

    def print_summary(self) -> None:
        """Print test summary."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)

        success_count = sum(1 for r in self.results.values() if r.get("status") == "SUCCESS")
        total_count = len(self.results)

        logger.info(f"Total APIs tested: {total_count}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {total_count - success_count}")
        logger.info("")

        for api_name, result in sorted(self.results.items()):
            status = result.get("status", "UNKNOWN")
            status_icon = "✅" if status == "SUCCESS" else "❌"
            logger.info(f"{status_icon} {api_name.upper()}: {status}")

            if status == "FAILED":
                error = result.get("error", "Unknown error")
                logger.info(f"   Error: {error}")

        logger.info("")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    tester = APITester()
    results = tester.test_all()

    # Exit with error code if any critical APIs failed
    critical_apis = ["massive", "database"]
    failed_critical = any(
        results.get(api, {}).get("status") != "SUCCESS" for api in critical_apis
    )

    if failed_critical:
        logger.error("Critical API tests failed!")
        sys.exit(1)
    else:
        logger.info("All critical API tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()


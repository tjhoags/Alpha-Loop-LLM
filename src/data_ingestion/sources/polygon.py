"""================================================================================
MASSIVE.COM DATA SOURCE (formerly Polygon.io)
================================================================================
Fetches market data from Massive.com API.
Note: Polygon.io rebranded to Massive.com - API endpoints remain compatible.
================================================================================
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings

# Massive.com API (rebranded from Polygon.io)
BASE_URL = "https://api.massive.com"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def fetch_aggregates(
    symbol: str,
    timespan: str = "minute",
    multiplier: int = 5,
    lookback_hours: Optional[int] = None,
    lookback_days: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch aggregate bars from Massive.com API.

    Args:
        symbol: Stock ticker symbol
        timespan: Bar timespan ("minute", "hour", "day")
        multiplier: Timespan multiplier (e.g., 5 for 5-minute bars)
        lookback_hours: Hours of history to fetch (default from settings)
        lookback_days: Days of history to fetch (overrides lookback_hours)

    Returns:
        DataFrame with OHLCV data
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)

    # Calculate date range
    if lookback_days is not None:
        start = now - timedelta(days=lookback_days)
    else:
        hrs = lookback_hours if lookback_hours is not None else settings.massive_lookback_hours
        start = now - timedelta(hours=hrs)

    url = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start.date()}/{now.date()}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": settings.massive_api_key,
    }

    resp = requests.get(url, params=params, timeout=45)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    records = []
    for row in results:
        ts = datetime.fromtimestamp(row["t"] / 1000, tz=timezone.utc)
        records.append({
            "symbol": symbol,
            "timestamp": ts,
            "open": row["o"],
            "high": row["h"],
            "low": row["l"],
            "close": row["c"],
            "volume": row["v"],
            "source": "massive",
        })

    df = pd.DataFrame(records)
    logger.info(f"Massive.com fetched {len(df)} rows for {symbol}")
    return df


def fetch_latest(symbol: str, timespan: str = "minute", multiplier: int = 5) -> Optional[pd.Series]:
    """Fetch the latest bar for a symbol.

    Args:
        symbol: Stock ticker symbol
        timespan: Bar timespan
        multiplier: Timespan multiplier

    Returns:
        Latest bar as Series, or None if unavailable
    """
    df = fetch_aggregates(symbol, timespan=timespan, multiplier=multiplier, lookback_hours=6)
    if df.empty:
        return None
    return df.sort_values("timestamp").iloc[-1]


def test_connection() -> bool:
    """Test Massive.com API connection.

    Returns:
        True if connection successful, False otherwise
    """
    settings = get_settings()
    if not settings.massive_api_key:
        logger.warning("No Massive.com API key configured")
        return False

    try:
        url = f"{BASE_URL}/v2/aggs/ticker/SPY/prev"
        params = {"apiKey": settings.massive_api_key}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        logger.info("✓ Massive.com API connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Massive.com API connection failed: {e}")
        return False

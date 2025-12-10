from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings

BASE_URL = "https://api.polygon.io"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def fetch_aggregates(symbol: str, timespan: str = "minute", multiplier: int = 5, lookback_hours: Optional[int] = None) -> pd.DataFrame:
    """Fetch aggregate bars from Polygon with extended lookback.
    For large ranges, Polygon caps at 50k bars per call. With 5m bars, 240h ~ 2880 bars.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    hrs = lookback_hours if lookback_hours is not None else settings.polygon_lookback_hours
    start = now - timedelta(hours=hrs)
    url = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start.date()}/{now.date()}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": settings.polygon_api_key}
    resp = requests.get(url, params=params, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    records = []
    for row in results:
        ts = datetime.fromtimestamp(row["t"] / 1000, tz=timezone.utc)
        records.append(
            {
                "symbol": symbol,
                "timestamp": ts,
                "open": row["o"],
                "high": row["h"],
                "low": row["l"],
                "close": row["c"],
                "volume": row["v"],
                "source": "polygon",
            },
        )
    df = pd.DataFrame(records)
    logger.info(f"Polygon fetched {len(df)} rows for {symbol} (lookback_hours={hrs})")
    return df


def fetch_latest(symbol: str, timespan: str = "minute", multiplier: int = 5) -> Optional[pd.Series]:
    df = fetch_aggregates(symbol, timespan=timespan, multiplier=multiplier, lookback_hours=6)
    if df.empty:
        return None
    return df.sort_values("timestamp").iloc[-1]


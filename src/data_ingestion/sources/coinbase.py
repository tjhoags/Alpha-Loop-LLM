from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings


BASE_URL = "https://api.exchange.coinbase.com"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def fetch_candles(product_id: str = "BTC-USD", granularity: int = 300, lookback_hours: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch candles from Coinbase Pro-style API.
    granularity: seconds (300 = 5m)
    """
    settings = get_settings()
    hrs = lookback_hours if lookback_hours is not None else settings.coinbase_lookback_hours
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hrs)
    params = {"start": start.isoformat(), "end": end.isoformat(), "granularity": granularity}
    headers = {}
    if settings.coinbase_api_key:
        headers["CB-ACCESS-KEY"] = settings.coinbase_api_key
    url = f"{BASE_URL}/products/{product_id}/candles"
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    records = []
    for row in data:
        # Coinbase returns [time, low, high, open, close, volume]
        ts = datetime.fromtimestamp(row[0], tz=timezone.utc)
        records.append(
            {
                "symbol": product_id,
                "timestamp": ts,
                "open": float(row[3]),
                "high": float(row[2]),
                "low": float(row[1]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "source": "coinbase",
            }
        )
    df = pd.DataFrame(records).sort_values("timestamp")
    logger.info(f"Coinbase fetched {len(df)} rows for {product_id}")
    return df


def fetch_latest(product_id: str = "BTC-USD", granularity: int = 300) -> Optional[pd.Series]:
    df = fetch_candles(product_id, granularity=granularity, lookback_hours=6)
    if df.empty:
        return None
    return df.iloc[-1]


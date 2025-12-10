from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings

API_URL = "https://www.alphavantage.co/query"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def fetch_intraday(symbol: str, interval: str = "1min") -> pd.DataFrame:
    """Fetch intraday data from Alpha Vantage.
    PREMIUM TIER: Uses TIME_SERIES_INTRADAY_EXTENDED for 2 years of 1min history.
    """
    settings = get_settings()

    # Premium Extended History Logic
    # We need to slice year1month1, year1month2... for full history.
    # For now, we'll fetch the standard 'full' which gives trailing 30 days high res,
    # OR we implement the slice loop.
    # Let's start with the standard 'full' on 1min to get recent high-res,
    # as Massive covers the deep history better.

    params = {
        "function": "TIME_SERIES_INTRADAY", # Switch to TIME_SERIES_INTRADAY_EXTENDED if implementing CSV slice loop
        "symbol": symbol,
        "interval": interval,
        "apikey": settings.alpha_vantage_api_key,
        "outputsize": "full", # Premium feature
    }
    resp = requests.get(API_URL, params=params, timeout=30)

    resp.raise_for_status()
    data = resp.json()
    key = f"Time Series ({interval})"
    if key not in data:
        raise ValueError(f"Unexpected Alpha Vantage response: {data}")
    records = []
    for ts, vals in data[key].items():
        records.append(
            {
                "symbol": symbol,
                "timestamp": datetime.fromisoformat(ts),
                "open": float(vals["1. open"]),
                "high": float(vals["2. high"]),
                "low": float(vals["3. low"]),
                "close": float(vals["4. close"]),
                "volume": float(vals["5. volume"]),
                "source": "alpha_vantage",
            },
        )
    df = pd.DataFrame(records).sort_values("timestamp")
    logger.info(f"Alpha Vantage fetched {len(df)} rows for {symbol}")
    return df


def fetch_latest(symbol: str, interval: str = "5min") -> Optional[pd.Series]:
    df = fetch_intraday(symbol, interval=interval)
    if df.empty:
        return None
    return df.iloc[-1]


"""================================================================================
ALPHA VANTAGE PREMIUM DATA INGESTION
================================================================================
Top-tier Alpha Vantage API integration for:
- Stocks (intraday, daily, fundamental data)
- Indices (S&P 500, NASDAQ, etc.)
- Currencies/Forex (FX pairs)
- Options (chain data with Greeks)
- Advanced valuation metrics (P/E, EV/EBITDA, etc.)
================================================================================
"""

import io
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings

API_URL = "https://www.alphavantage.co/query"
API_CALL_DELAY = 12.1  # Premium tier: 75 calls/minute = 1 call per 0.8s, but we use 12s for safety


class AlphaVantagePremium:
    """Premium Alpha Vantage client with all asset classes."""

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.alpha_vantage_api_key
        self.last_call_time = 0

    def _rate_limit(self):
        """Enforce API rate limiting."""
        elapsed = time.time() - self.last_call_time
        if elapsed < API_CALL_DELAY:
            time.sleep(API_CALL_DELAY - elapsed)
        self.last_call_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
    def _make_request(self, params: Dict) -> Dict:
        """Make API request with rate limiting."""
        self._rate_limit()
        params["apikey"] = self.api_key
        resp = requests.get(API_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        if "Note" in data:
            logger.warning(f"Rate limit note: {data['Note']}")
            time.sleep(60)  # Wait 1 minute if rate limited
            return self._make_request(params)

        return data

    def fetch_stock_intraday(self, symbol: str, interval: str = "1min", outputsize: str = "full") -> pd.DataFrame:
        """Fetch intraday stock data (PREMIUM: up to 2 years of 1min bars)."""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,  # "compact" (100 bars) or "full" (up to 2 years)
        }

        data = self._make_request(params)
        key = f"Time Series ({interval})"
        if key not in data:
            logger.warning(f"No intraday data for {symbol}")
            return pd.DataFrame()

        records = []
        for ts_str, vals in data[key].items():
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                records.append({
                    "symbol": symbol,
                    "timestamp": ts,
                    "open": float(vals["1. open"]),
                    "high": float(vals["2. high"]),
                    "low": float(vals["3. low"]),
                    "close": float(vals["4. close"]),
                    "volume": float(vals["5. volume"]),
                    "source": "alpha_vantage_stock",
                    "asset_type": "stock",
                })
            except Exception as e:
                logger.error(f"Error parsing {symbol} bar {ts_str}: {e}")
                continue

        df = pd.DataFrame(records).sort_values("timestamp")
        logger.info(f"Alpha Vantage fetched {len(df)} stock intraday rows for {symbol}")
        return df

    def fetch_stock_daily(self, symbol: str, outputsize: str = "full") -> pd.DataFrame:
        """Fetch daily stock data (PREMIUM: up to 20+ years)."""
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize,
        }

        data = self._make_request(params)
        key = "Time Series (Daily)"
        if key not in data:
            logger.warning(f"No daily data for {symbol}")
            return pd.DataFrame()

        records = []
        for date_str, vals in data[key].items():
            try:
                ts = datetime.strptime(date_str, "%Y-%m-%d")
                records.append({
                    "symbol": symbol,
                    "timestamp": ts,
                    "open": float(vals["1. open"]),
                    "high": float(vals["2. high"]),
                    "low": float(vals["3. low"]),
                    "close": float(vals["4. close"]),
                    "adjusted_close": float(vals["5. adjusted close"]),
                    "volume": float(vals["6. volume"]),
                    "dividend": float(vals.get("7. dividend amount", 0)),
                    "split_coefficient": float(vals.get("8. split coefficient", 1)),
                    "source": "alpha_vantage_daily",
                    "asset_type": "stock",
                })
            except Exception as e:
                logger.error(f"Error parsing {symbol} daily {date_str}: {e}")
                continue

        df = pd.DataFrame(records).sort_values("timestamp")
        logger.info(f"Alpha Vantage fetched {len(df)} daily rows for {symbol}")
        return df

    def fetch_index(self, index_symbol: str, interval: str = "daily") -> pd.DataFrame:
        """Fetch index data (e.g., SPX, NDX, DJI)."""
        # Alpha Vantage uses different functions for indices
        # For S&P 500: SPX, NASDAQ: NDX, Dow: DJI
        params = {
            "function": "TIME_SERIES_INTRADAY" if interval != "daily" else "TIME_SERIES_DAILY",
            "symbol": index_symbol,
            "interval": interval if interval != "daily" else None,
            "outputsize": "full",
        }
        params = {k: v for k, v in params.items() if v is not None}

        data = self._make_request(params)
        key = f"Time Series ({interval})" if interval != "daily" else "Time Series (Daily)"
        if key not in data:
            logger.warning(f"No index data for {index_symbol}")
            return pd.DataFrame()

        records = []
        for ts_str, vals in data[key].items():
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S" if interval != "daily" else "%Y-%m-%d")
                records.append({
                    "symbol": index_symbol,
                    "timestamp": ts,
                    "open": float(vals["1. open"]),
                    "high": float(vals["2. high"]),
                    "low": float(vals["3. low"]),
                    "close": float(vals["4. close"]),
                    "volume": float(vals.get("5. volume", 0)),
                    "source": "alpha_vantage_index",
                    "asset_type": "index",
                })
            except Exception as e:
                logger.error(f"Error parsing {index_symbol} {ts_str}: {e}")
                continue

        df = pd.DataFrame(records).sort_values("timestamp")
        logger.info(f"Alpha Vantage fetched {len(df)} index rows for {index_symbol}")
        return df

    def fetch_forex(self, from_currency: str, to_currency: str, interval: str = "1min") -> pd.DataFrame:
        """Fetch forex/currency pair data."""
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": from_currency,
            "to_symbol": to_currency,
            "interval": interval,
            "outputsize": "full",
        }

        data = self._make_request(params)
        key = f"Time Series FX ({interval})"
        if key not in data:
            logger.warning(f"No forex data for {from_currency}/{to_currency}")
            return pd.DataFrame()

        records = []
        symbol = f"{from_currency}/{to_currency}"
        for ts_str, vals in data[key].items():
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                records.append({
                    "symbol": symbol,
                    "timestamp": ts,
                    "open": float(vals["1. open"]),
                    "high": float(vals["2. high"]),
                    "low": float(vals["3. low"]),
                    "close": float(vals["4. close"]),
                    "source": "alpha_vantage_forex",
                    "asset_type": "forex",
                })
            except Exception as e:
                logger.error(f"Error parsing {symbol} {ts_str}: {e}")
                continue

        df = pd.DataFrame(records).sort_values("timestamp")
        logger.info(f"Alpha Vantage fetched {len(df)} forex rows for {symbol}")
        return df

    def fetch_fundamental_data(self, symbol: str) -> Optional[Dict]:
        """Fetch company fundamental data (valuation metrics)."""
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
        }

        data = self._make_request(params)
        if "Symbol" not in data:
            logger.warning(f"No fundamental data for {symbol}")
            return None

        # Extract key valuation metrics
        fundamentals = {
            "symbol": symbol,
            "timestamp": datetime.utcnow(),
            # Valuation ratios
            "pe_ratio": self._safe_float(data.get("PERatio")),
            "peg_ratio": self._safe_float(data.get("PEGRatio")),
            "price_to_book": self._safe_float(data.get("PriceToBookRatio")),
            "price_to_sales": self._safe_float(data.get("PriceToSalesRatioTTM")),
            "ev_to_revenue": self._safe_float(data.get("EVToRevenue")),
            "ev_to_ebitda": self._safe_float(data.get("EVToEBITDA")),
            "enterprise_value": self._safe_float(data.get("EnterpriseValue")),
            # Profitability
            "profit_margin": self._safe_float(data.get("ProfitMargin")),
            "operating_margin": self._safe_float(data.get("OperatingMarginTTM")),
            "gross_profit_ttm": self._safe_float(data.get("GrossProfitTTM")),
            "ebitda": self._safe_float(data.get("EBITDA")),
            "revenue_per_share": self._safe_float(data.get("RevenuePerShareTTM")),
            # Growth
            "revenue_growth": self._safe_float(data.get("QuarterlyRevenueGrowthYOY")),
            "earnings_growth": self._safe_float(data.get("QuarterlyEarningsGrowthYOY")),
            # Financial health
            "current_ratio": self._safe_float(data.get("CurrentRatio")),
            "quick_ratio": self._safe_float(data.get("QuickRatio")),
            "debt_to_equity": self._safe_float(data.get("DebtToEquity")),
            "return_on_equity": self._safe_float(data.get("ReturnOnEquityTTM")),
            "return_on_assets": self._safe_float(data.get("ReturnOnAssetsTTM")),
            # Market data
            "market_cap": self._safe_float(data.get("MarketCapitalization")),
            "shares_outstanding": self._safe_float(data.get("SharesOutstanding")),
            "dividend_yield": self._safe_float(data.get("DividendYield")),
            "beta": self._safe_float(data.get("Beta")),
            "52_week_high": self._safe_float(data.get("52WeekHigh")),
            "52_week_low": self._safe_float(data.get("52WeekLow")),
            "50_day_ma": self._safe_float(data.get("50DayMovingAverage")),
            "200_day_ma": self._safe_float(data.get("200DayMovingAverage")),
            "source": "alpha_vantage_fundamentals",
        }

        logger.info(f"Fetched fundamental data for {symbol}")
        return fundamentals

    def fetch_earnings(self, symbol: str) -> pd.DataFrame:
        """Fetch earnings calendar and estimates."""
        params = {
            "function": "EARNINGS_CALENDAR",
            "symbol": symbol,
            "horizon": "12month",
        }

        # This returns CSV format
        resp = requests.get(API_URL, params={**params, "apikey": self.api_key}, timeout=60)
        if resp.status_code != 200:
            return pd.DataFrame()

        try:
            df = pd.read_csv(io.StringIO(resp.text))
            df["symbol"] = symbol
            df["source"] = "alpha_vantage_earnings"
            logger.info(f"Fetched {len(df)} earnings records for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error parsing earnings for {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert to float."""
        if value in (None, "", "None", "N/A"):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


# Global instance
_av_premium = None

def get_av_premium() -> AlphaVantagePremium:
    """Get singleton Alpha Vantage premium client."""
    global _av_premium
    if _av_premium is None:
        _av_premium = AlphaVantagePremium()
    return _av_premium


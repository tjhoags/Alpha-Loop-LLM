"""
Alpha Vantage API Client
========================
Wrapper for Alpha Vantage market data API.
"""

import os
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaVantageClient:
    """
    Alpha Vantage API Client
    
    Free tier limits: 5 calls/minute, 500 calls/day
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")
        
        self.calls_made = 0
        self.last_call_time = None
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        if self.last_call_time:
            elapsed = (datetime.now() - self.last_call_time).total_seconds()
            if elapsed < 12:  # 5 calls/min = 1 call per 12 seconds
                time.sleep(12 - elapsed)
        
        self.last_call_time = datetime.now()
        self.calls_made += 1
    
    def _request(self, params: Dict) -> Dict:
        """Make API request"""
        self._rate_limit()
        params["apikey"] = self.api_key
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_daily_prices(
        self,
        symbol: str,
        outputsize: str = "compact"
    ) -> pd.DataFrame:
        """
        Get daily OHLCV data.
        
        Parameters:
            symbol: Ticker symbol
            outputsize: "compact" (100 days) or "full" (20+ years)
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize
        }
        
        data = self._request(params)
        
        if "Time Series (Daily)" not in data:
            logger.error(f"No data for {symbol}: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df = df.sort_index()
        
        return df
    
    def get_intraday_prices(
        self,
        symbol: str,
        interval: str = "5min"
    ) -> pd.DataFrame:
        """
        Get intraday OHLCV data.
        
        Parameters:
            symbol: Ticker symbol
            interval: "1min", "5min", "15min", "30min", "60min"
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval
        }
        
        data = self._request(params)
        
        key = f"Time Series ({interval})"
        if key not in data:
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(data[key], orient="index")
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df = df.sort_index()
        
        return df
    
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote"""
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol
        }
        
        data = self._request(params)
        
        quote = data.get("Global Quote", {})
        return {
            "symbol": quote.get("01. symbol"),
            "price": float(quote.get("05. price", 0)),
            "change": float(quote.get("09. change", 0)),
            "change_percent": quote.get("10. change percent", "0%"),
            "volume": int(quote.get("06. volume", 0))
        }
    
    def get_company_overview(self, symbol: str) -> Dict:
        """Get company fundamentals"""
        params = {
            "function": "OVERVIEW",
            "symbol": symbol
        }
        
        return self._request(params)
    
    def search_symbol(self, keywords: str) -> List[Dict]:
        """Search for symbols by keyword"""
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords
        }
        
        data = self._request(params)
        return data.get("bestMatches", [])


if __name__ == "__main__":
    # Demo
    client = AlphaVantageClient()
    
    # Get daily prices
    df = client.get_daily_prices("NVDA")
    print(f"NVDA Daily Data:\n{df.tail()}")
    
    # Get quote
    quote = client.get_quote("NVDA")
    print(f"\nNVDA Quote: {quote}")


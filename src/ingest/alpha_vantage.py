"""
Alpha Vantage data client.

Provides access to:
- Daily/Intraday price data
- Fundamental data (income statement, balance sheet, cash flow)
- Technical indicators
- Economic indicators
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Literal

import pandas as pd
import requests
from dotenv import load_dotenv

from .base import DataSource, DataSourceConfig

load_dotenv()
logger = logging.getLogger(__name__)


class AlphaVantageClient(DataSource):
    """
    Client for Alpha Vantage API.
    
    Supports free and premium tiers with automatic rate limiting.
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        premium: bool = False,
        config: Optional[DataSourceConfig] = None
    ):
        config = config or DataSourceConfig(
            name="alpha_vantage",
            rate_limit=75 if premium else 5
        )
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")
            
        self.premium = premium
        
    def validate_connection(self) -> bool:
        """Test API connection with a simple quote request."""
        try:
            df = self.get_quote("AAPL")
            return not df.empty
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
            
    def fetch_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "daily",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch price data for a symbol.
        
        Args:
            symbol: Stock ticker
            start_date: Start date (used for filtering)
            end_date: End date (used for filtering)
            interval: 'daily', 'weekly', 'monthly', or intraday intervals
            
        Returns:
            DataFrame with OHLCV data
        """
        if interval == "daily":
            df = self.get_daily_adjusted(symbol, outputsize="full")
        elif interval == "weekly":
            df = self.get_weekly_adjusted(symbol)
        elif interval == "monthly":
            df = self.get_monthly_adjusted(symbol)
        else:
            df = self.get_intraday(symbol, interval=interval)
            
        # Filter by date range
        if start_date and not df.empty:
            df = df[df.index >= start_date]
        if end_date and not df.empty:
            df = df[df.index <= end_date]
            
        return df
    
    def get_daily_adjusted(
        self,
        symbol: str,
        outputsize: Literal["compact", "full"] = "full"
    ) -> pd.DataFrame:
        """
        Get daily adjusted time series.
        
        Args:
            symbol: Stock ticker
            outputsize: 'compact' (100 days) or 'full' (20+ years)
            
        Returns:
            DataFrame with adjusted OHLCV data
        """
        self._apply_rate_limit()
        
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key
        }
        
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Error Message" in data:
            raise ValueError(f"API error: {data['Error Message']}")
            
        if "Time Series (Daily)" not in data:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        column_map = {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adj_close",
            "6. volume": "volume",
            "7. dividend amount": "dividend",
            "8. split coefficient": "split"
        }
        df = df.rename(columns=column_map)
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        return df
    
    def get_intraday(
        self,
        symbol: str,
        interval: str = "5min",
        outputsize: Literal["compact", "full"] = "compact"
    ) -> pd.DataFrame:
        """
        Get intraday time series.
        
        Args:
            symbol: Stock ticker
            interval: '1min', '5min', '15min', '30min', '60min'
            outputsize: 'compact' or 'full'
            
        Returns:
            DataFrame with OHLCV data
        """
        self._apply_rate_limit()
        
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key
        }
        
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        key = f"Time Series ({interval})"
        if key not in data:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(data[key], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        column_map = {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        }
        df = df.rename(columns=column_map)
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        return df
    
    def get_quote(self, symbol: str) -> pd.DataFrame:
        """Get current quote for a symbol."""
        self._apply_rate_limit()
        
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Global Quote" not in data or not data["Global Quote"]:
            return pd.DataFrame()
            
        quote = data["Global Quote"]
        return pd.DataFrame([{
            "symbol": quote.get("01. symbol"),
            "open": float(quote.get("02. open", 0)),
            "high": float(quote.get("03. high", 0)),
            "low": float(quote.get("04. low", 0)),
            "price": float(quote.get("05. price", 0)),
            "volume": int(quote.get("06. volume", 0)),
            "latest_trading_day": quote.get("07. latest trading day"),
            "previous_close": float(quote.get("08. previous close", 0)),
            "change": float(quote.get("09. change", 0)),
            "change_percent": quote.get("10. change percent", "0%").replace("%", "")
        }])
    
    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """Get company fundamental data."""
        self._apply_rate_limit()
        
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_income_statement(self, symbol: str) -> pd.DataFrame:
        """Get income statement data."""
        self._apply_rate_limit()
        
        params = {
            "function": "INCOME_STATEMENT",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "annualReports" not in data:
            return pd.DataFrame()
            
        return pd.DataFrame(data["annualReports"])
    
    def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """Get balance sheet data."""
        self._apply_rate_limit()
        
        params = {
            "function": "BALANCE_SHEET",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "annualReports" not in data:
            return pd.DataFrame()
            
        return pd.DataFrame(data["annualReports"])
    
    def get_weekly_adjusted(self, symbol: str) -> pd.DataFrame:
        """Get weekly adjusted time series."""
        self._apply_rate_limit()
        
        params = {
            "function": "TIME_SERIES_WEEKLY_ADJUSTED",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Weekly Adjusted Time Series" not in data:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(data["Weekly Adjusted Time Series"], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        column_map = {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adj_close",
            "6. volume": "volume",
            "7. dividend amount": "dividend"
        }
        df = df.rename(columns=column_map)
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        return df
    
    def get_monthly_adjusted(self, symbol: str) -> pd.DataFrame:
        """Get monthly adjusted time series."""
        self._apply_rate_limit()
        
        params = {
            "function": "TIME_SERIES_MONTHLY_ADJUSTED",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Monthly Adjusted Time Series" not in data:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(data["Monthly Adjusted Time Series"], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        column_map = {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adj_close",
            "6. volume": "volume",
            "7. dividend amount": "dividend"
        }
        df = df.rename(columns=column_map)
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        return df


"""
Yahoo Finance data client.

Provides access to:
- Historical price data
- Options chains
- Company info
- Analyst recommendations
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
import yfinance as yf

from .base import DataSource, DataSourceConfig

logger = logging.getLogger(__name__)


class YahooFinanceClient(DataSource):
    """
    Client for Yahoo Finance data via yfinance library.
    
    Free source for historical data and options chains.
    """
    
    def __init__(self, config: Optional[DataSourceConfig] = None):
        config = config or DataSourceConfig(
            name="yahoo_finance",
            rate_limit=2000  # Generous rate limit
        )
        super().__init__(config)
        
    def validate_connection(self) -> bool:
        """Test connection by fetching a simple quote."""
        try:
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            return "symbol" in info or "shortName" in info
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
            
    def fetch_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch historical data for a symbol.
        
        Args:
            symbol: Stock ticker
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval ('1m', '5m', '1h', '1d', '1wk', '1mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        self._apply_rate_limit()
        
        ticker = yf.Ticker(symbol)
        
        if start_date and end_date:
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval
            )
        else:
            # Default to max available history
            df = ticker.history(period="max", interval=interval)
            
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
            
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Add symbol column
        df["symbol"] = symbol
        
        return df
    
    def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Stock ticker
            expiration: Expiration date (YYYY-MM-DD) or None for nearest
            
        Returns:
            Dict with 'calls' and 'puts' DataFrames
        """
        self._apply_rate_limit()
        
        ticker = yf.Ticker(symbol)
        
        # Get available expirations
        expirations = ticker.options
        if not expirations:
            logger.warning(f"No options available for {symbol}")
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}
            
        # Use specified expiration or first available
        exp = expiration if expiration in expirations else expirations[0]
        
        try:
            opt = ticker.option_chain(exp)
            calls = opt.calls.copy()
            puts = opt.puts.copy()
            
            # Add metadata
            calls["symbol"] = symbol
            calls["expiration"] = exp
            calls["option_type"] = "call"
            
            puts["symbol"] = symbol
            puts["expiration"] = exp
            puts["option_type"] = "put"
            
            return {"calls": calls, "puts": puts}
            
        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}
    
    def get_all_options(self, symbol: str) -> pd.DataFrame:
        """
        Get all options across all expirations.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            DataFrame with all options data
        """
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        
        all_options = []
        
        for exp in expirations:
            self._apply_rate_limit()
            try:
                chain = self.get_options_chain(symbol, exp)
                all_options.append(chain["calls"])
                all_options.append(chain["puts"])
            except Exception as e:
                logger.warning(f"Error fetching options for {symbol} {exp}: {e}")
                
        if not all_options:
            return pd.DataFrame()
            
        return pd.concat(all_options, ignore_index=True)
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dictionary with company info
        """
        self._apply_rate_limit()
        ticker = yf.Ticker(symbol)
        return ticker.info
    
    def get_financials(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get financial statements.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dict with income_statement, balance_sheet, cash_flow DataFrames
        """
        self._apply_rate_limit()
        ticker = yf.Ticker(symbol)
        
        return {
            "income_statement": ticker.financials,
            "balance_sheet": ticker.balance_sheet,
            "cash_flow": ticker.cashflow
        }
    
    def get_analyst_recommendations(self, symbol: str) -> pd.DataFrame:
        """Get analyst recommendations."""
        self._apply_rate_limit()
        ticker = yf.Ticker(symbol)
        
        try:
            return ticker.recommendations
        except Exception as e:
            logger.warning(f"No recommendations for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_institutional_holders(self, symbol: str) -> pd.DataFrame:
        """Get institutional holders."""
        self._apply_rate_limit()
        ticker = yf.Ticker(symbol)
        
        try:
            return ticker.institutional_holders
        except Exception as e:
            logger.warning(f"No institutional holders for {symbol}: {e}")
            return pd.DataFrame()
    
    def download_multiple(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Download data for multiple symbols efficiently.
        
        Args:
            symbols: List of tickers
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame with multi-level columns (symbol, field)
        """
        self._apply_rate_limit()
        
        start = start_date.strftime("%Y-%m-%d") if start_date else None
        end = end_date.strftime("%Y-%m-%d") if end_date else None
        
        df = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            threads=True
        )
        
        return df
    
    def get_earnings_dates(self, symbol: str) -> pd.DataFrame:
        """Get earnings dates and estimates."""
        self._apply_rate_limit()
        ticker = yf.Ticker(symbol)
        
        try:
            return ticker.earnings_dates
        except Exception as e:
            logger.warning(f"No earnings dates for {symbol}: {e}")
            return pd.DataFrame()


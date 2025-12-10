"""
MARKET DATA COLLECTOR
=====================
Unified interface for collecting market data from multiple sources.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


class MarketDataCollector:
    """
    Unified market data collection from multiple sources.
    
    Priority order:
    1. Yahoo Finance (free, reliable)
    2. Alpha Vantage (rate limited)
    3. IBKR (requires connection)
    4. Polygon (paid)
    """
    
    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_price(self, ticker: str) -> Optional[float]:
        """Get current price for a ticker"""
        if not YF_AVAILABLE:
            return None
        
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('regularMarketPrice') or stock.info.get('currentPrice')
        except Exception as e:
            logger.error(f"Error getting price for {ticker}: {e}")
            return None
    
    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get current prices for multiple tickers"""
        prices = {}
        for ticker in tickers:
            price = self.get_price(ticker)
            if price:
                prices[ticker] = price
        return prices
    
    def get_historical(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
        period: str = "1y"
    ) -> pd.DataFrame:
        """Get historical OHLCV data"""
        if not YF_AVAILABLE:
            return pd.DataFrame()
        
        try:
            stock = yf.Ticker(ticker)
            
            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date)
            else:
                df = stock.history(period=period)
            
            df.columns = [c.lower() for c in df.columns]
            return df
            
        except Exception as e:
            logger.error(f"Error getting history for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_options_chain(self, ticker: str) -> Dict:
        """Get options chain data"""
        if not YF_AVAILABLE:
            return {}
        
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            
            chains = {}
            for exp in expirations[:3]:  # First 3 expirations
                opt = stock.option_chain(exp)
                chains[exp] = {
                    "calls": opt.calls.to_dict('records'),
                    "puts": opt.puts.to_dict('records')
                }
            
            return {
                "ticker": ticker,
                "expirations": list(expirations),
                "chains": chains
            }
            
        except Exception as e:
            logger.error(f"Error getting options for {ticker}: {e}")
            return {}
    
    def get_fundamentals(self, ticker: str) -> Dict:
        """Get fundamental data"""
        if not YF_AVAILABLE:
            return {}
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "ticker": ticker,
                "name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "52wk_high": info.get("fiftyTwoWeekHigh"),
                "52wk_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
                "short_ratio": info.get("shortRatio")
            }
            
        except Exception as e:
            logger.error(f"Error getting fundamentals for {ticker}: {e}")
            return {}


if __name__ == "__main__":
    collector = MarketDataCollector()
    
    # Get current prices
    prices = collector.get_prices(["NVDA", "CCJ", "SPY"])
    print(f"Current prices: {prices}")
    
    # Get historical data
    df = collector.get_historical("NVDA", period="1mo")
    print(f"\nNVDA last 5 days:\n{df.tail()}")


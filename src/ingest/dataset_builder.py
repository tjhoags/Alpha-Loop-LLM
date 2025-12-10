"""
DATASET BUILDER
===============
Build training datasets from multiple sources for ML models.

Integrates:
- Yahoo Finance (free historical data)
- Alpha Vantage (fundamentals)
- FRED (economic indicators)
- Custom datasets
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


class DatasetBuilder:
    """
    Build comprehensive datasets for algorithm training.
    
    Data Sources:
    1. Yahoo Finance - Price data, fundamentals (FREE)
    2. FRED - Economic indicators (FREE with API key)
    3. Alpha Vantage - Additional fundamentals (FREE tier)
    4. Custom CSVs - Your own data
    
    Output:
    - Training datasets in parquet/CSV format
    - Feature-engineered data ready for ML
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.datasets_dir = self.data_dir / "datasets"
        
        # Create directories
        for d in [self.raw_dir, self.processed_dir, self.datasets_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # FRED client
        self.fred = None
        if FRED_AVAILABLE and os.getenv("FRED_API_KEY"):
            self.fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    
    # =========================================================================
    # YAHOO FINANCE DATA (FREE)
    # =========================================================================
    
    def download_price_data(
        self,
        tickers: List[str],
        start_date: str = "2015-01-01",
        end_date: str = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Download price data for multiple tickers using Yahoo Finance.
        
        Parameters:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (defaults to today)
            interval: "1d", "1wk", "1mo"
        
        Returns:
            Dictionary of DataFrames keyed by ticker
        """
        if not YF_AVAILABLE:
            raise ImportError("yfinance required: pip install yfinance")
        
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        data = {}
        
        logger.info(f"Downloading data for {len(tickers)} tickers...")
        
        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False
                )
                
                if not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    data[ticker] = df
                    
                    # Save to raw
                    filepath = self.raw_dir / f"{ticker}_prices.parquet"
                    df.to_parquet(filepath)
                    logger.info(f"  {ticker}: {len(df)} rows")
                else:
                    logger.warning(f"  {ticker}: No data")
                    
            except Exception as e:
                logger.error(f"  {ticker}: Error - {e}")
        
        return data
    
    def download_fundamentals(
        self,
        tickers: List[str]
    ) -> Dict[str, Dict]:
        """
        Download fundamental data using Yahoo Finance.
        
        Returns company info, financials, and key metrics.
        """
        if not YF_AVAILABLE:
            raise ImportError("yfinance required")
        
        fundamentals = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                fundamentals[ticker] = {
                    "name": info.get("longName"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "price_to_book": info.get("priceToBook"),
                    "dividend_yield": info.get("dividendYield"),
                    "beta": info.get("beta"),
                    "52wk_high": info.get("fiftyTwoWeekHigh"),
                    "52wk_low": info.get("fiftyTwoWeekLow"),
                    "avg_volume": info.get("averageVolume"),
                    "shares_outstanding": info.get("sharesOutstanding"),
                    "short_ratio": info.get("shortRatio"),
                    "analyst_rating": info.get("recommendationKey")
                }
                
                logger.info(f"  {ticker}: Fundamentals loaded")
                
            except Exception as e:
                logger.error(f"  {ticker}: Error - {e}")
                fundamentals[ticker] = {}
        
        return fundamentals
    
    # =========================================================================
    # FRED ECONOMIC DATA (FREE WITH API KEY)
    # =========================================================================
    
    def download_economic_indicators(
        self,
        start_date: str = "2015-01-01"
    ) -> pd.DataFrame:
        """
        Download key economic indicators from FRED.
        
        Indicators:
        - DFF: Federal Funds Rate
        - UNRATE: Unemployment Rate
        - CPIAUCSL: CPI (Inflation)
        - UMCSENT: Consumer Sentiment
        - VIXCLS: VIX
        - T10Y2Y: 10Y-2Y Treasury Spread
        """
        if not self.fred:
            logger.warning("FRED not available. Set FRED_API_KEY environment variable.")
            return pd.DataFrame()
        
        indicators = {
            "fed_funds_rate": "DFF",
            "unemployment_rate": "UNRATE",
            "cpi": "CPIAUCSL",
            "consumer_sentiment": "UMCSENT",
            "vix": "VIXCLS",
            "yield_curve_10y2y": "T10Y2Y",
            "industrial_production": "INDPRO",
            "retail_sales": "RSAFS",
            "housing_starts": "HOUST",
            "initial_claims": "ICSA"
        }
        
        data = {}
        
        for name, series_id in indicators.items():
            try:
                series = self.fred.get_series(series_id, start_date)
                data[name] = series
                logger.info(f"  {name}: {len(series)} observations")
            except Exception as e:
                logger.error(f"  {name}: Error - {e}")
        
        df = pd.DataFrame(data)
        df.index.name = "date"
        
        # Save
        filepath = self.raw_dir / "economic_indicators.parquet"
        df.to_parquet(filepath)
        
        return df
    
    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    
    def add_technical_features(
        self,
        df: pd.DataFrame,
        include_all: bool = True
    ) -> pd.DataFrame:
        """
        Add technical analysis features to price data.
        
        Features:
        - Moving averages (SMA, EMA)
        - RSI
        - MACD
        - Bollinger Bands
        - Volume features
        - Returns
        """
        df = df.copy()
        
        # Ensure lowercase columns
        df.columns = [c.lower() for c in df.columns]
        
        # Returns
        df['return_1d'] = df['close'].pct_change()
        df['return_5d'] = df['close'].pct_change(5)
        df['return_20d'] = df['close'].pct_change(20)
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # Price relative to MAs
        df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
        df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
        df['price_vs_sma200'] = df['close'] / df['sma_200'] - 1
        
        # Volatility
        df['volatility_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
        df['volatility_60d'] = df['return_1d'].rolling(60).std() * np.sqrt(252)
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr_14'] = df['tr'].rolling(14).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Dollar volume
        df['dollar_volume'] = df['close'] * df['volume']
        
        # High-Low range
        df['daily_range'] = (df['high'] - df['low']) / df['close']
        df['range_sma_20'] = df['daily_range'].rolling(20).mean()
        
        return df
    
    def create_labels(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        threshold: float = 0.02
    ) -> pd.DataFrame:
        """
        Create classification labels for ML training.
        
        Labels:
        - 1: Price up more than threshold
        - 0: Price change within threshold
        - -1: Price down more than threshold
        """
        df = df.copy()
        
        # Forward return
        df['forward_return'] = df['close'].shift(-forward_days) / df['close'] - 1
        
        # Classification label
        df['label'] = 0
        df.loc[df['forward_return'] > threshold, 'label'] = 1
        df.loc[df['forward_return'] < -threshold, 'label'] = -1
        
        # Binary label (up/not up)
        df['label_binary'] = (df['forward_return'] > 0).astype(int)
        
        return df
    
    # =========================================================================
    # DATASET CREATION
    # =========================================================================
    
    def build_training_dataset(
        self,
        tickers: List[str],
        start_date: str = "2018-01-01",
        end_date: str = None,
        include_fundamentals: bool = True,
        include_economic: bool = True
    ) -> pd.DataFrame:
        """
        Build comprehensive training dataset.
        
        Combines:
        - Price data with technical features
        - Fundamental data
        - Economic indicators
        
        Returns single DataFrame ready for ML training.
        """
        logger.info("Building training dataset...")
        
        # Download price data
        price_data = self.download_price_data(tickers, start_date, end_date)
        
        # Add technical features
        processed_data = {}
        for ticker, df in price_data.items():
            df = self.add_technical_features(df)
            df = self.create_labels(df)
            df['ticker'] = ticker
            processed_data[ticker] = df
        
        # Combine all tickers
        all_data = pd.concat(processed_data.values())
        all_data = all_data.reset_index()
        all_data = all_data.rename(columns={'index': 'date'})
        
        # Add fundamentals
        if include_fundamentals:
            fundamentals = self.download_fundamentals(tickers)
            fund_df = pd.DataFrame(fundamentals).T
            fund_df.index.name = 'ticker'
            fund_df = fund_df.reset_index()
            all_data = all_data.merge(fund_df, on='ticker', how='left')
        
        # Add economic indicators
        if include_economic and self.fred:
            econ = self.download_economic_indicators(start_date)
            econ = econ.reset_index()
            econ['date'] = pd.to_datetime(econ['date'])
            all_data['date'] = pd.to_datetime(all_data['date'])
            all_data = all_data.merge(econ, on='date', how='left')
            all_data = all_data.ffill()  # Forward fill economic data
        
        # Drop rows with NaN labels
        all_data = all_data.dropna(subset=['label'])
        
        # Save dataset
        filepath = self.datasets_dir / f"training_dataset_{datetime.now().strftime('%Y%m%d')}.parquet"
        all_data.to_parquet(filepath)
        logger.info(f"Saved training dataset: {filepath}")
        logger.info(f"Dataset shape: {all_data.shape}")
        
        return all_data
    
    def get_sample_universe(self) -> List[str]:
        """Get a sample universe of tickers for testing"""
        return [
            # Mega caps
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
            # Energy
            "XOM", "CVX", "OXY",
            # Financials
            "JPM", "BAC", "GS",
            # Uranium thesis
            "CCJ", "UEC", "DNN", "UUUU",
            # Nuclear/Infrastructure
            "BWXT", "VST", "CEG",
            # ETFs
            "SPY", "QQQ", "IWM", "XLE", "XLF"
        ]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_dataset(tickers: List[str] = None) -> pd.DataFrame:
    """Quick function to build a dataset"""
    builder = DatasetBuilder()
    tickers = tickers or builder.get_sample_universe()[:10]
    return builder.build_training_dataset(tickers)


if __name__ == "__main__":
    # Demo
    builder = DatasetBuilder()
    
    # Download sample data
    sample_tickers = ["NVDA", "CCJ", "SPY"]
    
    print("Downloading price data...")
    prices = builder.download_price_data(sample_tickers, start_date="2023-01-01")
    
    print("\nAdding technical features...")
    for ticker, df in prices.items():
        df_features = builder.add_technical_features(df)
        print(f"\n{ticker} features: {list(df_features.columns)[:20]}...")
    
    print("\nBuilding full training dataset...")
    dataset = builder.build_training_dataset(
        sample_tickers,
        start_date="2023-01-01",
        include_fundamentals=True,
        include_economic=False  # Skip if no FRED key
    )
    
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Columns: {list(dataset.columns)}")


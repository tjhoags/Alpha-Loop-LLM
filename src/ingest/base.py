"""
Base classes for data ingestion.
"""

import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DataSourceConfig(BaseModel):
    """Configuration for a data source."""
    name: str
    enabled: bool = True
    rate_limit: int = 100
    timeout: int = 30
    retry_attempts: int = 3


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, config: Optional[DataSourceConfig] = None):
        self.config = config
        self._last_request_time: Optional[datetime] = None
        
    @abstractmethod
    def fetch_data(
        self, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch data for a given symbol."""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate the connection to the data source."""
        pass
    
    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        import time
        if self.config and self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            min_interval = 60.0 / self.config.rate_limit
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_request_time = datetime.now()


class DataIngester:
    """
    Main data ingestion orchestrator.
    
    Manages multiple data sources and provides unified interface
    for fetching and storing market data.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.sources: Dict[str, DataSource] = {}
        self._setup_directories()
        
    def _setup_directories(self) -> None:
        """Create necessary data directories."""
        subdirs = ["raw", "processed", "features", "models", "trades"]
        for subdir in subdirs:
            path = self.data_dir / subdir
            path.mkdir(parents=True, exist_ok=True)
            # Create .gitkeep to preserve empty directories
            gitkeep = path / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
                
    def register_source(self, name: str, source: DataSource) -> None:
        """Register a data source."""
        self.sources[name] = source
        logger.info(f"Registered data source: {name}")
        
    def fetch_historical_data(
        self,
        symbols: List[str],
        source: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        save: bool = True,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            source: Name of data source to use
            start_date: Start date for data
            end_date: End date (defaults to today)
            save: Whether to save data to disk
            **kwargs: Additional arguments for the data source
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if source not in self.sources:
            raise ValueError(f"Unknown data source: {source}")
            
        end_date = end_date or datetime.now()
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching {symbol} from {source}")
                df = self.sources[source].fetch_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs
                )
                results[symbol] = df
                
                if save and not df.empty:
                    self._save_data(df, symbol, source)
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                results[symbol] = pd.DataFrame()
                
        return results
    
    def _save_data(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        source: str,
        subdir: str = "raw"
    ) -> Path:
        """Save data to parquet file."""
        filepath = self.data_dir / subdir / f"{symbol}_{source}.parquet"
        df.to_parquet(filepath, index=True)
        logger.info(f"Saved data to {filepath}")
        return filepath
    
    def load_data(
        self, 
        symbol: str, 
        source: Optional[str] = None,
        subdir: str = "raw"
    ) -> pd.DataFrame:
        """Load data from disk."""
        pattern = f"{symbol}_*.parquet" if source is None else f"{symbol}_{source}.parquet"
        files = list((self.data_dir / subdir).glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No data found for {symbol}")
            
        # Return most recent if multiple files
        filepath = max(files, key=lambda p: p.stat().st_mtime)
        return pd.read_parquet(filepath)


def setup_directories():
    """Initialize data directories for the project."""
    ingester = DataIngester()
    logger.info("Data directories initialized successfully")
    return ingester


if __name__ == "__main__":
    setup_directories()


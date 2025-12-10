"""================================================================================
DATA TYPES - Unified Type Definitions for Data Ingestion
================================================================================

HOW TO USE:
-----------
Windows (PowerShell):
    cd "C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\sii"
    .\\venv\\Scripts\\Activate.ps1
    python -c "from src.data_ingestion.data_types import PriceBar, validate_dataframe; print('Types loaded successfully')"

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
    source venv/bin/activate
    python -c "from src.data_ingestion.data_types import PriceBar, validate_dataframe; print('Types loaded successfully')"

WHAT THIS MODULE PROVIDES:
--------------------------
Consistent type definitions for all data ingestion:
1. PriceBar - OHLCV price data
2. FundamentalData - Company fundamentals
3. OptionData - Options with Greeks
4. MacroIndicator - Economic indicators
5. Validation functions for DataFrames

WHY THIS MATTERS:
-----------------
- Prevents type mismatches between data sources
- Ensures consistent column names across the codebase
- Provides validation before database insertion
- Documents expected data shapes clearly

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, TypedDict, Union

import pandas as pd
from loguru import logger


# =============================================================================
# ENUMS - Data Source & Asset Type Classifications
# =============================================================================

class DataSource(str, Enum):
    """Supported data sources."""
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    COINBASE = "coinbase"
    FRED = "fred"
    SEC_EDGAR = "sec_edgar"
    IBKR = "ibkr"
    MASSIVE = "massive"
    YAHOO = "yahoo"
    MANUAL = "manual"


class AssetType(str, Enum):
    """Asset classifications."""
    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"
    FOREX = "forex"
    OPTION = "option"
    FUTURE = "future"
    INDEX = "index"
    BOND = "bond"
    COMMODITY = "commodity"


class TimeFrame(str, Enum):
    """Data timeframes/intervals."""
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1hour"
    HOUR_4 = "4hour"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class DataQuality(str, Enum):
    """Data quality indicators."""
    HIGH = "high"        # Verified, complete
    MEDIUM = "medium"    # Minor gaps possible
    LOW = "low"          # Significant gaps/issues
    UNKNOWN = "unknown"  # Not assessed


# =============================================================================
# TYPE DEFINITIONS - Core Data Structures
# =============================================================================

@dataclass
class PriceBar:
    """
    Standard OHLCV price bar.
    
    All price data should be normalized to this format regardless of source.
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: DataSource = DataSource.MANUAL
    asset_type: AssetType = AssetType.EQUITY
    timeframe: TimeFrame = TimeFrame.DAILY
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    adjusted_close: Optional[float] = None

    def __post_init__(self):
        """Validate price bar data."""
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be less than low ({self.low})")
        if self.open < 0 or self.close < 0:
            raise ValueError("Prices cannot be negative")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "source": self.source.value,
            "asset_type": self.asset_type.value,
            "timeframe": self.timeframe.value,
            "vwap": self.vwap,
            "trade_count": self.trade_count,
            "adjusted_close": self.adjusted_close,
        }


@dataclass
class OptionData:
    """
    Options data including Greeks.
    
    Comprehensive options chain data for derivatives analysis.
    """
    symbol: str              # Underlying symbol
    option_symbol: str       # Full option symbol (e.g., AAPL230120C00150000)
    timestamp: datetime
    expiration: datetime
    strike: float
    option_type: str         # "call" or "put"
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    implied_volatility: Optional[float] = None
    source: DataSource = DataSource.MANUAL

    def __post_init__(self):
        """Validate option data."""
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type}")
        if self.strike <= 0:
            raise ValueError("Strike price must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "option_symbol": self.option_symbol,
            "timestamp": self.timestamp,
            "expiration": self.expiration,
            "strike": self.strike,
            "option_type": self.option_type,
            "last_price": self.last_price,
            "bid": self.bid,
            "ask": self.ask,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "implied_volatility": self.implied_volatility,
            "source": self.source.value,
        }


@dataclass
class FundamentalData:
    """
    Company fundamental data.
    
    Financial metrics for fundamental analysis and valuation.
    """
    symbol: str
    timestamp: datetime
    # Valuation
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    ev_sales: Optional[float] = None
    # Profitability
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None
    # Growth
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    # Financial Health
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    interest_coverage: Optional[float] = None
    # Quality Scores
    altman_z_score: Optional[float] = None
    piotroski_f_score: Optional[int] = None
    graham_number: Optional[float] = None
    # Dividends
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    source: DataSource = DataSource.MANUAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v.value if isinstance(v, Enum) else v 
                for k, v in self.__dict__.items()}


@dataclass
class MacroIndicator:
    """
    Macroeconomic indicator data.
    
    Economic data from FRED and other sources.
    """
    series_id: str          # FRED series ID (e.g., "GDP", "UNRATE")
    timestamp: datetime
    value: float
    frequency: str          # "daily", "weekly", "monthly", "quarterly", "annual"
    units: str             # Description of units
    seasonally_adjusted: bool = True
    source: DataSource = DataSource.FRED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "series_id": self.series_id,
            "timestamp": self.timestamp,
            "value": self.value,
            "frequency": self.frequency,
            "units": self.units,
            "seasonally_adjusted": self.seasonally_adjusted,
            "source": self.source.value,
        }


# =============================================================================
# TYPED DICTS - DataFrame Schema Definitions
# =============================================================================

class PriceBarSchema(TypedDict):
    """Schema for price bar DataFrames."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str


class OptionSchema(TypedDict):
    """Schema for options DataFrames."""
    symbol: str
    option_symbol: str
    timestamp: datetime
    expiration: datetime
    strike: float
    option_type: str
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int


# =============================================================================
# COLUMN NAME CONSTANTS - Standardized Column Names
# =============================================================================

# Required columns for price data
PRICE_COLUMNS_REQUIRED = frozenset({
    "symbol", "timestamp", "open", "high", "low", "close", "volume"
})

# Optional columns for price data
PRICE_COLUMNS_OPTIONAL = frozenset({
    "source", "asset_type", "timeframe", "vwap", "trade_count", "adjusted_close"
})

# Required columns for options data
OPTION_COLUMNS_REQUIRED = frozenset({
    "symbol", "option_symbol", "timestamp", "expiration", "strike",
    "option_type", "last_price", "bid", "ask", "volume", "open_interest"
})

# Greek columns
GREEK_COLUMNS = frozenset({
    "delta", "gamma", "theta", "vega", "rho", "implied_volatility"
})

# Fundamental columns
FUNDAMENTAL_COLUMNS = frozenset({
    "symbol", "timestamp", "market_cap", "enterprise_value", "pe_ratio",
    "peg_ratio", "pb_ratio", "ps_ratio", "ev_ebitda", "ev_sales",
    "profit_margin", "operating_margin", "roe", "roa", "roic",
    "revenue_growth", "earnings_growth", "current_ratio", "quick_ratio",
    "debt_to_equity", "interest_coverage", "altman_z_score",
    "piotroski_f_score", "graham_number", "dividend_yield", "payout_ratio"
})


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_price_dataframe(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Validate and normalize a price DataFrame.
    
    Args:
        df: DataFrame to validate
        strict: If True, raise errors. If False, log warnings and continue.
        
    Returns:
        Validated DataFrame with consistent column types
        
    Raises:
        ValueError: If required columns are missing (strict mode)
    """
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return df
    
    # Check required columns
    missing = PRICE_COLUMNS_REQUIRED - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return df
    
    # Normalize column types
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Ensure numeric columns
    numeric_cols = ["open", "high", "low", "close"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Ensure volume is integer
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    
    # Validate price constraints
    invalid_prices = (df["high"] < df["low"]) | (df["open"] < 0) | (df["close"] < 0)
    if invalid_prices.any():
        count = invalid_prices.sum()
        msg = f"Found {count} rows with invalid prices"
        if strict:
            raise ValueError(msg)
        logger.warning(f"{msg}, dropping these rows")
        df = df[~invalid_prices]
    
    # Add source column if missing
    if "source" not in df.columns:
        df["source"] = DataSource.MANUAL.value
    
    return df


def validate_option_dataframe(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Validate and normalize an options DataFrame.
    
    Args:
        df: DataFrame to validate
        strict: If True, raise errors. If False, log warnings and continue.
        
    Returns:
        Validated DataFrame
    """
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return df
    
    # Check required columns
    missing = OPTION_COLUMNS_REQUIRED - set(df.columns)
    if missing:
        msg = f"Missing required option columns: {missing}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return df
    
    df = df.copy()
    
    # Ensure timestamps are datetime
    for col in ["timestamp", "expiration"]:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
    
    # Normalize option_type
    df["option_type"] = df["option_type"].str.lower()
    invalid_types = ~df["option_type"].isin(["call", "put"])
    if invalid_types.any():
        msg = f"Found {invalid_types.sum()} rows with invalid option types"
        if strict:
            raise ValueError(msg)
        logger.warning(f"{msg}, dropping these rows")
        df = df[~invalid_types]
    
    return df


def normalize_source_dataframe(
    df: pd.DataFrame,
    source: DataSource,
    column_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Normalize a DataFrame from a specific source to standard format.
    
    Args:
        df: Source-specific DataFrame
        source: The data source
        column_mapping: Optional mapping from source columns to standard columns
        
    Returns:
        Normalized DataFrame with standard column names
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Apply column mapping
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Add source if not present
    if "source" not in df.columns:
        df["source"] = source.value
    
    return df


# =============================================================================
# DATA CONVERSION UTILITIES
# =============================================================================

def price_bars_to_dataframe(bars: List[PriceBar]) -> pd.DataFrame:
    """Convert a list of PriceBar objects to a DataFrame."""
    if not bars:
        return pd.DataFrame()
    return pd.DataFrame([bar.to_dict() for bar in bars])


def dataframe_to_price_bars(df: pd.DataFrame) -> List[PriceBar]:
    """Convert a DataFrame to a list of PriceBar objects."""
    bars = []
    for _, row in df.iterrows():
        try:
            bar = PriceBar(
                symbol=row["symbol"],
                timestamp=row["timestamp"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                source=DataSource(row.get("source", "manual")),
            )
            bars.append(bar)
        except (ValueError, KeyError) as e:
            logger.warning(f"Could not convert row to PriceBar: {e}")
    return bars


# =============================================================================
# SOURCE-SPECIFIC COLUMN MAPPINGS
# =============================================================================

ALPHA_VANTAGE_MAPPING = {
    "1. open": "open",
    "2. high": "high",
    "3. low": "low",
    "4. close": "close",
    "5. volume": "volume",
    "5. adjusted close": "adjusted_close",
    "6. volume": "volume",
}

POLYGON_MAPPING = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vw": "vwap",
    "n": "trade_count",
    "t": "timestamp",
}

COINBASE_MAPPING = {
    "price": "close",
    "size": "volume",
    "time": "timestamp",
}


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "DataSource",
    "AssetType",
    "TimeFrame",
    "DataQuality",
    # Data classes
    "PriceBar",
    "OptionData",
    "FundamentalData",
    "MacroIndicator",
    # Schemas
    "PriceBarSchema",
    "OptionSchema",
    # Constants
    "PRICE_COLUMNS_REQUIRED",
    "PRICE_COLUMNS_OPTIONAL",
    "OPTION_COLUMNS_REQUIRED",
    "GREEK_COLUMNS",
    "FUNDAMENTAL_COLUMNS",
    # Validation
    "validate_price_dataframe",
    "validate_option_dataframe",
    "normalize_source_dataframe",
    # Conversion
    "price_bars_to_dataframe",
    "dataframe_to_price_bars",
    # Mappings
    "ALPHA_VANTAGE_MAPPING",
    "POLYGON_MAPPING",
    "COINBASE_MAPPING",
]


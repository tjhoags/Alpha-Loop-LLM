"""
Trade History Ingestion Module

This module handles the ingestion of historical trades and portfolio data
from Alpha Loop Capital to analyze past performance and improve trading strategies.

Supports multiple input formats:
- CSV exports from brokers (IBKR, TD Ameritrade, etc.)
- Excel spreadsheets
- JSON from trading journals
- Direct broker API connections
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Literal
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class TradeType(str, Enum):
    """Types of trades."""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_OPEN = "buy_to_open"
    SELL_TO_OPEN = "sell_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_CLOSE = "sell_to_close"


class AssetClass(str, Enum):
    """Asset classes."""
    EQUITY = "equity"
    OPTION = "option"
    FUTURE = "future"
    CRYPTO = "crypto"
    FOREX = "forex"


class Trade(BaseModel):
    """Schema for a single trade."""
    trade_id: Optional[str] = None
    timestamp: datetime
    symbol: str
    asset_class: AssetClass = AssetClass.EQUITY
    trade_type: TradeType
    quantity: float
    price: float
    commission: float = 0.0
    fees: float = 0.0
    
    # Options-specific fields
    option_type: Optional[Literal["call", "put"]] = None
    strike: Optional[float] = None
    expiration: Optional[datetime] = None
    
    # Computed fields
    total_value: Optional[float] = None
    net_value: Optional[float] = None
    
    # Metadata
    strategy: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    @validator("total_value", always=True)
    def compute_total_value(cls, v, values):
        """Compute total trade value."""
        if v is not None:
            return v
        return values.get("quantity", 0) * values.get("price", 0)
    
    @validator("net_value", always=True)
    def compute_net_value(cls, v, values):
        """Compute net value after costs."""
        if v is not None:
            return v
        total = values.get("quantity", 0) * values.get("price", 0)
        commission = values.get("commission", 0)
        fees = values.get("fees", 0)
        
        if values.get("trade_type") in [TradeType.BUY, TradeType.BUY_TO_OPEN, TradeType.BUY_TO_CLOSE]:
            return -(total + commission + fees)
        else:
            return total - commission - fees


class PortfolioSnapshot(BaseModel):
    """Schema for portfolio snapshot."""
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    positions: Dict[str, Dict[str, Any]]
    daily_pnl: Optional[float] = None
    total_pnl: Optional[float] = None


class TradeHistoryIngester:
    """
    Ingests and processes historical trade data from Alpha Loop Capital.
    
    This class provides functionality to:
    1. Load trades from various file formats
    2. Parse and standardize trade data
    3. Calculate performance metrics
    4. Generate analysis for improving trading strategies
    
    Example usage:
        >>> ingester = TradeHistoryIngester()
        >>> trades_df = ingester.load_from_csv("my_trades.csv", broker="ibkr")
        >>> metrics = ingester.calculate_metrics(trades_df)
        >>> ingester.save_trades(trades_df, "data/trades/alc_history.parquet")
    """
    
    # Column mappings for different broker formats
    BROKER_MAPPINGS = {
        "ibkr": {
            "columns": {
                "Date/Time": "timestamp",
                "Symbol": "symbol",
                "Buy/Sell": "trade_type",
                "Quantity": "quantity",
                "T. Price": "price",
                "Comm/Fee": "commission",
                "Asset Category": "asset_class"
            },
            "date_format": "%Y-%m-%d, %H:%M:%S",
            "trade_type_map": {
                "BUY": TradeType.BUY,
                "SELL": TradeType.SELL,
                "BOT": TradeType.BUY,
                "SLD": TradeType.SELL
            }
        },
        "td_ameritrade": {
            "columns": {
                "DATE": "timestamp",
                "SYMBOL": "symbol",
                "SIDE": "trade_type",
                "QTY": "quantity",
                "PRICE": "price",
                "COMMISSION": "commission"
            },
            "date_format": "%m/%d/%Y %H:%M:%S",
            "trade_type_map": {
                "BUY": TradeType.BUY,
                "SELL": TradeType.SELL
            }
        },
        "generic": {
            "columns": {
                "date": "timestamp",
                "symbol": "symbol",
                "side": "trade_type",
                "qty": "quantity",
                "price": "price",
                "commission": "commission"
            },
            "date_format": "%Y-%m-%d %H:%M:%S",
            "trade_type_map": {
                "buy": TradeType.BUY,
                "sell": TradeType.SELL,
                "BUY": TradeType.BUY,
                "SELL": TradeType.SELL
            }
        }
    }
    
    def __init__(self, data_dir: str = "data/trades"):
        """
        Initialize the trade history ingester.
        
        Args:
            data_dir: Directory for storing processed trade data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trades: List[Trade] = []
        self.trades_df: Optional[pd.DataFrame] = None
        
    def load_from_csv(
        self,
        filepath: Union[str, Path],
        broker: str = "generic",
        encoding: str = "utf-8",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load trades from a CSV file.
        
        Args:
            filepath: Path to CSV file
            broker: Broker format ('ibkr', 'td_ameritrade', 'generic')
            encoding: File encoding
            **kwargs: Additional arguments passed to pd.read_csv
            
        Returns:
            DataFrame with standardized trade data
        """
        logger.info(f"Loading trades from {filepath} (broker: {broker})")
        
        if broker not in self.BROKER_MAPPINGS:
            raise ValueError(f"Unknown broker: {broker}. Supported: {list(self.BROKER_MAPPINGS.keys())}")
            
        mapping = self.BROKER_MAPPINGS[broker]
        
        # Read CSV
        df = pd.read_csv(filepath, encoding=encoding, **kwargs)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Apply column mapping
        df = df.rename(columns=mapping["columns"])
        
        # Parse dates
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format=mapping.get("date_format"))
            
        # Map trade types
        if "trade_type" in df.columns:
            df["trade_type"] = df["trade_type"].map(
                lambda x: mapping["trade_type_map"].get(str(x).upper(), TradeType.BUY)
            )
            
        # Ensure numeric columns
        numeric_cols = ["quantity", "price", "commission", "fees"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                
        # Calculate derived fields
        df["total_value"] = df["quantity"] * df["price"]
        df["net_value"] = df.apply(self._calculate_net_value, axis=1)
        
        self.trades_df = df
        return df
    
    def load_from_excel(
        self,
        filepath: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        broker: str = "generic",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load trades from an Excel file.
        
        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name or index
            broker: Broker format for column mapping
            **kwargs: Additional arguments passed to pd.read_excel
            
        Returns:
            DataFrame with standardized trade data
        """
        logger.info(f"Loading trades from Excel: {filepath}")
        
        df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
        
        # Save to temp CSV and use CSV loader for standardization
        temp_csv = self.data_dir / "temp_import.csv"
        df.to_csv(temp_csv, index=False)
        
        result = self.load_from_csv(temp_csv, broker=broker)
        temp_csv.unlink()  # Clean up temp file
        
        return result
    
    def load_from_json(
        self,
        filepath: Union[str, Path],
        broker: str = "generic"
    ) -> pd.DataFrame:
        """
        Load trades from a JSON file.
        
        Args:
            filepath: Path to JSON file
            broker: Broker format for column mapping
            
        Returns:
            DataFrame with standardized trade data
        """
        logger.info(f"Loading trades from JSON: {filepath}")
        
        df = pd.read_json(filepath)
        
        mapping = self.BROKER_MAPPINGS.get(broker, self.BROKER_MAPPINGS["generic"])
        df = df.rename(columns=mapping["columns"])
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        self.trades_df = df
        return df
    
    def load_from_dropbox(
        self,
        dropbox_path: str,
        broker: str = "generic",
        file_type: str = "csv"
    ) -> pd.DataFrame:
        """
        Load trades from Dropbox.
        
        Args:
            dropbox_path: Path in Dropbox (e.g., "/ALC-Algo/trades/history.csv")
            broker: Broker format
            file_type: File type ('csv', 'excel', 'json')
            
        Returns:
            DataFrame with standardized trade data
        """
        try:
            import dropbox
        except ImportError:
            raise ImportError("dropbox package required. Install with: pip install dropbox")
            
        token = os.getenv("DROPBOX_ACCESS_TOKEN")
        if not token:
            raise ValueError("DROPBOX_ACCESS_TOKEN environment variable required")
            
        logger.info(f"Downloading trades from Dropbox: {dropbox_path}")
        
        dbx = dropbox.Dropbox(token)
        
        # Download file
        local_path = self.data_dir / f"dropbox_import.{file_type}"
        with open(local_path, "wb") as f:
            metadata, response = dbx.files_download(dropbox_path)
            f.write(response.content)
            
        # Load based on file type
        if file_type == "csv":
            result = self.load_from_csv(local_path, broker=broker)
        elif file_type in ["xlsx", "excel"]:
            result = self.load_from_excel(local_path, broker=broker)
        elif file_type == "json":
            result = self.load_from_json(local_path, broker=broker)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        # Clean up
        local_path.unlink()
        
        return result
    
    def add_trade(self, trade: Trade) -> None:
        """
        Add a single trade to the history.
        
        Args:
            trade: Trade object to add
        """
        self.trades.append(trade)
        
        # Update DataFrame
        trade_dict = trade.dict()
        if self.trades_df is None:
            self.trades_df = pd.DataFrame([trade_dict])
        else:
            self.trades_df = pd.concat([
                self.trades_df,
                pd.DataFrame([trade_dict])
            ], ignore_index=True)
    
    def add_trades_from_dict(self, trades: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Add multiple trades from a list of dictionaries.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Updated trades DataFrame
        """
        for trade_dict in trades:
            trade = Trade(**trade_dict)
            self.add_trade(trade)
            
        return self.trades_df
    
    def _calculate_net_value(self, row: pd.Series) -> float:
        """Calculate net value for a trade row."""
        total = row.get("total_value", 0)
        commission = row.get("commission", 0)
        fees = row.get("fees", 0)
        trade_type = row.get("trade_type")
        
        if trade_type in [TradeType.BUY, TradeType.BUY_TO_OPEN, TradeType.BUY_TO_CLOSE, "buy"]:
            return -(total + commission + fees)
        else:
            return total - commission - fees
    
    def calculate_metrics(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        group_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive trading performance metrics.
        
        Args:
            trades_df: DataFrame of trades (uses self.trades_df if None)
            group_by: Optional grouping column ('symbol', 'strategy', 'month')
            
        Returns:
            Dictionary of performance metrics
        """
        df = trades_df if trades_df is not None else self.trades_df
        
        if df is None or df.empty:
            logger.warning("No trades to analyze")
            return {}
            
        logger.info(f"Calculating metrics for {len(df)} trades")
        
        # Basic metrics
        total_trades = len(df)
        total_volume = df["total_value"].sum()
        total_commissions = df["commission"].sum() if "commission" in df.columns else 0
        
        # Separate buys and sells
        buys = df[df["trade_type"].isin([TradeType.BUY, TradeType.BUY_TO_OPEN, "buy"])]
        sells = df[df["trade_type"].isin([TradeType.SELL, TradeType.SELL_TO_CLOSE, "sell"])]
        
        # Calculate P&L (simplified - assumes FIFO matching)
        realized_pnl = sells["net_value"].sum() + buys["net_value"].sum()
        
        # Win rate calculation (by matching trades)
        pnl_by_symbol = df.groupby("symbol")["net_value"].sum()
        winning_symbols = (pnl_by_symbol > 0).sum()
        total_symbols = len(pnl_by_symbol)
        win_rate = winning_symbols / total_symbols if total_symbols > 0 else 0
        
        # Average trade size
        avg_trade_size = df["total_value"].mean()
        
        # Trading frequency
        if "timestamp" in df.columns:
            df_sorted = df.sort_values("timestamp")
            date_range = (df_sorted["timestamp"].max() - df_sorted["timestamp"].min()).days
            trades_per_day = total_trades / max(date_range, 1)
        else:
            date_range = None
            trades_per_day = None
            
        # Largest win/loss
        largest_win = pnl_by_symbol.max() if not pnl_by_symbol.empty else 0
        largest_loss = pnl_by_symbol.min() if not pnl_by_symbol.empty else 0
        
        # Profit factor
        gross_profit = pnl_by_symbol[pnl_by_symbol > 0].sum()
        gross_loss = abs(pnl_by_symbol[pnl_by_symbol < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        metrics = {
            "summary": {
                "total_trades": total_trades,
                "total_volume": round(total_volume, 2),
                "total_commissions": round(total_commissions, 2),
                "realized_pnl": round(realized_pnl, 2),
                "unique_symbols": total_symbols,
            },
            "performance": {
                "win_rate": round(win_rate * 100, 2),
                "profit_factor": round(profit_factor, 2),
                "avg_trade_size": round(avg_trade_size, 2),
                "largest_win": round(largest_win, 2),
                "largest_loss": round(largest_loss, 2),
            },
            "activity": {
                "date_range_days": date_range,
                "trades_per_day": round(trades_per_day, 2) if trades_per_day else None,
            },
            "by_symbol": pnl_by_symbol.to_dict()
        }
        
        # Add grouped metrics if requested
        if group_by and group_by in df.columns:
            grouped = df.groupby(group_by).agg({
                "total_value": "sum",
                "net_value": "sum",
                "commission": "sum" if "commission" in df.columns else "count"
            })
            metrics[f"by_{group_by}"] = grouped.to_dict()
            
        return metrics
    
    def analyze_patterns(
        self,
        trades_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Analyze trading patterns to identify areas for improvement.
        
        Args:
            trades_df: DataFrame of trades
            
        Returns:
            Dictionary with pattern analysis and recommendations
        """
        df = trades_df if trades_df is not None else self.trades_df
        
        if df is None or df.empty:
            return {}
            
        analysis = {
            "patterns": {},
            "recommendations": []
        }
        
        # Time-based analysis
        if "timestamp" in df.columns:
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
            df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
            
            # P&L by hour
            pnl_by_hour = df.groupby("hour")["net_value"].sum()
            best_hour = pnl_by_hour.idxmax() if not pnl_by_hour.empty else None
            worst_hour = pnl_by_hour.idxmin() if not pnl_by_hour.empty else None
            
            analysis["patterns"]["best_trading_hour"] = best_hour
            analysis["patterns"]["worst_trading_hour"] = worst_hour
            
            if worst_hour is not None and pnl_by_hour.get(worst_hour, 0) < 0:
                analysis["recommendations"].append(
                    f"Consider avoiding trades around {worst_hour}:00 - historically unprofitable"
                )
                
            # P&L by day of week
            pnl_by_day = df.groupby("day_of_week")["net_value"].sum()
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            best_day = days[pnl_by_day.idxmax()] if not pnl_by_day.empty else None
            
            analysis["patterns"]["best_trading_day"] = best_day
            
        # Position sizing analysis
        if "total_value" in df.columns:
            trade_sizes = df["total_value"]
            large_trades = trade_sizes[trade_sizes > trade_sizes.quantile(0.9)]
            small_trades = trade_sizes[trade_sizes < trade_sizes.quantile(0.1)]
            
            # Check if large trades perform worse
            if "net_value" in df.columns:
                large_trade_pnl = df[df["total_value"] > trade_sizes.quantile(0.9)]["net_value"].mean()
                small_trade_pnl = df[df["total_value"] < trade_sizes.quantile(0.1)]["net_value"].mean()
                
                if large_trade_pnl < small_trade_pnl:
                    analysis["recommendations"].append(
                        "Large trades underperforming - consider reducing position sizes"
                    )
                    
        # Symbol concentration
        if "symbol" in df.columns:
            symbol_counts = df["symbol"].value_counts()
            top_symbol_pct = symbol_counts.iloc[0] / len(df) * 100 if len(symbol_counts) > 0 else 0
            
            if top_symbol_pct > 30:
                analysis["recommendations"].append(
                    f"High concentration ({top_symbol_pct:.1f}%) in {symbol_counts.index[0]} - consider diversifying"
                )
                
        # Holding period analysis (if we can match buys/sells)
        analysis["patterns"]["total_unique_symbols"] = df["symbol"].nunique() if "symbol" in df.columns else None
        analysis["patterns"]["most_traded_symbol"] = df["symbol"].mode().iloc[0] if "symbol" in df.columns else None
        
        return analysis
    
    def generate_report(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate a comprehensive trade analysis report.
        
        Args:
            trades_df: DataFrame of trades
            output_path: Optional path to save report
            
        Returns:
            Report as markdown string
        """
        df = trades_df if trades_df is not None else self.trades_df
        
        if df is None or df.empty:
            return "No trades to analyze."
            
        metrics = self.calculate_metrics(df)
        patterns = self.analyze_patterns(df)
        
        report = f"""
# Alpha Loop Capital - Trade Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total Trades | {metrics['summary']['total_trades']} |
| Total Volume | ${metrics['summary']['total_volume']:,.2f} |
| Realized P&L | ${metrics['summary']['realized_pnl']:,.2f} |
| Total Commissions | ${metrics['summary']['total_commissions']:,.2f} |
| Unique Symbols | {metrics['summary']['unique_symbols']} |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Win Rate | {metrics['performance']['win_rate']}% |
| Profit Factor | {metrics['performance']['profit_factor']} |
| Avg Trade Size | ${metrics['performance']['avg_trade_size']:,.2f} |
| Largest Win | ${metrics['performance']['largest_win']:,.2f} |
| Largest Loss | ${metrics['performance']['largest_loss']:,.2f} |

## Trading Patterns

- Best Trading Hour: {patterns['patterns'].get('best_trading_hour', 'N/A')}
- Best Trading Day: {patterns['patterns'].get('best_trading_day', 'N/A')}
- Most Traded Symbol: {patterns['patterns'].get('most_traded_symbol', 'N/A')}

## Recommendations

"""
        for i, rec in enumerate(patterns.get('recommendations', []), 1):
            report += f"{i}. {rec}\n"
            
        if not patterns.get('recommendations'):
            report += "No specific recommendations at this time.\n"
            
        report += """
## Top Performing Symbols

"""
        if 'by_symbol' in metrics:
            sorted_symbols = sorted(metrics['by_symbol'].items(), key=lambda x: x[1], reverse=True)[:10]
            for symbol, pnl in sorted_symbols:
                report += f"- {symbol}: ${pnl:,.2f}\n"
                
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(report)
            logger.info(f"Report saved to {output_path}")
            
        return report
    
    def save_trades(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        filepath: Optional[Union[str, Path]] = None,
        format: str = "parquet"
    ) -> Path:
        """
        Save trades to file.
        
        Args:
            trades_df: DataFrame to save
            filepath: Output path
            format: File format ('parquet', 'csv', 'json')
            
        Returns:
            Path to saved file
        """
        df = trades_df if trades_df is not None else self.trades_df
        
        if df is None or df.empty:
            raise ValueError("No trades to save")
            
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.data_dir / f"trades_{timestamp}.{format}"
        else:
            filepath = Path(filepath)
            
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "parquet":
            df.to_parquet(filepath, index=False)
        elif format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient="records", date_format="iso")
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Saved {len(df)} trades to {filepath}")
        return filepath
    
    def load_saved_trades(
        self,
        filepath: Union[str, Path]
    ) -> pd.DataFrame:
        """
        Load previously saved trades.
        
        Args:
            filepath: Path to trade file
            
        Returns:
            DataFrame of trades
        """
        filepath = Path(filepath)
        
        if filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
        elif filepath.suffix == ".csv":
            df = pd.read_csv(filepath, parse_dates=["timestamp"])
        elif filepath.suffix == ".json":
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
        self.trades_df = df
        logger.info(f"Loaded {len(df)} trades from {filepath}")
        return df


# Convenience function for quick import
def ingest_alc_trades(
    filepath: str,
    broker: str = "ibkr",
    analyze: bool = True
) -> Dict[str, Any]:
    """
    Quick function to ingest Alpha Loop Capital trade history.
    
    Args:
        filepath: Path to trade history file
        broker: Broker format
        analyze: Whether to run analysis
        
    Returns:
        Dictionary with trades DataFrame and optional metrics
    """
    ingester = TradeHistoryIngester()
    
    # Determine file type from extension
    path = Path(filepath)
    if path.suffix == ".csv":
        trades = ingester.load_from_csv(filepath, broker=broker)
    elif path.suffix in [".xlsx", ".xls"]:
        trades = ingester.load_from_excel(filepath, broker=broker)
    elif path.suffix == ".json":
        trades = ingester.load_from_json(filepath, broker=broker)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
        
    result = {"trades": trades}
    
    if analyze:
        result["metrics"] = ingester.calculate_metrics()
        result["patterns"] = ingester.analyze_patterns()
        result["report"] = ingester.generate_report()
        
    return result


if __name__ == "__main__":
    # Example usage
    print("Trade History Ingester - Alpha Loop Capital")
    print("=" * 50)
    
    # Create sample trades for demonstration
    sample_trades = [
        {
            "timestamp": datetime(2024, 1, 15, 10, 30),
            "symbol": "AAPL",
            "trade_type": TradeType.BUY,
            "quantity": 100,
            "price": 185.50,
            "commission": 1.00
        },
        {
            "timestamp": datetime(2024, 1, 20, 14, 15),
            "symbol": "AAPL",
            "trade_type": TradeType.SELL,
            "quantity": 100,
            "price": 192.00,
            "commission": 1.00
        },
        {
            "timestamp": datetime(2024, 1, 18, 9, 45),
            "symbol": "MSFT",
            "trade_type": TradeType.BUY,
            "quantity": 50,
            "price": 390.00,
            "commission": 1.00
        },
    ]
    
    ingester = TradeHistoryIngester()
    ingester.add_trades_from_dict(sample_trades)
    
    print("\nSample Trades Loaded:")
    print(ingester.trades_df)
    
    print("\nMetrics:")
    metrics = ingester.calculate_metrics()
    for category, values in metrics.items():
        if isinstance(values, dict):
            print(f"\n{category.upper()}:")
            for key, value in values.items():
                print(f"  {key}: {value}")


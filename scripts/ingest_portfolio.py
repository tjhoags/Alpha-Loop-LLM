"""
Portfolio Data Ingestion Module
Author: Tom Hogan | Alpha Loop Capital, LLC

This module provides robust ingestion of historical portfolio data from various formats
including IBKR Flex Queries, generic CSV files, and Excel spreadsheets.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioIngestor:
    """
    Handles ingestion and normalization of portfolio data from multiple sources.
    """
    
    # Standard column mappings for various brokers
    COLUMN_MAPPINGS = {
        'ibkr': {
            'Symbol': 'ticker',
            'TradeDate': 'date',
            'BuySell': 'action',
            'Quantity': 'quantity',
            'TradePrice': 'price',
            'Commission': 'commission',
            'Proceeds': 'proceeds',
            'IBCommission': 'commission',
        },
        'generic': {
            'Date': 'date',
            'Symbol': 'ticker',
            'Ticker': 'ticker',
            'Action': 'action',
            'Side': 'action',
            'Qty': 'quantity',
            'Quantity': 'quantity',
            'Price': 'price',
            'Comm': 'commission',
            'Commission': 'commission',
        }
    }
    
    def __init__(self, output_dir: str = 'data/portfolio'):
        """
        Initialize the portfolio ingestor.
        
        Args:
            output_dir: Directory to save processed portfolio data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def detect_file_format(self, file_path: str) -> str:
        """
        Detect the file format based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File format ('csv', 'xlsx', 'xls', etc.)
        """
        extension = Path(file_path).suffix.lower()
        if extension in ['.xlsx', '.xls', '.xlsm']:
            return 'excel'
        elif extension == '.csv':
            return 'csv'
        elif extension == '.json':
            return 'json'
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def detect_broker_format(self, df: pd.DataFrame) -> str:
        """
        Detect the broker format based on column names.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Broker format ('ibkr', 'generic', etc.)
        """
        columns = set(df.columns)
        
        # Check for IBKR specific columns
        ibkr_columns = {'Symbol', 'TradeDate', 'IBCommission', 'AssetClass'}
        if len(columns.intersection(ibkr_columns)) >= 2:
            return 'ibkr'
        
        return 'generic'
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from file based on format.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DataFrame with loaded data
        """
        file_format = self.detect_file_format(file_path)
        
        logger.info(f"Loading {file_format} file: {file_path}")
        
        if file_format == 'csv':
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not decode CSV file with any common encoding")
            
        elif file_format == 'excel':
            df = pd.read_excel(file_path)
            return df
            
        elif file_format == 'json':
            df = pd.read_json(file_path)
            return df
        
        raise ValueError(f"Unsupported file format: {file_format}")
    
    def normalize_columns(self, df: pd.DataFrame, broker_format: str = 'generic') -> pd.DataFrame:
        """
        Normalize column names to standard format.
        
        Args:
            df: DataFrame to normalize
            broker_format: Broker format ('ibkr', 'generic')
            
        Returns:
            DataFrame with normalized columns
        """
        mapping = self.COLUMN_MAPPINGS.get(broker_format, self.COLUMN_MAPPINGS['generic'])
        
        # Rename columns based on mapping
        df = df.rename(columns=mapping)
        
        # Ensure required columns exist
        required_columns = ['date', 'ticker', 'action', 'quantity', 'price']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            # Try to infer from available columns
            for col in missing:
                if col == 'commission' and col not in df.columns:
                    df['commission'] = 0.0
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the data.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean ticker symbols (remove spaces, convert to uppercase)
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
        
        # Normalize action (BUY/SELL)
        if 'action' in df.columns:
            df['action'] = df['action'].astype(str).str.upper()
            df['action'] = df['action'].replace({
                'B': 'BUY',
                'BOT': 'BUY',
                'BOUGHT': 'BUY',
                'S': 'SELL',
                'SLD': 'SELL',
                'SOLD': 'SELL',
            })
        
        # Convert numeric columns
        numeric_columns = ['quantity', 'price', 'commission']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing critical data
        df = df.dropna(subset=['date', 'ticker', 'action', 'quantity', 'price'])
        
        # Add commission if not present
        if 'commission' not in df.columns:
            df['commission'] = 0.0
        
        # Calculate proceeds (price * quantity - commission)
        if 'proceeds' not in df.columns:
            df['proceeds'] = df['quantity'] * df['price']
            df.loc[df['action'] == 'SELL', 'proceeds'] -= df['commission']
            df.loc[df['action'] == 'BUY', 'proceeds'] = -df['proceeds'] - df['commission']
        
        return df
    
    def calculate_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate realized and unrealized P&L.
        
        Args:
            df: DataFrame with trades
            
        Returns:
            DataFrame with P&L calculations
        """
        df = df.copy()
        df = df.sort_values('date')
        
        # Track positions by ticker
        positions: Dict[str, List[Tuple[float, float]]] = {}  # ticker -> [(qty, price), ...]
        
        realized_pnl = []
        unrealized_pnl = []
        position_size = []
        avg_cost = []
        
        for idx, row in df.iterrows():
            ticker = row['ticker']
            action = row['action']
            qty = row['quantity']
            price = row['price']
            
            if ticker not in positions:
                positions[ticker] = []
            
            if action == 'BUY':
                # Add to position
                positions[ticker].append((qty, price))
                realized = 0.0
            
            elif action == 'SELL':
                # Close position (FIFO)
                remaining_qty = qty
                total_cost = 0.0
                
                while remaining_qty > 0 and positions[ticker]:
                    pos_qty, pos_price = positions[ticker][0]
                    
                    if pos_qty <= remaining_qty:
                        # Close entire position
                        total_cost += pos_qty * pos_price
                        remaining_qty -= pos_qty
                        positions[ticker].pop(0)
                    else:
                        # Partial close
                        total_cost += remaining_qty * pos_price
                        positions[ticker][0] = (pos_qty - remaining_qty, pos_price)
                        remaining_qty = 0
                
                # Calculate realized P&L
                realized = (qty * price) - total_cost - row['commission']
            else:
                realized = 0.0
            
            # Calculate current position
            current_qty = sum(q for q, p in positions[ticker])
            current_avg_cost = (
                sum(q * p for q, p in positions[ticker]) / current_qty
                if current_qty > 0 else 0
            )
            
            # Unrealized P&L (current position valued at current price)
            unrealized = current_qty * (price - current_avg_cost) if current_qty > 0 else 0
            
            realized_pnl.append(realized)
            unrealized_pnl.append(unrealized)
            position_size.append(current_qty)
            avg_cost.append(current_avg_cost)
        
        df['realized_pnl'] = realized_pnl
        df['unrealized_pnl'] = unrealized_pnl
        df['position_size'] = position_size
        df['avg_cost'] = avg_cost
        df['cumulative_pnl'] = df['realized_pnl'].cumsum()
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = 'history.parquet') -> str:
        """
        Save processed data to parquet format.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        # Also save as CSV for easy viewing
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV copy to {csv_path}")
        
        return str(output_path)
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the portfolio.
        
        Args:
            df: DataFrame with portfolio data
            
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_trades': len(df),
            'total_buys': len(df[df['action'] == 'BUY']),
            'total_sells': len(df[df['action'] == 'SELL']),
            'unique_tickers': df['ticker'].nunique(),
            'total_realized_pnl': df['realized_pnl'].sum(),
            'total_commission': df['commission'].sum(),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d'),
            },
            'top_tickers_by_volume': df.groupby('ticker')['quantity'].sum().nlargest(10).to_dict(),
            'win_rate': (
                len(df[df['realized_pnl'] > 0]) / len(df[df['realized_pnl'] != 0])
                if len(df[df['realized_pnl'] != 0]) > 0 else 0
            ) * 100,
        }
        
        return stats


def ingest_portfolio_history(file_path: str, output_dir: str = 'data/portfolio') -> pd.DataFrame:
    """
    Main function to ingest portfolio history from various formats.
    
    This function:
    1. Detects file format (CSV/Excel)
    2. Normalizes columns (Date, Ticker, Action, Quantity, Price, Commission)
    3. Calculates realized/unrealized P&L
    4. Stores cleaned data in data/portfolio/history.parquet
    
    Args:
        file_path: Path to the portfolio file
        output_dir: Directory to save processed data
        
    Returns:
        DataFrame with processed portfolio data
        
    Example:
        >>> df = ingest_portfolio_history("my_trades.csv")
        >>> print(f"Loaded {len(df)} trades")
        >>> print(f"Total P&L: ${df['cumulative_pnl'].iloc[-1]:,.2f}")
    """
    logger.info(f"Starting portfolio ingestion from: {file_path}")
    
    # Initialize ingestor
    ingestor = PortfolioIngestor(output_dir)
    
    # Load data
    df = ingestor.load_file(file_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Detect broker format
    broker_format = ingestor.detect_broker_format(df)
    logger.info(f"Detected broker format: {broker_format}")
    
    # Normalize columns
    df = ingestor.normalize_columns(df, broker_format)
    
    # Clean data
    df = ingestor.clean_data(df)
    logger.info(f"After cleaning: {len(df)} rows")
    
    # Calculate P&L
    df = ingestor.calculate_pnl(df)
    
    # Save processed data
    output_path = ingestor.save_data(df)
    
    # Generate summary statistics
    stats = ingestor.generate_summary_stats(df)
    
    # Log summary
    logger.info("=" * 60)
    logger.info("PORTFOLIO SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Trades: {stats['total_trades']}")
    logger.info(f"Buys: {stats['total_buys']} | Sells: {stats['total_sells']}")
    logger.info(f"Unique Tickers: {stats['unique_tickers']}")
    logger.info(f"Total Realized P&L: ${stats['total_realized_pnl']:,.2f}")
    logger.info(f"Total Commissions: ${stats['total_commission']:,.2f}")
    logger.info(f"Win Rate: {stats['win_rate']:.1f}%")
    logger.info(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    logger.info("=" * 60)
    
    return df


if __name__ == "__main__":
    """
    Example usage:
    python scripts/ingest_portfolio.py
    """
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        df = ingest_portfolio_history(file_path)
        print(f"\n✓ Successfully processed {len(df)} trades")
        print(f"✓ Output saved to data/portfolio/history.parquet")
    else:
        print("Usage: python scripts/ingest_portfolio.py <path_to_portfolio_file>")
        print("Example: python scripts/ingest_portfolio.py trades.csv")


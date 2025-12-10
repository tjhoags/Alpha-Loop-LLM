"""
ALPHA LOOP CAPITAL - DATA PIPELINE
Pulls market data from multiple sources and populates database.

Developer: Tom Hogan / Claude AI Assistant
Created: December 2025

USAGE:
    python data_pipeline.py --full          # Pull 5 years of data for all tickers
    python data_pipeline.py --daily         # Pull last 30 days
    python data_pipeline.py --ticker AAPL   # Pull single ticker
    python data_pipeline.py --fundamentals  # Pull fundamentals only
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

# Data libraries
import pandas as pd
import numpy as np

# Database
import psycopg2
from psycopg2.extras import execute_values

# Data sources
import yfinance as yf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Database connection - update these with your credentials
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alc_training',
    'user': 'postgres',
    'password': os.environ.get('POSTGRES_PASSWORD', 'your_password_here')
}

# Core universe - 80+ tickers for comprehensive coverage
CORE_UNIVERSE = [
    # Mega Caps / Market Leaders
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
    
    # Financials (Tom's expertise)
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'SCHW', 'BLK', 'KKR', 'APO',
    
    # Healthcare / Pharma
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT',
    
    # Energy (Nuclear thesis)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'CCJ', 'UEC', 'LEU', 'NNE',
    
    # Industrials
    'CAT', 'DE', 'HON', 'UNP', 'RTX', 'LMT', 'GE', 'BA',
    
    # Consumer
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'HD', 'NKE', 'MCD', 'SBUX',
    
    # Tech / Software
    'CRM', 'ADBE', 'ORCL', 'IBM', 'NOW', 'INTU', 'PANW', 'CRWD',
    
    # Semiconductors
    'AMD', 'INTC', 'AVGO', 'QCOM', 'TXN', 'MU', 'AMAT', 'LRCX',
    
    # ETFs for macro analysis
    'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI',
    'GLD', 'SLV', 'TLT', 'HYG', 'VXX', 'UVXY',
    
    # Crypto-adjacent (for MacBook Coinbase trading)
    'COIN', 'MSTR', 'RIOT', 'MARA', 'CLSK',
    
    # High-conviction positions (from your portfolio)
    'PLTR', 'SOFI', 'HOOD', 'ARM', 'SMCI'
]



# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_db_connection():
    """Get database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.error("Make sure PostgreSQL is running and credentials are correct")
        return None


def test_db_connection():
    """Test database connectivity."""
    conn = get_db_connection()
    if conn:
        logger.info("✓ Database connection successful")
        conn.close()
        return True
    return False


# ============================================================================
# PRICE DATA FUNCTIONS
# ============================================================================

def pull_price_history(symbols: List[str], years: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Pull historical price data from Yahoo Finance.
    
    Args:
        symbols: List of ticker symbols
        years: Years of history to pull
    
    Returns:
        Dict mapping symbol to DataFrame with OHLCV data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    results = {}
    total = len(symbols)
    
    logger.info(f"Pulling {years} years of data for {total} symbols...")
    
    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"[{i}/{total}] Pulling {symbol}...")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"  No data for {symbol}")
                continue
            
            # Clean column names
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            df['symbol'] = symbol
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            
            results[symbol] = df
            logger.info(f"  ✓ {len(df)} days of data")
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"  ✗ Error pulling {symbol}: {e}")
    
    logger.info(f"Pulled data for {len(results)}/{total} symbols")
    return results


def save_prices_to_db(price_data: Dict[str, pd.DataFrame]):
    """
    Save price data to database with upsert.
    
    Args:
        price_data: Dict of symbol -> DataFrame
    """
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    insert_sql = """
        INSERT INTO market_prices (symbol, date, open, high, low, close, volume, source)
        VALUES %s
        ON CONFLICT (symbol, date, source) 
        DO UPDATE SET 
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
    """
    
    total_rows = 0
    
    for symbol, df in price_data.items():
        try:
            # Prepare data
            records = []
            for _, row in df.iterrows():
                date_val = row.get('date') or row.get('index')
                if hasattr(date_val, 'date'):
                    date_val = date_val.date()
                
                records.append((
                    symbol,
                    date_val,
                    float(row.get('open', 0)),
                    float(row.get('high', 0)),
                    float(row.get('low', 0)),
                    float(row.get('close', 0)),
                    int(row.get('volume', 0)),
                    'yahoo'
                ))
            
            execute_values(cursor, insert_sql, records)
            total_rows += len(records)
            
        except Exception as e:
            logger.error(f"Error saving {symbol}: {e}")
            conn.rollback()
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logger.info(f"✓ Saved {total_rows} price records to database")



# ============================================================================
# FUNDAMENTALS FUNCTIONS
# ============================================================================

def pull_fundamentals(symbols: List[str]) -> List[Dict]:
    """
    Pull fundamental data from Yahoo Finance.
    
    Args:
        symbols: List of ticker symbols
    
    Returns:
        List of fundamental data dicts
    """
    results = []
    total = len(symbols)
    
    logger.info(f"Pulling fundamentals for {total} symbols...")
    
    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"[{i}/{total}] Pulling fundamentals for {symbol}...")
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                continue
            
            record = {
                'symbol': symbol,
                'period_end': datetime.now().date(),
                'period_type': 'latest',
                
                # Financials
                'revenue': info.get('totalRevenue'),
                'gross_profit': info.get('grossProfits'),
                'operating_income': info.get('operatingIncome'),
                'net_income': info.get('netIncomeToCommon'),
                'ebitda': info.get('ebitda'),
                
                # Per share
                'eps_basic': info.get('trailingEps'),
                
                # Balance sheet
                'total_assets': info.get('totalAssets'),
                'total_debt': info.get('totalDebt'),
                'cash_and_equivalents': info.get('totalCash'),
                
                # Cash flow
                'free_cash_flow': info.get('freeCashflow'),
                'operating_cash_flow': info.get('operatingCashflow'),
                
                # Ratios
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'gross_margin': info.get('grossMargins'),
                'operating_margin': info.get('operatingMargins'),
                'net_margin': info.get('profitMargins'),
            }
            
            results.append(record)
            logger.info(f"  ✓ Got fundamentals")
            
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
    
    logger.info(f"Pulled fundamentals for {len(results)}/{total} symbols")
    return results


def save_fundamentals_to_db(fundamentals: List[Dict]):
    """Save fundamentals to database."""
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    for record in fundamentals:
        try:
            # Build dynamic insert
            columns = [k for k, v in record.items() if v is not None]
            values = [record[k] for k in columns]
            
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            
            sql = f"""
                INSERT INTO company_fundamentals ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT (symbol, period_end, period_type) 
                DO UPDATE SET {', '.join(f'{c} = EXCLUDED.{c}' for c in columns if c not in ['symbol', 'period_end', 'period_type'])}
            """
            
            cursor.execute(sql, values)
            
        except Exception as e:
            logger.error(f"Error saving fundamentals for {record.get('symbol')}: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logger.info(f"✓ Saved {len(fundamentals)} fundamental records")


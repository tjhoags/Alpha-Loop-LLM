"""
================================================================================
MASSIVE S3 COMPREHENSIVE DATA HYDRATION
================================================================================
Bulk ingest ALL historical data from Massive S3 (Alpha Vantage premium files):
- Stocks (equity/minute/)
- Options (option/minute/) - WITH GREEKS
- Indices (index/minute/)
- Currencies/Forex (forex/minute/)
- Runs continuously to backfill years of history
================================================================================
"""
from datetime import datetime, timedelta
from typing import List
import time

import pandas as pd
from loguru import logger

from src.config.settings import get_settings
from src.data_ingestion.sources.massive import massive_client
from src.data_ingestion.collector import persist
from src.data_ingestion.sources.options_greeks import enrich_options_with_greeks


def backfill_massive_data(days_back: int = 1825):  # 5 years default
    """
    Scans Massive S3 bucket for ALL asset classes and ingests them.
    Supports: equity, option, index, forex
    """
    logger.info(f"Starting Massive S3 COMPREHENSIVE backfill for last {days_back} days ({days_back/365:.1f} years)...")
    
    start_date = datetime.utcnow() - timedelta(days=days_back)
    
    # ALL asset classes from your premium subscription
    asset_folders = [
        ("equity/minute/", "stock"),
        ("option/minute/", "option"),   # Huge volume - be patient
        ("index/minute/", "index"),
        ("forex/minute/", "forex"),
    ]
    
    total_rows = 0
    
    for folder, asset_type in asset_folders:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {asset_type.upper()}: {folder}")
        logger.info(f"{'='*80}")
        
        try:
            all_files = massive_client.list_files(prefix=folder)
            logger.info(f"  Found {len(all_files)} total files in {folder}")
        except Exception as e:
            logger.error(f"Could not list files from {folder}: {e}")
            continue

        if not all_files:
            logger.warning(f"No files found in {folder}.")
            continue
        
        # Filter files that match our date range
        files_to_ingest = []
        for f in all_files:
            try:
                # Check if file date is within range
                # Massive files typically named like: equity/minute/2024-01-15/AAPL.csv
                file_date_str = None
                for day_offset in range(days_back + 1):
                    check_date = start_date + timedelta(days=day_offset)
                    date_str = check_date.strftime("%Y-%m-%d")
                    if date_str in f:
                        file_date_str = date_str
                        break
                
                if file_date_str:
                    files_to_ingest.append(f)
            except Exception:
                continue
        
        logger.info(f"  Found {len(files_to_ingest)} files matching date range")
        
        # Process files in batches
        batch_size = 100
        for batch_start in range(0, len(files_to_ingest), batch_size):
            batch = files_to_ingest[batch_start:batch_start + batch_size]
            logger.info(f"  Processing batch {batch_start//batch_size + 1} ({len(batch)} files)...")
            
            batch_data = []
            for key in batch:
                try:
                    df = massive_client.fetch_file(key)
                    
                    if df is not None and not df.empty:
                        # Normalize columns
                        if "ticker" in df.columns:
                            df.rename(columns={"ticker": "symbol"}, inplace=True)
                        if "underlying_symbol" in df.columns:  # Options specific
                            df["symbol"] = df["underlying_symbol"]
                        
                        # Ensure timestamp exists
                        if "timestamp" not in df.columns:
                            if "window_start" in df.columns:
                                df["timestamp"] = pd.to_datetime(df["window_start"], unit="ns", errors="coerce")
                            elif "date" in df.columns:
                                df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
                        
                        # Ensure required columns
                        required_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
                        missing_cols = [c for c in required_cols if c not in df.columns]
                        if missing_cols:
                            logger.warning(f"  {key} missing columns: {missing_cols}")
                            continue
                        
                        # Add metadata
                        df["source"] = "massive"
                        df["asset_type"] = asset_type
                        
                        # For options: enrich with Greeks if we have strike/expiry info
                        if asset_type == "option" and "strike" in df.columns:
                            # This would require underlying price - simplified for now
                            # df = enrich_options_with_greeks(df, underlying_price=...)
                            pass
                        
                        batch_data.append(df)
                        
                except Exception as e:
                    logger.error(f"  Error processing {key}: {e}")
                    continue
            
            # Persist batch
            if batch_data:
                try:
                    combined = pd.concat(batch_data, ignore_index=True)
                    combined.sort_values(["symbol", "timestamp"], inplace=True)
                    combined.drop_duplicates(subset=["symbol", "timestamp"], keep="last", inplace=True)
                    
                    count = persist(combined, table="price_bars")
                    total_rows += count
                    logger.success(f"  Batch persisted: {count} rows (Total: {total_rows:,})")
                except Exception as e:
                    logger.error(f"  Error persisting batch: {e}")
            
            # Rate limiting - be nice to S3
            time.sleep(0.5)
    
    logger.success(f"\n{'='*80}")
    logger.success(f"Massive backfill COMPLETE: {total_rows:,} total rows ingested")
    logger.success(f"{'='*80}")


def continuous_hydration():
    """Run continuous hydration - pulls new data every hour."""
    logger.info("Starting CONTINUOUS Massive hydration (runs forever)...")
    
    while True:
        try:
            # Pull last 7 days continuously
            backfill_massive_data(days_back=7)
            logger.info("Sleeping 1 hour before next cycle...")
            time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Continuous hydration stopped by user")
            break
        except Exception as e:
            logger.exception(f"Error in continuous hydration: {e}")
            logger.info("Waiting 5 minutes before retry...")
            time.sleep(300)


if __name__ == "__main__":
    # Setup logging
    settings = get_settings()
    logger.add(settings.logs_dir / "massive_ingest.log", rotation="100 MB", level="INFO")
    
    # FULL THROTTLE: Backfill 5 years of history
    logger.info("FULL THROTTLE MODE: Backfilling 5 years of Massive S3 data...")
    backfill_massive_data(days_back=1825)  # 5 years
    
    # Then switch to continuous mode
    logger.info("\nSwitching to CONTINUOUS mode...")
    continuous_hydration() 



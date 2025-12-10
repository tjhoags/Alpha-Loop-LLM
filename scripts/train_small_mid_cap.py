"""================================================================================
SMALL/MID CAP TRAINING SCRIPT
================================================================================
Author: Alpha Loop Capital, LLC

Trains specialized models for small/mid cap trading strategies:
1. Retail Arbitrage - Exploits bad bid/ask, odd lots, retail flow
2. Conversion/Reversal - Options arbitrage detection
3. Mean Reversion - Fades overextended retail moves

Target Universe: Stocks <$10B market cap with high retail activity

Usage:
    python scripts/train_small_mid_cap.py [--continuous] [--symbols AAPL,SOFI,...]

Options:
    --continuous    Run in continuous mode (retrain every hour)
    --symbols       Specific symbols to train on (default: small/mid cap universe)
    --options       Include options arbitrage training
================================================================================
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import multiprocessing

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import get_settings
from src.database.connection import get_engine

# Feature engineering
from src.ml.feature_engineering import add_technical_indicators
from src.ml.options_arbitrage_features import (
    add_options_arbitrage_features,
)
from src.ml.retail_inefficiency_features import (
    add_retail_inefficiency_features,
    create_retail_arbitrage_target,
)

# Models
from src.ml.small_mid_cap_models import (
    ConversionReversalModel,
    MeanReversionModel,
    ModelMetrics,
    RetailArbitrageModel,
)

# Small/Mid Cap Universe - High Retail Activity Stocks
SMALL_MID_CAP_UNIVERSE = [
    # Meme / High Retail Stocks
    "GME", "AMC", "BBBY", "KOSS", "BB",

    # Fintech (Retail Favorites)
    "SOFI", "HOOD", "AFRM", "UPST", "LC", "COIN",

    # EV / Clean Energy (Retail Heavy)
    "RIVN", "LCID", "FSR", "NIO", "XPEV", "GOEV",
    "PLUG", "FCEL", "BE", "BLNK", "CHPT",

    # Tech Small/Mid
    "PLTR", "PATH", "IONQ", "RKLB", "JOBY", "LILM",
    "S", "DOCN", "GTLB", "CFLT", "ESTC",

    # Biotech Small Cap
    "RXRX", "DNA", "BEAM", "CRSP", "NTLA", "VERV",
    "EDIT", "FATE", "BLUE", "SGEN",

    # Uranium / Nuclear
    "UEC", "DNN", "NXE", "UUUU", "SMR", "LEU", "CCJ",

    # Cannabis
    "TLRY", "CGC", "ACB", "CRON", "HEXO",

    # SPACs / De-SPACs
    "OPEN", "RDFN", "WISH", "CLOV", "BARK",

    # Social / Gaming
    "RBLX", "DKNG", "PENN", "U", "SNAP",
]


def _load_single_symbol(args: Tuple[str, str, str]) -> Optional[pd.DataFrame]:
    """Load data for a single symbol (for parallel execution)."""
    symbol, start_date_str, connection_string = args
    try:
        from sqlalchemy import create_engine
        engine = create_engine(connection_string)

        query = f"""
        SELECT symbol, timestamp, [open], high, low, [close], volume
        FROM price_bars
        WHERE symbol = '{symbol}'
        AND timestamp >= '{start_date_str}'
        ORDER BY timestamp
        """
        df = pd.read_sql(query, engine)
        engine.dispose()

        if len(df) > 100:
            return df
        return None
    except Exception as e:
        return None


def load_price_data(
    symbols: List[str],
    lookback_days: int = 90,
    timeframe: str = "5min",
    parallel: bool = True,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Load price data from database for training.

    Args:
    ----
        symbols: List of symbols to load
        lookback_days: Days of history to load
        timeframe: Data granularity (not used for SQL, just for logging)
        parallel: Use parallel loading (faster for many symbols)
        max_workers: Number of parallel workers

    Returns:
    -------
        DataFrame with OHLCV data
    """
    settings = get_settings()
    start_date = datetime.now() - timedelta(days=lookback_days)
    start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")

    # Get connection string for parallel workers
    connection_string = f"mssql+pyodbc://{settings.db_user}:{settings.db_password}@{settings.db_host}/{settings.db_name}?driver=ODBC+Driver+17+for+SQL+Server"

    all_data = []

    if parallel and len(symbols) > 5:
        # Parallel loading with ThreadPoolExecutor
        logger.info(f"Loading {len(symbols)} symbols in parallel (workers={max_workers})...")

        args_list = [(s, start_date_str, connection_string) for s in symbols]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load_single_symbol, args): args[0] for args in args_list}

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result is not None and len(result) > 0:
                        all_data.append(result)
                        logger.info(f"  Loaded {len(result)} bars for {symbol}")
                except Exception as e:
                    logger.error(f"  Error loading {symbol}: {e}")
    else:
        # Sequential loading for small batches
        engine = get_engine()
        for symbol in symbols:
            try:
                query = f"""
                SELECT symbol, timestamp, [open], high, low, [close], volume
                FROM price_bars
                WHERE symbol = '{symbol}'
                AND timestamp >= '{start_date_str}'
                ORDER BY timestamp
                """
                df = pd.read_sql(query, engine)

                if len(df) > 100:
                    all_data.append(df)
                    logger.info(f"  Loaded {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"  Insufficient data for {symbol}: {len(df)} bars")

            except Exception as e:
                logger.error(f"  Error loading {symbol}: {e}")
                continue

    if not all_data:
        logger.error("No data loaded!")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total data loaded: {len(combined)} bars across {len(all_data)} symbols")

    return combined


def load_price_data_batch(
    symbols: List[str],
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Load price data using a single batch SQL query (most efficient for SQL Server).

    Args:
    ----
        symbols: List of symbols to load
        lookback_days: Days of history to load

    Returns:
    -------
        DataFrame with OHLCV data
    """
    engine = get_engine()
    start_date = datetime.now() - timedelta(days=lookback_days)
    start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")

    # Build IN clause
    symbols_str = ", ".join([f"'{s}'" for s in symbols])

    query = f"""
    SELECT symbol, timestamp, [open], high, low, [close], volume
    FROM price_bars
    WHERE symbol IN ({symbols_str})
    AND timestamp >= '{start_date_str}'
    ORDER BY symbol, timestamp
    """

    logger.info(f"Loading {len(symbols)} symbols in single batch query...")
    df = pd.read_sql(query, engine)

    # Filter to symbols with enough data
    symbol_counts = df.groupby("symbol").size()
    valid_symbols = symbol_counts[symbol_counts > 100].index.tolist()
    df = df[df["symbol"].isin(valid_symbols)]

    logger.info(f"Loaded {len(df)} bars across {len(valid_symbols)} symbols")
    return df


def load_options_data(symbols: List[str]) -> pd.DataFrame:
    """Load options chain data from database.

    Args:
    ----
        symbols: List of underlying symbols

    Returns:
    -------
        DataFrame with options data
    """
    engine = get_engine()

    try:
        query = """
        SELECT *
        FROM options_contracts
        WHERE underlying_symbol IN :symbols
        AND expiry > GETDATE()
        """

        # SQLAlchemy doesn't directly support IN with list, need to format
        symbols_str = ", ".join([f"'{s}'" for s in symbols])
        query = f"""
        SELECT *
        FROM options_contracts
        WHERE underlying_symbol IN ({symbols_str})
        """

        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} options contracts")
        return df

    except Exception as e:
        logger.warning(f"Could not load options data: {e}")
        return pd.DataFrame()


def train_retail_arbitrage_model(
    price_data: pd.DataFrame,
    model_id: str = "v1",
) -> Optional[ModelMetrics]:
    """Train the retail arbitrage model.

    Args:
    ----
        price_data: OHLCV data
        model_id: Model version identifier

    Returns:
    -------
        ModelMetrics or None if training failed
    """
    logger.info("=" * 60)
    logger.info("TRAINING RETAIL ARBITRAGE MODEL")
    logger.info("=" * 60)

    try:
        # Add technical indicators
        logger.info("Adding technical indicators...")
        df = add_technical_indicators(price_data.copy())

        # Add retail inefficiency features
        logger.info("Adding retail inefficiency features...")
        df = add_retail_inefficiency_features(df)

        # Create target
        logger.info("Creating training target...")
        df = create_retail_arbitrage_target(df, horizon=5, min_return=0.01)

        # Prepare features
        exclude_cols = [
            "symbol", "timestamp", "target", "target_long", "target_short",
            "target_binary", "future_return", "open", "high", "low", "close",
            "volume", "source", "asset_type",
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        logger.info(f"Training with {len(feature_cols)} features on {len(df)} samples")

        X = df[feature_cols]
        y = df["target_binary"]

        # Train model
        model = RetailArbitrageModel(model_id)
        metrics = model.train(X, y, cv_folds=5)

        # Save model
        model.save()

        # Log feature importance
        importance = model.get_feature_importance()
        logger.info("Top 10 Features:")
        for _, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        logger.info(f"Retail Arbitrage Model trained: AUC={metrics.auc:.4f}, Acc={metrics.accuracy:.4f}")

        return metrics

    except Exception as e:
        logger.error(f"Failed to train Retail Arbitrage Model: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_mean_reversion_model(
    price_data: pd.DataFrame,
    model_id: str = "v1",
) -> Optional[ModelMetrics]:
    """Train the mean reversion model.

    Targets overextended moves that are likely to revert.
    """
    logger.info("=" * 60)
    logger.info("TRAINING MEAN REVERSION MODEL")
    logger.info("=" * 60)

    try:
        # Add features
        df = add_technical_indicators(price_data.copy())
        df = add_retail_inefficiency_features(df)

        # Target: price reverts after overextension
        df["future_return"] = df["close"].pct_change(5).shift(-5)

        df["mean_rev_target"] = (
            (abs(df["price_extension"]) > 2) &  # Overextended
            # Price moves opposite to extension
            (df["future_return"] * np.sign(df["price_extension"]) < 0)
        ).astype(int)

        df.dropna(subset=["mean_rev_target", "future_return"], inplace=True)

        # Prepare features
        exclude_cols = [
            "symbol", "timestamp", "mean_rev_target", "future_return",
            "open", "high", "low", "close", "volume",
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y = df["mean_rev_target"]

        logger.info(f"Training with {len(feature_cols)} features on {len(df)} samples")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        # Train
        model = MeanReversionModel(model_id)
        metrics = model.train(X, y, cv_folds=5)
        model.save()

        logger.info(f"Mean Reversion Model trained: AUC={metrics.auc:.4f}, Acc={metrics.accuracy:.4f}")
        return metrics

    except Exception as e:
        logger.error(f"Failed to train Mean Reversion Model: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_conversion_reversal_model(
    options_data: pd.DataFrame,
    model_id: str = "v1",
) -> Optional[ModelMetrics]:
    """Train the conversion/reversal arbitrage model.
    """
    logger.info("=" * 60)
    logger.info("TRAINING CONVERSION/REVERSAL MODEL")
    logger.info("=" * 60)

    if options_data is None or len(options_data) == 0:
        logger.warning("No options data available for training")
        return None

    try:
        # Add features
        df = add_options_arbitrage_features(options_data.copy())

        # Target: arbitrage signal
        df["arb_target"] = (
            (df["conversion_signal"] == 1) |
            (df["reversal_signal"] == 1)
        ).astype(int)

        # Prepare features
        exclude_cols = [
            "symbol", "underlying_symbol", "underlying_price", "strike",
            "expiry", "call_bid", "call_ask", "put_bid", "put_ask",
            "arb_target", "T", "expiry_date",
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y = df["arb_target"]

        logger.info(f"Training with {len(feature_cols)} features on {len(df)} samples")

        model = ConversionReversalModel(model_id)
        metrics = model.train(X, y, cv_folds=3)
        model.save()

        logger.info(f"Conversion/Reversal Model trained: AUC={metrics.auc:.4f}")
        return metrics

    except Exception as e:
        logger.error(f"Failed to train Conversion/Reversal Model: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_training_cycle(
    symbols: List[str],
    include_options: bool = True,
) -> Dict[str, ModelMetrics]:
    """Run a complete training cycle for all small/mid cap models.

    Args:
    ----
        symbols: List of symbols to train on
        include_options: Whether to train options models

    Returns:
    -------
        Dict of model_name -> metrics
    """
    logger.info("=" * 70)
    logger.info("SMALL/MID CAP TRAINING CYCLE")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    results = {}

    # Load price data
    logger.info("\nLoading price data...")
    price_data = load_price_data(symbols, lookback_days=90)

    if len(price_data) == 0:
        logger.error("No price data loaded. Aborting training.")
        return results

    # Train retail arbitrage model
    retail_metrics = train_retail_arbitrage_model(price_data, model_id="v1")
    if retail_metrics:
        results["retail_arbitrage"] = retail_metrics

    # Train mean reversion model
    mean_rev_metrics = train_mean_reversion_model(price_data, model_id="v1")
    if mean_rev_metrics:
        results["mean_reversion"] = mean_rev_metrics

    # Train options models (if enabled)
    if include_options:
        logger.info("\nLoading options data...")
        options_data = load_options_data(symbols)

        if len(options_data) > 0:
            conv_rev_metrics = train_conversion_reversal_model(options_data, model_id="v1")
            if conv_rev_metrics:
                results["conversion_reversal"] = conv_rev_metrics

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    for model_name, metrics in results.items():
        passed = "PASS" if metrics.passes_threshold() else "FAIL"
        logger.info(
            f"  {model_name}: AUC={metrics.auc:.4f}, "
            f"Acc={metrics.accuracy:.4f}, F1={metrics.f1:.4f} [{passed}]",
        )

    logger.info("=" * 70)

    return results


def train_models_parallel(
    price_data: pd.DataFrame,
    options_data: Optional[pd.DataFrame] = None,
    include_options: bool = True,
) -> Dict[str, ModelMetrics]:
    """Train all models in parallel using ProcessPoolExecutor.

    This provides significant speedup when training multiple models.
    """
    from functools import partial

    results = {}
    model_tasks = []

    # Queue up model training tasks
    model_tasks.append(("retail_arbitrage", train_retail_arbitrage_model, price_data, "v1"))
    model_tasks.append(("mean_reversion", train_mean_reversion_model, price_data, "v1"))

    if include_options and options_data is not None and len(options_data) > 0:
        model_tasks.append(("conversion_reversal", train_conversion_reversal_model, options_data, "v1"))

    logger.info(f"Training {len(model_tasks)} models in parallel...")

    # Use ThreadPoolExecutor for I/O bound model training
    with ThreadPoolExecutor(max_workers=min(len(model_tasks), 4)) as executor:
        futures = {}
        for name, func, data, model_id in model_tasks:
            future = executor.submit(func, data, model_id)
            futures[future] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                metrics = future.result()
                if metrics:
                    results[name] = metrics
                    logger.info(f"  {name} completed: AUC={metrics.auc:.4f}")
            except Exception as e:
                logger.error(f"  {name} failed: {e}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Small/Mid Cap Models")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--options", action="store_true", help="Include options training")
    parser.add_argument("--interval", type=int, default=3600, help="Training interval (seconds)")

    # New high-compute options
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for data loading")
    parser.add_argument("--batch-load", action="store_true", help="Use single batch SQL query (fastest)")
    parser.add_argument("--parallel-train", action="store_true", help="Train models in parallel")
    parser.add_argument("--lookback", type=int, default=90, help="Days of historical data")
    parser.add_argument("--full-universe", action="store_true", help="Use full 70+ symbol universe")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration (requires CUDA)")

    args = parser.parse_args()

    # Setup logging
    logger.add(
        "logs/small_mid_cap_training_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
    )

    # Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.full_universe:
        symbols = SMALL_MID_CAP_UNIVERSE
    else:
        # Default to smaller set for faster iteration
        symbols = SMALL_MID_CAP_UNIVERSE[:20]

    # GPU configuration
    if args.gpu:
        logger.info("GPU acceleration enabled - configuring XGBoost/LightGBM for CUDA")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    logger.info("=" * 70)
    logger.info("SMALL/MID CAP MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Lookback: {args.lookback} days")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Batch SQL: {args.batch_load}")
    logger.info(f"Parallel Training: {args.parallel_train}")
    logger.info(f"GPU: {args.gpu}")
    logger.info("=" * 70)

    def run_high_compute_cycle():
        """Run a single training cycle with high-compute options."""
        # Load data
        if args.batch_load:
            price_data = load_price_data_batch(symbols, lookback_days=args.lookback)
        else:
            price_data = load_price_data(
                symbols,
                lookback_days=args.lookback,
                parallel=True,
                max_workers=args.workers,
            )

        if len(price_data) == 0:
            logger.error("No price data loaded!")
            return {}

        # Load options if needed
        options_data = None
        if args.options:
            options_data = load_options_data(symbols)

        # Train models
        if args.parallel_train:
            return train_models_parallel(price_data, options_data, args.options)
        else:
            return run_training_cycle(symbols, include_options=args.options)

    if args.continuous:
        logger.info(f"Running in continuous mode (interval: {args.interval}s)")

        while True:
            try:
                results = run_high_compute_cycle()
                if results:
                    passing = sum(1 for m in results.values() if m.passes_threshold())
                    logger.info(f"Cycle complete: {passing}/{len(results)} passed")
                logger.info(f"Next training in {args.interval} seconds...")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                break
            except Exception as e:
                logger.error(f"Training cycle failed: {e}")
                import traceback
                traceback.print_exc()
                logger.info("Retrying in 60 seconds...")
                time.sleep(60)
    else:
        # Single training run
        results = run_high_compute_cycle()

        # Exit code based on results
        if results:
            passing = sum(1 for m in results.values() if m.passes_threshold())
            logger.info(f"\n{passing}/{len(results)} models passed thresholds")
            sys.exit(0 if passing > 0 else 1)
        else:
            logger.error("No models trained successfully")
            sys.exit(1)


if __name__ == "__main__":
    main()

"""================================================================================
MODEL TRAINING - Continuous ML Model Training Pipeline
================================================================================
Trains ensemble ML models (XGBoost, LightGBM, CatBoost) on market data.
Runs continuously, retraining hourly to capture recent market dynamics.

Features:
- Time-series cross-validation to prevent lookahead bias
- Automatic feature engineering with 100+ indicators
- Model versioning with metadata
- Graceful error recovery
================================================================================
"""

from datetime import datetime

import pandas as pd
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError

from src.config.settings import get_settings
from src.database.connection import get_engine
from src.ml.feature_engineering import prepare_features
from src.ml.models import build_models, save_model, time_series_cv


def load_data(symbol: str) -> pd.DataFrame:
    """Load price data for a symbol from the database.

    Args:
        symbol: Stock/crypto ticker symbol

    Returns:
        DataFrame with OHLCV data sorted by timestamp
    """
    try:
        engine = get_engine()
        query = """
        SELECT symbol, timestamp, [open], high, low, [close], volume
        FROM price_bars
        WHERE symbol = :symbol
        ORDER BY timestamp ASC
        """
        return pd.read_sql(query, engine, params={"symbol": symbol})
    except SQLAlchemyError as e:
        logger.error(f"Database error loading data for {symbol}: {e}")
        return pd.DataFrame()


def train_for_symbol(symbol: str, settings) -> bool:
    """Train all models for a single symbol.

    Args:
        symbol: Stock/crypto ticker symbol
        settings: Application settings

    Returns:
        True if training succeeded, False otherwise
    """
    MIN_ROWS = 200

    df = load_data(symbol)
    if df.empty or len(df) < MIN_ROWS:
        logger.warning(f"Not enough data to train for {symbol} (found {len(df)} rows, need {MIN_ROWS}).")
        return False

    # Use enhanced features with valuation metrics
    try:
        X, y = prepare_features(df, include_valuation=True)
    except ValueError as e:
        logger.error(f"Feature preparation failed for {symbol}: {e}")
        return False

    if len(X) < MIN_ROWS:
        logger.warning(f"Not enough engineered rows to train for {symbol} (found {len(X)} rows).")
        return False

    models = build_models()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    trained_count = 0

    for name, model in models.items():
        try:
            acc, auc = time_series_cv(model, X, y, splits=3)

            # Skip models with poor performance
            if auc < settings.min_auc:
                logger.warning(f"{symbol} | {name} | AUC {auc:.3f} below threshold {settings.min_auc}")
                continue

            model.fit(X, y)
            meta = {
                "symbol": symbol,
                "type": name,
                "cv_acc": acc,
                "cv_auc": auc,
                "timestamp": timestamp,
                "n_features": X.shape[1],
                "n_samples": len(X),
            }
            save_model(model, f"{symbol}_{name}_{timestamp}", settings.models_dir, meta)
            logger.info(f"{symbol} | {name} | CV acc={acc:.3f} auc={auc:.3f} | Features: {X.shape[1]}")
            trained_count += 1

        except Exception as e:
            logger.error(f"Training failed for {symbol}/{name}: {e}")
            continue

    return trained_count > 0


def main() -> None:
    """Main entry point for continuous model training."""
    import time

    settings = get_settings()
    logger.add(settings.logs_dir / "model_training.log", rotation="50 MB", level=settings.log_level)

    # Log configuration status at startup
    settings.log_configuration_status()
    logger.info("Starting continuous model training pipeline...")

    cycle = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5

    while True:
        cycle += 1
        logger.info(f"{'='*80}")
        logger.info(f"TRAINING CYCLE #{cycle} - {datetime.now()}")
        logger.info(f"{'='*80}")

        try:
            symbols = settings.target_symbols
            if not symbols:
                logger.warning("No target symbols configured. Check settings.target_symbols")
                time.sleep(60)
                continue

            successful = 0
            failed = 0

            for sym in symbols:
                if train_for_symbol(sym, settings):
                    successful += 1
                else:
                    failed += 1

            logger.info(f"Cycle #{cycle} complete: {successful} succeeded, {failed} failed")

            if successful > 0:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.critical(f"Too many consecutive failures ({consecutive_failures}). Check data sources.")

            logger.info("Sleeping 1 hour before next cycle...")
            time.sleep(3600)

        except KeyboardInterrupt:
            logger.info("Training stopped by user")
            break
        except SQLAlchemyError as e:
            logger.error(f"Database error in cycle #{cycle}: {e}")
            logger.info("Waiting 5 minutes before retry...")
            time.sleep(300)
        except Exception as e:
            logger.exception(f"Unexpected error in training cycle #{cycle}: {e}")
            logger.info("Waiting 5 minutes before retry...")
            time.sleep(300)


if __name__ == "__main__":
    main()


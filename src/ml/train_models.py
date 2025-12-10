from datetime import datetime

import pandas as pd
from loguru import logger

from src.config.settings import get_settings
from src.database.connection import get_engine
from src.ml.feature_engineering import prepare_features
from src.ml.models import build_models, save_model, time_series_cv


def load_data(symbol: str) -> pd.DataFrame:
    engine = get_engine()
    query = f"""
    SELECT symbol, timestamp, [open], high, low, [close], volume
    FROM price_bars
    WHERE symbol = :symbol
    ORDER BY timestamp ASC
    """
    return pd.read_sql(query, engine, params={"symbol": symbol})


def train_for_symbol(symbol: str, settings) -> None:
    df = load_data(symbol)
    if len(df) < 200:
        logger.warning(f"Not enough data to train for {symbol} (found {len(df)} rows).")
        return

    # Use enhanced features with valuation metrics
    X, y = prepare_features(df, include_valuation=True)
    if len(X) < 200:
        logger.warning(f"Not enough engineered rows to train for {symbol} (found {len(X)} rows).")
        return

    models = build_models()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for name, model in models.items():
        acc, auc = time_series_cv(model, X, y, splits=3)
        logger.info(f"{symbol} | {name} | CV acc={acc:.3f} auc={auc:.3f} | Features: {X.shape[1]}")
        model.fit(X, y)
        meta = {"symbol": symbol, "cv_acc": acc, "cv_auc": auc, "timestamp": timestamp, "n_features": X.shape[1]}
        save_model(model, f"{symbol}_{name}_{timestamp}", settings.models_dir, meta)


def main() -> None:
    import time
    
    settings = get_settings()
    logger.add(settings.logs_dir / "model_training.log", rotation="50 MB", level=settings.log_level)
    logger.info("🚀 Starting CONTINUOUS model training with valuation metrics...")
    
    cycle = 0
    while True:
        cycle += 1
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING CYCLE #{cycle} - {datetime.now()}")
        logger.info(f"{'='*80}\n")
        
        try:
            for sym in settings.target_symbols:
                train_for_symbol(sym, settings)
            
            logger.success(f"✅ Cycle #{cycle} complete. Sleeping 1 hour before next cycle...")
            time.sleep(3600)  # Retrain every hour
            
        except KeyboardInterrupt:
            logger.info("🛑 Training stopped by user")
            break
        except Exception as e:
            logger.exception(f"❌ Error in training cycle #{cycle}: {e}")
            logger.info("⏳ Waiting 5 minutes before retry...")
            time.sleep(300)


if __name__ == "__main__":
    main()


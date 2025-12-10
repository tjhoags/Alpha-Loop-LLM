"""================================================================================
STRATEGY TRAINING SCRIPT - All Trading Strategies
================================================================================
Alpha Loop Capital, LLC

Trains all trading strategy models continuously:
- Momentum Strategy
- Mean Reversion Strategy
- Value Strategy
- Arbitrage Strategy
- Conversion/Reversal (Options)
- Pairs Trading
- Trend Following
- Volatility Strategy

Usage:
    python scripts/train_all_strategies.py --continuous --interval 15
================================================================================
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import get_settings
from src.database.connection import get_engine


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

STRATEGIES = {
    "momentum": {
        "description": "Trend following / breakout detection",
        "symbols": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD"],
        "lookback": 60,
        "features": ["rsi", "macd", "adx", "momentum", "breakout"],
    },
    "mean_reversion": {
        "description": "Fades overextended moves",
        "symbols": ["SPY", "QQQ", "IWM", "XLF", "XLE", "XLK"],
        "lookback": 30,
        "features": ["bollinger", "zscore", "rsi_extreme", "volume_spike"],
    },
    "value": {
        "description": "Fundamental undervaluation",
        "symbols": ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "JPM", "BAC", "WFC"],
        "lookback": 90,
        "features": ["pe_ratio", "pb_ratio", "dividend_yield", "earnings_growth"],
    },
    "arbitrage": {
        "description": "Cross-market price discrepancies",
        "symbols": ["SPY", "ES=F", "QQQ", "NQ=F", "IWM", "RTY=F"],
        "lookback": 5,
        "features": ["spread", "basis", "correlation", "cointegration"],
    },
    "pairs_trading": {
        "description": "Statistical arbitrage between correlated pairs",
        "symbols": ["XOM", "CVX", "KO", "PEP", "GLD", "SLV", "JPM", "BAC"],
        "lookback": 60,
        "features": ["spread_zscore", "half_life", "correlation", "cointegration"],
    },
    "trend_following": {
        "description": "Long-term trend capture",
        "symbols": ["SPY", "QQQ", "GLD", "TLT", "UUP", "USO"],
        "lookback": 200,
        "features": ["sma_cross", "channel_breakout", "atr", "trend_strength"],
    },
    "volatility": {
        "description": "Volatility regime trading",
        "symbols": ["VXX", "UVXY", "SVXY", "SPY", "QQQ"],
        "lookback": 30,
        "features": ["vix_level", "vix_term_structure", "realized_vol", "iv_rv_spread"],
    },
}

# Full universe for comprehensive training
FULL_UNIVERSE = [
    # Large Cap Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
    # Fixed Income ETFs
    "TLT", "IEF", "SHY", "LQD", "HYG", "AGG",
    # Commodities
    "GLD", "SLV", "USO", "UNG", "DBA",
    # Volatility
    "VXX", "UVXY", "SVXY",
    # Crypto ETFs
    "BITO", "GBTC",
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(symbols: List[str], lookback_days: int = 90) -> pd.DataFrame:
    """Load price data for training."""
    from sqlalchemy import text

    try:
        engine = get_engine()
        start_date = datetime.now() - timedelta(days=lookback_days)

        # Use parameterized query to prevent SQL injection
        # SQLAlchemy doesn't support IN clause with bind params directly,
        # so we sanitize symbols to alphanumeric only
        safe_symbols = [s for s in symbols if s.isalnum() or s.replace("-", "").replace(".", "").isalnum()]
        if not safe_symbols:
            return pd.DataFrame()

        symbols_str = ", ".join([f"'{s}'" for s in safe_symbols])

        query = text(f"""
        SELECT symbol, timestamp, [open], high, low, [close], volume
        FROM price_bars
        WHERE symbol IN ({symbols_str})
        AND timestamp >= :start_date
        ORDER BY symbol, timestamp
        """)

        df = pd.read_sql(query, engine, params={"start_date": start_date})
        logger.info(f"Loaded {len(df)} bars for {len(df['symbol'].unique())} symbols")
        return df

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return pd.DataFrame()


def add_features(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Add strategy-specific features."""
    if df.empty:
        return df

    df = df.copy()

    # Basic features for all strategies
    df["returns"] = df.groupby("symbol")["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df.groupby("symbol")["returns"].transform(lambda x: x.rolling(20).std())

    # RSI
    delta = df.groupby("symbol")["close"].diff()
    gain = delta.where(delta > 0, 0).groupby(df["symbol"]).transform(lambda x: x.rolling(14).mean())
    loss = (-delta.where(delta < 0, 0)).groupby(df["symbol"]).transform(lambda x: x.rolling(14).mean())
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Moving averages
    df["sma_20"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(20).mean())
    df["sma_50"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(50).mean())
    df["sma_200"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(200).mean())

    # Bollinger Bands
    df["bb_middle"] = df["sma_20"]
    df["bb_std"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(20).std())
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # MACD
    ema_12 = df.groupby("symbol")["close"].transform(lambda x: x.ewm(span=12).mean())
    ema_26 = df.groupby("symbol")["close"].transform(lambda x: x.ewm(span=26).mean())
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df.groupby("symbol")["macd"].transform(lambda x: x.ewm(span=9).mean())
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Volume features
    df["volume_sma"] = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(20).mean())
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # Momentum
    df["momentum_5"] = df.groupby("symbol")["close"].pct_change(5)
    df["momentum_10"] = df.groupby("symbol")["close"].pct_change(10)
    df["momentum_20"] = df.groupby("symbol")["close"].pct_change(20)

    # Strategy-specific features
    if strategy == "mean_reversion":
        df["zscore"] = (df["close"] - df["sma_20"]) / df["bb_std"]
        df["rsi_extreme"] = ((df["rsi"] < 30) | (df["rsi"] > 70)).astype(int)

    elif strategy == "momentum":
        df["breakout_up"] = (df["close"] > df["bb_upper"]).astype(int)
        df["breakout_down"] = (df["close"] < df["bb_lower"]).astype(int)
        df["trend_strength"] = abs(df["close"] - df["sma_50"]) / df["volatility"]

    elif strategy == "trend_following":
        df["above_sma_200"] = (df["close"] > df["sma_200"]).astype(int)
        df["sma_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)

    elif strategy == "volatility":
        df["vol_regime"] = pd.cut(df["volatility"], bins=3, labels=["low", "medium", "high"])
        df["vol_expanding"] = (df["volatility"] > df["volatility"].shift(5)).astype(int)

    return df


def create_target(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.01) -> pd.DataFrame:
    """Create training target."""
    df = df.copy()

    # Future return
    df["future_return"] = df.groupby("symbol")["close"].pct_change(horizon).shift(-horizon)

    # Binary target: 1 if positive return above threshold
    df["target"] = (df["future_return"] > threshold).astype(int)

    # Multi-class target
    df["target_direction"] = pd.cut(
        df["future_return"],
        bins=[-np.inf, -threshold, threshold, np.inf],
        labels=["short", "hold", "long"]
    )

    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_strategy_model(
    strategy_name: str,
    df: pd.DataFrame,
    model_id: str = "v1"
) -> Dict:
    """Train a single strategy model."""
    logger.info(f"Training {strategy_name} strategy...")

    try:
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        import joblib

        # Add features
        df = add_features(df, strategy_name)
        df = create_target(df)

        # Drop NaN
        df = df.dropna()

        if len(df) < 100:
            logger.warning(f"Insufficient data for {strategy_name}: {len(df)} rows")
            return {"success": False, "error": "Insufficient data"}

        # Feature columns
        exclude_cols = [
            "symbol", "timestamp", "open", "high", "low", "close", "volume",
            "target", "target_direction", "future_return", "vol_regime"
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64]]

        X = df[feature_cols].fillna(0)
        y = df["target"]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        cv_mean = cv_scores.mean()

        # Save model
        settings = get_settings()
        model_path = settings.models_dir / f"{strategy_name}_strategy_{model_id}.pkl"
        joblib.dump({
            "model": model,
            "feature_cols": feature_cols,
            "metrics": {
                "accuracy": accuracy,
                "auc": auc,
                "f1": f1,
                "cv_auc": cv_mean,
            },
            "trained_at": datetime.now().isoformat(),
        }, model_path)

        logger.info(f"  {strategy_name}: AUC={auc:.4f}, Acc={accuracy:.4f}, F1={f1:.4f}, CV={cv_mean:.4f}")

        return {
            "success": True,
            "strategy": strategy_name,
            "accuracy": accuracy,
            "auc": auc,
            "f1": f1,
            "cv_auc": cv_mean,
            "samples": len(df),
            "features": len(feature_cols),
        }

    except Exception as e:
        logger.error(f"Failed to train {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "strategy": strategy_name, "error": str(e)}


def train_all_strategies(
    symbols: List[str],
    lookback_days: int = 90,
    parallel: bool = True
) -> Dict[str, Dict]:
    """Train all strategy models."""
    logger.info("=" * 70)
    logger.info("STRATEGY TRAINING CYCLE")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info("=" * 70)

    # Load data
    df = load_training_data(symbols, lookback_days)

    if df.empty:
        logger.error("No data loaded - cannot train")
        return {}

    results = {}

    if parallel:
        # Parallel training
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for strategy_name in STRATEGIES:
                strategy_symbols = STRATEGIES[strategy_name]["symbols"]
                strategy_df = df[df["symbol"].isin(strategy_symbols)].copy()

                if len(strategy_df) > 0:
                    future = executor.submit(train_strategy_model, strategy_name, strategy_df)
                    futures[future] = strategy_name

            for future in as_completed(futures):
                strategy_name = futures[future]
                try:
                    result = future.result()
                    results[strategy_name] = result
                except Exception as e:
                    logger.error(f"{strategy_name} failed: {e}")
                    results[strategy_name] = {"success": False, "error": str(e)}
    else:
        # Sequential training
        for strategy_name, strategy_config in STRATEGIES.items():
            strategy_symbols = strategy_config["symbols"]
            strategy_df = df[df["symbol"].isin(strategy_symbols)].copy()

            if len(strategy_df) > 0:
                result = train_strategy_model(strategy_name, strategy_df)
                results[strategy_name] = result
            else:
                logger.warning(f"No data for {strategy_name} symbols")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    successful = 0
    for name, result in results.items():
        if result.get("success"):
            successful += 1
            auc = result.get("auc", 0)
            status = "[PASS]" if auc >= 0.52 else "[FAIL]"
            logger.info(f"  {name}: AUC={auc:.4f} {status}")
        else:
            logger.info(f"  {name}: FAILED - {result.get('error', 'Unknown error')}")

    logger.info(f"\nSuccessful: {successful}/{len(STRATEGIES)}")
    logger.info("=" * 70)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train All Trading Strategies")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=15, help="Training interval (seconds)")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--lookback", type=int, default=90, help="Days of historical data")
    parser.add_argument("--parallel", action="store_true", default=True, help="Train in parallel")
    parser.add_argument("--strategies", type=str, help="Comma-separated strategies to train")

    args = parser.parse_args()

    # Setup logging
    logger.add(
        "logs/strategy_training_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
    )

    # Symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = FULL_UNIVERSE

    # Strategies filter
    if args.strategies:
        strategy_filter = [s.strip().lower() for s in args.strategies.split(",")]
        global STRATEGIES
        STRATEGIES = {k: v for k, v in STRATEGIES.items() if k in strategy_filter}

    logger.info("=" * 70)
    logger.info("ALPHA LOOP CAPITAL - STRATEGY TRAINING")
    logger.info("=" * 70)
    logger.info(f"Strategies: {list(STRATEGIES.keys())}")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Interval: {args.interval}s")
    logger.info("=" * 70)

    if args.continuous:
        logger.info(f"Running in continuous mode (interval: {args.interval}s)")

        while True:
            try:
                results = train_all_strategies(symbols, args.lookback, args.parallel)

                if results:
                    successful = sum(1 for r in results.values() if r.get("success"))
                    logger.info(f"Cycle complete: {successful}/{len(results)} successful")

                logger.info(f"Next training in {args.interval} seconds...")
                time.sleep(args.interval)

            except KeyboardInterrupt:
                logger.info("Training interrupted")
                break
            except Exception as e:
                logger.error(f"Training cycle failed: {e}")
                import traceback
                traceback.print_exc()
                logger.info(f"Retrying in {args.interval} seconds...")
                time.sleep(args.interval)
    else:
        # Single run
        results = train_all_strategies(symbols, args.lookback, args.parallel)

        if results:
            successful = sum(1 for r in results.values() if r.get("success"))
            sys.exit(0 if successful > 0 else 1)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()

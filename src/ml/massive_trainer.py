"""================================================================================
MASSIVE PARALLEL TRAINING SYSTEM
================================================================================
Trains ML models on the FULL universe:
- 8,000+ stocks
- 2,500+ ETFs
- Options chains
- Crypto
- Forex

Features:
- Dynamic universe loading from database
- Parallel model training
- Behavioral + Technical features
- ACA (Agents Creating Agents) integration
- Auto-checkpoint and resume
- Multi-tier grading system

================================================================================
"""

import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.settings import get_settings
from src.database.connection import get_engine
from src.ml.behavioral_features import BehavioralFeatureEngine
from src.ml.feature_engineering import add_technical_indicators, make_supervised
from src.ml.models import (
    build_models,
    calculate_risk_metrics,
    save_model,
    time_series_cv,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for massive training."""

    min_rows: int = 252           # Minimum 1 year of daily data
    cv_splits: int = 5            # Cross-validation splits
    max_workers: int = 4          # Parallel processes
    batch_size: int = 50          # Symbols per batch
    checkpoint_interval: int = 100  # Save progress every N symbols

    # Grading thresholds
    min_auc: float = 0.52
    min_accuracy: float = 0.51
    min_sharpe: float = 0.5
    max_drawdown: float = 0.20

    # Feature configuration
    use_behavioral: bool = True
    use_technical: bool = True

    # Universe filters
    asset_classes: List[str] = field(default_factory=lambda: ["stock", "etf", "crypto"])
    market_cap_min: float = 0          # Minimum market cap (0 = no filter)
    market_cap_max: float = 25e9       # Maximum market cap (<$25B as requested)
    min_volume: float = 100000         # Minimum average volume


@dataclass
class TrainingProgress:
    """Tracks training progress for resume."""

    completed_symbols: List[str] = field(default_factory=list)
    failed_symbols: List[str] = field(default_factory=list)
    total_symbols: int = 0
    models_saved: int = 0
    start_time: str = ""
    last_checkpoint: str = ""

    def to_dict(self) -> Dict:
        return {
            "completed_symbols": self.completed_symbols,
            "failed_symbols": self.failed_symbols,
            "total_symbols": self.total_symbols,
            "models_saved": self.models_saved,
            "start_time": self.start_time,
            "last_checkpoint": self.last_checkpoint,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingProgress":
        return cls(**data)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_full_universe(config: TrainingConfig) -> List[str]:
    """Load ALL symbols from the database.
    """
    engine = get_engine()

    query = """
    SELECT DISTINCT symbol
    FROM price_bars
    GROUP BY symbol
    HAVING COUNT(*) >= :min_rows
    ORDER BY symbol
    """

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"min_rows": config.min_rows})

    symbols = df["symbol"].tolist()
    logger.info(f"Loaded {len(symbols)} symbols from universe")
    return symbols


def load_symbol_data(symbol: str) -> pd.DataFrame:
    """Load price data for a single symbol.
    """
    engine = get_engine()

    query = """
    SELECT symbol, timestamp, [open], high, low, [close], volume
    FROM price_bars
    WHERE symbol = :symbol
    ORDER BY timestamp ASC
    """

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": symbol})

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def prepare_features(df: pd.DataFrame, config: TrainingConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare full feature set (technical + behavioral).
    """
    if df.empty or len(df) < config.min_rows:
        return pd.DataFrame(), pd.Series()

    try:
        # Add technical indicators
        if config.use_technical:
            df = add_technical_indicators(df)

        # Add behavioral features
        if config.use_behavioral:
            df = BehavioralFeatureEngine.add_emotional_features(df)
            df = BehavioralFeatureEngine.add_crowd_psychology_features(df)
            df = BehavioralFeatureEngine.add_cognitive_bias_features(df)
            df = BehavioralFeatureEngine.add_game_theory_features(df)

        # Create supervised target
        df = make_supervised(df, horizon=1)

        # Define features (exclude target and identifiers)
        exclude = ["symbol", "timestamp", "target", "future_return",
                   "open", "high", "low", "close", "volume"]
        feature_cols = [c for c in df.columns if c not in exclude]

        X = df[feature_cols].copy()
        y = df["target"].copy()

        # Handle infinities and NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        # Drop columns that are all NaN
        X = X.dropna(axis=1, how="all")

        # Fill remaining NaN with median
        X = X.fillna(X.median())

        # Align y with X
        valid_idx = X.index
        y = y.loc[valid_idx]

        return X, y

    except Exception as e:
        logger.error(f"Feature preparation error: {e}")
        return pd.DataFrame(), pd.Series()


# =============================================================================
# MODEL TRAINING
# =============================================================================

def grade_model(metrics: Dict, config: TrainingConfig) -> Tuple[bool, str]:
    """Grade model against thresholds.
    Returns (passed, grade_note).
    """
    auc = metrics.get("auc", 0)
    acc = metrics.get("accuracy", 0)
    sharpe = metrics.get("sharpe", 0)
    drawdown = metrics.get("max_drawdown", 1)

    checks = []

    if auc >= config.min_auc:
        checks.append(f"AUC={auc:.3f} OK")
    else:
        return False, f"FAIL: AUC={auc:.3f} < {config.min_auc}"

    if acc >= config.min_accuracy:
        checks.append(f"Acc={acc:.3f} OK")
    else:
        return False, f"FAIL: Acc={acc:.3f} < {config.min_accuracy}"

    if sharpe >= config.min_sharpe:
        checks.append(f"Sharpe={sharpe:.2f} OK")
    else:
        return False, f"FAIL: Sharpe={sharpe:.2f} < {config.min_sharpe}"

    if drawdown <= config.max_drawdown:
        checks.append(f"DD={drawdown:.1%} OK")
    else:
        return False, f"FAIL: DD={drawdown:.1%} > {config.max_drawdown}"

    return True, f"PASS: {', '.join(checks)}"


def train_single_symbol(args: Tuple) -> Dict:
    """Train models for a single symbol.
    Designed for multiprocessing.
    """
    symbol, config_dict = args

    # Reconstruct config (can't pickle dataclass directly in some cases)
    config = TrainingConfig(**config_dict)
    settings = get_settings()

    result = {
        "symbol": symbol,
        "status": "unknown",
        "models_saved": 0,
        "best_model": None,
        "best_auc": 0,
        "error": None,
    }

    try:
        # Load data
        df = load_symbol_data(symbol)

        if len(df) < config.min_rows:
            result["status"] = "insufficient_data"
            return result

        # Prepare features
        X, y = prepare_features(df, config)

        if X.empty or len(X) < config.min_rows:
            result["status"] = "insufficient_features"
            return result

        # Build models
        models = build_models()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        best_model_name = None
        best_score = 0

        for name, model in models.items():
            try:
                # Time-series cross-validation
                metrics = time_series_cv(model, X, y, splits=config.cv_splits)

                # Add risk metrics
                risk_metrics = calculate_risk_metrics(y.values)
                metrics.update(risk_metrics)

                # Grade
                passed, grade_note = grade_model(metrics, config)

                if passed:
                    # Train on full data
                    model.fit(X, y)

                    # Save model
                    meta = {
                        "symbol": symbol,
                        "type": name,
                        "metrics": metrics,
                        "timestamp": timestamp,
                        "features": list(X.columns),
                        "n_samples": len(X),
                        "status": "active",
                    }

                    model_name = f"{symbol}_{name}_{timestamp}"
                    save_model(model, model_name, settings.models_dir, meta)

                    result["models_saved"] += 1

                    if metrics["auc"] > best_score:
                        best_score = metrics["auc"]
                        best_model_name = name

            except Exception as e:
                logger.debug(f"Model {name} failed for {symbol}: {e}")

        if result["models_saved"] > 0:
            result["status"] = "success"
            result["best_model"] = best_model_name
            result["best_auc"] = best_score
        else:
            result["status"] = "no_passing_models"

        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result


# =============================================================================
# MASSIVE TRAINER
# =============================================================================

class MassiveTrainer:
    """Orchestrates training across the full universe.
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.settings = get_settings()
        self.progress = TrainingProgress()
        self.checkpoint_file = self.settings.models_dir / "training_checkpoint.json"

        # Ensure directories exist
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        self.settings.logs_dir.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self) -> bool:
        """Load training progress from checkpoint."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    data = json.load(f)
                self.progress = TrainingProgress.from_dict(data)
                logger.info(f"Loaded checkpoint: {len(self.progress.completed_symbols)} completed")
                return True
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return False

    def save_checkpoint(self):
        """Save training progress to checkpoint."""
        self.progress.last_checkpoint = datetime.utcnow().isoformat()
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.progress.to_dict(), f, indent=2)

    def train_universe(self, resume: bool = True) -> Dict:
        """Train on the full universe.

        Args:
        ----
            resume: If True, resume from checkpoint
        """
        start_time = time.time()
        self.progress.start_time = datetime.utcnow().isoformat()

        # Load checkpoint if resuming
        if resume:
            self.load_checkpoint()

        # Load universe
        all_symbols = load_full_universe(self.config)
        self.progress.total_symbols = len(all_symbols)

        # Filter out completed symbols
        remaining = [s for s in all_symbols if s not in self.progress.completed_symbols]
        logger.info(f"Training {len(remaining)} symbols ({len(self.progress.completed_symbols)} already done)")

        # Prepare config dict for multiprocessing
        config_dict = {
            "min_rows": self.config.min_rows,
            "cv_splits": self.config.cv_splits,
            "min_auc": self.config.min_auc,
            "min_accuracy": self.config.min_accuracy,
            "min_sharpe": self.config.min_sharpe,
            "max_drawdown": self.config.max_drawdown,
            "use_behavioral": self.config.use_behavioral,
            "use_technical": self.config.use_technical,
        }

        # Process in batches
        batch_num = 0

        for i in range(0, len(remaining), self.config.batch_size):
            batch = remaining[i:i+self.config.batch_size]
            batch_num += 1

            logger.info(f"\n{'='*60}")
            logger.info(f"BATCH {batch_num} | Symbols {i+1}-{i+len(batch)} of {len(remaining)}")
            logger.info(f"{'='*60}")

            # Parallel training
            args = [(symbol, config_dict) for symbol in batch]

            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(train_single_symbol, arg): arg[0] for arg in args}

                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result()

                        if result["status"] == "success":
                            self.progress.completed_symbols.append(symbol)
                            self.progress.models_saved += result["models_saved"]
                            logger.info(f"[OK] {symbol}: {result['best_model']} AUC={result['best_auc']:.3f}")
                        elif result["status"] == "no_passing_models":
                            self.progress.completed_symbols.append(symbol)
                            logger.warning(f"[--] {symbol}: No passing models")
                        elif result["status"] in ["insufficient_data", "insufficient_features"]:
                            self.progress.failed_symbols.append(symbol)
                            logger.debug(f"[SKIP] {symbol}: {result['status']}")
                        else:
                            self.progress.failed_symbols.append(symbol)
                            logger.error(f"[ERR] {symbol}: {result.get('error', 'Unknown')}")

                    except Exception as e:
                        self.progress.failed_symbols.append(symbol)
                        logger.error(f"[ERR] {symbol}: {e}")

            # Checkpoint every N batches
            if batch_num % (self.config.checkpoint_interval // self.config.batch_size) == 0:
                self.save_checkpoint()
                elapsed = (time.time() - start_time) / 3600
                progress_pct = len(self.progress.completed_symbols) / self.progress.total_symbols * 100
                logger.info(f"\n[CHECKPOINT] {progress_pct:.1f}% complete | {elapsed:.2f} hours elapsed")

        # Final checkpoint
        self.save_checkpoint()

        # Summary
        elapsed_hours = (time.time() - start_time) / 3600

        summary = {
            "total_symbols": self.progress.total_symbols,
            "completed": len(self.progress.completed_symbols),
            "failed": len(self.progress.failed_symbols),
            "models_saved": self.progress.models_saved,
            "elapsed_hours": elapsed_hours,
            "symbols_per_hour": len(self.progress.completed_symbols) / max(elapsed_hours, 0.01),
        }

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total symbols: {summary['total_symbols']}")
        logger.info(f"Completed: {summary['completed']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Models saved: {summary['models_saved']}")
        logger.info(f"Time elapsed: {summary['elapsed_hours']:.2f} hours")
        logger.info(f"Speed: {summary['symbols_per_hour']:.1f} symbols/hour")

        return summary


# =============================================================================
# CLI INTERFACE
# =============================================================================

def run_massive_training(
    max_workers: int = None,
    batch_size: int = None,
    resume: bool = True,
):
    """Run massive training from command line.
    """
    # Auto-detect workers
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)

    config = TrainingConfig(
        max_workers=max_workers,
        batch_size=batch_size or 50,
        use_behavioral=True,
        use_technical=True,
    )

    logger.info("\n" + "=" * 70)
    logger.info("MASSIVE PARALLEL TRAINING SYSTEM")
    logger.info("=" * 70)
    logger.info(f"Workers: {config.max_workers}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Features: Technical={config.use_technical}, Behavioral={config.use_behavioral}")
    logger.info("=" * 70 + "\n")

    trainer = MassiveTrainer(config)
    return trainer.train_universe(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Massive Parallel Training")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=50, help="Symbols per batch")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore checkpoint)")

    args = parser.parse_args()

    run_massive_training(
        max_workers=args.workers,
        batch_size=args.batch_size,
        resume=not args.fresh,
    )



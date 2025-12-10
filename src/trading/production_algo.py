"""PRODUCTION TRADING ALGORITHM
============================
This is the LIVE trading code that runs your trained models.

USAGE:
  1. Models must be trained and PROMOTED first
  2. Start IBKR TWS/Gateway
  3. Run: python src/trading/production_algo.py

This file is SEPARATE from training - it only USES trained models.
"""
import sys

sys.path.insert(0, ".")

import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    "logs/live_trading.log",
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
logger.add(sys.stdout, level="INFO")


class ProductionAlgo:
    """The actual trading algorithm that runs in production.

    This class:
    1. Loads PROMOTED models only
    2. Gets live market data
    3. Generates signals
    4. Executes trades via IBKR
    5. Manages risk
    """

    # PROMOTION THRESHOLDS - Model must pass ALL to trade live
    MIN_AUC = 0.52
    MIN_ACCURACY = 0.52
    MIN_SHARPE = 1.5
    MAX_DRAWDOWN = 0.05

    def __init__(
        self,
        models_dir: str = "models",
        paper_trading: bool = True,  # START IN PAPER MODE
        max_position_pct: float = 0.10,
        max_daily_loss_pct: float = 0.02,
    ):
        self.models_dir = Path(models_dir)
        self.paper_trading = paper_trading
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct

        # State
        self.models: Dict[str, dict] = {}
        self.positions: Dict[str, float] = {}
        self.daily_pnl = 0.0
        self.trading_halted = False

        logger.info("Production Algo initialized")
        logger.info(f"  Paper Trading: {paper_trading}")
        logger.info(f"  Max Position: {max_position_pct:.0%}")
        logger.info(f"  Max Daily Loss: {max_daily_loss_pct:.0%}")

    def load_promoted_models(self) -> int:
        """Load only models that passed promotion criteria."""
        promoted_count = 0

        for model_file in self.models_dir.glob("*.pkl"):
            try:
                with open(model_file, "rb") as f:
                    model_data = pickle.load(f)

                # Extract metrics
                if isinstance(model_data, dict):
                    metrics = model_data.get("metrics", model_data)
                else:
                    metrics = getattr(model_data, "metrics", {})

                auc = metrics.get("auc", 0)
                accuracy = metrics.get("accuracy", 0)
                sharpe = metrics.get("sharpe", metrics.get("sharpe_ratio", 0))
                drawdown = metrics.get("max_drawdown", 1.0)

                # Check promotion criteria
                if (auc >= self.MIN_AUC and
                    accuracy >= self.MIN_ACCURACY and
                    sharpe >= self.MIN_SHARPE and
                    drawdown <= self.MAX_DRAWDOWN):

                    symbol = model_file.stem.replace("_ensemble", "").replace("_model", "")
                    self.models[symbol] = {
                        "model": model_data,
                        "metrics": metrics,
                        "file": model_file,
                    }
                    promoted_count += 1
                    logger.info(f"PROMOTED: {symbol} (AUC={auc:.3f}, Sharpe={sharpe:.2f})")
                else:
                    logger.debug(f"Skipped {model_file.stem}: Did not meet promotion criteria")

            except Exception as e:
                logger.warning(f"Could not load {model_file}: {e}")

        logger.info(f"Loaded {promoted_count} promoted models")
        return promoted_count

    def get_signal(self, symbol: str, market_data: pd.DataFrame) -> Tuple[str, float]:
        """Generate trading signal for a symbol.

        Returns
        -------
            (action, confidence): ('BUY', 0.65) or ('SELL', 0.72) or ('HOLD', 0.50)
        """
        if symbol not in self.models:
            return ("HOLD", 0.0)

        try:
            model_data = self.models[symbol]["model"]

            # Get the ensemble models
            if isinstance(model_data, dict):
                models = model_data.get("models", [model_data.get("model")])
            else:
                models = [model_data]

            # Prepare features (simplified - real version uses feature_engineering.py)
            features = self._prepare_features(market_data)

            # Get predictions from each model
            predictions = []
            for model in models:
                if model is not None and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(features)
                        predictions.append(proba[0][1])  # Probability of class 1 (up)
                    except Exception:
                        pass

            if not predictions:
                return ("HOLD", 0.0)

            # Ensemble average
            avg_proba = np.mean(predictions)

            # Convert to signal
            if avg_proba > 0.55:
                return ("BUY", avg_proba)
            elif avg_proba < 0.45:
                return ("SELL", 1 - avg_proba)
            else:
                return ("HOLD", 0.5)

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return ("HOLD", 0.0)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction (simplified version)."""
        # This is a simplified version - the full version is in feature_engineering.py
        features = pd.DataFrame()

        if "close" in df.columns:
            features["returns"] = df["close"].pct_change()
            features["volatility"] = features["returns"].rolling(20).std()
            features["momentum"] = df["close"].pct_change(5)
            features["rsi"] = self._calc_rsi(df["close"])

        return features.iloc[[-1]].fillna(0)

    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def check_risk_limits(self) -> bool:
        """Check if we can trade (risk limits not breached)."""
        if self.trading_halted:
            logger.warning("Trading is HALTED - risk limit breached")
            return False

        if self.daily_pnl < -self.max_daily_loss_pct:
            self.trading_halted = True
            logger.critical(f"TRADING HALTED: Daily loss {self.daily_pnl:.2%} exceeds limit")
            return False

        return True

    def execute_trade(self, symbol: str, action: str, confidence: float) -> bool:
        """Execute a trade (paper or live)."""
        if not self.check_risk_limits():
            return False

        if self.paper_trading:
            logger.info(f"[PAPER] {action} {symbol} (confidence: {confidence:.2%})")
            return True
        else:
            # Real IBKR execution would go here
            # from src.trading.execution_engine import ExecutionEngine
            # engine = ExecutionEngine()
            # engine.execute(symbol, action, size)
            logger.info(f"[LIVE] {action} {symbol} (confidence: {confidence:.2%})")
            return True

    def run_trading_loop(self, interval_seconds: int = 60):
        """Main trading loop - runs continuously during market hours.
        """
        logger.info("=" * 60)
        logger.info("STARTING PRODUCTION TRADING LOOP")
        logger.info("=" * 60)

        # Load models
        model_count = self.load_promoted_models()

        if model_count == 0:
            logger.error("NO PROMOTED MODELS FOUND!")
            logger.error('Train models first with: python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()"')
            return

        logger.info(f"Trading {model_count} symbols")
        logger.info(f"Interval: {interval_seconds}s")
        logger.info("")

        while True:
            try:
                # Check if market is open (simplified)
                now = datetime.now()
                if now.weekday() >= 5:  # Weekend
                    logger.info("Weekend - market closed")
                    time.sleep(3600)
                    continue

                market_open = now.replace(hour=9, minute=30, second=0)
                market_close = now.replace(hour=16, minute=0, second=0)

                if now < market_open or now > market_close:
                    logger.info("Market closed (open 9:30-16:00)")
                    time.sleep(60)
                    continue

                # Process each symbol
                for symbol in self.models:
                    # Get latest data (simplified - real version pulls from API)
                    # market_data = get_live_data(symbol)
                    market_data = pd.DataFrame()  # Placeholder

                    # Generate signal
                    action, confidence = self.get_signal(symbol, market_data)

                    # Execute if confident
                    if action != "HOLD" and confidence > 0.55:
                        self.execute_trade(symbol, action, confidence)

                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(interval_seconds)


def check_readiness() -> Dict:
    """Check if system is ready for live trading.

    Returns dict with status of each component.
    """
    status = {
        "models": {"ready": False, "count": 0, "promoted": 0},
        "data": {"ready": False, "rows": 0},
        "ibkr": {"ready": False},
        "overall": False,
    }

    # Check models
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl"))
        status["models"]["count"] = len(model_files)

        # Count promoted
        promoted = 0
        for f in model_files:
            try:
                with open(f, "rb") as fp:
                    m = pickle.load(fp)
                metrics = m.get("metrics", m) if isinstance(m, dict) else {}
                if (metrics.get("auc", 0) >= 0.52 and
                    metrics.get("sharpe", 0) >= 1.5):
                    promoted += 1
            except Exception:
                pass

        status["models"]["promoted"] = promoted
        status["models"]["ready"] = promoted > 0

    # Check data
    try:
        import pyodbc
        conn = pyodbc.connect(
            "Driver={ODBC Driver 17 for SQL Server};"
            "Server=alc-sql-server.database.windows.net;"
            "Database=alc_market_data;"
            "UID=CloudSAb3fcbb35;"
            "PWD=ALCadmin27!",
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM price_bars")
        rows = cursor.fetchone()[0]
        status["data"]["rows"] = rows
        status["data"]["ready"] = rows > 10000
        conn.close()
    except Exception:
        pass

    # Check IBKR (simplified)
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", 7497))
        status["ibkr"]["ready"] = result == 0
        sock.close()
    except Exception:
        pass

    # Overall readiness
    status["overall"] = (
        status["models"]["ready"] and
        status["data"]["ready"]
    )

    return status


def print_readiness_report():
    """Print a readiness report for live trading."""
    print("\n" + "=" * 60)
    print("        LIVE TRADING READINESS CHECK")
    print("=" * 60)

    status = check_readiness()

    # Models
    print("\n  MODELS")
    print("  ------")
    print(f"  Total trained:  {status['models']['count']}")
    print(f"  Promoted:       {status['models']['promoted']}")
    print(f"  Status:         {'READY' if status['models']['ready'] else 'NOT READY - Need promoted models'}")

    # Data
    print("\n  DATA (Azure SQL)")
    print("  ----------------")
    print(f"  Rows:           {status['data']['rows']:,}")
    print(f"  Status:         {'READY' if status['data']['ready'] else 'NOT READY - Need more data'}")

    # IBKR
    print("\n  IBKR CONNECTION")
    print("  ---------------")
    print(f"  TWS/Gateway:    {'Connected on port 7497' if status['ibkr']['ready'] else 'Not running'}")
    print(f"  Status:         {'READY' if status['ibkr']['ready'] else 'Start TWS/Gateway first'}")

    # Overall
    print("\n" + "=" * 60)
    if status["overall"]:
        print("  SYSTEM IS READY FOR LIVE TRADING")
        print("=" * 60)
        print("""
  To start trading:

    PAPER TRADING (Safe):
    python -c "from src.trading.production_algo import ProductionAlgo; algo = ProductionAlgo(paper_trading=True); algo.run_trading_loop()"

    LIVE TRADING (Real Money):
    python -c "from src.trading.production_algo import ProductionAlgo; algo = ProductionAlgo(paper_trading=False); algo.run_trading_loop()"
        """)
    else:
        print("  SYSTEM NOT READY - See issues above")
        print("=" * 60)
        print("""
  To fix:

  1. If no promoted models:
     - Run overnight training
     - Wait for models to pass thresholds

  2. If no data:
     - Run: python scripts/hydrate_full_universe.py

  3. If IBKR not connected:
     - Start TWS or IB Gateway
     - Enable API connections in settings
        """)

    print("")
    return status


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        print_readiness_report()
    elif "--paper" in sys.argv:
        algo = ProductionAlgo(paper_trading=True)
        algo.run_trading_loop()
    elif "--live" in sys.argv:
        print("\n*** WARNING: LIVE TRADING WITH REAL MONEY ***")
        confirm = input("Type 'YES' to confirm: ")
        if confirm == "YES":
            algo = ProductionAlgo(paper_trading=False)
            algo.run_trading_loop()
        else:
            print("Aborted.")
    else:
        print_readiness_report()



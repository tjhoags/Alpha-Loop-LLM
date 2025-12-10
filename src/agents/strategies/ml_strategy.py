"""Machine Learning-Based Trading Strategy
Author: Tom Hogan | Alpha Loop Capital, LLC

Advanced ML trading strategy that uses trained models to generate signals.

Features:
- Multiple ML models (ensemble voting)
- Confidence-based position sizing
- Risk management with stop-loss and take-profit
- Regime detection for model selection
- Real-time model updates
- Performance tracking

This strategy loads trained ML models and uses them to predict future price movements.
Positions are sized based on prediction confidence and current volatility.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MLTradingStrategy:
    """Machine Learning-based trading strategy.

    Uses trained ML models to generate trading signals with confidence-based sizing.
    """

    def __init__(
        self,
        model_dir: str = "data/models",
        min_confidence: float = 0.6,
        max_position_size: float = 0.15,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        use_ensemble: bool = True,
    ):
        """Initialize ML trading strategy.

        Args:
        ----
            model_dir: Directory containing trained models
            min_confidence: Minimum prediction confidence to trade (0-1)
            max_position_size: Maximum position size as fraction of portfolio
            stop_loss_pct: Stop loss percentage (e.g., 0.05 = 5%)
            take_profit_pct: Take profit percentage
            use_ensemble: Use ensemble of models vs single best model
        """
        self.model_dir = Path(model_dir)
        self.min_confidence = min_confidence
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_ensemble = use_ensemble

        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.positions: Dict[str, Dict] = {}

        logger.info("ML Trading Strategy initialized")

    def load_models(self, ticker: str, algorithm: Optional[str] = None) -> bool:
        """Load trained models for a ticker.

        Args:
        ----
            ticker: Stock ticker
            algorithm: Specific algorithm to load (None = load all)

        Returns:
        -------
            True if models loaded successfully
        """
        import json
        import pickle

        # Find model files for this ticker
        pattern = f"{ticker}_*.pkl"
        model_files = list(self.model_dir.glob(pattern))

        if not model_files:
            logger.error(f"No models found for {ticker} in {self.model_dir}")
            return False

        # Filter by algorithm if specified
        if algorithm:
            model_files = [f for f in model_files if algorithm in f.stem]

        loaded_count = 0

        for model_file in model_files:
            try:
                # Load model
                with open(model_file, "rb") as f:
                    model = pickle.load(f)

                # Load metadata if exists
                metadata_file = model_file.with_suffix(".json")
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                model_key = f"{ticker}_{model_file.stem}"
                self.models[model_key] = model
                self.model_metadata[model_key] = metadata

                loaded_count += 1
                logger.info(f"Loaded model: {model_file.name}")

            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")

        logger.info(f"Loaded {loaded_count} models for {ticker}")
        return loaded_count > 0

    def engineer_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for prediction (same as training).

        Args:
        ----
            df: DataFrame with OHLCV data

        Returns:
        -------
            DataFrame with engineered features
        """
        from src.models.train import MLTrainingPipeline

        pipeline = MLTrainingPipeline()
        features_df = pipeline.engineer_features(df)

        return features_df

    def predict(
        self,
        ticker: str,
        current_data: pd.DataFrame,
    ) -> Tuple[str, float, Dict]:
        """Generate prediction for a ticker.

        Args:
        ----
            ticker: Stock ticker
            current_data: Recent OHLCV data

        Returns:
        -------
            Tuple of (signal, confidence, details)
            signal: 'BUY', 'SELL', or 'HOLD'
            confidence: Prediction confidence (0-1)
            details: Additional information
        """
        # Engineer features
        try:
            features_df = self.engineer_features_for_prediction(current_data)
        except Exception as e:
            logger.error(f"Feature engineering failed for {ticker}: {e}")
            return "HOLD", 0.0, {"error": str(e)}

        # Get latest features for prediction
        if features_df.empty:
            return "HOLD", 0.0, {"error": "No features available"}

        latest_features = features_df.iloc[-1:]

        # Find models for this ticker
        ticker_models = {k: v for k, v in self.models.items() if k.startswith(ticker)}

        if not ticker_models:
            logger.warning(f"No models loaded for {ticker}")
            return "HOLD", 0.0, {"error": "No models loaded"}

        # Get feature columns from metadata
        predictions = []
        probabilities = []

        for model_key, model in ticker_models.items():
            metadata = self.model_metadata.get(model_key, {})
            feature_cols = metadata.get("features", [])

            if not feature_cols:
                logger.warning(f"No feature columns in metadata for {model_key}")
                continue

            # Select features
            try:
                X = latest_features[feature_cols].values

                # Make prediction
                pred = model.predict(X)[0]
                predictions.append(pred)

                # Get probability if available
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]
                    probabilities.append(proba)

            except Exception as e:
                logger.error(f"Prediction failed for {model_key}: {e}")
                continue

        if not predictions:
            return "HOLD", 0.0, {"error": "All predictions failed"}

        # Aggregate predictions
        if self.use_ensemble:
            # Majority voting for classification
            pred_value = int(np.round(np.mean(predictions)))

            # Average probabilities if available
            if probabilities:
                avg_proba = np.mean(probabilities, axis=0)
                confidence = float(avg_proba[pred_value])
            else:
                # Use agreement rate as confidence
                agreement = np.sum(np.array(predictions) == pred_value) / len(predictions)
                confidence = float(agreement)
        else:
            # Use best model (first one)
            pred_value = int(predictions[0])
            confidence = float(probabilities[0][pred_value]) if probabilities else 0.5

        # Convert prediction to signal
        signal = "BUY" if pred_value == 1 else "HOLD"

        # Additional details
        details = {
            "num_models": len(predictions),
            "raw_predictions": predictions,
            "ensemble_prediction": pred_value,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"{ticker} Prediction: {signal} (confidence={confidence:.2f})")

        return signal, confidence, details

    def calculate_position_size(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        current_price: float,
        portfolio_value: float,
        current_volatility: float = 0.02,
    ) -> float:
        """Calculate position size based on confidence and risk.

        Uses Kelly Criterion modified for ML confidence.

        Args:
        ----
            ticker: Stock ticker
            signal: 'BUY' or 'SELL'
            confidence: Prediction confidence (0-1)
            current_price: Current stock price
            portfolio_value: Total portfolio value
            current_volatility: Current price volatility

        Returns:
        -------
            Position size in dollars
        """
        if signal == "HOLD" or confidence < self.min_confidence:
            return 0.0

        # Base position size on confidence
        # Kelly Criterion: f = (p*b - q) / b
        # where p = win probability, q = 1-p, b = odds (assume 2:1)
        p = confidence
        q = 1 - p
        b = 2.0  # Assume 2:1 payoff ratio

        kelly_fraction = max(0, (p * b - q) / b)

        # Apply safety factor (half Kelly)
        position_fraction = kelly_fraction * 0.5

        # Cap at max position size
        position_fraction = min(position_fraction, self.max_position_size)

        # Adjust for volatility (reduce size in high volatility)
        volatility_adjustment = max(0.5, 1.0 - (current_volatility / 0.05))
        position_fraction *= volatility_adjustment

        # Calculate dollar amount
        position_size = portfolio_value * position_fraction

        logger.info(
            f"{ticker} Position Size: ${position_size:,.2f} "
            f"({position_fraction*100:.1f}% of portfolio, confidence={confidence:.2f})",
        )

        return position_size

    def generate_signals(
        self,
        tickers: List[str],
        market_data: Dict[str, pd.DataFrame],
        portfolio_value: float,
    ) -> List[Dict]:
        """Generate trading signals for multiple tickers.

        Args:
        ----
            tickers: List of stock tickers
            market_data: Dictionary mapping tickers to OHLCV DataFrames
            portfolio_value: Current portfolio value

        Returns:
        -------
            List of trading signals
        """
        signals = []

        for ticker in tickers:
            if ticker not in market_data:
                logger.warning(f"No market data for {ticker}")
                continue

            df = market_data[ticker]
            current_price = float(df["close"].iloc[-1])

            # Calculate current volatility
            returns = df["close"].pct_change()
            current_volatility = float(returns.tail(20).std())

            # Generate prediction
            signal, confidence, details = self.predict(ticker, df)

            # Calculate position size
            position_size = self.calculate_position_size(
                ticker=ticker,
                signal=signal,
                confidence=confidence,
                current_price=current_price,
                portfolio_value=portfolio_value,
                current_volatility=current_volatility,
            )

            # Calculate number of shares
            shares = int(position_size / current_price) if position_size > 0 else 0

            if shares > 0:
                signals.append({
                    "ticker": ticker,
                    "signal": signal,
                    "confidence": confidence,
                    "price": current_price,
                    "shares": shares,
                    "position_value": shares * current_price,
                    "stop_loss": current_price * (1 - self.stop_loss_pct),
                    "take_profit": current_price * (1 + self.take_profit_pct),
                    "timestamp": datetime.now(),
                    "details": details,
                })

                logger.info(
                    f"Signal: {signal} {shares} shares of {ticker} @ ${current_price:.2f} "
                    f"(confidence={confidence:.2f})",
                )

        return signals

    def check_risk_management(
        self,
        ticker: str,
        entry_price: float,
        current_price: float,
    ) -> Optional[str]:
        """Check if stop-loss or take-profit triggered.

        Args:
        ----
            ticker: Stock ticker
            entry_price: Entry price
            current_price: Current price

        Returns:
        -------
            'STOP_LOSS', 'TAKE_PROFIT', or None
        """
        pnl_pct = (current_price - entry_price) / entry_price

        if pnl_pct <= -self.stop_loss_pct:
            logger.warning(f"{ticker} STOP LOSS triggered: {pnl_pct*100:.2f}%")
            return "STOP_LOSS"

        if pnl_pct >= self.take_profit_pct:
            logger.info(f"{ticker} TAKE PROFIT triggered: {pnl_pct*100:.2f}%")
            return "TAKE_PROFIT"

        return None


def backtest_ml_strategy(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000.0,
) -> pd.DataFrame:
    """Backtest the ML trading strategy.

    Args:
    ----
        tickers: List of tickers to trade
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital

    Returns:
    -------
        DataFrame with backtest results
    """
    logger.info("="*70)
    logger.info("ML STRATEGY BACKTEST")
    logger.info("="*70)
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Capital: ${initial_capital:,.2f}")
    logger.info("="*70)

    # Initialize strategy
    strategy = MLTradingStrategy()

    # Load models for all tickers
    for ticker in tickers:
        strategy.load_models(ticker)

    # Load market data
    import yfinance as yf

    market_data = {}
    for ticker in tickers:
        logger.info(f"Loading data for {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        df.columns = df.columns.str.lower()
        market_data[ticker] = df

    # Run backtest
    portfolio_value = initial_capital
    positions = {}
    trades = []

    # Iterate through trading days
    trading_days = market_data[tickers[0]].index

    for current_date in trading_days[252:]:  # Start after 1 year for feature calculation
        logger.info(f"\nDate: {current_date.strftime('%Y-%m-%d')}")

        # Get data up to current date
        current_market_data = {
            ticker: df.loc[:current_date]
            for ticker, df in market_data.items()
        }

        # Generate signals
        signals = strategy.generate_signals(tickers, current_market_data, portfolio_value)

        # Execute trades
        for signal_dict in signals:
            ticker = signal_dict["ticker"]
            signal = signal_dict["signal"]
            shares = signal_dict["shares"]
            price = signal_dict["price"]

            if signal == "BUY" and ticker not in positions:
                # Open position
                cost = shares * price
                if cost <= portfolio_value:
                    positions[ticker] = {
                        "shares": shares,
                        "entry_price": price,
                        "entry_date": current_date,
                    }
                    portfolio_value -= cost

                    trades.append({
                        "date": current_date,
                        "ticker": ticker,
                        "action": "BUY",
                        "shares": shares,
                        "price": price,
                        "value": cost,
                    })

                    logger.info(f"  BUY {shares} {ticker} @ ${price:.2f}")

        # Check existing positions for exit signals
        for ticker in list(positions.keys()):
            if ticker in current_market_data:
                current_price = float(current_market_data[ticker]["close"].iloc[-1])
                entry_price = positions[ticker]["entry_price"]

                exit_signal = strategy.check_risk_management(ticker, entry_price, current_price)

                if exit_signal:
                    # Close position
                    shares = positions[ticker]["shares"]
                    proceeds = shares * current_price
                    portfolio_value += proceeds

                    pnl = proceeds - (shares * entry_price)

                    trades.append({
                        "date": current_date,
                        "ticker": ticker,
                        "action": exit_signal,
                        "shares": shares,
                        "price": current_price,
                        "value": proceeds,
                        "pnl": pnl,
                    })

                    logger.info(f"  {exit_signal} {shares} {ticker} @ ${current_price:.2f} (P&L: ${pnl:.2f})")

                    del positions[ticker]

    # Close all remaining positions
    for ticker, position in positions.items():
        current_price = float(market_data[ticker]["close"].iloc[-1])
        shares = position["shares"]
        proceeds = shares * current_price
        portfolio_value += proceeds

        pnl = proceeds - (shares * position["entry_price"])

        trades.append({
            "date": end_date,
            "ticker": ticker,
            "action": "CLOSE",
            "shares": shares,
            "price": current_price,
            "value": proceeds,
            "pnl": pnl,
        })

    # Create results DataFrame
    results_df = pd.DataFrame(trades)

    # Calculate performance metrics
    total_pnl = results_df["pnl"].sum() if "pnl" in results_df.columns else 0
    final_value = portfolio_value
    total_return = (final_value - initial_capital) / initial_capital

    logger.info("\n" + "="*70)
    logger.info("BACKTEST RESULTS")
    logger.info("="*70)
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Final Value: ${final_value:,.2f}")
    logger.info(f"Total P&L: ${total_pnl:,.2f}")
    logger.info(f"Total Return: {total_return*100:.2f}%")
    logger.info(f"Number of Trades: {len(results_df)}")
    logger.info("="*70)

    return results_df


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = "AAPL"

    # Initialize strategy
    strategy = MLTradingStrategy()

    # Load models
    if strategy.load_models(ticker):
        logger.info(f"Models loaded for {ticker}")

        # Example: Run backtest
        backtest_ml_strategy(
            tickers=[ticker],
            start_date=datetime.now() - timedelta(days=365*2),
            end_date=datetime.now(),
            initial_capital=100000.0,
        )
    else:
        logger.error(f"Failed to load models for {ticker}")
        logger.info("Run training first: python scripts/automated_training_workflow.py")

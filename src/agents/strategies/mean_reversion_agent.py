"""
Mean Reversion Agent with Regime Detection
Exploit short-term oversold/overbought conditions

Strategy:
- Identify oversold conditions using RSI, Z-score
- Enter when mean reversion likely
- Exit quickly (3-10 days typical hold)
- Use regime detection to avoid trending markets

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReversionSignal:
    """Mean reversion signal"""
    symbol: str
    z_score: float
    rsi: float
    regime: str
    signal_strength: float  # -1 to 1
    target_weight: float
    entry_price: float
    expected_days_to_revert: int
    confidence: float
    timestamp: datetime


class MeanReversionAgent:
    """
    Mean reversion strategy for choppy markets.

    Entry Triggers:
    - RSI < 30 (oversold) or RSI > 70 (overbought)
    - Z-score < -2 or > 2 (price deviation from mean)
    - Not in strong trend (regime = choppy)

    Exit Triggers:
    - Price returns to mean
    - Max hold period reached (10 days)
    - Stop loss (-5%)

    Performance Targets:
    - Win Rate: 65%+
    - Average Hold: 5 days
    - Max Drawdown: <10%
    """

    def __init__(
        self,
        rsi_period: int = 14,
        z_score_period: int = 20,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        z_score_threshold: float = 2.0,
        max_positions: int = 8,
        position_size: float = 0.08,  # 8% per position
        max_hold_days: int = 10,
        stop_loss: float = 0.05,
    ):
        self.rsi_period = rsi_period
        self.z_score_period = z_score_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.z_score_threshold = z_score_threshold
        self.max_positions = max_positions
        self.position_size = position_size
        self.max_hold_days = max_hold_days
        self.stop_loss = stop_loss

        self.entry_dates: Dict[str, datetime] = {}
        self.entry_prices: Dict[str, float] = {}

        logger.info(f"Mean Reversion Agent initialized: RSI={rsi_period}, Z-score={z_score_period}")

    def generate_signals(
        self,
        price_data: pd.DataFrame,
        current_positions: Dict[str, float],
        current_date: datetime
    ) -> Dict[str, float]:
        """Generate mean reversion signals"""

        signals = []

        for symbol in price_data.columns:
            try:
                # Skip if already at max hold period
                if symbol in self.entry_dates:
                    days_held = (current_date - self.entry_dates[symbol]).days
                    if days_held >= self.max_hold_days:
                        logger.info(f"{symbol}: Max hold period reached, closing")
                        continue

                signal = self._calculate_reversion_signal(
                    price_data[symbol],
                    symbol,
                    current_date
                )

                if signal:
                    signals.append(signal)

            except Exception as e:
                logger.warning(f"Error calculating reversion signal for {symbol}: {e}")

        # Select top signals by confidence
        signals.sort(key=lambda x: abs(x.signal_strength) * x.confidence, reverse=True)
        top_signals = signals[:self.max_positions]

        # Convert to weights
        target_weights = self._signals_to_weights(top_signals)

        # Check exit conditions for existing positions
        exit_symbols = self._check_exits(price_data, current_positions, current_date)
        for symbol in exit_symbols:
            target_weights[symbol] = 0.0

        logger.info(f"Generated {len(top_signals)} reversion signals, {len(exit_symbols)} exits")

        return target_weights

    def _calculate_reversion_signal(
        self,
        prices: pd.Series,
        symbol: str,
        current_date: datetime
    ) -> Optional[ReversionSignal]:
        """Calculate mean reversion signal"""

        if len(prices) < max(self.rsi_period, self.z_score_period) + 10:
            return None

        # Calculate RSI
        rsi = self._calculate_rsi(prices, self.rsi_period)

        # Calculate Z-score
        z_score = self._calculate_z_score(prices, self.z_score_period)

        # Detect regime
        regime = self._detect_regime(prices)

        # Only trade in choppy markets (avoid strong trends)
        if regime != "choppy":
            return None

        # Check for oversold (buy signal)
        if rsi < self.rsi_oversold and z_score < -self.z_score_threshold:
            signal_strength = -1.0 * (1.0 - rsi / 100)  # Stronger signal = lower RSI
            target_weight = self.position_size
            expected_days = self._estimate_reversion_time(prices)

        # Check for overbought (sell signal - or no position)
        elif rsi > self.rsi_overbought and z_score > self.z_score_threshold:
            signal_strength = 1.0 * (rsi / 100)
            target_weight = 0.0  # Close position if we have one
            expected_days = self._estimate_reversion_time(prices)

        else:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(rsi, z_score, regime)

        # Record entry
        if target_weight > 0 and symbol not in self.entry_dates:
            self.entry_dates[symbol] = current_date
            self.entry_prices[symbol] = prices.iloc[-1]

        return ReversionSignal(
            symbol=symbol,
            z_score=z_score,
            rsi=rsi,
            regime=regime,
            signal_strength=signal_strength,
            target_weight=target_weight,
            entry_price=prices.iloc[-1],
            expected_days_to_revert=expected_days,
            confidence=confidence,
            timestamp=current_date
        )

    def _calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1]

    def _calculate_z_score(self, prices: pd.Series, period: int) -> float:
        """Calculate Z-score (standard deviations from mean)"""
        mean = prices.tail(period).mean()
        std = prices.tail(period).std()

        if std == 0:
            return 0.0

        z_score = (prices.iloc[-1] - mean) / std
        return z_score

    def _detect_regime(self, prices: pd.Series) -> str:
        """Detect market regime"""
        if len(prices) < 50:
            return "choppy"

        # Linear regression slope
        x = np.arange(50)
        y = prices.tail(50).values
        slope = np.polyfit(x, y, 1)[0]

        # Volatility
        returns = prices.pct_change().tail(50)
        volatility = returns.std()

        # Strong trend if significant slope and low volatility
        if abs(slope) > 0.5 and volatility < 0.02:
            return "trending"
        else:
            return "choppy"

    def _estimate_reversion_time(self, prices: pd.Series) -> int:
        """Estimate days until mean reversion (heuristic)"""
        # Typical mean reversion: 3-7 days
        # More extreme deviation = faster reversion expected
        z_score = abs(self._calculate_z_score(prices, self.z_score_period))

        if z_score > 3.0:
            return 3
        elif z_score > 2.5:
            return 5
        else:
            return 7

    def _calculate_confidence(self, rsi: float, z_score: float, regime: str) -> float:
        """Calculate signal confidence"""
        # Higher confidence for:
        # - More extreme RSI
        # - Larger Z-score deviation
        # - Choppy regime

        rsi_confidence = min(abs(50 - rsi) / 50, 1.0)  # Distance from neutral 50
        z_confidence = min(abs(z_score) / 3.0, 1.0)  # Larger deviation = higher confidence
        regime_confidence = 1.0 if regime == "choppy" else 0.5

        confidence = (
            0.40 * rsi_confidence +
            0.40 * z_confidence +
            0.20 * regime_confidence
        )

        return np.clip(confidence, 0.0, 1.0)

    def _check_exits(
        self,
        price_data: pd.DataFrame,
        current_positions: Dict[str, float],
        current_date: datetime
    ) -> List[str]:
        """Check if any positions should be exited"""
        exit_symbols = []

        for symbol, shares in current_positions.items():
            if shares == 0 or symbol not in price_data.columns:
                continue

            current_price = price_data[symbol].iloc[-1]

            # Exit if max hold period reached
            if symbol in self.entry_dates:
                days_held = (current_date - self.entry_dates[symbol]).days
                if days_held >= self.max_hold_days:
                    exit_symbols.append(symbol)
                    logger.info(f"{symbol}: Max hold period exit ({days_held} days)")
                    continue

            # Exit if stop loss hit
            if symbol in self.entry_prices:
                entry_price = self.entry_prices[symbol]
                return_pct = (current_price / entry_price) - 1.0

                if return_pct <= -self.stop_loss:
                    exit_symbols.append(symbol)
                    logger.info(f"{symbol}: Stop loss exit ({return_pct:.2%})")
                    continue

                # Exit if reverted to mean (RSI back to neutral)
                rsi = self._calculate_rsi(price_data[symbol], self.rsi_period)
                if 45 <= rsi <= 55:  # Neutral zone
                    exit_symbols.append(symbol)
                    logger.info(f"{symbol}: Mean reversion complete ({return_pct:.2%})")

        # Clear entry tracking for exited positions
        for symbol in exit_symbols:
            if symbol in self.entry_dates:
                del self.entry_dates[symbol]
            if symbol in self.entry_prices:
                del self.entry_prices[symbol]

        return exit_symbols

    def _signals_to_weights(self, signals: List[ReversionSignal]) -> Dict[str, float]:
        """Convert signals to portfolio weights"""
        weights = {}
        for signal in signals:
            weights[signal.symbol] = signal.target_weight
        return weights


# Example usage
if __name__ == "__main__":
    agent = MeanReversionAgent(
        max_positions=8,
        position_size=0.08,
        max_hold_days=10
    )

    # Simulate price data
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

    np.random.seed(42)
    price_data = pd.DataFrame({
        symbol: 100 + 10 * np.sin(np.arange(len(dates)) * 0.1) + np.random.randn(len(dates)) * 2
        for symbol in symbols
    }, index=dates)

    # Generate signals
    target_weights = agent.generate_signals(price_data, {}, dates[-1])

    print("\n=== Mean Reversion Agent ===")
    print(f"Target Weights: {target_weights}")

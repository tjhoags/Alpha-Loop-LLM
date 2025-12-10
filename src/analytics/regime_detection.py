"""
Market Regime Detection System

Identifying market regimes (bull, bear, high vol, low vol, crisis) for adaptive strategy.

Why Basic Approaches Fail:
- Fixed strategies break in regime changes
- Simple moving averages lag too much
- Ignoring regime shifts causes catastrophic losses
- Static risk management fails in crises

Our Creative Philosophy:
- Hidden Markov Models for statistical regime detection
- Change-point detection for regime shifts
- Volatility regime classification (VIX-based)
- Correlation regime monitoring (breakdown = crisis)
- Multi-timeframe regime analysis
- Forward-looking indicators (not just backward)

Elite institutions use regime detection:
- Bridgewater: Regime-dependent portfolios (All Weather)
- AQR: Regime-conditional factor exposures
- Two Sigma: Real-time regime classification for execution

Author: Tom Hogan
Date: 2025-12-09
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""

    BULL_LOW_VOL = "bull_low_vol"  # Rising + Low volatility (best)
    BULL_HIGH_VOL = "bull_high_vol"  # Rising + High volatility (choppy)
    BEAR_LOW_VOL = "bear_low_vol"  # Falling + Low volatility (grinding down)
    BEAR_HIGH_VOL = "bear_high_vol"  # Falling + High volatility (crisis)
    SIDEWAYS_LOW_VOL = "sideways_low_vol"  # Range-bound + Low vol
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"  # Range-bound + High vol
    CRISIS = "crisis"  # Extreme volatility + correlation breakdown
    RECOVERY = "recovery"  # Post-crisis recovery


class VolatilityRegime(Enum):
    """Volatility regime classifications"""

    EXTREMELY_LOW = "extremely_low"  # VIX < 12
    LOW = "low"  # VIX 12-16
    NORMAL = "normal"  # VIX 16-20
    ELEVATED = "elevated"  # VIX 20-30
    HIGH = "high"  # VIX 30-40
    EXTREME = "extreme"  # VIX > 40 (panic)


class CorrelationRegime(Enum):
    """Correlation regime classifications"""

    DIVERSIFIED = "diversified"  # Low avg correlation (< 0.3)
    NORMAL = "normal"  # Normal correlation (0.3 - 0.6)
    ELEVATED = "elevated"  # High correlation (0.6 - 0.8)
    BREAKDOWN = "breakdown"  # Correlation breakdown (> 0.8, crisis)


class TrendRegime(Enum):
    """Trend regime classifications"""

    STRONG_UPTREND = "strong_uptrend"  # Clear trend up
    WEAK_UPTREND = "weak_uptrend"  # Choppy trend up
    SIDEWAYS = "sideways"  # No clear trend
    WEAK_DOWNTREND = "weak_downtrend"  # Choppy trend down
    STRONG_DOWNTREND = "strong_downtrend"  # Clear trend down


@dataclass
class RegimeDetectionResult:
    """Market regime detection result"""

    current_regime: MarketRegime
    volatility_regime: VolatilityRegime
    correlation_regime: CorrelationRegime
    trend_regime: TrendRegime
    regime_probability: float  # Probability of current regime (HMM)
    time_in_regime: int  # Days in current regime
    change_point_detected: bool  # Recent regime change?
    expected_regime_duration: int  # Expected days remaining in regime
    forward_indicators: Dict[str, float]  # Leading indicators
    regime_history: List[Tuple[datetime, MarketRegime]]  # Recent regime changes
    metadata: Dict[str, Any] = None
    timestamp: datetime = datetime.now()

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RegimeCharacteristics:
    """Characteristics of a market regime"""

    regime: MarketRegime
    avg_return: float  # Average daily return in this regime
    avg_volatility: float  # Average volatility
    avg_sharpe: float  # Average Sharpe ratio
    avg_max_drawdown: float  # Average max drawdown
    avg_correlation: float  # Average cross-asset correlation
    avg_duration: int  # Average regime duration (days)
    transition_probabilities: Dict[MarketRegime, float]  # Transition matrix


class MarketRegimeDetector:
    """
    Market regime detection using multiple methods.

    Features:
    - Hidden Markov Models for statistical regime detection
    - Change-point detection for regime shifts
    - Volatility regime classification
    - Correlation breakdown detection
    - Multi-timeframe analysis
    """

    def __init__(
        self,
        lookback_days: int = 252,  # 1 year lookback
        n_regimes: int = 4,  # Number of HMM states
        vix_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.lookback_days = lookback_days
        self.n_regimes = n_regimes
        self.vix_thresholds = vix_thresholds or {
            "extremely_low": 12,
            "low": 16,
            "normal": 20,
            "elevated": 30,
            "high": 40,
        }

        # State tracking
        self.current_regime: Optional[MarketRegime] = None
        self.regime_start_date: Optional[datetime] = None
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []

        # Cached regime characteristics (learned from history)
        self.regime_characteristics: Dict[MarketRegime, RegimeCharacteristics] = {}

        logger.info("MarketRegimeDetector initialized")

    def detect_regime(
        self,
        returns: pd.Series,  # Market returns (SPY or portfolio)
        vix: Optional[pd.Series] = None,  # VIX data
        cross_asset_returns: Optional[pd.DataFrame] = None,  # For correlation
        prices: Optional[pd.Series] = None,  # For trend detection
    ) -> RegimeDetectionResult:
        """
        Detect current market regime.

        Args:
            returns: Market returns (daily)
            vix: VIX levels (optional)
            cross_asset_returns: Cross-asset returns for correlation (optional)
            prices: Price series for trend detection (optional)

        Returns:
            RegimeDetectionResult with current regime and diagnostics
        """
        # Detect volatility regime
        vol_regime = self._detect_volatility_regime(returns, vix)

        # Detect correlation regime
        corr_regime = self._detect_correlation_regime(cross_asset_returns or returns.to_frame())

        # Detect trend regime
        trend_regime = self._detect_trend_regime(prices if prices is not None else returns.cumsum())

        # Combine into overall market regime
        market_regime = self._classify_market_regime(vol_regime, corr_regime, trend_regime)

        # Hidden Markov Model for regime probability
        regime_prob, expected_duration = self._hmm_regime_probability(
            returns, market_regime
        )

        # Change-point detection
        change_point_detected = self._detect_change_point(returns)

        # Update regime tracking
        if self.current_regime != market_regime:
            if self.current_regime is not None:
                self.regime_history.append((datetime.now(), self.current_regime))
            self.current_regime = market_regime
            self.regime_start_date = datetime.now()

        # Time in regime
        time_in_regime = (
            (datetime.now() - self.regime_start_date).days
            if self.regime_start_date
            else 0
        )

        # Forward-looking indicators
        forward_indicators = self._calculate_forward_indicators(
            returns, vix, cross_asset_returns
        )

        return RegimeDetectionResult(
            current_regime=market_regime,
            volatility_regime=vol_regime,
            correlation_regime=corr_regime,
            trend_regime=trend_regime,
            regime_probability=regime_prob,
            time_in_regime=time_in_regime,
            change_point_detected=change_point_detected,
            expected_regime_duration=expected_duration,
            forward_indicators=forward_indicators,
            regime_history=self.regime_history[-10:],  # Last 10 regime changes
        )

    def _detect_volatility_regime(
        self, returns: pd.Series, vix: Optional[pd.Series] = None
    ) -> VolatilityRegime:
        """Detect volatility regime using VIX or realized volatility"""
        if vix is not None and len(vix) > 0:
            current_vix = vix.iloc[-1]
        else:
            # Use realized volatility as proxy
            realized_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            current_vix = realized_vol

        # Classify based on VIX thresholds
        if current_vix < self.vix_thresholds["extremely_low"]:
            return VolatilityRegime.EXTREMELY_LOW
        elif current_vix < self.vix_thresholds["low"]:
            return VolatilityRegime.LOW
        elif current_vix < self.vix_thresholds["normal"]:
            return VolatilityRegime.NORMAL
        elif current_vix < self.vix_thresholds["elevated"]:
            return VolatilityRegime.ELEVATED
        elif current_vix < self.vix_thresholds["high"]:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    def _detect_correlation_regime(
        self, cross_asset_returns: pd.DataFrame
    ) -> CorrelationRegime:
        """Detect correlation regime (diversification breakdown = crisis)"""
        if len(cross_asset_returns.columns) < 2:
            return CorrelationRegime.NORMAL

        # Rolling correlation matrix
        window = min(60, len(cross_asset_returns))  # 60 days or less
        recent_returns = cross_asset_returns.iloc[-window:]

        # Calculate average pairwise correlation
        corr_matrix = recent_returns.corr()
        n_assets = len(corr_matrix)

        # Average correlation (excluding diagonal)
        avg_corr = (corr_matrix.sum().sum() - n_assets) / (n_assets * (n_assets - 1))

        # Classify
        if avg_corr < 0.3:
            return CorrelationRegime.DIVERSIFIED
        elif avg_corr < 0.6:
            return CorrelationRegime.NORMAL
        elif avg_corr < 0.8:
            return CorrelationRegime.ELEVATED
        else:
            return CorrelationRegime.BREAKDOWN  # Crisis!

    def _detect_trend_regime(self, prices: pd.Series) -> TrendRegime:
        """Detect trend regime using multiple moving averages"""
        if len(prices) < 200:
            return TrendRegime.SIDEWAYS

        current_price = prices.iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1]
        sma_200 = prices.rolling(200).mean().iloc[-1]

        # Trend strength (% above/below moving average)
        trend_strength_50 = (current_price - sma_50) / sma_50
        trend_strength_200 = (current_price - sma_200) / sma_200

        # Golden cross / death cross
        golden_cross = sma_50 > sma_200  # Bullish
        death_cross = sma_50 < sma_200  # Bearish

        # Classify
        if golden_cross and trend_strength_50 > 0.05:
            return TrendRegime.STRONG_UPTREND
        elif golden_cross and trend_strength_50 > 0:
            return TrendRegime.WEAK_UPTREND
        elif death_cross and trend_strength_50 < -0.05:
            return TrendRegime.STRONG_DOWNTREND
        elif death_cross and trend_strength_50 < 0:
            return TrendRegime.WEAK_DOWNTREND
        else:
            return TrendRegime.SIDEWAYS

    def _classify_market_regime(
        self,
        vol_regime: VolatilityRegime,
        corr_regime: CorrelationRegime,
        trend_regime: TrendRegime,
    ) -> MarketRegime:
        """Combine sub-regimes into overall market regime"""
        # Crisis detection (highest priority)
        if corr_regime == CorrelationRegime.BREAKDOWN:
            return MarketRegime.CRISIS
        if vol_regime == VolatilityRegime.EXTREME:
            return MarketRegime.CRISIS

        # Recovery detection
        if (
            vol_regime in [VolatilityRegime.ELEVATED, VolatilityRegime.HIGH]
            and trend_regime
            in [TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND]
            and corr_regime == CorrelationRegime.ELEVATED
        ):
            return MarketRegime.RECOVERY

        # Bull regimes
        if trend_regime in [TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND]:
            if vol_regime in [VolatilityRegime.EXTREMELY_LOW, VolatilityRegime.LOW]:
                return MarketRegime.BULL_LOW_VOL
            else:
                return MarketRegime.BULL_HIGH_VOL

        # Bear regimes
        if trend_regime in [TrendRegime.STRONG_DOWNTREND, TrendRegime.WEAK_DOWNTREND]:
            if vol_regime in [VolatilityRegime.EXTREMELY_LOW, VolatilityRegime.LOW]:
                return MarketRegime.BEAR_LOW_VOL
            else:
                return MarketRegime.BEAR_HIGH_VOL

        # Sideways regimes
        if vol_regime in [VolatilityRegime.EXTREMELY_LOW, VolatilityRegime.LOW]:
            return MarketRegime.SIDEWAYS_LOW_VOL
        else:
            return MarketRegime.SIDEWAYS_HIGH_VOL

    def _hmm_regime_probability(
        self, returns: pd.Series, current_regime: MarketRegime
    ) -> Tuple[float, int]:
        """
        Use Hidden Markov Model to estimate regime probability.

        Simplified implementation using clustering as proxy.
        Full HMM would use hmmlearn library.
        """
        # Features for clustering
        window = min(self.lookback_days, len(returns))
        recent_returns = returns.iloc[-window:]

        # Feature engineering
        features = []
        for i in range(20, len(recent_returns)):
            window_returns = recent_returns.iloc[i - 20 : i]
            features.append(
                [
                    window_returns.mean(),  # Average return
                    window_returns.std(),  # Volatility
                    window_returns.skew(),  # Skew
                    window_returns.kurtosis(),  # Kurtosis
                ]
            )

        if len(features) < self.n_regimes:
            return 0.5, 60  # Default: 50% confidence, 60 day duration

        X = np.array(features)

        # K-means clustering (proxy for HMM states)
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Current state
        current_state = labels[-1]

        # State probability (% of recent observations in this state)
        recent_labels = labels[-20:]  # Last 20 observations
        state_prob = (recent_labels == current_state).sum() / len(recent_labels)

        # Expected duration (average run length)
        state_runs = []
        current_run = 1
        for i in range(1, len(labels)):
            if labels[i] == labels[i - 1]:
                current_run += 1
            else:
                if labels[i - 1] == current_state:
                    state_runs.append(current_run)
                current_run = 1

        expected_duration = int(np.mean(state_runs)) if state_runs else 60

        return float(state_prob), expected_duration

    def _detect_change_point(self, returns: pd.Series, threshold: float = 2.0) -> bool:
        """
        Detect regime change using CUSUM (Cumulative Sum) method.

        Change point = sudden shift in mean or variance.
        """
        window = min(60, len(returns))
        recent_returns = returns.iloc[-window:]

        if len(recent_returns) < 30:
            return False

        # Split into two halves
        mid = len(recent_returns) // 2
        first_half = recent_returns.iloc[:mid]
        second_half = recent_returns.iloc[mid:]

        # Test for mean change
        mean_diff = abs(second_half.mean() - first_half.mean())
        pooled_std = np.sqrt(
            (first_half.var() * len(first_half) + second_half.var() * len(second_half))
            / (len(first_half) + len(second_half))
        )

        if pooled_std > 0:
            t_stat = mean_diff / (pooled_std * np.sqrt(1 / len(first_half) + 1 / len(second_half)))
            if t_stat > threshold:
                return True

        # Test for variance change
        f_stat = second_half.var() / first_half.var() if first_half.var() > 0 else 1.0
        if f_stat > 2.0 or f_stat < 0.5:
            return True

        return False

    def _calculate_forward_indicators(
        self,
        returns: pd.Series,
        vix: Optional[pd.Series],
        cross_asset_returns: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Calculate forward-looking regime indicators.

        Leading indicators (not lagging):
        - VIX term structure (contango vs backwardation)
        - Credit spreads
        - Yield curve slope
        - Put/call ratio
        - Market breadth
        """
        indicators = {}

        # VIX term structure (if available)
        if vix is not None and len(vix) > 20:
            # Simplified: compare current VIX to 20-day average
            current_vix = vix.iloc[-1]
            avg_vix = vix.rolling(20).mean().iloc[-1]
            indicators["vix_vs_average"] = (current_vix - avg_vix) / avg_vix

            # VIX trend (rising = fear building)
            vix_change = (vix.iloc[-1] - vix.iloc[-5]) / vix.iloc[-5]
            indicators["vix_5day_change"] = vix_change

        # Momentum (20-day return)
        if len(returns) > 20:
            momentum = returns.iloc[-20:].sum()
            indicators["momentum_20d"] = momentum

        # Volatility trend
        if len(returns) > 60:
            recent_vol = returns.iloc[-20:].std() * np.sqrt(252)
            past_vol = returns.iloc[-60:-20].std() * np.sqrt(252)
            indicators["volatility_trend"] = (recent_vol - past_vol) / past_vol if past_vol > 0 else 0.0

        # Correlation trend (breakdown building?)
        if cross_asset_returns is not None and len(cross_asset_returns) > 60:
            recent_corr = cross_asset_returns.iloc[-20:].corr().values
            past_corr = cross_asset_returns.iloc[-60:-20].corr().values

            avg_recent = (recent_corr.sum() - len(recent_corr)) / (
                len(recent_corr) * (len(recent_corr) - 1)
            )
            avg_past = (past_corr.sum() - len(past_corr)) / (
                len(past_corr) * (len(past_corr) - 1)
            )

            indicators["correlation_trend"] = (avg_recent - avg_past) / avg_past if avg_past > 0 else 0.0

        # Drawdown (current drawdown from peak)
        if len(returns) > 0:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            indicators["current_drawdown"] = drawdown.iloc[-1]

        return indicators

    def get_regime_strategy_adjustments(
        self, regime: MarketRegime
    ) -> Dict[str, Any]:
        """
        Get recommended strategy adjustments for current regime.

        Different regimes require different approaches:
        - Bull Low Vol: Maximize exposure, momentum strategies
        - Bear High Vol: Reduce exposure, defensive positioning
        - Crisis: Cash/bonds, no leverage, survive
        """
        adjustments = {
            MarketRegime.BULL_LOW_VOL: {
                "equity_allocation": 1.0,  # Fully invested
                "leverage": 1.2,  # Modest leverage OK
                "strategies": ["momentum", "growth", "breakout"],
                "risk_level": "moderate",
                "rebalance_frequency": "monthly",
            },
            MarketRegime.BULL_HIGH_VOL: {
                "equity_allocation": 0.8,  # Slightly reduce
                "leverage": 1.0,  # No leverage
                "strategies": ["quality", "low_vol", "dividend"],
                "risk_level": "moderate_low",
                "rebalance_frequency": "weekly",
            },
            MarketRegime.BEAR_LOW_VOL: {
                "equity_allocation": 0.5,  # Defensive
                "leverage": 0.0,  # No leverage
                "strategies": ["defensive", "dividend", "market_neutral"],
                "risk_level": "low",
                "rebalance_frequency": "weekly",
            },
            MarketRegime.BEAR_HIGH_VOL: {
                "equity_allocation": 0.3,  # Very defensive
                "leverage": 0.0,
                "strategies": ["market_neutral", "short", "cash"],
                "risk_level": "very_low",
                "rebalance_frequency": "daily",
            },
            MarketRegime.CRISIS: {
                "equity_allocation": 0.2,  # Survival mode
                "leverage": 0.0,
                "strategies": ["cash", "treasuries", "gold"],
                "risk_level": "minimum",
                "rebalance_frequency": "daily",
                "special_instructions": "CAPITAL PRESERVATION MODE - Survive to trade another day",
            },
            MarketRegime.RECOVERY: {
                "equity_allocation": 0.6,  # Gradually re-risk
                "leverage": 0.0,
                "strategies": ["value", "cyclical", "small_cap"],
                "risk_level": "moderate_low",
                "rebalance_frequency": "weekly",
            },
            MarketRegime.SIDEWAYS_LOW_VOL: {
                "equity_allocation": 0.7,
                "leverage": 1.0,
                "strategies": ["mean_reversion", "pairs", "options_income"],
                "risk_level": "moderate",
                "rebalance_frequency": "monthly",
            },
            MarketRegime.SIDEWAYS_HIGH_VOL: {
                "equity_allocation": 0.5,
                "leverage": 0.0,
                "strategies": ["market_neutral", "pairs", "defensive"],
                "risk_level": "low",
                "rebalance_frequency": "weekly",
            },
        }

        return adjustments.get(regime, adjustments[MarketRegime.SIDEWAYS_LOW_VOL])


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Sample data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    n_days = len(dates)

    # Simulate market returns with regime changes
    returns_data = []
    for i in range(n_days):
        if i < 500:  # Bull low vol
            returns_data.append(np.random.randn() * 0.008 + 0.0005)
        elif i < 700:  # Crisis
            returns_data.append(np.random.randn() * 0.03 - 0.002)
        elif i < 1000:  # Recovery
            returns_data.append(np.random.randn() * 0.015 + 0.001)
        else:  # Bull low vol
            returns_data.append(np.random.randn() * 0.01 + 0.0004)

    returns = pd.Series(returns_data, index=dates)

    # Simulate VIX
    vix_data = []
    for i in range(n_days):
        if i < 500:
            vix_data.append(15 + np.random.randn() * 2)
        elif i < 700:
            vix_data.append(50 + np.random.randn() * 10)
        elif i < 1000:
            vix_data.append(25 + np.random.randn() * 5)
        else:
            vix_data.append(16 + np.random.randn() * 3)

    vix = pd.Series(vix_data, index=dates)

    # Simulate cross-asset returns
    assets = ["SPY", "TLT", "GLD"]
    cross_returns = pd.DataFrame(
        np.random.randn(n_days, len(assets)) * 0.01, columns=assets, index=dates
    )

    # Initialize detector
    detector = MarketRegimeDetector()

    # Detect regime
    result = detector.detect_regime(
        returns=returns,
        vix=vix,
        cross_asset_returns=cross_returns,
        prices=returns.cumsum(),
    )

    print("\n=== Market Regime Detection ===")
    print(f"Current Regime: {result.current_regime.value}")
    print(f"Volatility Regime: {result.volatility_regime.value}")
    print(f"Correlation Regime: {result.correlation_regime.value}")
    print(f"Trend Regime: {result.trend_regime.value}")
    print(f"Regime Probability: {result.regime_probability:.2%}")
    print(f"Time in Regime: {result.time_in_regime} days")
    print(f"Change Point Detected: {result.change_point_detected}")
    print(f"Expected Duration: {result.expected_regime_duration} days")

    print("\nForward Indicators:")
    for indicator, value in result.forward_indicators.items():
        print(f"  {indicator}: {value:.4f}")

    # Get strategy adjustments
    adjustments = detector.get_regime_strategy_adjustments(result.current_regime)
    print("\nRecommended Strategy Adjustments:")
    print(f"  Equity Allocation: {adjustments['equity_allocation']:.0%}")
    print(f"  Leverage: {adjustments['leverage']:.1f}x")
    print(f"  Strategies: {', '.join(adjustments['strategies'])}")
    print(f"  Risk Level: {adjustments['risk_level']}")
    print(f"  Rebalance: {adjustments['rebalance_frequency']}")

    print("\n✅ Market Regime Detection - Tom Hogan")

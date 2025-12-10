"""================================================================================
QUANTITATIVE SIGNALS - Pure Math Edge
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Pure quantitative signals - math doesn't lie.
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QuantSignal:
    """Signal from quantitative analysis."""

    signal_id: str
    signal_type: str
    ticker: str
    direction: str
    confidence: float
    description: str
    quant_data: Dict[str, Any]
    backtest_sharpe: Optional[float] = None
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class QuantSignals:
    """QUANTITATIVE SIGNALS

    1. Statistical Arbitrage - Pairs divergence
    2. Mean-Variance Optimization Signal - Portfolio efficiency
    3. Risk Parity Rebalance - Factor exposure drift
    4. Momentum Crash Risk - Momentum factor stress
    5. Value Spread Extremes - Value/Growth spread
    6. Quality Factor Rotation - Quality vs Junk spread
    7. Size Factor Timing - Small vs Large timing
    8. Volatility Risk Premium - Implied vs Realized
    9. Carry Trade Signal - FX and rates carry
    10. Trend Following Signal - Multi-timeframe trend
    """

    def __init__(self):
        self.signals_detected: List[QuantSignal] = []

    def statistical_arbitrage_pairs(
        self,
        ticker1: str,
        ticker2: str,
        current_spread: float,
        historical_mean: float,
        historical_std: float,
        correlation: float,
        half_life_days: float,
    ) -> Optional[QuantSignal]:
        """Statistical arbitrage - pairs trading signal.

        Z-score based entry with mean reversion assumption.
        """
        if correlation < 0.7:
            return None  # Not correlated enough

        z_score = (current_spread - historical_mean) / historical_std if historical_std > 0 else 0

        if abs(z_score) < 1.5:
            return None

        if z_score > 2.0:
            direction = "short_spread"  # Short ticker1, long ticker2
            confidence = min(0.55 + (z_score - 2) * 0.1, 0.75)
            desc = f"PAIRS TRADE: {ticker1}/{ticker2} spread +{z_score:.1f}σ - short spread (long {ticker2}, short {ticker1})"
        elif z_score < -2.0:
            direction = "long_spread"
            confidence = min(0.55 + (abs(z_score) - 2) * 0.1, 0.75)
            desc = f"PAIRS TRADE: {ticker1}/{ticker2} spread {z_score:.1f}σ - long spread (long {ticker1}, short {ticker2})"
        else:
            return None

        return QuantSignal(
            signal_id=f"stat_{hashlib.sha256(f'{ticker1}{ticker2}'.encode()).hexdigest()[:8]}",
            signal_type="statistical_arbitrage",
            ticker=f"{ticker1}/{ticker2}",
            direction=direction,
            confidence=confidence,
            description=desc,
            quant_data={"z_score": z_score, "correlation": correlation, "half_life": half_life_days},
            backtest_sharpe=1.2 if half_life_days < 20 else 0.8,
        )

    def momentum_crash_risk(
        self,
        momentum_factor_return_1m: float,
        momentum_factor_return_3m: float,
        market_volatility: float,
        momentum_factor_volatility: float,
    ) -> Optional[QuantSignal]:
        """Momentum factor crash risk detection.

        Strong momentum in low vol + rising market vol = crash risk.
        """
        momentum_beta = momentum_factor_volatility / market_volatility if market_volatility > 0 else 1

        # Momentum crash risk indicators
        crash_risk_score = (
            max(0, momentum_factor_return_3m - 0.10) * 2 +  # Extended momentum
            max(0, market_volatility - 0.15) * 5 +  # Rising market vol
            max(0, 1.5 - momentum_beta) * 0.5  # Low recent momentum vol = complacent
        )

        if crash_risk_score < 0.3:
            return None

        if crash_risk_score > 0.6:
            severity = "HIGH"
            confidence = 0.68
        else:
            severity = "ELEVATED"
            confidence = 0.58

        return QuantSignal(
            signal_id=f"momcrash_{hashlib.sha256(b'momcrash').hexdigest()[:8]}",
            signal_type="momentum_crash_risk",
            ticker="MOM_FACTOR",
            direction="bearish",
            confidence=confidence,
            description=f"{severity} MOMENTUM CRASH RISK: Momentum extended ({momentum_factor_return_3m:.0%} 3m) while vol rising",
            quant_data={"crash_risk_score": crash_risk_score, "mom_3m": momentum_factor_return_3m, "mkt_vol": market_volatility},
        )

    def value_spread_extreme(
        self,
        value_spread: float,  # Cheap vs Expensive valuation spread
        value_spread_percentile: float,  # Historical percentile
        recent_value_performance: float,
    ) -> Optional[QuantSignal]:
        """Value/Growth spread extremes signal rotation.

        Extreme cheap = value rotation coming
        Extreme expensive = growth exhaustion
        """
        if value_spread_percentile < 0.10:
            direction = "value"
            confidence = 0.62
            desc = f"VALUE EXTREME: Value spread at {value_spread_percentile:.0%} percentile - value rotation likely"
        elif value_spread_percentile > 0.90:
            direction = "growth"
            confidence = 0.58
            desc = f"GROWTH EXTREME: Value spread at {value_spread_percentile:.0%} percentile - growth may continue"
        else:
            return None

        return QuantSignal(
            signal_id=f"val_{hashlib.sha256(b'valspread').hexdigest()[:8]}",
            signal_type="value_spread_extreme",
            ticker="VALUE_FACTOR",
            direction=direction,
            confidence=confidence,
            description=desc,
            quant_data={"spread": value_spread, "percentile": value_spread_percentile, "recent_perf": recent_value_performance},
        )

    def quality_factor_rotation(
        self,
        quality_junk_spread: float,
        quality_junk_spread_6m_avg: float,
        credit_spreads_widening: bool,
        recession_probability: float,
    ) -> Optional[QuantSignal]:
        """Quality vs Junk factor rotation signal.

        Quality outperforms in stress, junk in recovery.
        """
        spread_deviation = quality_junk_spread - quality_junk_spread_6m_avg

        if credit_spreads_widening and spread_deviation > 0.02:
            direction = "quality"
            confidence = 0.65
            desc = f"QUALITY ROTATION: Quality outperforming junk by {spread_deviation:.1%}, credit stress rising"
        elif not credit_spreads_widening and spread_deviation < -0.02 and recession_probability < 0.3:
            direction = "junk"
            confidence = 0.58
            desc = "JUNK ROTATION: Quality underperforming, low recession risk - risk-on rotation"
        else:
            return None

        return QuantSignal(
            signal_id=f"qual_{hashlib.sha256(b'quality').hexdigest()[:8]}",
            signal_type="quality_rotation",
            ticker="QUALITY_FACTOR",
            direction=direction,
            confidence=confidence,
            description=desc,
            quant_data={"spread": quality_junk_spread, "deviation": spread_deviation, "recession_prob": recession_probability},
        )

    def volatility_risk_premium(
        self,
        implied_vol: float,
        realized_vol_30d: float,
        vix_percentile: float,
        vrp_historical_avg: float,
    ) -> Optional[QuantSignal]:
        """Volatility risk premium signal.

        VRP = Implied - Realized. Persistently positive = sell vol strategy.
        """
        vrp = implied_vol - realized_vol_30d
        vrp_percentile = (vrp - vrp_historical_avg) / vrp_historical_avg if vrp_historical_avg > 0 else 0

        if vrp > 0.05 and vrp_percentile > 0.5:
            direction = "sell_vol"
            confidence = 0.62
            desc = f"VRP HIGH: IV {implied_vol:.0%} vs RV {realized_vol_30d:.0%} ({vrp:.0%} premium) - sell vol"
        elif vrp < -0.02:
            direction = "buy_vol"
            confidence = 0.58
            desc = f"VRP NEGATIVE: IV {implied_vol:.0%} < RV {realized_vol_30d:.0%} - buy vol (rare)"
        else:
            return None

        return QuantSignal(
            signal_id=f"vrp_{hashlib.sha256(b'vrp').hexdigest()[:8]}",
            signal_type="volatility_risk_premium",
            ticker="VIX",
            direction=direction,
            confidence=confidence,
            description=desc,
            quant_data={"implied": implied_vol, "realized": realized_vol_30d, "vrp": vrp},
            backtest_sharpe=0.8 if direction == "sell_vol" else 0.5,
        )

    def trend_following_signal(
        self,
        ticker: str,
        price: float,
        sma_20: float,
        sma_50: float,
        sma_200: float,
        atr_14: float,
    ) -> Optional[QuantSignal]:
        """Multi-timeframe trend following signal.

        All timeframes aligned = strong trend, trade with it.
        """
        above_20 = price > sma_20
        above_50 = price > sma_50
        above_200 = price > sma_200
        sma_20_above_50 = sma_20 > sma_50
        sma_50_above_200 = sma_50 > sma_200

        bullish_score = sum([above_20, above_50, above_200, sma_20_above_50, sma_50_above_200])
        bearish_score = 5 - bullish_score

        if bullish_score >= 4:
            direction = "bullish"
            confidence = 0.55 + bullish_score * 0.05
            trend_strength = "STRONG" if bullish_score == 5 else "MODERATE"
            desc = f"TREND BULLISH: {ticker} {trend_strength} uptrend (score {bullish_score}/5)"
        elif bearish_score >= 4:
            direction = "bearish"
            confidence = 0.55 + bearish_score * 0.05
            trend_strength = "STRONG" if bearish_score == 5 else "MODERATE"
            desc = f"TREND BEARISH: {ticker} {trend_strength} downtrend (score {bearish_score}/5)"
        else:
            return None

        return QuantSignal(
            signal_id=f"trend_{hashlib.sha256(f'{ticker}trend'.encode()).hexdigest()[:8]}",
            signal_type="trend_following",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            quant_data={"bullish_score": bullish_score, "above_200": above_200, "atr": atr_14},
            backtest_sharpe=0.7,
        )

    def mean_variance_optimization(
        self,
        ticker: str,
        current_weight: float,
        optimal_weight: float,
        expected_return: float,
        volatility: float,
        correlation_to_portfolio: float,
    ) -> Optional[QuantSignal]:
        """Mean-variance optimization signal.

        Current vs optimal weight divergence = rebalance signal.
        """
        weight_diff = optimal_weight - current_weight
        sharpe = expected_return / volatility if volatility > 0 else 0

        if abs(weight_diff) < 0.02:
            return None

        # Diversification benefit from low correlation
        diversification_benefit = (1 - correlation_to_portfolio) * 0.1

        if weight_diff > 0.03:
            direction = "add"
            confidence = 0.58 + diversification_benefit
            desc = f"MVO ADD: {ticker} underweight by {weight_diff:.1%} (Sharpe {sharpe:.2f})"
        elif weight_diff < -0.03:
            direction = "reduce"
            confidence = 0.55
            desc = f"MVO REDUCE: {ticker} overweight by {abs(weight_diff):.1%}"
        else:
            return None

        return QuantSignal(
            signal_id=f"mvo_{hashlib.sha256(f'{ticker}mvo'.encode()).hexdigest()[:8]}",
            signal_type="mean_variance_optimization",
            ticker=ticker,
            direction=direction,
            confidence=min(confidence, 0.72),
            description=desc,
            quant_data={"weight_diff": weight_diff, "sharpe": sharpe, "corr": correlation_to_portfolio},
        )

    def size_factor_timing(
        self,
        small_cap_return_1m: float,
        large_cap_return_1m: float,
        credit_conditions: str,  # "tight", "normal", "easy"
        economic_cycle: str,  # "early", "mid", "late", "recession"
    ) -> Optional[QuantSignal]:
        """Small vs Large cap timing signal.

        Small caps outperform in early cycle with easy credit.
        """
        size_spread = small_cap_return_1m - large_cap_return_1m

        # Economic regime score for small caps
        regime_score = {
            "early": 1.0,
            "mid": 0.5,
            "late": -0.3,
            "recession": -1.0,
        }.get(economic_cycle, 0)

        credit_score = {
            "easy": 1.0,
            "normal": 0,
            "tight": -1.0,
        }.get(credit_conditions, 0)

        small_cap_signal = regime_score * 0.6 + credit_score * 0.4

        if small_cap_signal > 0.5:
            direction = "small_cap"
            confidence = 0.60
            desc = f"SIZE ROTATION: Favor small caps (cycle: {economic_cycle}, credit: {credit_conditions})"
        elif small_cap_signal < -0.5:
            direction = "large_cap"
            confidence = 0.62
            desc = f"SIZE ROTATION: Favor large caps (cycle: {economic_cycle}, credit: {credit_conditions})"
        else:
            return None

        return QuantSignal(
            signal_id=f"size_{hashlib.sha256(b'size').hexdigest()[:8]}",
            signal_type="size_factor_timing",
            ticker="SIZE_FACTOR",
            direction=direction,
            confidence=confidence,
            description=desc,
            quant_data={"size_spread": size_spread, "regime_score": regime_score, "credit_score": credit_score},
        )

    def carry_trade_signal(
        self,
        high_yield_currency: str,
        low_yield_currency: str,
        yield_differential: float,
        fx_volatility: float,
        carry_to_vol_ratio: float,
    ) -> Optional[QuantSignal]:
        """FX carry trade signal.

        High yield differential + low vol = attractive carry.
        """
        if carry_to_vol_ratio < 0.3:
            return None  # Not enough carry per unit vol

        if carry_to_vol_ratio > 0.5:
            direction = "long_carry"
            confidence = 0.60
            desc = f"CARRY ATTRACTIVE: {high_yield_currency}/{low_yield_currency} yield diff {yield_differential:.1%}, carry/vol {carry_to_vol_ratio:.2f}"
        else:
            return None

        return QuantSignal(
            signal_id=f"carry_{hashlib.sha256(f'{high_yield_currency}{low_yield_currency}'.encode()).hexdigest()[:8]}",
            signal_type="carry_trade",
            ticker=f"{high_yield_currency}/{low_yield_currency}",
            direction=direction,
            confidence=confidence,
            description=desc,
            quant_data={"yield_diff": yield_differential, "vol": fx_volatility, "carry_vol": carry_to_vol_ratio},
            backtest_sharpe=0.6,
        )

    def risk_parity_rebalance(
        self,
        asset_class: str,
        current_risk_contribution: float,
        target_risk_contribution: float,
        volatility_change_30d: float,
    ) -> Optional[QuantSignal]:
        """Risk parity rebalance signal.

        Volatility changes require rebalancing for equal risk contribution.
        """
        risk_diff = current_risk_contribution - target_risk_contribution

        if abs(risk_diff) < 0.03:
            return None

        if risk_diff > 0.05:
            direction = "reduce"
            confidence = 0.58
            desc = f"RISK PARITY REDUCE: {asset_class} risk contribution {current_risk_contribution:.0%} > target {target_risk_contribution:.0%}"
        elif risk_diff < -0.05:
            direction = "add"
            confidence = 0.58
            desc = f"RISK PARITY ADD: {asset_class} risk contribution {current_risk_contribution:.0%} < target {target_risk_contribution:.0%}"
        else:
            return None

        return QuantSignal(
            signal_id=f"rp_{hashlib.sha256(f'{asset_class}rp'.encode()).hexdigest()[:8]}",
            signal_type="risk_parity_rebalance",
            ticker=asset_class,
            direction=direction,
            confidence=confidence,
            description=desc,
            quant_data={"current_risk": current_risk_contribution, "target_risk": target_risk_contribution, "vol_change": volatility_change_30d},
        )

    def get_stats(self) -> Dict[str, Any]:
        return {"total_signals": len(self.signals_detected)}



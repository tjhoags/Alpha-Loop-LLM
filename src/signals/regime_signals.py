"""================================================================================
REGIME SIGNALS - Market State Detection
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Different regimes require different strategies. Detect the regime first.
================================================================================
"""

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class RegimeSignal:
    """Signal from regime detection."""

    signal_id: str
    signal_type: str
    regime_name: str
    confidence: float
    description: str
    regime_characteristics: Dict[str, Any]
    recommended_strategies: List[str]
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class RegimeSignals:
    """REGIME DETECTION SIGNALS

    1. Volatility Regime - Low/Mid/High vol environment
    2. Correlation Regime - Correlated vs Dispersed
    3. Trend Regime - Trending vs Mean-Reverting
    4. Liquidity Regime - Abundant vs Scarce
    5. Risk-On/Risk-Off - Risk appetite state
    6. Economic Cycle Phase - Early/Mid/Late/Recession
    7. Fed Policy Regime - Hawkish/Neutral/Dovish
    8. Credit Regime - Expansion vs Contraction
    9. Momentum Regime - Strong vs Weak momentum
    10. Value/Growth Regime - Which factor dominates
    """

    def __init__(self):
        self.signals_detected: List[RegimeSignal] = []
        self.current_regime: Dict[str, str] = {}

    def volatility_regime(
        self,
        vix_level: float,
        vix_percentile_1y: float,
        realized_vol_30d: float,
        vol_of_vol: float,
    ) -> RegimeSignal:
        """Detect volatility regime.

        Low vol: VIX < 15, different strategies
        High vol: VIX > 25, different strategies
        """
        if vix_level < 13:
            regime = "ULTRA_LOW_VOL"
            strategies = ["sell_vol", "carry_trades", "momentum"]
            desc = f"ULTRA LOW VOL: VIX {vix_level:.0f} - complacency, sell vol but watch for spikes"
        elif vix_level < 18:
            regime = "LOW_VOL"
            strategies = ["sell_vol", "trend_following", "carry"]
            desc = f"LOW VOL: VIX {vix_level:.0f} - benign environment, trend following works"
        elif vix_level < 25:
            regime = "MID_VOL"
            strategies = ["balanced", "stock_picking", "pairs"]
            desc = f"MID VOL: VIX {vix_level:.0f} - normal environment, diversified approach"
        elif vix_level < 35:
            regime = "HIGH_VOL"
            strategies = ["buy_vol", "mean_reversion", "quality"]
            desc = f"HIGH VOL: VIX {vix_level:.0f} - elevated fear, mean reversion opportunities"
        else:
            regime = "EXTREME_VOL"
            strategies = ["cash", "puts", "quality", "treasuries"]
            desc = f"EXTREME VOL: VIX {vix_level:.0f} - crisis mode, capital preservation first"

        confidence = 0.80 if abs(vix_level - 20) > 5 else 0.65

        self.current_regime["volatility"] = regime

        return RegimeSignal(
            signal_id=f"volreg_{hashlib.sha256(regime.encode()).hexdigest()[:8]}",
            signal_type="volatility_regime",
            regime_name=regime,
            confidence=confidence,
            description=desc,
            regime_characteristics={"vix": vix_level, "percentile": vix_percentile_1y, "vol_of_vol": vol_of_vol},
            recommended_strategies=strategies,
        )

    def correlation_regime(
        self,
        avg_stock_correlation: float,
        sector_dispersion: float,
        correlation_change_30d: float,
    ) -> RegimeSignal:
        """Detect correlation regime.

        High correlation = macro driven, factor bets
        Low correlation = stock picking works
        """
        if avg_stock_correlation > 0.7:
            regime = "HIGH_CORRELATION"
            strategies = ["macro_bets", "factor_timing", "hedged_positions"]
            desc = f"HIGH CORRELATION: Stocks moving together ({avg_stock_correlation:.0%}) - macro driven"
        elif avg_stock_correlation > 0.5:
            regime = "MODERATE_CORRELATION"
            strategies = ["balanced", "sector_rotation", "pairs"]
            desc = f"MODERATE CORRELATION: Normal dispersion ({avg_stock_correlation:.0%})"
        else:
            regime = "LOW_CORRELATION"
            strategies = ["stock_picking", "long_short", "alpha_generation"]
            desc = f"LOW CORRELATION: High dispersion ({avg_stock_correlation:.0%}) - stock picking works"

        confidence = 0.75

        self.current_regime["correlation"] = regime

        return RegimeSignal(
            signal_id=f"corrreg_{hashlib.sha256(regime.encode()).hexdigest()[:8]}",
            signal_type="correlation_regime",
            regime_name=regime,
            confidence=confidence,
            description=desc,
            regime_characteristics={"avg_corr": avg_stock_correlation, "dispersion": sector_dispersion},
            recommended_strategies=strategies,
        )

    def trend_regime(
        self,
        pct_stocks_above_200ma: float,
        adv_decline_ratio_20d: float,
        new_highs_lows_ratio: float,
    ) -> RegimeSignal:
        """Detect trend regime.

        Strong trends = momentum works
        Choppy = mean reversion works
        """
        trend_score = (
            (pct_stocks_above_200ma - 0.5) * 2 +
            (adv_decline_ratio_20d - 1) * 0.5 +
            math.log(new_highs_lows_ratio + 0.1) * 0.3
        )

        if trend_score > 0.8:
            regime = "STRONG_UPTREND"
            strategies = ["momentum", "trend_following", "dip_buying"]
            desc = f"STRONG UPTREND: {pct_stocks_above_200ma:.0%} above 200MA - momentum works"
        elif trend_score > 0.2:
            regime = "MILD_UPTREND"
            strategies = ["balanced", "quality_growth", "selective_momentum"]
            desc = f"MILD UPTREND: {pct_stocks_above_200ma:.0%} above 200MA - selective approach"
        elif trend_score > -0.2:
            regime = "CHOPPY"
            strategies = ["mean_reversion", "range_trading", "volatility_selling"]
            desc = f"CHOPPY: Mixed signals ({pct_stocks_above_200ma:.0%} above 200MA) - mean reversion"
        elif trend_score > -0.8:
            regime = "MILD_DOWNTREND"
            strategies = ["defensive", "quality", "hedged"]
            desc = f"MILD DOWNTREND: {pct_stocks_above_200ma:.0%} above 200MA - defensive positioning"
        else:
            regime = "STRONG_DOWNTREND"
            strategies = ["short", "puts", "cash", "inverse_etfs"]
            desc = f"STRONG DOWNTREND: Only {pct_stocks_above_200ma:.0%} above 200MA - preservation mode"

        confidence = min(0.60 + abs(trend_score) * 0.15, 0.85)

        self.current_regime["trend"] = regime

        return RegimeSignal(
            signal_id=f"trendreg_{hashlib.sha256(regime.encode()).hexdigest()[:8]}",
            signal_type="trend_regime",
            regime_name=regime,
            confidence=confidence,
            description=desc,
            regime_characteristics={"pct_above_200ma": pct_stocks_above_200ma, "trend_score": trend_score},
            recommended_strategies=strategies,
        )

    def risk_appetite_regime(
        self,
        high_yield_spread: float,
        high_yield_spread_change_30d: float,
        equity_flows_30d: float,
        vix_skew: float,
    ) -> RegimeSignal:
        """Detect risk-on vs risk-off regime.
        """
        risk_appetite_score = (
            -(high_yield_spread - 4) * 0.3 +  # Lower spread = risk on
            -high_yield_spread_change_30d * 2 +  # Tightening = risk on
            equity_flows_30d * 0.2 +  # Inflows = risk on
            -(vix_skew - 0.1) * 3  # Lower skew = risk on
        )

        if risk_appetite_score > 0.5:
            regime = "RISK_ON"
            strategies = ["growth", "small_cap", "high_beta", "cyclicals"]
            desc = f"RISK ON: HY spread {high_yield_spread:.0%}, flows positive - risk seeking"
        elif risk_appetite_score > -0.2:
            regime = "NEUTRAL"
            strategies = ["balanced", "barbell", "quality"]
            desc = "NEUTRAL: Mixed signals - balanced approach"
        else:
            regime = "RISK_OFF"
            strategies = ["defensive", "treasuries", "gold", "utilities", "healthcare"]
            desc = f"RISK OFF: HY spread {high_yield_spread:.0%}, widening - flight to safety"

        confidence = min(0.60 + abs(risk_appetite_score) * 0.15, 0.82)

        self.current_regime["risk_appetite"] = regime

        return RegimeSignal(
            signal_id=f"riskreg_{hashlib.sha256(regime.encode()).hexdigest()[:8]}",
            signal_type="risk_appetite_regime",
            regime_name=regime,
            confidence=confidence,
            description=desc,
            regime_characteristics={"hy_spread": high_yield_spread, "risk_score": risk_appetite_score},
            recommended_strategies=strategies,
        )

    def economic_cycle_regime(
        self,
        gdp_growth: float,
        gdp_acceleration: float,
        unemployment_rate: float,
        unemployment_change: float,
        pmi_manufacturing: float,
        yield_curve_slope: float,
    ) -> RegimeSignal:
        """Detect economic cycle phase.

        Different phases favor different sectors/factors.
        """
        # Simplified cycle detection
        expansion = gdp_growth > 0 and gdp_acceleration > 0
        contraction = gdp_growth < 0 or gdp_acceleration < -0.5
        labor_improving = unemployment_change < 0

        if gdp_growth > 2 and gdp_acceleration > 0 and labor_improving:
            regime = "EARLY_CYCLE"
            strategies = ["cyclicals", "small_cap", "financials", "industrials"]
            desc = f"EARLY CYCLE: GDP +{gdp_growth:.1%} accelerating, labor improving"
        elif gdp_growth > 1.5 and not labor_improving and pmi_manufacturing > 50:
            regime = "MID_CYCLE"
            strategies = ["quality_growth", "tech", "industrials", "balanced"]
            desc = f"MID CYCLE: GDP +{gdp_growth:.1%}, mature expansion"
        elif gdp_growth > 0 and gdp_acceleration < 0:
            regime = "LATE_CYCLE"
            strategies = ["defensive", "staples", "healthcare", "utilities", "quality"]
            desc = f"LATE CYCLE: GDP +{gdp_growth:.1%} but decelerating - defensive positioning"
        elif gdp_growth < 0 or (yield_curve_slope < -0.5 and pmi_manufacturing < 48):
            regime = "RECESSION"
            strategies = ["cash", "treasuries", "gold", "utilities", "inverse"]
            desc = f"RECESSION: GDP {gdp_growth:+.1%}, inverted curve - capital preservation"
        else:
            regime = "RECOVERY"
            strategies = ["cyclicals", "value", "small_cap", "high_beta"]
            desc = "RECOVERY: Transitioning out of weakness"

        confidence = 0.70

        self.current_regime["economic_cycle"] = regime

        return RegimeSignal(
            signal_id=f"econreg_{hashlib.sha256(regime.encode()).hexdigest()[:8]}",
            signal_type="economic_cycle_regime",
            regime_name=regime,
            confidence=confidence,
            description=desc,
            regime_characteristics={"gdp": gdp_growth, "unemployment": unemployment_rate, "pmi": pmi_manufacturing},
            recommended_strategies=strategies,
        )

    def fed_policy_regime(
        self,
        fed_funds_rate: float,
        dot_plot_median_change: float,
        balance_sheet_change: float,
        inflation_rate: float,
        unemployment_gap: float,
    ) -> RegimeSignal:
        """Detect Fed policy regime.

        Hawkish vs Dovish affects all asset classes.
        """
        policy_score = (
            dot_plot_median_change * 2 +  # Raising = hawkish
            -balance_sheet_change * 10 +  # Shrinking = hawkish
            (inflation_rate - 2) * 0.5 +  # High inflation = hawkish
            -unemployment_gap * 0.3  # High unemployment = dovish
        )

        if policy_score > 0.5:
            regime = "HAWKISH"
            strategies = ["short_duration", "value", "financials", "cash"]
            desc = f"HAWKISH FED: Dot plot +{dot_plot_median_change:.0%}, inflation {inflation_rate:.1%}"
        elif policy_score > -0.3:
            regime = "NEUTRAL"
            strategies = ["balanced", "quality", "moderate_duration"]
            desc = "NEUTRAL FED: Data dependent, balanced approach"
        elif policy_score > -1.0:
            regime = "DOVISH"
            strategies = ["growth", "long_duration", "high_beta", "risk_on"]
            desc = "DOVISH FED: Accommodative stance, risk-on"
        else:
            regime = "EMERGENCY_EASING"
            strategies = ["risk_assets", "gold", "growth", "long_duration"]
            desc = "EMERGENCY EASING: Crisis response mode"

        confidence = 0.75

        self.current_regime["fed_policy"] = regime

        return RegimeSignal(
            signal_id=f"fedreg_{hashlib.sha256(regime.encode()).hexdigest()[:8]}",
            signal_type="fed_policy_regime",
            regime_name=regime,
            confidence=confidence,
            description=desc,
            regime_characteristics={"fed_funds": fed_funds_rate, "policy_score": policy_score},
            recommended_strategies=strategies,
        )

    def credit_regime(
        self,
        ig_spread: float,
        hy_spread: float,
        default_rate: float,
        lending_standards: str,  # "tightening", "stable", "easing"
    ) -> RegimeSignal:
        """Detect credit regime.

        Credit cycles lead economic cycles.
        """
        credit_score = (
            -(ig_spread - 1) * 0.3 +
            -(hy_spread - 4) * 0.2 +
            -default_rate * 5 +
            {"easing": 1, "stable": 0, "tightening": -1}.get(lending_standards, 0) * 0.3
        )

        if credit_score > 0.3:
            regime = "CREDIT_EXPANSION"
            strategies = ["high_yield", "leveraged_loans", "credit_spreads", "cyclicals"]
            desc = f"CREDIT EXPANSION: IG {ig_spread:.0%}, HY {hy_spread:.0%} - spreads tight"
        elif credit_score > -0.3:
            regime = "CREDIT_STABLE"
            strategies = ["investment_grade", "balanced_credit", "quality"]
            desc = "CREDIT STABLE: Normal conditions"
        else:
            regime = "CREDIT_CONTRACTION"
            strategies = ["treasuries", "short_credit", "cash", "defensive_equity"]
            desc = "CREDIT CONTRACTION: Spreads widening, defaults rising - risk off"

        confidence = 0.72

        self.current_regime["credit"] = regime

        return RegimeSignal(
            signal_id=f"credreg_{hashlib.sha256(regime.encode()).hexdigest()[:8]}",
            signal_type="credit_regime",
            regime_name=regime,
            confidence=confidence,
            description=desc,
            regime_characteristics={"ig_spread": ig_spread, "hy_spread": hy_spread, "default_rate": default_rate},
            recommended_strategies=strategies,
        )

    def get_current_regime_summary(self) -> Dict[str, str]:
        """Get summary of all current regimes."""
        return self.current_regime.copy()

    def get_stats(self) -> Dict[str, Any]:
        return {"total_signals": len(self.signals_detected), "current_regimes": self.current_regime}



"""================================================================================
RISK AGENT - "TAIL RISK & CORRELATION SIGNALS"
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Detect when the market is about to break.
Correlations, vol-of-vol, dispersion - these warn before it's too late.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TailRiskSignal:
    """A signal from tail risk/correlation analysis."""

    signal_id: str
    signal_type: str
    direction: str
    severity: str  # low, moderate, elevated, high, extreme
    confidence: float
    description: str
    risk_metric: str
    current_value: float
    threshold: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class TailRiskSignals:
    """TAIL RISK & CORRELATION SIGNALS - Don't get caught

    1. Correlation Breakdown Early Warning - Crisis brewing
    2. Sector Dispersion Opportunity - Stock picking vs macro
    3. Vol of Vol Regime Signal - VVIX/VIX ratio
    4. Skew Crash Probability Model - Risk-neutral crash prob
    5. Cross Asset Dislocation Scanner - Abnormal relationships
    """

    def __init__(self):
        self.signals_detected: List[TailRiskSignal] = []
        self.current_regime: str = "normal"

    def correlation_breakdown_early_warning(
        self,
        rolling_30d_corr: float,
        two_year_avg_corr: float,
        corr_velocity: float,
    ) -> Optional[TailRiskSignal]:
        """Track 30-day rolling correlations vs 2-year average.
        When correlations spike toward 1 = crisis brewing.
        """
        import hashlib

        corr_deviation = rolling_30d_corr - two_year_avg_corr

        if rolling_30d_corr > 0.8 and corr_velocity > 0.1:
            return TailRiskSignal(
                signal_id=f"cor_{hashlib.sha256(f'corr{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="correlation_spike",
                direction="risk_off",
                severity="high" if rolling_30d_corr > 0.85 else "elevated",
                confidence=0.75,
                description=f"CORRELATION CRISIS WARNING: 30d corr at {rolling_30d_corr:.2f} (avg {two_year_avg_corr:.2f}), rising fast",
                risk_metric="cross_asset_correlation",
                current_value=rolling_30d_corr,
                threshold=0.80,
                evidence={"avg_corr": two_year_avg_corr, "velocity": corr_velocity},
            )
        elif rolling_30d_corr < two_year_avg_corr * 0.6:
            return TailRiskSignal(
                signal_id=f"cor_{hashlib.sha256(f'decorr{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="correlation_breakdown",
                direction="stock_picking",
                severity="moderate",
                confidence=0.65,
                description=f"DECORRELATION: 30d corr at {rolling_30d_corr:.2f} - stock picking opportunity",
                risk_metric="cross_asset_correlation",
                current_value=rolling_30d_corr,
                threshold=two_year_avg_corr * 0.7,
                evidence={"avg_corr": two_year_avg_corr},
            )

        return None

    def sector_dispersion_opportunity(
        self,
        intra_sector_dispersion: float,
        historical_dispersion: float,
        market_regime: str,
    ) -> Optional[TailRiskSignal]:
        """High intra-sector dispersion = stock picking environment.
        Low dispersion = macro regime, go passive.
        """
        import hashlib

        dispersion_ratio = intra_sector_dispersion / historical_dispersion if historical_dispersion > 0 else 1

        if dispersion_ratio > 1.5:
            return TailRiskSignal(
                signal_id=f"dsp_{hashlib.sha256(f'disp{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="sector_dispersion",
                direction="active_management",
                severity="opportunity",
                confidence=0.68,
                description=f"HIGH DISPERSION: Sector dispersion {dispersion_ratio:.1f}x normal - stock picking environment",
                risk_metric="intra_sector_dispersion",
                current_value=intra_sector_dispersion,
                threshold=historical_dispersion * 1.3,
                evidence={"hist_dispersion": historical_dispersion, "regime": market_regime},
            )
        elif dispersion_ratio < 0.6:
            return TailRiskSignal(
                signal_id=f"dsp_{hashlib.sha256(f'lowdisp{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="sector_dispersion",
                direction="passive_tilt",
                severity="low",
                confidence=0.62,
                description=f"LOW DISPERSION: Sector dispersion {dispersion_ratio:.1f}x normal - macro driven, consider passive",
                risk_metric="intra_sector_dispersion",
                current_value=intra_sector_dispersion,
                threshold=historical_dispersion * 0.7,
                evidence={"hist_dispersion": historical_dispersion, "regime": market_regime},
            )

        return None

    def vol_of_vol_regime_signal(
        self,
        vvix: float,
        vix: float,
        vvix_vix_ratio: float,
        historical_ratio: float,
    ) -> Optional[TailRiskSignal]:
        """VVIX/VIX ratio predicts vol regime changes.
        High ratio = vol spike coming.
        """
        import hashlib

        ratio_deviation = (vvix_vix_ratio - historical_ratio) / historical_ratio if historical_ratio > 0 else 0

        if vvix_vix_ratio > 6.0 and ratio_deviation > 0.2:
            return TailRiskSignal(
                signal_id=f"vov_{hashlib.sha256(f'volofvol{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="vol_of_vol_spike",
                direction="vol_expansion",
                severity="elevated",
                confidence=0.70,
                description=f"VOL SPIKE WARNING: VVIX/VIX at {vvix_vix_ratio:.1f} (hist {historical_ratio:.1f}) - vol expansion likely",
                risk_metric="vvix_vix_ratio",
                current_value=vvix_vix_ratio,
                threshold=5.5,
                evidence={"vvix": vvix, "vix": vix, "hist_ratio": historical_ratio},
            )
        elif vvix_vix_ratio < 4.0 and vix > 25:
            return TailRiskSignal(
                signal_id=f"vov_{hashlib.sha256(f'volcrush{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="vol_of_vol_compression",
                direction="vol_contraction",
                severity="moderate",
                confidence=0.65,
                description=f"VOL NORMALIZATION: VVIX/VIX low at {vvix_vix_ratio:.1f} with elevated VIX - mean reversion",
                risk_metric="vvix_vix_ratio",
                current_value=vvix_vix_ratio,
                threshold=4.5,
                evidence={"vvix": vvix, "vix": vix},
            )

        return None

    def skew_crash_probability_model(
        self,
        put_skew_25d: float,
        put_skew_10d: float,
        historical_crash_rate: float,
    ) -> Optional[TailRiskSignal]:
        """Extract risk-neutral crash probability from options surface.
        Compare to historical base rates.
        """
        import hashlib

        # Simplified crash prob extraction
        implied_crash_prob = min(put_skew_25d / 100, 0.15)  # Crude estimate

        prob_vs_base = implied_crash_prob / historical_crash_rate if historical_crash_rate > 0 else 1

        if prob_vs_base > 2.0:
            return TailRiskSignal(
                signal_id=f"skw_{hashlib.sha256(f'crash{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="crash_probability",
                direction="elevated_tail_risk",
                severity="high" if prob_vs_base > 3 else "elevated",
                confidence=0.68,
                description=f"CRASH PROB ELEVATED: Implied crash prob {prob_vs_base:.1f}x historical base rate",
                risk_metric="implied_crash_probability",
                current_value=implied_crash_prob,
                threshold=historical_crash_rate * 1.5,
                evidence={"put_skew_25d": put_skew_25d, "put_skew_10d": put_skew_10d, "base_rate": historical_crash_rate},
            )

        return None

    def cross_asset_dislocation_scanner(
        self,
        gold_copper_ratio: float,
        historical_gc_ratio: float,
        equity_level: float,
        regime_indicator: str,
    ) -> Optional[TailRiskSignal]:
        """Scan for abnormal cross-asset relationships.
        Gold/copper ratio vs equity for regime detection.
        """
        import hashlib

        gc_deviation = (gold_copper_ratio - historical_gc_ratio) / historical_gc_ratio if historical_gc_ratio > 0 else 0

        if gc_deviation > 0.25:  # Gold rich vs copper - risk-off signal
            return TailRiskSignal(
                signal_id=f"xas_{hashlib.sha256(f'goldcopper{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="cross_asset_dislocation",
                direction="risk_off",
                severity="elevated",
                confidence=0.65,
                description=f"GOLD/COPPER DISLOCATION: Ratio {gc_deviation:+.0%} above normal - risk-off signal",
                risk_metric="gold_copper_ratio",
                current_value=gold_copper_ratio,
                threshold=historical_gc_ratio * 1.15,
                evidence={"hist_ratio": historical_gc_ratio, "regime": regime_indicator},
            )
        elif gc_deviation < -0.2:  # Copper rich - growth signal
            return TailRiskSignal(
                signal_id=f"xas_{hashlib.sha256(f'copper{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="cross_asset_dislocation",
                direction="risk_on",
                severity="moderate",
                confidence=0.62,
                description=f"COPPER STRENGTH: Gold/copper {abs(gc_deviation):.0%} below normal - growth signal",
                risk_metric="gold_copper_ratio",
                current_value=gold_copper_ratio,
                threshold=historical_gc_ratio * 0.85,
                evidence={"hist_ratio": historical_gc_ratio, "regime": regime_indicator},
            )

        return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_signals": len(self.signals_detected),
            "current_regime": self.current_regime,
            "by_severity": self._count_by_severity(),
        }

    def _count_by_severity(self) -> Dict[str, int]:
        counts = {}
        for sig in self.signals_detected:
            counts[sig.severity] = counts.get(sig.severity, 0) + 1
        return counts



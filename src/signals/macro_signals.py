"""================================================================================
MACRO AGENT - "REGIME & INFLECTION SIGNALS"
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Detect regime changes before they become obvious.
The market changes character - catch it early or get run over.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MacroSignal:
    """A macro regime or inflection signal."""

    signal_id: str
    signal_type: str
    direction: str
    confidence: float
    description: str
    current_regime: str
    predicted_regime: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class MacroRegimeSignals:
    """MACRO REGIME SIGNALS - Catch inflections early

    1. Fed Dot Plot Dispersion - High dispersion = uncertainty = vol
    2. Yield Curve Butterfly Momentum - 5y belly predicts better
    3. Corporate Credit Steepening - BBB-BB spreads lead equity
    4. Central Bank Balance Sheet Velocity - Rate of change matters
    5. Real Rates Equity Correlation Regime - Correlation flip = regime change
    6. Currency Vol Term Structure Inversion - Event risk priced
    7. Sovereign CDS Basis Stress - Counterparty stress emerging
    """

    def __init__(self):
        self.signals_detected: List[MacroSignal] = []
        self.current_regime: str = "normal"
        self.regime_history: List[Dict] = []

    def fed_dot_plot_dispersion_trade(
        self,
        dot_median: float,
        dot_std: float,
        previous_std: float,
    ) -> Optional[MacroSignal]:
        """Track standard deviation of Fed dots, not just median.
        High dispersion = uncertainty = volatility coming.
        Low dispersion AFTER high = regime change signal.
        """
        import hashlib

        dispersion_change = (dot_std - previous_std) / previous_std if previous_std > 0 else 0

        if dot_std > 0.75 and dispersion_change > 0.2:
            return MacroSignal(
                signal_id=f"fed_{hashlib.sha256(f'dotdispersion{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="fed_dot_dispersion",
                direction="volatility_increase",
                confidence=0.72,
                description=f"FED DISPERSION WIDENING: Dot plot std dev {dot_std:.2f} (+{dispersion_change:.0%}) - vol regime ahead",
                current_regime="low_vol" if previous_std < 0.5 else "normal",
                predicted_regime="high_vol",
                evidence={"dot_std": dot_std, "previous_std": previous_std, "median": dot_median},
            )
        elif dot_std < 0.4 and previous_std > 0.6:
            return MacroSignal(
                signal_id=f"fed_{hashlib.sha256(f'dotconverge{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="fed_dot_convergence",
                direction="volatility_decrease",
                confidence=0.68,
                description=f"FED CONVERGENCE: Dots converging (std {dot_std:.2f}) - clarity emerging",
                current_regime="high_uncertainty",
                predicted_regime="policy_clarity",
                evidence={"dot_std": dot_std, "previous_std": previous_std},
            )

        return None

    def yield_curve_butterfly_momentum(
        self,
        twos: float,
        fives: float,
        tens: float,
        twos_tens_slope: float,
    ) -> Optional[MacroSignal]:
        """2s10s gets attention, but track the 5y belly.
        Butterfly spread velocity predicts recession inflection better.
        """
        import hashlib

        butterfly = (twos + tens) / 2 - fives  # Simplified butterfly

        # Belly rich (negative butterfly) = risk-off positioning
        if butterfly < -0.3:
            return MacroSignal(
                signal_id=f"yc_{hashlib.sha256(f'butterfly{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="yield_curve_butterfly",
                direction="risk_off",
                confidence=0.65,
                description=f"BUTTERFLY RISK-OFF: 5y belly rich ({butterfly:.2f}bps) - flight to quality",
                current_regime="normal",
                predicted_regime="risk_off",
                evidence={"twos": twos, "fives": fives, "tens": tens, "butterfly": butterfly},
            )
        elif butterfly > 0.3 and twos_tens_slope > 0.5:
            return MacroSignal(
                signal_id=f"yc_{hashlib.sha256(f'steepen{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="yield_curve_butterfly",
                direction="risk_on",
                confidence=0.62,
                description="CURVE STEEPENING: Butterfly cheap, curve steep - growth expectations rising",
                current_regime="normal",
                predicted_regime="risk_on",
                evidence={"twos": twos, "fives": fives, "tens": tens, "slope": twos_tens_slope},
            )

        return None

    def central_bank_balance_sheet_velocity(
        self,
        current_bs: float,
        previous_bs: float,
        second_derivative: float,
    ) -> Optional[MacroSignal]:
        """Not level of Fed balance sheet - rate of change.
        Second derivative turning positive = liquidity inflection.
        """
        import hashlib

        first_derivative = (current_bs - previous_bs) / previous_bs if previous_bs > 0 else 0

        if second_derivative > 0.02 and first_derivative < 0:
            # QT slowing - potential pivot
            return MacroSignal(
                signal_id=f"cb_{hashlib.sha256(f'bsvelocity{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="central_bank_bs_velocity",
                direction="bullish",
                confidence=0.70,
                description=f"LIQUIDITY INFLECTION: Balance sheet contraction SLOWING (2nd deriv +{second_derivative:.2%})",
                current_regime="qt",
                predicted_regime="qt_taper",
                evidence={"first_deriv": first_derivative, "second_deriv": second_derivative},
            )
        elif second_derivative < -0.02 and first_derivative > 0:
            # QE slowing - tightening ahead
            return MacroSignal(
                signal_id=f"cb_{hashlib.sha256(f'bsslowing{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="central_bank_bs_velocity",
                direction="bearish",
                confidence=0.68,
                description=f"QE SLOWING: Balance sheet expansion decelerating (2nd deriv {second_derivative:.2%})",
                current_regime="qe",
                predicted_regime="qe_taper",
                evidence={"first_deriv": first_derivative, "second_deriv": second_derivative},
            )

        return None

    def real_rates_equity_correlation_regime(
        self,
        rolling_correlation: float,
        previous_correlation: float,
        real_rate: float,
    ) -> Optional[MacroSignal]:
        """Track rolling correlation of real rates to equity.
        When correlation flips sign = major regime change.
        """
        import hashlib

        # Correlation sign flip detection
        if rolling_correlation * previous_correlation < 0:
            flip_direction = "positive_to_negative" if previous_correlation > 0 else "negative_to_positive"

            return MacroSignal(
                signal_id=f"rr_{hashlib.sha256(f'corrflip{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="real_rate_correlation_flip",
                direction="regime_change",
                confidence=0.75,
                description=f"CORRELATION REGIME FLIP: Real rate/equity corr flipped {flip_direction}",
                current_regime=f"corr_{previous_correlation:.2f}",
                predicted_regime=f"corr_{rolling_correlation:.2f}",
                evidence={"rolling_corr": rolling_correlation, "previous_corr": previous_correlation, "real_rate": real_rate},
            )

        return None

    def currency_vol_term_structure_inversion(
        self,
        short_vol: float,
        long_vol: float,
        currency_pair: str,
    ) -> Optional[MacroSignal]:
        """When short-dated FX vol exceeds long-dated.
        Event risk priced in - but which event?
        """
        import hashlib

        if short_vol > long_vol * 1.15:
            return MacroSignal(
                signal_id=f"fx_{hashlib.sha256(f'{currency_pair}vol{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="fx_vol_inversion",
                direction="risk_elevated",
                confidence=0.68,
                description=f"FX VOL INVERTED: {currency_pair} short vol ({short_vol:.1f}) > long vol ({long_vol:.1f}) - event risk priced",
                current_regime="normal_vol",
                predicted_regime="event_risk",
                evidence={"short_vol": short_vol, "long_vol": long_vol, "pair": currency_pair},
            )

        return None

    def sovereign_cds_basis_stress_signal(
        self,
        country: str,
        cds_spread: float,
        bond_spread: float,
        basis: float,
    ) -> Optional[MacroSignal]:
        """Basis between CDS and bond spreads widening.
        Counterparty stress emerging before headlines.
        """
        import hashlib

        if abs(basis) > 50:  # 50bps basis is significant
            direction = "bearish" if basis > 0 else "bullish"

            return MacroSignal(
                signal_id=f"cds_{hashlib.sha256(f'{country}basis{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="sovereign_cds_basis",
                direction=direction,
                confidence=0.65,
                description=f"CDS BASIS STRESS: {country} CDS-bond basis at {basis:.0f}bps - counterparty concern",
                current_regime="normal",
                predicted_regime="stress",
                evidence={"cds_spread": cds_spread, "bond_spread": bond_spread, "basis": basis},
            )

        return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_signals": len(self.signals_detected),
            "current_regime": self.current_regime,
            "regime_changes": len(self.regime_history),
        }



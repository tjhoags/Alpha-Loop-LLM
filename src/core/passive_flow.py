"""MODULE 2: PASSIVE FLOW DETECTOR
================================
Alpha Loop Capital - Consequence Engine

Purpose: Detect when the "stocks only go up" passive bid is weakening
         Track structural vulnerability before consensus recognizes it

Core Edge: See flow reversals before they hit headlines
           Front-run the transition from passive inflows to outflows

Author: Tom Hogan
Version: 1.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowRegime(Enum):
    """Classification of passive flow environment"""

    STRONG_INFLOW = "strong_inflow"      # >$10B weekly, accelerating
    MODERATE_INFLOW = "moderate_inflow"   # $5-10B weekly, stable
    WEAK_INFLOW = "weak_inflow"          # $0-5B weekly, decelerating
    NEUTRAL = "neutral"                   # Flat flows, mixed signals
    WEAK_OUTFLOW = "weak_outflow"        # $0-5B weekly outflows
    MODERATE_OUTFLOW = "moderate_outflow" # $5-10B weekly outflows
    STRONG_OUTFLOW = "strong_outflow"    # >$10B weekly outflows, accelerating


class FlowVelocity(Enum):
    """Rate of change in flows"""

    ACCELERATING = "accelerating"
    STABLE = "stable"
    DECELERATING = "decelerating"
    REVERSING = "reversing"


class StressLevel(Enum):
    """Consumer/retirement financial stress level"""

    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FlowDataPoint:
    """Single observation of flow data"""

    date: str
    source: str              # "ici", "etf_creations", "401k", etc.
    value_mm: float          # Value in millions
    category: str            # "equity", "bond", "money_market"
    direction: str           # "inflow", "outflow"

    def to_dict(self) -> Dict:
        return {
            "date": self.date,
            "source": self.source,
            "value_mm": self.value_mm,
            "category": self.category,
            "direction": self.direction,
        }


@dataclass
class StressIndicator:
    """Single stress metric observation"""

    name: str
    date: str
    value: float
    prior_value: float
    threshold_warning: float
    threshold_critical: float
    weight: float = 1.0

    @property
    def level(self) -> StressLevel:
        """Determine stress level from thresholds"""
        if self.value >= self.threshold_critical:
            return StressLevel.CRITICAL
        elif self.value >= self.threshold_warning:
            return StressLevel.HIGH
        elif self.value > self.prior_value:
            return StressLevel.ELEVATED
        return StressLevel.LOW

    @property
    def change_pct(self) -> float:
        """Percentage change from prior"""
        if self.prior_value == 0:
            return 0
        return ((self.value - self.prior_value) / self.prior_value) * 100


@dataclass
class FlowSnapshot:
    """Complete picture of passive flows at a point in time"""

    date: str

    # Flow data
    weekly_equity_flow_mm: float
    weekly_bond_flow_mm: float
    weekly_mm_flow_mm: float  # Money market

    # 4-week averages
    avg_4wk_equity_mm: float
    avg_4wk_bond_mm: float

    # Regime classification
    regime: FlowRegime
    velocity: FlowVelocity

    # Stress composite
    stress_index: float  # 0-100

    def to_dict(self) -> Dict:
        return {
            "date": self.date,
            "weekly_equity_flow_mm": self.weekly_equity_flow_mm,
            "weekly_bond_flow_mm": self.weekly_bond_flow_mm,
            "regime": self.regime.value,
            "velocity": self.velocity.value,
            "stress_index": self.stress_index,
        }


class PassiveFlowDetector:
    """PASSIVE FLOW DETECTOR

    Monitors structural flows into passive investments and
    detects when the bid is weakening.

    Key Data Sources:
    - ICI weekly fund flows
    - ETF creation/redemption data
    - 401k contribution estimates
    - Hardship withdrawal data
    - Consumer stress indicators

    Signals:
    - Flow regime (inflow/outflow and strength)
    - Flow velocity (accelerating/decelerating)
    - Stress index (composite of leading indicators)
    """

    def __init__(self):
        self.flow_history: List[FlowSnapshot] = []
        self.stress_indicators: List[StressIndicator] = []
        self.current_regime: FlowRegime = FlowRegime.NEUTRAL

        # Stress indicator definitions
        self.stress_metrics = {
            "credit_card_delinquency_rate": {
                "warning": 3.0,
                "critical": 4.5,
                "weight": 1.5,
            },
            "401k_hardship_withdrawal_rate": {
                "warning": 2.5,
                "critical": 4.0,
                "weight": 2.0,  # High weight - directly affects flows
            },
            "unemployment_claims_4wk_avg": {
                "warning": 250000,
                "critical": 350000,
                "weight": 1.2,
            },
            "consumer_sentiment_index": {
                "warning": 65,  # Inverted - low is bad
                "critical": 55,
                "weight": 1.0,
                "inverted": True,
            },
            "savings_rate": {
                "warning": 4.0,  # Inverted - low is bad
                "critical": 2.5,
                "weight": 1.3,
                "inverted": True,
            },
        }

    def add_flow_data(self, data: FlowDataPoint) -> None:
        """Add a flow data observation"""
        # In production, would update rolling calculations
        logger.info(f"Added flow data: {data.source} {data.date} ${data.value_mm}M")

    def add_stress_indicator(self, indicator: StressIndicator) -> None:
        """Add a stress indicator observation"""
        self.stress_indicators.append(indicator)
        logger.info(f"Added stress indicator: {indicator.name} = {indicator.value} ({indicator.level.value})")

    def calculate_stress_index(self) -> float:
        """Calculate composite stress index (0-100).
        Higher = more stress = more likely outflows.
        """
        if not self.stress_indicators:
            return 50.0  # Neutral if no data

        weighted_sum = 0.0
        total_weight = 0.0

        for indicator in self.stress_indicators:
            metric_config = self.stress_metrics.get(indicator.name, {})
            weight = metric_config.get("weight", 1.0)
            inverted = metric_config.get("inverted", False)

            # Score each indicator 0-100
            warning = metric_config.get("warning", indicator.threshold_warning)
            critical = metric_config.get("critical", indicator.threshold_critical)

            if inverted:
                # Lower is worse
                if indicator.value <= critical:
                    score = 100
                elif indicator.value <= warning:
                    score = 50 + 50 * (warning - indicator.value) / (warning - critical)
                else:
                    score = max(0, 50 - (indicator.value - warning) * 2)
            else:
                # Higher is worse
                if indicator.value >= critical:
                    score = 100
                elif indicator.value >= warning:
                    score = 50 + 50 * (indicator.value - warning) / (critical - warning)
                else:
                    score = max(0, 50 * indicator.value / warning)

            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 50.0

    def classify_regime(
        self,
        weekly_flow_mm: float,
        avg_4wk_flow_mm: float,
    ) -> Tuple[FlowRegime, FlowVelocity]:
        """Classify current flow regime and velocity.
        """
        # Determine regime
        if weekly_flow_mm > 10000:
            regime = FlowRegime.STRONG_INFLOW
        elif weekly_flow_mm > 5000:
            regime = FlowRegime.MODERATE_INFLOW
        elif weekly_flow_mm > 0:
            regime = FlowRegime.WEAK_INFLOW
        elif weekly_flow_mm > -5000:
            regime = FlowRegime.WEAK_OUTFLOW
        elif weekly_flow_mm > -10000:
            regime = FlowRegime.MODERATE_OUTFLOW
        else:
            regime = FlowRegime.STRONG_OUTFLOW

        # Determine velocity
        if abs(weekly_flow_mm) > abs(avg_4wk_flow_mm) * 1.2:
            if (weekly_flow_mm > 0 and avg_4wk_flow_mm > 0) or (weekly_flow_mm < 0 and avg_4wk_flow_mm < 0):
                velocity = FlowVelocity.ACCELERATING
            else:
                velocity = FlowVelocity.REVERSING
        elif abs(weekly_flow_mm) < abs(avg_4wk_flow_mm) * 0.8:
            velocity = FlowVelocity.DECELERATING
        else:
            velocity = FlowVelocity.STABLE

        return regime, velocity

    def get_current_snapshot(
        self,
        weekly_equity_mm: float,
        weekly_bond_mm: float,
        weekly_mm_mm: float,
        avg_4wk_equity_mm: float,
        avg_4wk_bond_mm: float,
    ) -> FlowSnapshot:
        """Generate current flow snapshot with regime classification.
        """
        regime, velocity = self.classify_regime(weekly_equity_mm, avg_4wk_equity_mm)
        stress_index = self.calculate_stress_index()

        snapshot = FlowSnapshot(
            date=datetime.now().strftime("%Y-%m-%d"),
            weekly_equity_flow_mm=weekly_equity_mm,
            weekly_bond_flow_mm=weekly_bond_mm,
            weekly_mm_flow_mm=weekly_mm_mm,
            avg_4wk_equity_mm=avg_4wk_equity_mm,
            avg_4wk_bond_mm=avg_4wk_bond_mm,
            regime=regime,
            velocity=velocity,
            stress_index=stress_index,
        )

        self.flow_history.append(snapshot)
        self.current_regime = regime

        return snapshot

    def get_trading_signal(self) -> Dict:
        """Generate trading signal based on flow analysis.

        Returns positioning guidance based on:
        - Current regime
        - Velocity
        - Stress index
        """
        if not self.flow_history:
            return {"signal": "NEUTRAL", "confidence": 0, "rationale": "No data"}

        latest = self.flow_history[-1]

        # Decision matrix
        if latest.regime in [FlowRegime.STRONG_OUTFLOW, FlowRegime.MODERATE_OUTFLOW]:
            if latest.velocity == FlowVelocity.ACCELERATING:
                signal = "DEFENSIVE"
                confidence = 85
                actions = [
                    "Reduce equity exposure by 20-30%",
                    "Favor dividend payers over growth",
                    "Add protective puts on winners",
                    "Increase cash allocation",
                ]
            else:
                signal = "CAUTIOUS"
                confidence = 70
                actions = [
                    "Reduce equity exposure by 10-15%",
                    "Tighten stops",
                    "Favor quality",
                ]
        elif latest.regime in [FlowRegime.STRONG_INFLOW, FlowRegime.MODERATE_INFLOW]:
            if latest.velocity == FlowVelocity.ACCELERATING:
                signal = "RISK_ON"
                confidence = 75
                actions = [
                    "Maintain/increase equity exposure",
                    "Lean into momentum",
                    "Can remove hedges",
                ]
            else:
                signal = "NEUTRAL_BULLISH"
                confidence = 60
                actions = [
                    "Maintain positions",
                    "Selective additions",
                ]
        else:
            signal = "NEUTRAL"
            confidence = 50
            actions = ["Focus on stock selection over macro"]

        # Adjust for stress index
        if latest.stress_index > 75 and signal not in ["DEFENSIVE", "CAUTIOUS"]:
            signal = "CAUTIOUS"
            confidence = min(confidence + 10, 90)
            actions.insert(0, "STRESS WARNING: Consumer stress elevated")

        return {
            "signal": signal,
            "confidence": confidence,
            "regime": latest.regime.value,
            "velocity": latest.velocity.value,
            "stress_index": round(latest.stress_index, 1),
            "actions": actions,
            "date": latest.date,
        }

    def generate_report(self) -> str:
        """Generate human-readable flow report"""
        if not self.flow_history:
            return "No flow data available"

        latest = self.flow_history[-1]
        signal = self.get_trading_signal()

        lines = [
            "=" * 60,
            "PASSIVE FLOW DETECTOR - WEEKLY REPORT",
            f"Date: {latest.date}",
            "=" * 60,
            "",
            f"📊 FLOW REGIME: {latest.regime.value.upper()}",
            f"📈 VELOCITY: {latest.velocity.value}",
            f"⚠️ STRESS INDEX: {latest.stress_index:.1f}/100",
            "",
            "WEEKLY FLOWS:",
            f"  Equity: ${latest.weekly_equity_flow_mm:,.0f}M",
            f"  Bond: ${latest.weekly_bond_flow_mm:,.0f}M",
            f"  Money Market: ${latest.weekly_mm_flow_mm:,.0f}M",
            "",
            f"🎯 SIGNAL: {signal['signal']} (Confidence: {signal['confidence']}%)",
            "",
            "RECOMMENDED ACTIONS:",
        ]

        for action in signal["actions"]:
            lines.append(f"  • {action}")

        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    detector = PassiveFlowDetector()

    # Add some stress indicators
    detector.add_stress_indicator(StressIndicator(
        name="credit_card_delinquency_rate",
        date="2024-12-01",
        value=3.8,
        prior_value=3.2,
        threshold_warning=3.0,
        threshold_critical=4.5,
    ))

    detector.add_stress_indicator(StressIndicator(
        name="401k_hardship_withdrawal_rate",
        date="2024-12-01",
        value=3.1,
        prior_value=2.6,
        threshold_warning=2.5,
        threshold_critical=4.0,
    ))

    # Generate snapshot
    snapshot = detector.get_current_snapshot(
        weekly_equity_mm=-2500,
        weekly_bond_mm=1500,
        weekly_mm_mm=8000,
        avg_4wk_equity_mm=1000,
        avg_4wk_bond_mm=2000,
    )

    # Print report
    print(detector.generate_report())


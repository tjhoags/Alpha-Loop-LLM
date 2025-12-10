"""================================================================================
INFLECTION POINT DETECTOR - Core Holdings Focus
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

PURPOSE: Detect inflection points in core holdings (8-16% positions, >6 month
horizon, sub-$25bn market cap) using BOTH conventional AND unconventional signals.

THIS IS YOUR EDGE. Most funds look at the same data. We look at everything.

INFLECTION TYPES WE DETECT:
1. Revenue Acceleration - Growth rate increasing
2. Margin Expansion - Operating leverage kicking in
3. Market Share Gain - Taking share from competitors
4. Product Cycle Beginning - New product/service launch
5. Management Quality Improvement - Better execution
6. Competitive Position Strengthening - Moat widening
7. Sentiment Inflection - Perception changing before fundamentals
8. Institutional Discovery - Smart money arriving

SIGNAL SOURCES:
- Traditional: Financials, estimates, technicals, sentiment
- Unconventional: Employee reviews, patents, supplier filings, local news, etc.

PHILOSOPHY:
"The best time to buy is when nobody wants it. The best time to sell is
when everyone wants it. Inflection detection is about finding the turn."

TARGET: Sub-$25bn market cap (where inefficiencies exist)
HORIZON: 6-12 months (fundamental inflections take time)
POSITION SIZE: 8-16% (concentrated, conviction-based)
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.core.agent_base import AgentTier, BaseAgent

logger = logging.getLogger(__name__)


class InflectionType(Enum):
    """Types of inflection points we can detect."""

    REVENUE_ACCELERATION = "revenue_acceleration"
    MARGIN_EXPANSION = "margin_expansion"
    MARKET_SHARE_GAIN = "market_share_gain"
    PRODUCT_CYCLE = "product_cycle"
    MANAGEMENT_IMPROVEMENT = "management_improvement"
    COMPETITIVE_STRENGTHENING = "competitive_strengthening"
    SENTIMENT_INFLECTION = "sentiment_inflection"
    INSTITUTIONAL_DISCOVERY = "institutional_discovery"

    # Negative inflections
    REVENUE_DECELERATION = "revenue_deceleration"
    MARGIN_COMPRESSION = "margin_compression"
    MARKET_SHARE_LOSS = "market_share_loss"
    COMPETITIVE_WEAKENING = "competitive_weakening"


class ConvictionLevel(Enum):
    """How confident are we in this inflection?"""

    VERY_HIGH = "very_high"    # 80%+ confidence, multiple signals
    HIGH = "high"              # 70-80% confidence
    MODERATE = "moderate"      # 55-70% confidence
    LOW = "low"               # 40-55% confidence
    SPECULATIVE = "speculative"  # <40% confidence


class TimeHorizon(Enum):
    """Expected time for inflection to play out."""

    IMMEDIATE = "immediate"    # 0-3 months
    NEAR_TERM = "near_term"    # 3-6 months
    MEDIUM_TERM = "medium_term"  # 6-12 months
    LONG_TERM = "long_term"    # 12-24 months


@dataclass
class Signal:
    """An individual signal contributing to inflection detection."""

    signal_id: str
    signal_type: str  # "traditional" or "unconventional"
    signal_source: str  # e.g., "fundamentals", "glassdoor", "patents"
    timestamp: datetime

    # Signal details
    name: str
    description: str

    # Quantification
    raw_value: float  # The actual measurement
    normalized_score: float  # -1 to 1 scale
    confidence: float  # 0 to 1

    # Metadata
    is_leading: bool  # Does this lead fundamental changes?
    lead_time_months: float  # How far ahead does it signal?
    historical_accuracy: float  # How accurate has this signal type been?

    def to_dict(self) -> Dict:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "source": self.signal_source,
            "name": self.name,
            "normalized_score": self.normalized_score,
            "confidence": self.confidence,
            "is_leading": self.is_leading,
        }


@dataclass
class InflectionDetection:
    """A detected inflection point."""

    detection_id: str
    ticker: str
    company_name: str
    market_cap_billions: float

    timestamp: datetime

    # Inflection classification
    inflection_type: InflectionType
    direction: str  # "positive" or "negative"
    time_horizon: TimeHorizon

    # Scoring
    composite_score: float  # 0 to 100
    conviction_level: ConvictionLevel

    # Supporting signals
    traditional_signals: List[Signal]
    unconventional_signals: List[Signal]

    # Action recommendation
    position_action: str  # "add", "hold", "trim", "exit"
    target_weight_pct: float  # Recommended portfolio weight
    current_weight_pct: float  # Current portfolio weight

    # Scenarios
    bull_case: str
    bear_case: str
    base_case: str

    # Risk/reward
    upside_target_pct: float  # Expected upside
    downside_risk_pct: float  # Expected downside
    risk_reward_ratio: float

    # Monitoring
    key_metrics_to_watch: List[str]
    catalyst_events: List[Dict]
    next_review_date: datetime

    def to_dict(self) -> Dict:
        return {
            "detection_id": self.detection_id,
            "ticker": self.ticker,
            "company": self.company_name,
            "market_cap_bn": self.market_cap_billions,
            "inflection_type": self.inflection_type.value,
            "direction": self.direction,
            "time_horizon": self.time_horizon.value,
            "composite_score": self.composite_score,
            "conviction": self.conviction_level.value,
            "position_action": self.position_action,
            "target_weight_pct": self.target_weight_pct,
            "traditional_signals": len(self.traditional_signals),
            "unconventional_signals": len(self.unconventional_signals),
            "risk_reward": self.risk_reward_ratio,
        }


class InflectionDetector(BaseAgent):
    """INFLECTION POINT DETECTOR

    Specialized for Tom's investment style:
    - Core holdings: 8-16% positions
    - Long horizon: >6 months
    - Sub-$25bn market cap (where inefficiencies exist)
    - Concentrated portfolio (70% in core holdings)

    This agent combines:
    1. Traditional signals (financials, estimates, technicals)
    2. Unconventional signals (employee sentiment, patents, local news, etc.)

    To detect inflection points BEFORE they're obvious.

    KEY PRINCIPLE:
    The best inflections are the ones where the market hasn't caught on yet.
    By the time it's in the Wall Street Journal, the move is mostly done.
    """

    # Signal weights for composite scoring
    SIGNAL_WEIGHTS = {
        # Traditional signals
        "revenue_growth": 0.12,
        "earnings_revision": 0.10,
        "margin_trend": 0.08,
        "guidance_change": 0.10,
        "analyst_revision": 0.06,
        "technical_momentum": 0.05,
        "insider_activity": 0.08,
        "institutional_flows": 0.06,

        # Unconventional signals (THESE ARE THE EDGE)
        "employee_sentiment": 0.08,
        "supplier_signals": 0.06,
        "patent_activity": 0.04,
        "regulatory_filings": 0.05,
        "local_news": 0.03,
        "geographic_hiring": 0.04,
        "competitor_distress": 0.05,
    }

    # Conviction thresholds
    CONVICTION_THRESHOLDS = {
        ConvictionLevel.VERY_HIGH: 75,
        ConvictionLevel.HIGH: 60,
        ConvictionLevel.MODERATE: 45,
        ConvictionLevel.LOW: 30,
        ConvictionLevel.SPECULATIVE: 0,
    }

    def __init__(self, user_id: str = "TJH"):
        super().__init__(
            name="InflectionDetector",
            tier=AgentTier.SENIOR,
            capabilities=[
                # Core detection
                "inflection_detection",
                "multi_signal_synthesis",
                "conviction_scoring",
                "time_horizon_estimation",

                # Signal analysis
                "traditional_signal_analysis",
                "unconventional_signal_analysis",
                "signal_correlation_detection",
                "leading_indicator_identification",

                # Position management
                "position_recommendation",
                "target_weight_calculation",
                "risk_reward_assessment",

                # Monitoring
                "key_metric_identification",
                "catalyst_tracking",
                "review_scheduling",

                # Reporting
                "hoags_escalation",
                "inflection_report_generation",
            ],
            user_id=user_id,
        )

        # Core holdings configuration
        self.core_holdings: List[str] = []
        self.max_market_cap_billions = 25.0
        self.min_position_pct = 0.08  # 8%
        self.max_position_pct = 0.16  # 16%
        self.target_holding_period_months = 6

        # Detection storage
        self.detections: List[InflectionDetection] = []
        self.signals: Dict[str, List[Signal]] = {}  # ticker -> signals

        # Performance tracking
        self.detections_total = 0
        self.detections_correct = 0
        self.total_alpha_captured_bps = 0

        self.logger.info("InflectionDetector initialized - Core holdings focused")

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process inflection detection tasks."""
        action = task.get("action", task.get("type", ""))
        params = task.get("parameters", task)

        self.log_action(action, f"InflectionDetector processing: {action}")

        handlers = {
            "detect_inflection": self._handle_detect_inflection,
            "scan_core_holdings": self._handle_scan_holdings,
            "add_signal": self._handle_add_signal,
            "get_detections": self._handle_get_detections,
            "set_core_holdings": self._handle_set_holdings,
            "get_recommendations": self._handle_get_recommendations,
        }

        handler = handlers.get(action, self._handle_unknown)
        return handler(params)

    def get_capabilities(self) -> List[str]:
        return self.capabilities

    # =========================================================================
    # SIGNAL COLLECTION
    # =========================================================================

    def add_traditional_signal(
        self,
        ticker: str,
        signal_source: str,
        name: str,
        raw_value: float,
        description: str = "",
    ) -> Signal:
        """Add a traditional (fundamental/technical) signal.

        TRADITIONAL SIGNALS:
        - Revenue growth rate change
        - Earnings revision direction
        - Margin expansion/compression
        - Guidance changes
        - Analyst estimate revisions
        - Technical breakouts
        - Insider buying clusters
        - Institutional accumulation
        """
        # Normalize value to -1 to 1 scale based on signal type
        normalized = self._normalize_traditional_signal(signal_source, raw_value)

        signal = Signal(
            signal_id=f"trad_{hashlib.sha256(f'{ticker}{signal_source}{datetime.now()}'.encode()).hexdigest()[:8]}",
            signal_type="traditional",
            signal_source=signal_source,
            timestamp=datetime.now(),
            name=name,
            description=description,
            raw_value=raw_value,
            normalized_score=normalized,
            confidence=0.7,  # Traditional signals are generally more reliable
            is_leading=signal_source in ["earnings_revision", "guidance_change", "insider_activity"],
            lead_time_months=3.0,  # Traditional signals lead by ~3 months
            historical_accuracy=0.6,  # Traditional signals are ~60% accurate
        )

        if ticker not in self.signals:
            self.signals[ticker] = []
        self.signals[ticker].append(signal)

        return signal

    def add_unconventional_signal(
        self,
        ticker: str,
        signal_source: str,
        name: str,
        raw_value: float,
        description: str = "",
        lead_time_months: float = 6.0,
    ) -> Signal:
        """Add an unconventional signal (THIS IS YOUR EDGE).

        UNCONVENTIONAL SIGNALS:
        - Employee sentiment (Glassdoor, Blind)
        - Supplier filing signals
        - Patent activity changes
        - Regulatory filing patterns
        - Local news mentions
        - Geographic hiring patterns
        - Competitor distress signals
        """
        # Normalize based on unconventional signal type
        normalized = self._normalize_unconventional_signal(signal_source, raw_value)

        signal = Signal(
            signal_id=f"unconv_{hashlib.sha256(f'{ticker}{signal_source}{datetime.now()}'.encode()).hexdigest()[:8]}",
            signal_type="unconventional",
            signal_source=signal_source,
            timestamp=datetime.now(),
            name=name,
            description=description,
            raw_value=raw_value,
            normalized_score=normalized,
            confidence=0.55,  # Unconventional signals are noisier
            is_leading=True,  # By definition, these lead traditional signals
            lead_time_months=lead_time_months,
            historical_accuracy=0.5,  # Lower hit rate but higher payoff
        )

        if ticker not in self.signals:
            self.signals[ticker] = []
        self.signals[ticker].append(signal)

        return signal

    def _normalize_traditional_signal(self, source: str, value: float) -> float:
        """Normalize traditional signal to -1 to 1 scale."""
        normalizers = {
            "revenue_growth": lambda x: max(-1, min(1, x / 30)),  # 30% growth = max
            "earnings_revision": lambda x: max(-1, min(1, x / 20)),  # 20% revision = max
            "margin_trend": lambda x: max(-1, min(1, x / 5)),  # 5% margin change = max
            "guidance_change": lambda x: max(-1, min(1, x / 15)),  # 15% guidance change = max
            "analyst_revision": lambda x: max(-1, min(1, x / 10)),  # 10% estimate change = max
            "technical_momentum": lambda x: max(-1, min(1, x)),  # Already scaled
            "insider_activity": lambda x: max(-1, min(1, x / 100)),  # 100 = max score
            "institutional_flows": lambda x: max(-1, min(1, x)),  # Already scaled
        }

        normalizer = normalizers.get(source, lambda x: max(-1, min(1, x)))
        return normalizer(value)

    def _normalize_unconventional_signal(self, source: str, value: float) -> float:
        """Normalize unconventional signal to -1 to 1 scale."""
        normalizers = {
            "employee_sentiment": lambda x: max(-1, min(1, x / 2)),  # -2 to 2 rating change
            "supplier_signals": lambda x: max(-1, min(1, x)),  # Already scaled
            "patent_activity": lambda x: max(-1, min(1, x / 50)),  # 50% YoY change = max
            "regulatory_filings": lambda x: max(-1, min(1, x)),  # Already scaled
            "local_news": lambda x: max(-1, min(1, x)),  # Sentiment score
            "geographic_hiring": lambda x: max(-1, min(1, x / 100)),  # 100% increase = max
            "competitor_distress": lambda x: max(-1, min(1, x)),  # Already scaled
        }

        normalizer = normalizers.get(source, lambda x: max(-1, min(1, x)))
        return normalizer(value)

    # =========================================================================
    # INFLECTION DETECTION
    # =========================================================================

    def detect_inflection(
        self,
        ticker: str,
        company_name: str = None,
        market_cap_billions: float = None,
        current_weight_pct: float = 0.0,
    ) -> Optional[InflectionDetection]:
        """Detect inflection point for a stock.

        Combines all signals (traditional + unconventional) to identify
        potential inflection points.

        DETECTION LOGIC:
        1. Collect all recent signals (last 90 days)
        2. Weight by signal type and recency
        3. Calculate composite score
        4. Determine conviction level
        5. Classify inflection type
        6. Generate recommendation
        """
        ticker_signals = self.signals.get(ticker, [])

        # Filter to recent signals (last 90 days)
        cutoff = datetime.now() - timedelta(days=90)
        recent_signals = [s for s in ticker_signals if s.timestamp > cutoff]

        if len(recent_signals) < 3:
            self.logger.info(f"{ticker}: Not enough signals for detection ({len(recent_signals)})")
            return None

        # Separate by type
        traditional = [s for s in recent_signals if s.signal_type == "traditional"]
        unconventional = [s for s in recent_signals if s.signal_type == "unconventional"]

        # Calculate weighted composite score
        weighted_sum = 0
        total_weight = 0

        for signal in recent_signals:
            weight = self.SIGNAL_WEIGHTS.get(signal.signal_source, 0.05)

            # Adjust weight by recency (more recent = more weight)
            days_old = (datetime.now() - signal.timestamp).days
            recency_factor = max(0.5, 1 - days_old / 90)

            # Adjust weight by confidence
            confidence_factor = signal.confidence

            adjusted_weight = weight * recency_factor * confidence_factor
            weighted_sum += signal.normalized_score * adjusted_weight
            total_weight += adjusted_weight

        if total_weight == 0:
            return None

        # Scale to 0-100
        composite_score = (weighted_sum / total_weight + 1) * 50

        # Determine direction
        direction = "positive" if composite_score > 50 else "negative"

        # Determine conviction level
        conviction = self._determine_conviction(composite_score, len(traditional), len(unconventional))

        # Determine inflection type
        inflection_type = self._determine_inflection_type(recent_signals, direction)

        # Determine time horizon
        avg_lead_time = sum(s.lead_time_months for s in recent_signals) / len(recent_signals)
        if avg_lead_time < 3:
            time_horizon = TimeHorizon.NEAR_TERM
        elif avg_lead_time < 9:
            time_horizon = TimeHorizon.MEDIUM_TERM
        else:
            time_horizon = TimeHorizon.LONG_TERM

        # Generate position recommendation
        position_action, target_weight = self._determine_position_action(
            composite_score, conviction, current_weight_pct,
        )

        # Generate scenarios
        bull_case, bear_case, base_case = self._generate_scenarios(
            ticker, inflection_type, composite_score,
        )

        # Calculate risk/reward
        upside = self._estimate_upside(composite_score, inflection_type)
        downside = self._estimate_downside(composite_score, inflection_type)
        risk_reward = upside / max(0.01, downside)

        detection = InflectionDetection(
            detection_id=f"inf_{hashlib.sha256(f'{ticker}{datetime.now()}'.encode()).hexdigest()[:8]}",
            ticker=ticker,
            company_name=company_name or ticker,
            market_cap_billions=market_cap_billions or 0,
            timestamp=datetime.now(),
            inflection_type=inflection_type,
            direction=direction,
            time_horizon=time_horizon,
            composite_score=composite_score,
            conviction_level=conviction,
            traditional_signals=traditional,
            unconventional_signals=unconventional,
            position_action=position_action,
            target_weight_pct=target_weight,
            current_weight_pct=current_weight_pct,
            bull_case=bull_case,
            bear_case=bear_case,
            base_case=base_case,
            upside_target_pct=upside,
            downside_risk_pct=downside,
            risk_reward_ratio=risk_reward,
            key_metrics_to_watch=self._identify_key_metrics(inflection_type),
            catalyst_events=[],
            next_review_date=datetime.now() + timedelta(days=30),
        )

        self.detections.append(detection)
        self.detections_total += 1

        # Escalate high conviction detections to HOAGS
        if conviction in [ConvictionLevel.HIGH, ConvictionLevel.VERY_HIGH]:
            self._escalate_to_hoags(detection)

        return detection

    def _determine_conviction(
        self,
        score: float,
        traditional_count: int,
        unconventional_count: int,
    ) -> ConvictionLevel:
        """Determine conviction level.

        CONVICTION REQUIRES:
        - High score AND multiple corroborating signals
        - Bonus for unconventional signals (edge)
        """
        # Score-based thresholds
        base_conviction = None
        for level, threshold in sorted(self.CONVICTION_THRESHOLDS.items(),
                                       key=lambda x: x[1], reverse=True):
            if abs(score - 50) >= threshold / 2:  # Distance from neutral
                base_conviction = level
                break

        if base_conviction is None:
            base_conviction = ConvictionLevel.SPECULATIVE

        # Adjust based on signal count
        total_signals = traditional_count + unconventional_count

        if total_signals < 3:
            # Not enough signals, downgrade
            if base_conviction == ConvictionLevel.VERY_HIGH:
                return ConvictionLevel.HIGH
            elif base_conviction == ConvictionLevel.HIGH:
                return ConvictionLevel.MODERATE
        elif total_signals >= 5 and unconventional_count >= 2:
            # Multiple unconventional signals confirm, upgrade
            if base_conviction == ConvictionLevel.HIGH:
                return ConvictionLevel.VERY_HIGH
            elif base_conviction == ConvictionLevel.MODERATE:
                return ConvictionLevel.HIGH

        return base_conviction

    def _determine_inflection_type(
        self,
        signals: List[Signal],
        direction: str,
    ) -> InflectionType:
        """Determine what type of inflection this is."""
        # Count signal sources
        source_counts: Dict[str, float] = {}
        for signal in signals:
            source_counts[signal.signal_source] = source_counts.get(signal.signal_source, 0) + abs(signal.normalized_score)

        # Find dominant signal source
        dominant_source = max(source_counts.items(), key=lambda x: x[1])[0] if source_counts else ""

        # Map to inflection type
        type_mapping = {
            "revenue_growth": (InflectionType.REVENUE_ACCELERATION, InflectionType.REVENUE_DECELERATION),
            "margin_trend": (InflectionType.MARGIN_EXPANSION, InflectionType.MARGIN_COMPRESSION),
            "competitor_distress": (InflectionType.MARKET_SHARE_GAIN, InflectionType.MARKET_SHARE_LOSS),
            "patent_activity": (InflectionType.PRODUCT_CYCLE, InflectionType.COMPETITIVE_WEAKENING),
            "employee_sentiment": (InflectionType.MANAGEMENT_IMPROVEMENT, InflectionType.COMPETITIVE_WEAKENING),
            "institutional_flows": (InflectionType.INSTITUTIONAL_DISCOVERY, InflectionType.INSTITUTIONAL_DISCOVERY),
        }

        if dominant_source in type_mapping:
            positive_type, negative_type = type_mapping[dominant_source]
            return positive_type if direction == "positive" else negative_type

        # Default
        return InflectionType.SENTIMENT_INFLECTION if direction == "positive" else InflectionType.COMPETITIVE_WEAKENING

    def _determine_position_action(
        self,
        score: float,
        conviction: ConvictionLevel,
        current_weight: float,
    ) -> Tuple[str, float]:
        """Determine position action and target weight."""
        # Score is 0-100, 50 is neutral

        if score > 70 and conviction in [ConvictionLevel.HIGH, ConvictionLevel.VERY_HIGH]:
            # Strong positive inflection
            target = self.max_position_pct
            action = "add" if current_weight < target else "hold"
        elif score > 60:
            # Moderate positive inflection
            target = min(0.12, max(current_weight, 0.10))
            action = "add" if current_weight < target else "hold"
        elif score > 40:
            # Neutral
            target = current_weight
            action = "hold"
        elif score > 30:
            # Moderate negative
            target = max(0.05, current_weight * 0.7)  # Reduce 30%
            action = "trim" if current_weight > target else "hold"
        else:
            # Strong negative
            target = 0.0
            action = "exit"

        return action, target

    def _generate_scenarios(
        self,
        ticker: str,
        inflection_type: InflectionType,
        score: float,
    ) -> Tuple[str, str, str]:
        """Generate bull/bear/base case scenarios."""
        scenarios = {
            InflectionType.REVENUE_ACCELERATION: (
                "Revenue growth accelerates to 25%+, multiple expands",
                "Acceleration proves temporary, growth normalizes",
                "Revenue growth improves moderately, stock re-rates",
            ),
            InflectionType.MARGIN_EXPANSION: (
                "Operating leverage kicks in, margins expand 500bps+",
                "Cost pressures offset gains, margins flat",
                "Gradual margin expansion as scale benefits emerge",
            ),
            InflectionType.MARKET_SHARE_GAIN: (
                "Significant market share gains from struggling competitors",
                "Competition responds effectively, share gains modest",
                "Steady share gains as competitive position strengthens",
            ),
            InflectionType.PRODUCT_CYCLE: (
                "New product is a hit, drives significant revenue growth",
                "Product launch disappoints, limited revenue impact",
                "New product contributes incrementally to growth",
            ),
        }

        return scenarios.get(inflection_type, (
            "Inflection plays out positively, stock up 40%+",
            "Inflection fails to materialize, stock down 15%",
            "Gradual improvement, stock up 20%",
        ))

    def _estimate_upside(self, score: float, inflection_type: InflectionType) -> float:
        """Estimate upside potential."""
        # Base upside by inflection type
        base_upside = {
            InflectionType.REVENUE_ACCELERATION: 0.50,
            InflectionType.MARGIN_EXPANSION: 0.40,
            InflectionType.MARKET_SHARE_GAIN: 0.35,
            InflectionType.PRODUCT_CYCLE: 0.45,
            InflectionType.INSTITUTIONAL_DISCOVERY: 0.30,
        }.get(inflection_type, 0.25)

        # Scale by score strength
        score_factor = (score - 50) / 50 if score > 50 else 0

        return base_upside * (1 + score_factor)

    def _estimate_downside(self, score: float, inflection_type: InflectionType) -> float:
        """Estimate downside risk."""
        # Base downside (false positive case)
        return 0.15  # 15% downside if inflection fails

    def _identify_key_metrics(self, inflection_type: InflectionType) -> List[str]:
        """Identify key metrics to monitor for this inflection type."""
        metrics = {
            InflectionType.REVENUE_ACCELERATION: [
                "Quarterly revenue growth rate",
                "New customer acquisition",
                "Same-store/organic growth",
                "Bookings/backlog",
            ],
            InflectionType.MARGIN_EXPANSION: [
                "Gross margin %",
                "Operating margin %",
                "SG&A as % of revenue",
                "Cost of goods sold trend",
            ],
            InflectionType.MARKET_SHARE_GAIN: [
                "Market share estimates",
                "Competitor revenue trends",
                "Win rates",
                "Customer churn",
            ],
            InflectionType.PRODUCT_CYCLE: [
                "New product revenue %",
                "Product reviews/ratings",
                "Patent filings",
                "R&D spending efficiency",
            ],
        }

        return metrics.get(inflection_type, ["Revenue", "Margins", "Growth rate"])

    def _escalate_to_hoags(self, detection: InflectionDetection):
        """Escalate high conviction detection to HOAGS."""
        self.logger.critical(
            f"INFLECTION DETECTION -> HOAGS\n"
            f"Ticker: {detection.ticker}\n"
            f"Type: {detection.inflection_type.value}\n"
            f"Direction: {detection.direction.upper()}\n"
            f"Score: {detection.composite_score:.0f}/100\n"
            f"Conviction: {detection.conviction_level.value.upper()}\n"
            f"Action: {detection.position_action.upper()}\n"
            f"Target Weight: {detection.target_weight_pct:.1%}\n"
            f"Signals: {len(detection.traditional_signals)} traditional, "
            f"{len(detection.unconventional_signals)} unconventional\n"
            f"Risk/Reward: {detection.risk_reward_ratio:.1f}x",
        )

    # =========================================================================
    # TASK HANDLERS
    # =========================================================================

    def _handle_detect_inflection(self, params: Dict) -> Dict:
        detection = self.detect_inflection(
            ticker=params.get("ticker", ""),
            company_name=params.get("company_name"),
            market_cap_billions=params.get("market_cap_billions"),
            current_weight_pct=params.get("current_weight_pct", 0),
        )
        return {
            "status": "success",
            "detection": detection.to_dict() if detection else None,
        }

    def _handle_scan_holdings(self, params: Dict) -> Dict:
        """Scan all core holdings for inflections."""
        results = []

        for ticker in self.core_holdings:
            detection = self.detect_inflection(ticker)
            if detection:
                results.append(detection.to_dict())

        return {
            "status": "success",
            "holdings_scanned": len(self.core_holdings),
            "inflections_detected": len(results),
            "results": results,
        }

    def _handle_add_signal(self, params: Dict) -> Dict:
        signal_type = params.get("signal_type", "traditional")

        if signal_type == "traditional":
            signal = self.add_traditional_signal(
                ticker=params.get("ticker", ""),
                signal_source=params.get("signal_source", ""),
                name=params.get("name", ""),
                raw_value=params.get("raw_value", 0),
                description=params.get("description", ""),
            )
        else:
            signal = self.add_unconventional_signal(
                ticker=params.get("ticker", ""),
                signal_source=params.get("signal_source", ""),
                name=params.get("name", ""),
                raw_value=params.get("raw_value", 0),
                description=params.get("description", ""),
                lead_time_months=params.get("lead_time_months", 6.0),
            )

        return {"status": "success", "signal": signal.to_dict()}

    def _handle_get_detections(self, params: Dict) -> Dict:
        ticker = params.get("ticker")

        if ticker:
            detections = [d for d in self.detections if d.ticker == ticker]
        else:
            detections = self.detections[-20:]  # Last 20

        return {
            "status": "success",
            "detections": [d.to_dict() for d in detections],
            "total": len(detections),
        }

    def _handle_set_holdings(self, params: Dict) -> Dict:
        self.core_holdings = params.get("tickers", [])
        return {
            "status": "success",
            "core_holdings": self.core_holdings,
            "count": len(self.core_holdings),
        }

    def _handle_get_recommendations(self, params: Dict) -> Dict:
        """Get current recommendations from all detections."""
        recs = []

        for detection in self.detections:
            if detection.timestamp > datetime.now() - timedelta(days=30):
                recs.append({
                    "ticker": detection.ticker,
                    "action": detection.position_action,
                    "target_weight": detection.target_weight_pct,
                    "conviction": detection.conviction_level.value,
                    "score": detection.composite_score,
                    "risk_reward": detection.risk_reward_ratio,
                })

        # Sort by conviction and score
        recs.sort(key=lambda x: x["score"], reverse=True)

        return {
            "status": "success",
            "recommendations": recs,
            "add": [r for r in recs if r["action"] == "add"],
            "trim": [r for r in recs if r["action"] == "trim"],
            "exit": [r for r in recs if r["action"] == "exit"],
        }

    def _handle_unknown(self, params: Dict) -> Dict:
        return {"status": "error", "message": "Unknown action"}

    def log_action(self, action: str, description: str):
        self.logger.info(f"[InflectionDetector] {action}: {description}")


# =============================================================================
# SINGLETON
# =============================================================================

_inflection_detector: Optional[InflectionDetector] = None


def get_inflection_detector() -> InflectionDetector:
    """Get inflection detector singleton."""
    global _inflection_detector
    if _inflection_detector is None:
        _inflection_detector = InflectionDetector()
    return _inflection_detector


"""================================================================================
SIGNAL AGGREGATOR - Central Signal Orchestration
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Aggregates and coordinates all signal sources for unified alpha generation.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from .alternative_signals import *
from .behavioral_signals import BehavioralSignals
from .cross_asset_signals import CrossAssetSignals
from .event_driven_signals import EventDrivenSignals

# Import all signal modules
from .insider_signals import *
from .insider_signals_extended import InsiderSignalsExtended
from .macro_signals import *
from .momentum_signals import *
from .nlp_sentiment_signals import NLPSentimentSignals
from .options_derivatives_signals import OptionsDerivativesSignals
from .quant_signals import QuantSignals
from .regime_signals import RegimeSignals
from .research_signals import *
from .risk_signals import *
from .supply_chain_signals import SupplyChainSignals
from .valuation_signals import *

logger = logging.getLogger(__name__)


@dataclass
class AggregatedSignal:
    """Unified signal from aggregation."""

    signal_id: str
    source_module: str
    ticker: str
    direction: str
    raw_confidence: float
    adjusted_confidence: float
    description: str
    category: str
    subcategory: str
    supporting_signals: List[str] = field(default_factory=list)
    conflicting_signals: List[str] = field(default_factory=list)
    regime_alignment: float = 1.0
    final_score: float = 0.0
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class SignalAggregator:
    """MASTER SIGNAL AGGREGATOR

    Coordinates all signal sources:
    - Insider Intelligence (basic + extended)
    - Research/Fundamental Analysis
    - Alternative Data
    - Macro/Regime Analysis
    - Valuation Signals
    - Momentum/Technical Signals
    - Risk Signals
    - NLP/Sentiment Analysis
    - Supply Chain/Real Economy
    - Options/Derivatives
    - Event-Driven/Special Situations
    - Cross-Asset Relationships
    - Behavioral Finance
    - Quantitative Factors
    - Regime Detection
    """

    SIGNAL_CATEGORIES = {
        "insider": ["insider_signals", "insider_signals_extended"],
        "fundamental": ["research_signals", "valuation_signals"],
        "alternative": ["alternative_signals", "supply_chain_signals"],
        "macro": ["macro_signals", "regime_signals"],
        "technical": ["momentum_signals", "quant_signals"],
        "sentiment": ["nlp_sentiment_signals"],
        "derivatives": ["options_derivatives_signals"],
        "event": ["event_driven_signals"],
        "cross_asset": ["cross_asset_signals"],
        "behavioral": ["behavioral_signals"],
        "risk": ["risk_signals"],
    }

    def __init__(self):
        # Initialize all signal generators
        self.insider_extended = InsiderSignalsExtended()
        self.nlp_sentiment = NLPSentimentSignals()
        self.supply_chain = SupplyChainSignals()
        self.options = OptionsDerivativesSignals()
        self.events = EventDrivenSignals()
        self.cross_asset = CrossAssetSignals()
        self.behavioral = BehavioralSignals()
        self.quant = QuantSignals()
        self.regime = RegimeSignals()

        # Signal storage
        self.raw_signals: List[Any] = []
        self.aggregated_signals: List[AggregatedSignal] = []
        self.signal_history: List[Dict] = []

        # Configuration
        self.min_confidence = 0.50
        self.regime_weight = 0.15
        self.supporting_signal_boost = 0.05
        self.conflicting_signal_penalty = 0.08

        logger.info("SignalAggregator initialized with all signal modules")

    def collect_all_signals(self, ticker: str, market_data: Dict[str, Any]) -> List[Any]:
        """Collect signals from all sources for a given ticker.
        """
        all_signals = []

        # Each module would be called here with appropriate data
        # This is the orchestration point

        # Example: Options signals
        if "options_data" in market_data:
            opts = market_data["options_data"]
            # Would call various options signal methods

        # Example: NLP signals
        if "earnings_transcript" in market_data:
            # Would call NLP analysis methods
            pass

        # Example: Supply chain
        if "supply_chain_data" in market_data:
            # Would call supply chain signal methods
            pass

        # Example: Event driven
        if "corporate_events" in market_data:
            # Would call event signal methods
            pass

        self.raw_signals.extend(all_signals)
        return all_signals

    def aggregate_signals(
        self,
        ticker: str,
        signals: List[Any],
        current_regime: Dict[str, str] = None,
    ) -> List[AggregatedSignal]:
        """Aggregate and score signals for a ticker.

        1. Group signals by direction
        2. Check for confirmation/contradiction
        3. Apply regime adjustment
        4. Calculate final scores
        """
        if not signals:
            return []

        aggregated = []

        # Group by direction
        bullish_signals = [s for s in signals if getattr(s, "direction", "") == "bullish"]
        bearish_signals = [s for s in signals if getattr(s, "direction", "") == "bearish"]

        # Process each signal
        for signal in signals:
            direction = getattr(signal, "direction", "uncertain")
            raw_confidence = getattr(signal, "confidence", 0.5)

            # Count supporting/conflicting signals
            if direction == "bullish":
                supporting = len(bullish_signals) - 1
                conflicting = len(bearish_signals)
            elif direction == "bearish":
                supporting = len(bearish_signals) - 1
                conflicting = len(bullish_signals)
            else:
                supporting = 0
                conflicting = 0

            # Adjust confidence
            adjusted = raw_confidence
            adjusted += supporting * self.supporting_signal_boost
            adjusted -= conflicting * self.conflicting_signal_penalty
            adjusted = max(0.1, min(0.95, adjusted))

            # Regime alignment
            regime_alignment = self._calculate_regime_alignment(direction, current_regime)
            adjusted *= (1 + (regime_alignment - 0.5) * self.regime_weight)

            # Final score combines everything
            final_score = adjusted * (1 + supporting * 0.1) * regime_alignment

            agg = AggregatedSignal(
                signal_id=getattr(signal, "signal_id", "unknown"),
                source_module=signal.__class__.__name__,
                ticker=ticker,
                direction=direction,
                raw_confidence=raw_confidence,
                adjusted_confidence=adjusted,
                description=getattr(signal, "description", ""),
                category=self._get_signal_category(signal),
                subcategory=getattr(signal, "signal_type", "unknown"),
                supporting_signals=[getattr(s, "signal_id", "") for s in (bullish_signals if direction == "bullish" else bearish_signals)],
                conflicting_signals=[getattr(s, "signal_id", "") for s in (bearish_signals if direction == "bullish" else bullish_signals)],
                regime_alignment=regime_alignment,
                final_score=final_score,
            )

            aggregated.append(agg)

        # Sort by final score
        aggregated.sort(key=lambda x: x.final_score, reverse=True)

        self.aggregated_signals.extend(aggregated)
        return aggregated

    def _calculate_regime_alignment(self, direction: str, regime: Dict[str, str] = None) -> float:
        """Calculate how well a signal direction aligns with current regime.
        """
        if not regime:
            return 1.0

        alignment = 1.0

        # Risk appetite alignment
        risk_regime = regime.get("risk_appetite", "NEUTRAL")
        if direction == "bullish" and risk_regime == "RISK_ON":
            alignment *= 1.15
        elif direction == "bullish" and risk_regime == "RISK_OFF":
            alignment *= 0.85
        elif direction == "bearish" and risk_regime == "RISK_OFF":
            alignment *= 1.10
        elif direction == "bearish" and risk_regime == "RISK_ON":
            alignment *= 0.90

        # Trend alignment
        trend_regime = regime.get("trend", "CHOPPY")
        if direction == "bullish" and "UPTREND" in trend_regime:
            alignment *= 1.12
        elif direction == "bearish" and "DOWNTREND" in trend_regime:
            alignment *= 1.12

        # Volatility adjustment
        vol_regime = regime.get("volatility", "MID_VOL")
        if "HIGH" in vol_regime or "EXTREME" in vol_regime:
            alignment *= 0.95  # Reduce confidence in high vol

        return alignment

    def _get_signal_category(self, signal: Any) -> str:
        """Determine signal category from signal type."""
        signal_class = signal.__class__.__name__

        for category, modules in self.SIGNAL_CATEGORIES.items():
            for module in modules:
                if module.lower() in signal_class.lower():
                    return category

        return "other"

    def get_top_signals(
        self,
        ticker: str = None,
        direction: str = None,
        category: str = None,
        min_score: float = 0.0,
        limit: int = 10,
    ) -> List[AggregatedSignal]:
        """Get top signals with optional filters.
        """
        signals = self.aggregated_signals

        if ticker:
            signals = [s for s in signals if s.ticker == ticker]
        if direction:
            signals = [s for s in signals if s.direction == direction]
        if category:
            signals = [s for s in signals if s.category == category]
        if min_score > 0:
            signals = [s for s in signals if s.final_score >= min_score]

        return sorted(signals, key=lambda x: x.final_score, reverse=True)[:limit]

    def get_signal_summary(self, ticker: str = None) -> Dict[str, Any]:
        """Get summary statistics of signals.
        """
        signals = self.aggregated_signals
        if ticker:
            signals = [s for s in signals if s.ticker == ticker]

        if not signals:
            return {"total": 0}

        bullish = [s for s in signals if s.direction == "bullish"]
        bearish = [s for s in signals if s.direction == "bearish"]

        return {
            "total": len(signals),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "avg_bullish_score": sum(s.final_score for s in bullish) / len(bullish) if bullish else 0,
            "avg_bearish_score": sum(s.final_score for s in bearish) / len(bearish) if bearish else 0,
            "net_direction": "bullish" if len(bullish) > len(bearish) else "bearish" if len(bearish) > len(bullish) else "neutral",
            "by_category": self._count_by_category(signals),
            "top_signal": signals[0].to_dict() if signals else None,
        }

    def _count_by_category(self, signals: List[AggregatedSignal]) -> Dict[str, int]:
        """Count signals by category."""
        counts = {}
        for s in signals:
            counts[s.category] = counts.get(s.category, 0) + 1
        return counts

    def clear_signals(self):
        """Clear all stored signals."""
        self.raw_signals.clear()
        self.aggregated_signals.clear()

    def get_all_module_stats(self) -> Dict[str, Dict]:
        """Get statistics from all signal modules."""
        return {
            "insider_extended": self.insider_extended.get_stats(),
            "nlp_sentiment": self.nlp_sentiment.get_stats(),
            "supply_chain": self.supply_chain.get_stats(),
            "options": self.options.get_stats(),
            "events": self.events.get_stats(),
            "cross_asset": self.cross_asset.get_stats(),
            "behavioral": self.behavioral.get_stats(),
            "quant": self.quant.get_stats(),
            "regime": self.regime.get_stats(),
        }

    def get_current_regime(self) -> Dict[str, str]:
        """Get current regime from regime detector."""
        return self.regime.get_current_regime_summary()


# Convenience function
def create_signal_aggregator() -> SignalAggregator:
    """Create and return a configured SignalAggregator."""
    return SignalAggregator()

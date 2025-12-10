"""================================================================================
ALPHA LOOP SIGNALS PACKAGE
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Comprehensive signal generation framework for alpha discovery.

SIGNAL MODULES:
---------------
1. insider_signals - CFO/Audit Committee/Executive signals
2. insider_signals_extended - 10b5-1 modifications, 13D patterns
3. research_signals - Anti-signals, footnote analysis, hedging detection
4. alternative_signals - Freight, satellite, patent, job posting signals
5. macro_signals - Fed, yield curve, credit, regime signals
6. valuation_signals - Credit-equity, skew, convertible, activist signals
7. momentum_signals - Dark pool, odd-lot, options MM, MOC signals
8. risk_signals - Correlation breakdown, dispersion, vol-of-vol signals
9. nlp_sentiment_signals - Earnings call tone, WSB sentiment, CEO language
10. supply_chain_signals - Credit card, web traffic, supplier, inventory
11. options_derivatives_signals - UOA, skew, gamma, walls, LEAPS
12. event_driven_signals - M&A arb, spinoff, activist, lockup, debt cliff
13. cross_asset_signals - Credit-equity, FX, commodity, factor crowding
14. behavioral_signals - Anchoring, recency, herding, disposition effect
15. quant_signals - Stat arb, momentum crash, value spread, carry
16. regime_signals - Vol, correlation, trend, risk-on/off, economic cycle

USAGE:
------
from src.signals import SignalAggregator

aggregator = SignalAggregator()
signals = aggregator.collect_all_signals(ticker, market_data)
scored = aggregator.aggregate_signals(ticker, signals, current_regime)
top_signals = aggregator.get_top_signals(direction="bullish", limit=5)
================================================================================
"""

# =============================================================================
# CORE SIGNAL CLASSES
# =============================================================================

# Insider signals (CFO, Audit Committee, Executive patterns)
# Alternative data signals (freight, satellite, patents, job postings)
from .alternative_signals import (
    AlternativeDataSignals,
    PhysicalSignal,
)
from .behavioral_signals import BehavioralSignal, BehavioralSignals
from .cross_asset_signals import CrossAssetSignal, CrossAssetSignals
from .event_driven_signals import EventDrivenSignals, EventSignal
from .insider_signals import (
    InsiderSignalStrength,
    InsiderSkepticSignals,
    SkepticSignal,
)

# Extended signal classes
from .insider_signals_extended import ExtendedInsiderSignal, InsiderSignalsExtended

# Macro regime signals (Fed, yield curve, credit spreads)
from .macro_signals import (
    MacroRegimeSignals,
    MacroSignal,
)

# Microstructure/Momentum signals (dark pool, odd-lot, market-on-close)
from .momentum_signals import (
    MicrostructureSignal,
    MicrostructureSignals,
)
from .nlp_sentiment_signals import NLPSentimentSignals, NLPSignal
from .options_derivatives_signals import OptionsDerivativesSignals, OptionsSignal
from .quant_signals import QuantSignal, QuantSignals
from .regime_signals import RegimeSignal, RegimeSignals

# Research anti-signals (footnotes, hedging language, auditor changes)
from .research_signals import (
    AbsenceType,
    AntiSignal,
    ResearchAntiSignals,
)

# Tail risk signals (correlation breakdown, dispersion, vol-of-vol)
from .risk_signals import (
    TailRiskSignal,
    TailRiskSignals,
)

# Master aggregator
from .signal_aggregator import (
    AggregatedSignal,
    SignalAggregator,
    create_signal_aggregator,
)
from .supply_chain_signals import SupplyChainSignal, SupplyChainSignals

# Valuation discovery signals (credit-equity, skew, convertibles)
from .valuation_signals import (
    ValuationDiscoverySignals,
    ValuationSignal,
)

# =============================================================================
# CONVENIENCE FUNCTION WRAPPERS
# =============================================================================
# These create singletons and wrap class methods as functions for ease of use

_insider_signals = None
_research_signals = None
_alternative_signals = None
_macro_signals = None
_valuation_signals = None
_momentum_signals = None
_risk_signals = None


def get_insider_signals():
    """Get InsiderSkepticSignals singleton."""
    global _insider_signals
    if _insider_signals is None:
        _insider_signals = InsiderSkepticSignals()
    return _insider_signals


def get_research_signals():
    """Get ResearchAntiSignals singleton."""
    global _research_signals
    if _research_signals is None:
        _research_signals = ResearchAntiSignals()
    return _research_signals


def get_alternative_signals():
    """Get AlternativeDataSignals singleton."""
    global _alternative_signals
    if _alternative_signals is None:
        _alternative_signals = AlternativeDataSignals()
    return _alternative_signals


def get_macro_signals():
    """Get MacroRegimeSignals singleton."""
    global _macro_signals
    if _macro_signals is None:
        _macro_signals = MacroRegimeSignals()
    return _macro_signals


def get_valuation_signals():
    """Get ValuationDiscoverySignals singleton."""
    global _valuation_signals
    if _valuation_signals is None:
        _valuation_signals = ValuationDiscoverySignals()
    return _valuation_signals


def get_momentum_signals():
    """Get MicrostructureSignals singleton."""
    global _momentum_signals
    if _momentum_signals is None:
        _momentum_signals = MicrostructureSignals()
    return _momentum_signals


def get_risk_signals():
    """Get TailRiskSignals singleton."""
    global _risk_signals
    if _risk_signals is None:
        _risk_signals = TailRiskSignals()
    return _risk_signals


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core signal classes
    "InsiderSkepticSignals",
    "InsiderSignalStrength",
    "SkepticSignal",
    "ResearchAntiSignals",
    "AbsenceType",
    "AntiSignal",
    "AlternativeDataSignals",
    "PhysicalSignal",
    "MacroRegimeSignals",
    "MacroSignal",
    "ValuationDiscoverySignals",
    "ValuationSignal",
    "MicrostructureSignals",
    "MicrostructureSignal",
    "TailRiskSignals",
    "TailRiskSignal",

    # Extended signal classes
    "InsiderSignalsExtended",
    "ExtendedInsiderSignal",
    "NLPSentimentSignals",
    "NLPSignal",
    "SupplyChainSignals",
    "SupplyChainSignal",
    "OptionsDerivativesSignals",
    "OptionsSignal",
    "EventDrivenSignals",
    "EventSignal",
    "CrossAssetSignals",
    "CrossAssetSignal",
    "BehavioralSignals",
    "BehavioralSignal",
    "QuantSignals",
    "QuantSignal",
    "RegimeSignals",
    "RegimeSignal",

    # Factory functions
    "get_insider_signals",
    "get_research_signals",
    "get_alternative_signals",
    "get_macro_signals",
    "get_valuation_signals",
    "get_momentum_signals",
    "get_risk_signals",

    # Aggregator
    "SignalAggregator",
    "AggregatedSignal",
    "create_signal_aggregator",
]

# Module stats
SIGNAL_MODULE_COUNT = 16
SIGNAL_CLASS_COUNT = len([c for c in __all__ if c[0].isupper() and "Signal" in c])

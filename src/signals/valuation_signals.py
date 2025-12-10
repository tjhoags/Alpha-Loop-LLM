"""================================================================================
VALUATION AGENT - "PRICE DISCOVERY SIGNALS"
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Find mispricings by triangulating across markets.
Credit, options, converts - they all tell us something equity doesn't.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValuationSignal:
    """A signal from valuation divergence."""

    signal_id: str
    signal_type: str
    ticker: str
    direction: str
    confidence: float
    description: str
    fair_value_estimate: float
    current_price: float
    mispricing_pct: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class ValuationDiscoverySignals:
    """VALUATION DISCOVERY SIGNALS - Triangulate across markets

    1. Implied Equity from Credit - Back out equity from where credit trades
    2. Options Skew Fundamental Contradiction - Skew vs fundamentals divergence
    3. Convertible Bond Gamma Trap - Delta hedging forced flows
    4. SPAC Merger PIPE Discount - Smart money skepticism
    5. ETF Premium Discount Persistence - Discount reflects future
    6. Appraisal Arbitrage Predictor - Sophisticated shareholders disagree
    7. Activist Entry Point Predictor - Front-run the 13D
    """

    def __init__(self):
        self.signals_detected: List[ValuationSignal] = []

    def implied_equity_from_credit(
        self,
        ticker: str,
        credit_spread: float,
        recovery_rate: float,
        risk_free: float,
        current_equity_price: float,
        enterprise_value: float,
    ) -> Optional[ValuationSignal]:
        """Back out equity value implied by where credit trades.
        Credit markets are often smarter than equity.
        Divergence = opportunity in one or both.
        """
        import hashlib

        # Simplified Merton model - implied default prob from spread
        implied_default_prob = credit_spread / (1 - recovery_rate) / 100

        # Implied equity value (simplified)
        implied_equity_value = enterprise_value * (1 - implied_default_prob * 0.5)
        implied_per_share = implied_equity_value / 1e9  # Rough per share estimate

        mispricing = (implied_per_share - current_equity_price) / current_equity_price

        if abs(mispricing) < 0.15:
            return None

        direction = "bullish" if mispricing > 0 else "bearish"

        return ValuationSignal(
            signal_id=f"crd_{hashlib.sha256(f'{ticker}credit'.encode()).hexdigest()[:8]}",
            signal_type="credit_implied_equity",
            ticker=ticker,
            direction=direction,
            confidence=0.70,
            description=f"CREDIT-EQUITY DIVERGENCE: {ticker} credit implies equity {'undervalued' if mispricing > 0 else 'overvalued'} by {abs(mispricing):.0%}",
            fair_value_estimate=implied_per_share,
            current_price=current_equity_price,
            mispricing_pct=mispricing,
            evidence={"credit_spread": credit_spread, "implied_default": implied_default_prob},
        )

    def options_skew_fundamental_contradiction(
        self,
        ticker: str,
        put_skew_25delta: float,
        fundamental_health: str,
        news_sentiment: float,
    ) -> Optional[ValuationSignal]:
        """When 25-delta put skew explodes but fundamentals look fine.
        Someone knows something - or overreaction opportunity.
        """
        import hashlib

        # High put skew (>10) with good fundamentals = contradiction
        if put_skew_25delta > 10 and fundamental_health == "strong":
            return ValuationSignal(
                signal_id=f"skw_{hashlib.sha256(f'{ticker}skew'.encode()).hexdigest()[:8]}",
                signal_type="skew_fundamental_contradiction",
                ticker=ticker,
                direction="uncertain",  # Need investigation
                confidence=0.62,
                description=f"SKEW CONTRADICTION: {ticker} put skew at {put_skew_25delta:.1f} but fundamentals strong - investigate",
                fair_value_estimate=0,
                current_price=0,
                mispricing_pct=0,
                evidence={"put_skew": put_skew_25delta, "fundamentals": fundamental_health, "sentiment": news_sentiment},
            )
        elif put_skew_25delta < 5 and fundamental_health == "weak":
            return ValuationSignal(
                signal_id=f"skw_{hashlib.sha256(f'{ticker}lowskew'.encode()).hexdigest()[:8]}",
                signal_type="skew_fundamental_contradiction",
                ticker=ticker,
                direction="bearish",
                confidence=0.65,
                description=f"SKEW COMPLACENCY: {ticker} put skew low ({put_skew_25delta:.1f}) but fundamentals weak",
                fair_value_estimate=0,
                current_price=0,
                mispricing_pct=0,
                evidence={"put_skew": put_skew_25delta, "fundamentals": fundamental_health},
            )

        return None

    def convertible_bond_gamma_trap(
        self,
        ticker: str,
        convert_price: float,
        current_stock: float,
        delta: float,
        gamma: float,
        outstanding_converts: float,
    ) -> Optional[ValuationSignal]:
        """Track delta-hedging flows from convert desks.
        Near conversion prices = forced buying/selling.
        """
        import hashlib

        pct_to_convert = (convert_price - current_stock) / current_stock

        # Within 10% of conversion and high gamma
        if abs(pct_to_convert) < 0.10 and gamma > 0.05:
            direction = "bullish" if current_stock < convert_price else "bearish"
            hedge_flow = "buying" if direction == "bullish" else "selling"

            return ValuationSignal(
                signal_id=f"cvt_{hashlib.sha256(f'{ticker}gamma'.encode()).hexdigest()[:8]}",
                signal_type="convertible_gamma_trap",
                ticker=ticker,
                direction=direction,
                confidence=0.68,
                description=f"GAMMA TRAP: {ticker} near convert price (${convert_price:.2f}) - forced {hedge_flow} ahead",
                fair_value_estimate=convert_price,
                current_price=current_stock,
                mispricing_pct=pct_to_convert,
                evidence={"delta": delta, "gamma": gamma, "convert_price": convert_price},
            )

        return None

    def spac_merger_pipe_discount_signal(
        self,
        ticker: str,
        deal_price: float,
        pipe_price: float,
        pipe_size: float,
    ) -> Optional[ValuationSignal]:
        """PIPE discount to deal price predicts post-merger performance.
        Large PIPE discounts = smart money skepticism.
        """
        import hashlib

        discount = (deal_price - pipe_price) / deal_price

        if discount > 0.15:
            return ValuationSignal(
                signal_id=f"spc_{hashlib.sha256(f'{ticker}pipe'.encode()).hexdigest()[:8]}",
                signal_type="spac_pipe_discount",
                ticker=ticker,
                direction="bearish",
                confidence=0.72,
                description=f"PIPE SKEPTICISM: {ticker} PIPE at {discount:.0%} discount - smart money skeptical",
                fair_value_estimate=pipe_price,
                current_price=deal_price,
                mispricing_pct=-discount,
                evidence={"deal_price": deal_price, "pipe_price": pipe_price, "pipe_size": pipe_size},
            )
        elif discount < 0:  # Premium
            return ValuationSignal(
                signal_id=f"spc_{hashlib.sha256(f'{ticker}premium'.encode()).hexdigest()[:8]}",
                signal_type="spac_pipe_premium",
                ticker=ticker,
                direction="bullish",
                confidence=0.65,
                description=f"PIPE CONFIDENCE: {ticker} PIPE at {abs(discount):.0%} PREMIUM - smart money bullish",
                fair_value_estimate=pipe_price,
                current_price=deal_price,
                mispricing_pct=-discount,
                evidence={"deal_price": deal_price, "pipe_price": pipe_price},
            )

        return None

    def activist_entry_point_predictor(
        self,
        ticker: str,
        current_price: float,
        typical_activist_entry_discount: float,
        fundamental_triggers: List[str],
    ) -> Optional[ValuationSignal]:
        """Model typical activist entry points based on historical patterns.
        Front-run the 13D by predicting 13F accumulation.
        """
        import hashlib

        # If price is at typical activist entry and triggers present
        if len(fundamental_triggers) >= 2:
            return ValuationSignal(
                signal_id=f"act_{hashlib.sha256(f'{ticker}activist'.encode()).hexdigest()[:8]}",
                signal_type="activist_entry_predictor",
                ticker=ticker,
                direction="bullish",
                confidence=0.60,
                description=f"ACTIVIST SETUP: {ticker} at typical activist entry with {len(fundamental_triggers)} triggers",
                fair_value_estimate=current_price * (1 + typical_activist_entry_discount),
                current_price=current_price,
                mispricing_pct=typical_activist_entry_discount,
                evidence={"triggers": fundamental_triggers, "entry_discount": typical_activist_entry_discount},
            )

        return None

    def get_stats(self) -> Dict[str, Any]:
        return {"total_signals": len(self.signals_detected)}



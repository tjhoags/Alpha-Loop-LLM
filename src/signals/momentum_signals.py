"""================================================================================
MOMENTUM/TECHNICAL AGENT - "MICROSTRUCTURE SIGNALS"
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

The plumbing of the market tells us things.
Dark pools, odd lots, order imbalances - this is where the edge lives.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MicrostructureSignal:
    """A signal from market microstructure."""

    signal_id: str
    signal_type: str
    ticker: str
    direction: str
    confidence: float
    description: str
    timeframe: str
    expected_move: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class MicrostructureSignals:
    """MICROSTRUCTURE SIGNALS - Read the tape

    1. Dark Pool Print Sentiment - Large blocks tell direction
    2. Odd Lot Retail Capitulation - Small orders mark bottoms
    3. Options MM Inventory Stress - Hedging flows coming
    4. After Hours Volume Conviction - Sustained vs knee-jerk
    5. Short Sale Circuit Breaker Exploitation - SSR dynamics
    6. Market on Close Imbalance Momentum - Predict overnight
    """

    def __init__(self):
        self.signals_detected: List[MicrostructureSignal] = []

    def dark_pool_print_sentiment(
        self,
        ticker: str,
        large_prints_at_bid: int,
        large_prints_at_ask: int,
        total_dark_volume: float,
    ) -> Optional[MicrostructureSignal]:
        """Track dark pool trade sizes and timing.
        Large blocks at bid = distribution.
        Large blocks at ask = accumulation.
        """
        import hashlib

        if large_prints_at_bid + large_prints_at_ask < 10:
            return None

        bid_ratio = large_prints_at_bid / (large_prints_at_bid + large_prints_at_ask)

        if bid_ratio > 0.65:
            return MicrostructureSignal(
                signal_id=f"dp_{hashlib.sha256(f'{ticker}dark'.encode()).hexdigest()[:8]}",
                signal_type="dark_pool_sentiment",
                ticker=ticker,
                direction="bearish",
                confidence=0.65,
                description=f"DARK POOL DISTRIBUTION: {ticker} {bid_ratio:.0%} of large prints at bid",
                timeframe="intraday to 2 days",
                expected_move="Continued selling pressure",
                evidence={"at_bid": large_prints_at_bid, "at_ask": large_prints_at_ask},
            )
        elif bid_ratio < 0.35:
            return MicrostructureSignal(
                signal_id=f"dp_{hashlib.sha256(f'{ticker}accum'.encode()).hexdigest()[:8]}",
                signal_type="dark_pool_sentiment",
                ticker=ticker,
                direction="bullish",
                confidence=0.65,
                description=f"DARK POOL ACCUMULATION: {ticker} {1-bid_ratio:.0%} of large prints at ask",
                timeframe="intraday to 2 days",
                expected_move="Continued buying pressure",
                evidence={"at_bid": large_prints_at_bid, "at_ask": large_prints_at_ask},
            )

        return None

    def odd_lot_retail_capitulation(
        self,
        ticker: str,
        odd_lot_sell_ratio: float,
        normal_odd_lot_ratio: float,
        recent_price_decline: float,
    ) -> Optional[MicrostructureSignal]:
        """Spike in odd-lot selling = retail capitulation.
        Often marks local bottoms in quality names.
        """
        import hashlib

        ratio_spike = odd_lot_sell_ratio / normal_odd_lot_ratio if normal_odd_lot_ratio > 0 else 1

        if ratio_spike > 2.0 and recent_price_decline < -0.10:
            return MicrostructureSignal(
                signal_id=f"odd_{hashlib.sha256(f'{ticker}capit'.encode()).hexdigest()[:8]}",
                signal_type="odd_lot_capitulation",
                ticker=ticker,
                direction="bullish",
                confidence=0.68,
                description=f"RETAIL CAPITULATION: {ticker} odd-lot selling {ratio_spike:.1f}x normal after {recent_price_decline:.0%} decline",
                timeframe="1-5 days",
                expected_move="Potential local bottom forming",
                evidence={"odd_lot_spike": ratio_spike, "price_decline": recent_price_decline},
            )

        return None

    def options_market_maker_inventory_stress(
        self,
        ticker: str,
        quote_width_change: float,
        inventory_direction: str,
        gamma_exposure: float,
    ) -> Optional[MicrostructureSignal]:
        """Infer MM inventory from quote behavior.
        Stressed inventory = hedging flows coming.
        """
        import hashlib

        if quote_width_change > 0.5:  # Spreads widened 50%+
            hedge_direction = "selling" if inventory_direction == "long" else "buying"

            return MicrostructureSignal(
                signal_id=f"mm_{hashlib.sha256(f'{ticker}stress'.encode()).hexdigest()[:8]}",
                signal_type="mm_inventory_stress",
                ticker=ticker,
                direction="bearish" if hedge_direction == "selling" else "bullish",
                confidence=0.60,
                description=f"MM STRESS: {ticker} option spreads widened {quote_width_change:.0%} - expect {hedge_direction}",
                timeframe="intraday",
                expected_move=f"MM hedge {hedge_direction} flows",
                evidence={"spread_change": quote_width_change, "gamma": gamma_exposure},
            )

        return None

    def after_hours_volume_conviction(
        self,
        ticker: str,
        ah_volume: float,
        normal_ah_volume: float,
        volume_decay_rate: float,
        earnings_reaction: str,
    ) -> Optional[MicrostructureSignal]:
        """Post-earnings, track AH volume decay rate.
        Fast decay = knee-jerk reaction, fadeable.
        Sustained volume = real information.
        """
        import hashlib

        volume_ratio = ah_volume / normal_ah_volume if normal_ah_volume > 0 else 1

        if volume_ratio > 3 and volume_decay_rate < 0.3:
            return MicrostructureSignal(
                signal_id=f"ah_{hashlib.sha256(f'{ticker}sustained'.encode()).hexdigest()[:8]}",
                signal_type="ah_volume_conviction",
                ticker=ticker,
                direction=earnings_reaction,
                confidence=0.72,
                description=f"SUSTAINED AH: {ticker} volume {volume_ratio:.0f}x normal with slow decay - real information",
                timeframe="next day open",
                expected_move=f"Continue {earnings_reaction} trend",
                evidence={"volume_ratio": volume_ratio, "decay_rate": volume_decay_rate},
            )
        elif volume_ratio > 3 and volume_decay_rate > 0.7:
            return MicrostructureSignal(
                signal_id=f"ah_{hashlib.sha256(f'{ticker}kneejerk'.encode()).hexdigest()[:8]}",
                signal_type="ah_volume_conviction",
                ticker=ticker,
                direction="fade_" + earnings_reaction,
                confidence=0.62,
                description=f"KNEE-JERK AH: {ticker} volume spiked but fast decay - consider fading",
                timeframe="next day open",
                expected_move=f"Potential fade of initial {earnings_reaction} move",
                evidence={"volume_ratio": volume_ratio, "decay_rate": volume_decay_rate},
            )

        return None

    def market_on_close_imbalance_momentum(
        self,
        ticker: str,
        moc_imbalance: float,
        imbalance_source: str,
        historical_overnight_correlation: float,
    ) -> Optional[MicrostructureSignal]:
        """MOC imbalances predict overnight moves.
        Track WHO is causing imbalance (index rebal vs active).
        """
        import hashlib

        if abs(moc_imbalance) < 1_000_000:  # Less than $1M imbalance
            return None

        direction = "bullish" if moc_imbalance > 0 else "bearish"

        # Active flow more predictive than index rebal
        confidence = 0.58 if imbalance_source == "index_rebal" else 0.68

        return MicrostructureSignal(
            signal_id=f"moc_{hashlib.sha256(f'{ticker}imbal'.encode()).hexdigest()[:8]}",
            signal_type="moc_imbalance",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=f"MOC IMBALANCE: {ticker} ${abs(moc_imbalance/1e6):.1f}M {'buy' if moc_imbalance > 0 else 'sell'} ({imbalance_source})",
            timeframe="overnight",
            expected_move=f"Overnight {direction} bias",
            evidence={"imbalance": moc_imbalance, "source": imbalance_source, "hist_corr": historical_overnight_correlation},
        )

    def get_stats(self) -> Dict[str, Any]:
        return {"total_signals": len(self.signals_detected)}



"""================================================================================
OPTIONS & DERIVATIVES SIGNALS - The Smartest Money Speaks First
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Options and derivatives markets lead equity markets. Always.
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OptionsSignal:
    """Signal from options/derivatives market."""

    signal_id: str
    signal_type: str
    ticker: str
    direction: str
    confidence: float
    description: str
    option_evidence: Dict[str, Any]
    time_horizon: str  # "immediate", "1_week", "1_month", "multi_month"
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class OptionsDerivativesSignals:
    """OPTIONS & DERIVATIVES SIGNALS

    1. Unusual Options Activity Scanner - Large directional bets
    2. Put/Call Skew Divergence - Skew vs realized vol
    3. Term Structure Inversion - Front-month vs back-month
    4. Gamma Exposure Flip - When MMs flip from long to short gamma
    5. Variance Swap Fair Value - Implied vs realized variance
    6. Call Wall Detection - Strike price magnetism
    7. Put Wall Detection - Support from options
    8. Whale Options Flow - Block trades by smart money
    9. Earnings Straddle Pricing - Implied move vs historical
    10. LEAPS Accumulation Pattern - Long-term conviction bets
    """

    def __init__(self):
        self.signals_detected: List[OptionsSignal] = []

    def unusual_options_activity(
        self,
        ticker: str,
        contract_type: str,  # "call" or "put"
        strike: float,
        expiry_days: int,
        volume: int,
        open_interest: int,
        premium_paid: float,
        trade_side: str,  # "buy", "sell", "unknown"
    ) -> Optional[OptionsSignal]:
        """Large directional options bets signal informed money.

        Volume >> OI = new position
        Premium paid on offer = urgency
        """
        vol_to_oi = volume / open_interest if open_interest > 0 else volume

        if vol_to_oi < 2.0:
            return None

        is_buying = trade_side == "buy"

        if contract_type == "call" and is_buying:
            direction = "bullish"
            desc = f"UNUSUAL CALL: {ticker} ${strike} {expiry_days}d - {volume:,} contracts ({vol_to_oi:.1f}x OI)"
        elif contract_type == "put" and is_buying:
            direction = "bearish"
            desc = f"UNUSUAL PUT: {ticker} ${strike} {expiry_days}d - {volume:,} contracts ({vol_to_oi:.1f}x OI)"
        elif contract_type == "call" and not is_buying:
            direction = "bearish"  # Selling calls = bearish
            desc = f"UNUSUAL CALL SALE: {ticker} ${strike} {expiry_days}d - {volume:,} contracts sold"
        elif contract_type == "put" and not is_buying:
            direction = "bullish"  # Selling puts = bullish
            desc = f"UNUSUAL PUT SALE: {ticker} ${strike} {expiry_days}d - {volume:,} contracts sold"
        else:
            return None

        # Higher conviction for larger premium
        confidence = min(0.55 + (premium_paid / 1_000_000) * 0.1, 0.80)

        return OptionsSignal(
            signal_id=f"uoa_{hashlib.sha256(f'{ticker}{strike}{expiry_days}'.encode()).hexdigest()[:8]}",
            signal_type="unusual_options_activity",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            option_evidence={"vol_to_oi": vol_to_oi, "premium": premium_paid, "trade_side": trade_side},
            time_horizon="1_week" if expiry_days < 14 else "1_month",
        )

    def put_call_skew_divergence(
        self,
        ticker: str,
        current_skew: float,  # 25-delta put IV - 25-delta call IV
        historical_skew: float,
        realized_vol_30d: float,
        implied_vol_atm: float,
    ) -> Optional[OptionsSignal]:
        """Skew divergence from historical norm signals fear/greed.

        High skew = fear, but if realized vol is low = possible overreaction
        Low skew = complacency, possible tail risk ahead
        """
        skew_deviation = current_skew - historical_skew
        vol_ratio = implied_vol_atm / realized_vol_30d if realized_vol_30d > 0 else 1

        if abs(skew_deviation) < 0.05:
            return None

        if skew_deviation > 0.10 and vol_ratio < 1.0:
            # High skew but low implied/realized = puts may be cheap
            direction = "bearish"
            confidence = 0.65
            desc = f"SKEW SIGNAL: {ticker} puts expensive (skew +{skew_deviation:.0%}) but IV/RV low"
        elif skew_deviation < -0.05:
            # Low skew = complacency
            direction = "bearish"
            confidence = 0.60
            desc = f"COMPLACENCY: {ticker} skew collapsed ({skew_deviation:.0%}) - tail risk underpriced"
        elif skew_deviation > 0.15 and vol_ratio > 1.3:
            # Extreme fear, possibly overdone
            direction = "bullish"
            confidence = 0.58
            desc = f"FEAR EXTREME: {ticker} skew +{skew_deviation:.0%}, IV/RV {vol_ratio:.1f}x - possible overreaction"
        else:
            return None

        return OptionsSignal(
            signal_id=f"skew_{hashlib.sha256(f'{ticker}skew'.encode()).hexdigest()[:8]}",
            signal_type="put_call_skew",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            option_evidence={"skew": current_skew, "skew_dev": skew_deviation, "vol_ratio": vol_ratio},
            time_horizon="1_month",
        )

    def term_structure_inversion(
        self,
        ticker: str,
        front_month_iv: float,
        second_month_iv: float,
        third_month_iv: float,
    ) -> Optional[OptionsSignal]:
        """Term structure inversion = event expected.

        Normal: IV increases with time (uncertainty)
        Inverted: Near-term IV > far-term = event expected
        """
        inversion_1_2 = front_month_iv - second_month_iv
        inversion_2_3 = second_month_iv - third_month_iv

        if inversion_1_2 < 0.02:
            return None  # Not inverted enough

        if inversion_1_2 > 0.10:
            severity = "STRONG"
            confidence = 0.72
        elif inversion_1_2 > 0.05:
            severity = "MODERATE"
            confidence = 0.62
        else:
            severity = "WEAK"
            confidence = 0.55

        return OptionsSignal(
            signal_id=f"term_{hashlib.sha256(f'{ticker}term'.encode()).hexdigest()[:8]}",
            signal_type="term_structure_inversion",
            ticker=ticker,
            direction="uncertain",  # Inversion doesn't tell direction, just magnitude
            confidence=confidence,
            description=f"{severity} TERM INVERSION: {ticker} front IV {front_month_iv:.0%} > back {third_month_iv:.0%} - event expected",
            option_evidence={"front_iv": front_month_iv, "back_iv": third_month_iv, "spread": inversion_1_2},
            time_horizon="immediate",
        )

    def gamma_exposure_flip(
        self,
        ticker: str,
        dealer_gamma: float,  # Positive = long gamma, negative = short gamma
        previous_dealer_gamma: float,
        current_price: float,
    ) -> Optional[OptionsSignal]:
        """When dealers flip from long to short gamma, dynamics change.

        Long gamma: Dealers stabilize (buy dips, sell rips)
        Short gamma: Dealers amplify (sell dips, buy rips)
        """
        flip_detected = (dealer_gamma * previous_dealer_gamma) < 0

        if not flip_detected:
            return None

        if previous_dealer_gamma > 0 and dealer_gamma < 0:
            direction = "uncertain"
            risk_type = "AMPLIFICATION"
            desc = f"GAMMA FLIP: {ticker} dealers now SHORT gamma - expect amplified moves"
            confidence = 0.68
        else:  # previous < 0 and current > 0
            direction = "uncertain"
            risk_type = "STABILIZATION"
            desc = f"GAMMA FLIP: {ticker} dealers now LONG gamma - expect stabilization"
            confidence = 0.65

        return OptionsSignal(
            signal_id=f"gex_{hashlib.sha256(f'{ticker}gex'.encode()).hexdigest()[:8]}",
            signal_type="gamma_exposure_flip",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            option_evidence={"prev_gamma": previous_dealer_gamma, "curr_gamma": dealer_gamma},
            time_horizon="1_week",
        )

    def call_wall_detection(
        self,
        ticker: str,
        strike_open_interest: Dict[float, int],  # strike -> OI
        current_price: float,
        expiry_days: int,
    ) -> Optional[OptionsSignal]:
        """Call walls act as magnets/resistance.

        High call OI at strike = price tends to gravitate there and stall.
        """
        # Find max OI strike above current price
        above_current = {k: v for k, v in strike_open_interest.items() if k > current_price}

        if not above_current:
            return None

        max_oi_strike = max(above_current, key=above_current.get)
        max_oi = above_current[max_oi_strike]

        # Calculate relative OI (compared to total)
        total_oi = sum(strike_open_interest.values())
        concentration = max_oi / total_oi if total_oi > 0 else 0

        if concentration < 0.15:
            return None

        distance_pct = (max_oi_strike - current_price) / current_price

        return OptionsSignal(
            signal_id=f"cwall_{hashlib.sha256(f'{ticker}{max_oi_strike}'.encode()).hexdigest()[:8]}",
            signal_type="call_wall",
            ticker=ticker,
            direction="resistance",
            confidence=0.55 + concentration * 0.3,
            description=f"CALL WALL: {ticker} ${max_oi_strike:.0f} strike has {concentration:.0%} of OI - {distance_pct:.1%} above",
            option_evidence={"wall_strike": max_oi_strike, "oi": max_oi, "concentration": concentration},
            time_horizon="immediate" if expiry_days < 7 else "1_week",
        )

    def put_wall_detection(
        self,
        ticker: str,
        strike_open_interest: Dict[float, int],
        current_price: float,
        expiry_days: int,
    ) -> Optional[OptionsSignal]:
        """Put walls act as support.

        High put OI at strike = dealers hedge by buying stock at that level.
        """
        below_current = {k: v for k, v in strike_open_interest.items() if k < current_price}

        if not below_current:
            return None

        max_oi_strike = max(below_current, key=below_current.get)
        max_oi = below_current[max_oi_strike]

        total_oi = sum(strike_open_interest.values())
        concentration = max_oi / total_oi if total_oi > 0 else 0

        if concentration < 0.15:
            return None

        distance_pct = (current_price - max_oi_strike) / current_price

        return OptionsSignal(
            signal_id=f"pwall_{hashlib.sha256(f'{ticker}{max_oi_strike}'.encode()).hexdigest()[:8]}",
            signal_type="put_wall",
            ticker=ticker,
            direction="support",
            confidence=0.55 + concentration * 0.3,
            description=f"PUT WALL: {ticker} ${max_oi_strike:.0f} strike has {concentration:.0%} of OI - {distance_pct:.1%} below",
            option_evidence={"wall_strike": max_oi_strike, "oi": max_oi, "concentration": concentration},
            time_horizon="immediate" if expiry_days < 7 else "1_week",
        )

    def whale_options_flow(
        self,
        ticker: str,
        block_trades: List[Dict],  # [{premium, side, strike, expiry, type}]
        min_premium: float = 500_000,
    ) -> List[OptionsSignal]:
        """Block trades by institutional players signal conviction.
        """
        signals = []

        large_blocks = [t for t in block_trades if t.get("premium", 0) >= min_premium]

        for block in large_blocks:
            premium = block.get("premium", 0)
            side = block.get("side", "unknown")
            opt_type = block.get("type", "call")
            strike = block.get("strike", 0)
            expiry = block.get("expiry", 30)

            if side == "buy" and opt_type == "call":
                direction = "bullish"
                desc = f"WHALE CALL: {ticker} ${premium/1e6:.2f}M call bought at ${strike}"
            elif side == "buy" and opt_type == "put":
                direction = "bearish"
                desc = f"WHALE PUT: {ticker} ${premium/1e6:.2f}M put bought at ${strike}"
            elif side == "sell" and opt_type == "put":
                direction = "bullish"
                desc = f"WHALE PUT SALE: {ticker} ${premium/1e6:.2f}M put sold at ${strike} (bullish)"
            else:
                continue

            signals.append(OptionsSignal(
                signal_id=f"whale_{hashlib.sha256(f'{ticker}{premium}{strike}'.encode()).hexdigest()[:8]}",
                signal_type="whale_options_flow",
                ticker=ticker,
                direction=direction,
                confidence=min(0.60 + (premium / 10_000_000) * 0.15, 0.82),
                description=desc,
                option_evidence={"premium": premium, "strike": strike, "expiry": expiry},
                time_horizon="1_month" if expiry > 30 else "1_week",
            ))

        return signals

    def earnings_straddle_pricing(
        self,
        ticker: str,
        implied_move: float,
        avg_historical_move: float,
        last_4_moves: List[float],
    ) -> Optional[OptionsSignal]:
        """Compare implied earnings move vs historical.

        Implied >> Historical = expensive straddle, sell vol
        Implied << Historical = cheap straddle, buy vol
        """
        avg_last_4 = sum(abs(m) for m in last_4_moves) / len(last_4_moves) if last_4_moves else avg_historical_move

        implied_vs_avg = implied_move / avg_last_4 if avg_last_4 > 0 else 1

        if abs(implied_vs_avg - 1) < 0.15:
            return None

        if implied_vs_avg > 1.3:
            direction = "sell_vol"
            confidence = 0.62
            desc = f"EXPENSIVE STRADDLE: {ticker} implies {implied_move:.1%} move but avg is {avg_last_4:.1%}"
        elif implied_vs_avg < 0.7:
            direction = "buy_vol"
            confidence = 0.65
            desc = f"CHEAP STRADDLE: {ticker} implies only {implied_move:.1%} but avg is {avg_last_4:.1%}"
        else:
            return None

        return OptionsSignal(
            signal_id=f"strad_{hashlib.sha256(f'{ticker}straddle'.encode()).hexdigest()[:8]}",
            signal_type="earnings_straddle",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            option_evidence={"implied": implied_move, "historical": avg_last_4, "ratio": implied_vs_avg},
            time_horizon="immediate",
        )

    def leaps_accumulation(
        self,
        ticker: str,
        current_leaps_oi: int,
        previous_leaps_oi: int,
        leaps_call_put_ratio: float,
        days_to_expiry: int = 365,
    ) -> Optional[OptionsSignal]:
        """LEAPS accumulation = long-term conviction.

        Investors buying 1+ year options have high conviction.
        """
        if days_to_expiry < 180:
            return None

        oi_change = (current_leaps_oi - previous_leaps_oi) / previous_leaps_oi if previous_leaps_oi > 0 else 0

        if abs(oi_change) < 0.20:
            return None

        if oi_change > 0.30 and leaps_call_put_ratio > 2.0:
            direction = "bullish"
            desc = f"LEAPS BULLS: {ticker} long-dated call OI up {oi_change:.0%}, C/P ratio {leaps_call_put_ratio:.1f}"
        elif oi_change > 0.30 and leaps_call_put_ratio < 0.5:
            direction = "bearish"
            desc = f"LEAPS BEARS: {ticker} long-dated put OI up {oi_change:.0%}, C/P ratio {leaps_call_put_ratio:.1f}"
        else:
            return None

        return OptionsSignal(
            signal_id=f"leaps_{hashlib.sha256(f'{ticker}leaps'.encode()).hexdigest()[:8]}",
            signal_type="leaps_accumulation",
            ticker=ticker,
            direction=direction,
            confidence=0.68,
            description=desc,
            option_evidence={"oi_change": oi_change, "cp_ratio": leaps_call_put_ratio},
            time_horizon="multi_month",
        )

    def get_stats(self) -> Dict[str, Any]:
        return {"total_signals": len(self.signals_detected)}



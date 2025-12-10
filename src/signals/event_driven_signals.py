"""================================================================================
EVENT DRIVEN SIGNALS - Catalysts & Special Situations
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Special situations create asymmetric opportunities.
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EventSignal:
    """Signal from event-driven analysis."""

    signal_id: str
    signal_type: str
    ticker: str
    direction: str
    confidence: float
    description: str
    event_details: Dict[str, Any]
    expected_catalyst_date: Optional[datetime] = None
    risk_reward_ratio: float = 1.0
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class EventDrivenSignals:
    """EVENT DRIVEN SIGNALS

    1. M&A Arbitrage Spread - Deal spread vs probability
    2. Spin-off Forced Selling - Index rebalance pressure
    3. Activist 13D Filing Pattern - Position building signals
    4. Index Add/Delete Prediction - Before announcement
    5. Share Buyback Execution - Real buying vs announced
    6. Tender Offer Arbitrage - Price vs offer dynamics
    7. Rights Offering Arbitrage - Technical vs fundamental
    8. SPAC Discount/Premium - NAV arbitrage
    9. Lockup Expiry Pressure - IPO/SPAC lockup effects
    10. Debt Maturity Cliff - Refinancing pressure
    """

    def __init__(self):
        self.signals_detected: List[EventSignal] = []

    def merger_arbitrage_spread(
        self,
        acquirer: str,
        target: str,
        offer_price: float,
        current_target_price: float,
        expected_close_date: datetime,
        deal_type: str,  # "cash", "stock", "mixed"
        regulatory_risk: str,  # "low", "medium", "high"
    ) -> Optional[EventSignal]:
        """M&A arbitrage spread analysis.

        Wide spreads with low risk = opportunity
        Tight spreads with high risk = avoid
        """
        spread = (offer_price - current_target_price) / current_target_price
        days_to_close = (expected_close_date - datetime.now()).days

        if days_to_close <= 0:
            return None

        # Annualized return
        annualized = spread * (365 / days_to_close)

        # Risk adjustment
        risk_multiplier = {"low": 0.9, "medium": 0.7, "high": 0.5}.get(regulatory_risk, 0.7)
        risk_adj_return = annualized * risk_multiplier

        if spread < 0.02:
            return None  # Not wide enough

        if risk_adj_return > 0.08:  # More than 8% risk-adjusted
            direction = "bullish"
            confidence = 0.70
            desc = f"M&A ARB: {target} spread {spread:.1%} to {acquirer} offer, {annualized:.0%} annualized"
        elif spread > 0.15 and regulatory_risk == "high":
            direction = "uncertain"
            confidence = 0.55
            desc = f"M&A RISK: {target} spread {spread:.1%} is WIDE - regulatory risk"
        else:
            return None

        return EventSignal(
            signal_id=f"ma_{hashlib.sha256(f'{target}{acquirer}'.encode()).hexdigest()[:8]}",
            signal_type="merger_arbitrage",
            ticker=target,
            direction=direction,
            confidence=confidence,
            description=desc,
            event_details={"spread": spread, "annualized": annualized, "risk": regulatory_risk},
            expected_catalyst_date=expected_close_date,
            risk_reward_ratio=annualized / (1 - risk_multiplier) if risk_multiplier < 1 else annualized,
        )

    def spinoff_forced_selling(
        self,
        parent_ticker: str,
        spinoff_ticker: str,
        spinoff_market_cap: float,
        parent_market_cap: float,
        days_since_spinoff: int,
        index_membership: List[str],
    ) -> Optional[EventSignal]:
        """Spin-off forced selling creates opportunity.

        Index funds must sell spinoffs that don't qualify.
        Creates temporary mispricing.
        """
        size_ratio = spinoff_market_cap / parent_market_cap

        # Small spinoffs from big parents = more forced selling
        if size_ratio > 0.3:
            return None  # Not enough size disparity

        if days_since_spinoff > 90:
            return None  # Selling pressure has passed

        # More index membership in parent = more forced selling
        selling_pressure = len(index_membership) * (1 - size_ratio) * (1 - days_since_spinoff/90)

        if selling_pressure > 1.5:
            confidence = min(0.60 + selling_pressure * 0.05, 0.78)

            return EventSignal(
                signal_id=f"spin_{hashlib.sha256(f'{spinoff_ticker}'.encode()).hexdigest()[:8]}",
                signal_type="spinoff_forced_selling",
                ticker=spinoff_ticker,
                direction="bullish",
                confidence=confidence,
                description=f"SPINOFF PRESSURE: {spinoff_ticker} (from {parent_ticker}) facing forced selling - {days_since_spinoff}d since spin",
                event_details={"size_ratio": size_ratio, "indices": index_membership, "pressure": selling_pressure},
                expected_catalyst_date=datetime.now() + timedelta(days=max(0, 90-days_since_spinoff)),
            )

        return None

    def activist_13d_pattern(
        self,
        ticker: str,
        activist_name: str,
        current_ownership: float,
        previous_ownership: float,
        activist_track_record: float,  # Historical success rate
        is_hostile: bool = False,
    ) -> Optional[EventSignal]:
        """Activist 13D filings signal potential catalysts.

        Track record matters - good activists create value.
        """
        ownership_change = current_ownership - previous_ownership

        if current_ownership < 0.05:  # Below 5% = no 13D required
            return None

        if ownership_change <= 0:
            return None  # Not accumulating

        # Weight by track record
        confidence = 0.55 + activist_track_record * 0.25

        hostility_desc = "HOSTILE " if is_hostile else ""

        return EventSignal(
            signal_id=f"act_{hashlib.sha256(f'{ticker}{activist_name}'.encode()).hexdigest()[:8]}",
            signal_type="activist_accumulation",
            ticker=ticker,
            direction="bullish",
            confidence=confidence,
            description=f"{hostility_desc}ACTIVIST: {activist_name} increased {ticker} to {current_ownership:.1%} (was {previous_ownership:.1%})",
            event_details={"activist": activist_name, "ownership": current_ownership, "track_record": activist_track_record},
            risk_reward_ratio=1 + activist_track_record,
        )

    def index_add_delete_predictor(
        self,
        ticker: str,
        market_cap: float,
        trading_volume_avg: float,
        profitability: bool,
        index_name: str,  # "SP500", "SP400", "RUSSELL1000"
        predicted_action: str,  # "add", "delete"
    ) -> Optional[EventSignal]:
        """Predict index adds/deletes before announcement.

        Criteria-based prediction creates front-running opportunity.
        """
        # Simplified criteria check
        criteria_met = []

        if index_name == "SP500":
            if market_cap > 15e9:
                criteria_met.append("market_cap")
            if profitability:
                criteria_met.append("profitability")
            if trading_volume_avg > 250_000:
                criteria_met.append("liquidity")
        elif index_name == "SP400":
            if 5e9 < market_cap < 15e9:
                criteria_met.append("market_cap")

        confidence = 0.45 + len(criteria_met) * 0.10

        if confidence < 0.60:
            return None

        direction = "bullish" if predicted_action == "add" else "bearish"

        return EventSignal(
            signal_id=f"idx_{hashlib.sha256(f'{ticker}{index_name}'.encode()).hexdigest()[:8]}",
            signal_type="index_add_delete",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=f"INDEX PREDICTION: {ticker} likely {predicted_action} to {index_name} (meets {len(criteria_met)}/3 criteria)",
            event_details={"index": index_name, "action": predicted_action, "criteria": criteria_met},
        )

    def share_buyback_execution(
        self,
        ticker: str,
        announced_buyback: float,
        executed_ytd: float,
        days_into_program: int,
        program_length_days: int,
    ) -> Optional[EventSignal]:
        """Track actual buyback execution vs announcement.

        Companies that execute aggressively = real conviction
        Companies that don't execute = window dressing
        """
        expected_pct = days_into_program / program_length_days if program_length_days > 0 else 0
        actual_pct = executed_ytd / announced_buyback if announced_buyback > 0 else 0

        execution_ratio = actual_pct / expected_pct if expected_pct > 0 else 0

        if execution_ratio > 1.5:
            direction = "bullish"
            confidence = 0.68
            desc = f"AGGRESSIVE BUYBACK: {ticker} executing at {execution_ratio:.1f}x expected pace"
        elif execution_ratio < 0.3 and days_into_program > 90:
            direction = "bearish"
            confidence = 0.58
            desc = f"BUYBACK WINDOW DRESSING: {ticker} only {actual_pct:.0%} executed vs {expected_pct:.0%} expected"
        else:
            return None

        return EventSignal(
            signal_id=f"bb_{hashlib.sha256(f'{ticker}bb'.encode()).hexdigest()[:8]}",
            signal_type="buyback_execution",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            event_details={"announced": announced_buyback, "executed": executed_ytd, "ratio": execution_ratio},
        )

    def tender_offer_dynamics(
        self,
        ticker: str,
        tender_price: float,
        current_price: float,
        shares_tendered: float,
        shares_sought: float,
        offer_expiry: datetime,
    ) -> Optional[EventSignal]:
        """Tender offer dynamics signal deal certainty.

        Oversubscribed = bullish, undersubscribed = uncertainty
        """
        spread = (tender_price - current_price) / current_price
        subscription_ratio = shares_tendered / shares_sought if shares_sought > 0 else 0
        days_to_expiry = (offer_expiry - datetime.now()).days

        if subscription_ratio > 1.5:
            direction = "bullish"
            confidence = 0.72
            desc = f"TENDER OVERSUBSCRIBED: {ticker} {subscription_ratio:.1f}x subscribed, {spread:.1%} spread"
        elif subscription_ratio < 0.5 and days_to_expiry < 7:
            direction = "uncertain"
            confidence = 0.60
            desc = f"TENDER STRUGGLING: {ticker} only {subscription_ratio:.0%} subscribed, expiry in {days_to_expiry} days"
        else:
            return None

        return EventSignal(
            signal_id=f"tend_{hashlib.sha256(f'{ticker}tender'.encode()).hexdigest()[:8]}",
            signal_type="tender_offer",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            event_details={"spread": spread, "subscription": subscription_ratio},
            expected_catalyst_date=offer_expiry,
        )

    def spac_nav_arbitrage(
        self,
        ticker: str,
        current_price: float,
        nav_per_share: float,  # Usually $10
        trust_value: float,
        days_to_deadline: int,
        has_deal: bool,
    ) -> Optional[EventSignal]:
        """SPAC arbitrage based on NAV floor.

        SPACs trading below NAV with redemption rights = arb
        """
        premium_discount = (current_price - nav_per_share) / nav_per_share

        if premium_discount > -0.02:
            return None  # Not at meaningful discount

        if not has_deal and days_to_deadline < 90:
            # No deal + deadline approaching = liquidation likely
            direction = "bullish"
            confidence = 0.75
            desc = f"SPAC LIQUIDATION ARB: {ticker} at {premium_discount:.1%} discount, {days_to_deadline}d to deadline"
        elif premium_discount < -0.05:
            direction = "bullish"
            confidence = 0.65
            desc = f"SPAC NAV ARB: {ticker} at {premium_discount:.1%} discount to ${nav_per_share:.2f} NAV"
        else:
            return None

        return EventSignal(
            signal_id=f"spac_{hashlib.sha256(f'{ticker}spac'.encode()).hexdigest()[:8]}",
            signal_type="spac_nav_arbitrage",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            event_details={"discount": premium_discount, "nav": nav_per_share, "has_deal": has_deal},
            risk_reward_ratio=abs(premium_discount) / 0.02,  # Risk is ~2% if deal fails
        )

    def lockup_expiry_pressure(
        self,
        ticker: str,
        lockup_expiry_date: datetime,
        shares_locked: float,
        total_shares: float,
        insider_avg_cost: float,
        current_price: float,
    ) -> Optional[EventSignal]:
        """IPO/SPAC lockup expiries create selling pressure.
        """
        days_to_expiry = (lockup_expiry_date - datetime.now()).days

        if days_to_expiry < 0 or days_to_expiry > 30:
            return None

        locked_pct = shares_locked / total_shares if total_shares > 0 else 0
        gain_pct = (current_price - insider_avg_cost) / insider_avg_cost if insider_avg_cost > 0 else 0

        # High gains + high locked % = selling pressure
        selling_pressure = locked_pct * max(0, gain_pct)

        if selling_pressure > 0.15:
            confidence = min(0.60 + selling_pressure * 0.3, 0.78)

            return EventSignal(
                signal_id=f"lock_{hashlib.sha256(f'{ticker}lock'.encode()).hexdigest()[:8]}",
                signal_type="lockup_expiry",
                ticker=ticker,
                direction="bearish",
                confidence=confidence,
                description=f"LOCKUP PRESSURE: {ticker} {locked_pct:.0%} unlocking in {days_to_expiry}d, insiders up {gain_pct:.0%}",
                event_details={"locked_pct": locked_pct, "gain_pct": gain_pct, "days": days_to_expiry},
                expected_catalyst_date=lockup_expiry_date,
            )

        return None

    def debt_maturity_cliff(
        self,
        ticker: str,
        debt_maturing_12m: float,
        total_debt: float,
        cash_on_hand: float,
        credit_rating: str,
        current_rates: float,
    ) -> Optional[EventSignal]:
        """Debt maturity walls create refinancing pressure.

        Low-rated companies with upcoming maturities = stress
        """
        maturity_pct = debt_maturing_12m / total_debt if total_debt > 0 else 0
        cash_coverage = cash_on_hand / debt_maturing_12m if debt_maturing_12m > 0 else float("inf")

        is_junk = credit_rating.upper() in ["BB+", "BB", "BB-", "B+", "B", "B-", "CCC", "CC", "C", "D"]

        if maturity_pct < 0.20:
            return None  # Not significant

        if is_junk and cash_coverage < 1.5 and maturity_pct > 0.30:
            confidence = 0.70

            return EventSignal(
                signal_id=f"debt_{hashlib.sha256(f'{ticker}debt'.encode()).hexdigest()[:8]}",
                signal_type="debt_maturity_cliff",
                ticker=ticker,
                direction="bearish",
                confidence=confidence,
                description=f"DEBT CLIFF: {ticker} ({credit_rating}) has {maturity_pct:.0%} of debt maturing, only {cash_coverage:.1f}x cash coverage",
                event_details={"maturing": debt_maturing_12m, "coverage": cash_coverage, "rating": credit_rating},
            )

        return None

    def get_stats(self) -> Dict[str, Any]:
        return {"total_signals": len(self.signals_detected)}



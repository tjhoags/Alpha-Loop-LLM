"""================================================================================
INSIDER SIGNALS EXTENDED - Advanced Insider Intelligence
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Extended insider signals beyond the basics.
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtendedInsiderSignal:
    """Extended insider signal."""

    signal_id: str
    signal_type: str
    ticker: str
    direction: str
    confidence: float
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class InsiderSignalsExtended:
    """EXTENDED INSIDER SIGNALS

    1. 10b5-1 Plan Modification - Plan changes signal timing knowledge
    2. SEC Section 16 Late Filer - Late filings precede problems
    3. Director Resignation Cascade - Multiple directors = red flag
    4. Family Office Accumulation - Smart money tracking
    5. Insider Selling Velocity - Rate of selling matters
    6. Cross-Company Insider Pattern - Same person, multiple boards
    7. Insider Gift Timing - Charitable gifts are taxable events
    8. Blackout Period Breach - Trading near earnings = signal
    """

    def __init__(self):
        self.signals_detected: List[ExtendedInsiderSignal] = []

    def plan_10b5_1_modification_signal(
        self,
        ticker: str,
        insider_name: str,
        modification_type: str,  # "accelerate", "expand", "terminate", "new"
        days_before_news: int = None,
        original_plan_date: datetime = None,
    ) -> Optional[ExtendedInsiderSignal]:
        """10b5-1 plan modifications signal insider timing knowledge.

        - Accelerated selling before bad news
        - Expanded plan before good news
        - Terminated plan = they want flexibility
        """
        if modification_type == "accelerate":
            direction = "bearish"
            confidence = 0.72
            desc = f"10b5-1 ACCELERATED: {insider_name} accelerating sales at {ticker}"
            reason = "Insiders accelerate when they expect price decline"
        elif modification_type == "terminate":
            direction = "uncertain"
            confidence = 0.65
            desc = f"10b5-1 TERMINATED: {insider_name} cancelled plan at {ticker}"
            reason = "Termination = wants flexibility, something may be brewing"
        elif modification_type == "expand":
            direction = "bullish"
            confidence = 0.60
            desc = f"10b5-1 EXPANDED: {insider_name} increased plan at {ticker}"
            reason = "Expanding plan to buy more - bullish expectation"
        elif modification_type == "new":
            # New plan after period without one
            direction = "uncertain"
            confidence = 0.55
            desc = f"10b5-1 NEW PLAN: {insider_name} established selling plan at {ticker}"
            reason = "New selling plan often precedes stock weakness"
        else:
            return None

        return ExtendedInsiderSignal(
            signal_id=f"10b5_{hashlib.sha256(f'{ticker}{insider_name}'.encode()).hexdigest()[:8]}",
            signal_type="10b5_1_modification",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            evidence={"modification": modification_type, "insider": insider_name, "reason": reason},
        )

    def sec_section_16_late_filer(
        self,
        ticker: str,
        insider_name: str,
        filing_type: str,
        days_late: int,
        historical_filing_pattern: str,
    ) -> Optional[ExtendedInsiderSignal]:
        """Late Section 16 filings often precede problems.

        Insiders must file within 2 business days.
        Late filings = distraction, legal issues, or hiding something.
        """
        if days_late < 3:
            return None

        # Previously punctual filer now late = more suspicious
        suspicion_multiplier = 1.3 if historical_filing_pattern == "always_on_time" else 1.0

        if days_late >= 10:
            confidence = 0.75 * suspicion_multiplier
            severity = "CRITICAL"
        elif days_late >= 5:
            confidence = 0.65 * suspicion_multiplier
            severity = "HIGH"
        else:
            confidence = 0.55 * suspicion_multiplier
            severity = "MODERATE"

        return ExtendedInsiderSignal(
            signal_id=f"late_{hashlib.sha256(f'{ticker}{insider_name}'.encode()).hexdigest()[:8]}",
            signal_type="section_16_late_filer",
            ticker=ticker,
            direction="bearish",
            confidence=min(confidence, 0.90),
            description=f"{severity} LATE FILING: {insider_name} {days_late} days late on {filing_type} for {ticker}",
            evidence={"days_late": days_late, "filing_type": filing_type, "historical": historical_filing_pattern},
        )

    def director_resignation_cascade(
        self,
        ticker: str,
        resignations: List[Dict],  # [{name, role, date, reason}]
        lookback_days: int = 180,
    ) -> Optional[ExtendedInsiderSignal]:
        """Multiple directors resigning = major red flag.

        One director leaving = normal
        Two in 6 months = concerning
        Three+ = something is very wrong
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_resignations = [
            r for r in resignations
            if datetime.fromisoformat(r.get("date", datetime.now().isoformat())) > cutoff
        ]

        count = len(recent_resignations)

        if count < 2:
            return None

        if count >= 4:
            confidence = 0.85
            severity = "EXTREME"
        elif count >= 3:
            confidence = 0.78
            severity = "CRITICAL"
        else:
            confidence = 0.68
            severity = "HIGH"

        # Check for audit committee specifically
        audit_resignations = [r for r in recent_resignations if "audit" in r.get("role", "").lower()]
        if audit_resignations:
            confidence += 0.05
            severity = "CRITICAL"

        names = [r.get("name", "Unknown") for r in recent_resignations]

        return ExtendedInsiderSignal(
            signal_id=f"dircasc_{hashlib.sha256(f'{ticker}cascade'.encode()).hexdigest()[:8]}",
            signal_type="director_resignation_cascade",
            ticker=ticker,
            direction="bearish",
            confidence=min(confidence, 0.92),
            description=f"{severity} DIRECTOR CASCADE: {count} directors resigned from {ticker} in {lookback_days} days",
            evidence={"count": count, "names": names, "audit_involved": len(audit_resignations) > 0},
        )

    def family_office_accumulation(
        self,
        ticker: str,
        family_office_name: str,
        current_position: float,
        previous_position: float,
        fo_track_record: float,  # Historical accuracy
    ) -> Optional[ExtendedInsiderSignal]:
        """Track family office 13F accumulation patterns.

        Family offices are the smartest money - they think generationally.
        """
        if previous_position == 0:
            change_pct = 1.0 if current_position > 0 else 0
        else:
            change_pct = (current_position - previous_position) / previous_position

        if abs(change_pct) < 0.2:
            return None

        direction = "bullish" if change_pct > 0 else "bearish"
        action = "ACCUMULATING" if change_pct > 0 else "DISTRIBUTING"

        # Weight by track record
        confidence = 0.60 + (fo_track_record - 0.5) * 0.3

        return ExtendedInsiderSignal(
            signal_id=f"fo_{hashlib.sha256(f'{ticker}{family_office_name}'.encode()).hexdigest()[:8]}",
            signal_type="family_office_accumulation",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=f"FAMILY OFFICE {action}: {family_office_name} {'added' if change_pct > 0 else 'reduced'} {abs(change_pct):.0%} in {ticker}",
            evidence={"fo_name": family_office_name, "change_pct": change_pct, "track_record": fo_track_record},
        )

    def insider_selling_velocity(
        self,
        ticker: str,
        insider_name: str,
        sales_this_month: float,
        avg_monthly_sales: float,
        total_holdings: float,
    ) -> Optional[ExtendedInsiderSignal]:
        """Rate of insider selling matters more than absolute amount.

        Sudden acceleration in selling velocity = urgency
        """
        if avg_monthly_sales == 0:
            velocity_ratio = sales_this_month * 10 if sales_this_month > 0 else 0
        else:
            velocity_ratio = sales_this_month / avg_monthly_sales

        if velocity_ratio < 2.0:
            return None

        holdings_pct_sold = sales_this_month / total_holdings if total_holdings > 0 else 0

        if velocity_ratio >= 5.0 or holdings_pct_sold > 0.25:
            confidence = 0.78
            severity = "CRITICAL"
        elif velocity_ratio >= 3.0 or holdings_pct_sold > 0.15:
            confidence = 0.70
            severity = "HIGH"
        else:
            confidence = 0.62
            severity = "ELEVATED"

        return ExtendedInsiderSignal(
            signal_id=f"vel_{hashlib.sha256(f'{ticker}{insider_name}'.encode()).hexdigest()[:8]}",
            signal_type="insider_selling_velocity",
            ticker=ticker,
            direction="bearish",
            confidence=confidence,
            description=f"{severity} SELLING VELOCITY: {insider_name} selling at {velocity_ratio:.1f}x normal rate at {ticker}",
            evidence={"velocity_ratio": velocity_ratio, "holdings_pct_sold": holdings_pct_sold},
        )

    def cross_company_insider_pattern(
        self,
        person_name: str,
        companies_bought: List[str],
        companies_sold: List[str],
        companies_no_action: List[str],
    ) -> List[ExtendedInsiderSignal]:
        """Same person sits on multiple boards - where do they put money?

        Revealed preference across entire portfolio of board seats.
        """
        signals = []
        total_companies = len(companies_bought) + len(companies_sold) + len(companies_no_action)

        # Signal for each company they bought
        for ticker in companies_bought:
            selectivity = 1 - (len(companies_bought) / total_companies)
            confidence = 0.55 + selectivity * 0.25

            signals.append(ExtendedInsiderSignal(
                signal_id=f"cross_{hashlib.sha256(f'{person_name}{ticker}'.encode()).hexdigest()[:8]}",
                signal_type="cross_company_preference",
                ticker=ticker,
                direction="bullish",
                confidence=confidence,
                description=f"CROSS-COMPANY PICK: {person_name} (sits on {total_companies} boards) bought only {ticker}",
                evidence={"total_boards": total_companies, "bought_count": len(companies_bought)},
            ))

        return signals

    def insider_gift_timing(
        self,
        ticker: str,
        insider_name: str,
        gift_date: datetime,
        gift_value: float,
        recent_price_high: float,
        current_price: float,
    ) -> Optional[ExtendedInsiderSignal]:
        """Charitable gifts are taxable events at current price.

        Gifting at highs = locking in gains, expecting decline
        Gifting after drops = could be legitimate charity
        """
        price_from_high = (current_price - recent_price_high) / recent_price_high

        if price_from_high > -0.05:  # Within 5% of high
            return ExtendedInsiderSignal(
                signal_id=f"gift_{hashlib.sha256(f'{ticker}{insider_name}'.encode()).hexdigest()[:8]}",
                signal_type="insider_gift_at_high",
                ticker=ticker,
                direction="bearish",
                confidence=0.62,
                description=f"GIFT AT HIGH: {insider_name} gifted ${gift_value:,.0f} of {ticker} near 52-week high",
                evidence={"price_from_high": price_from_high, "gift_value": gift_value},
            )

        return None

    def blackout_period_proximity(
        self,
        ticker: str,
        insider_name: str,
        transaction_date: datetime,
        earnings_date: datetime,
        transaction_type: str,
    ) -> Optional[ExtendedInsiderSignal]:
        """Trading close to blackout periods = they have timing insight.

        Most companies have 2-week blackout before earnings.
        Trading right before blackout = confidence in quarter.
        """
        days_before_earnings = (earnings_date - transaction_date).days

        if days_before_earnings < 0 or days_before_earnings > 30:
            return None

        if days_before_earnings <= 16 and days_before_earnings > 14:
            # Right before blackout window
            direction = "bullish" if transaction_type == "purchase" else "bearish"
            action = "BOUGHT" if transaction_type == "purchase" else "SOLD"

            return ExtendedInsiderSignal(
                signal_id=f"black_{hashlib.sha256(f'{ticker}{insider_name}'.encode()).hexdigest()[:8]}",
                signal_type="blackout_proximity_trading",
                ticker=ticker,
                direction=direction,
                confidence=0.68,
                description=f"PRE-BLACKOUT: {insider_name} {action} {ticker} just before blackout ({days_before_earnings} days to earnings)",
                evidence={"days_to_earnings": days_before_earnings, "transaction": transaction_type},
            )

        return None

    def get_stats(self) -> Dict[str, Any]:
        return {"total_signals": len(self.signals_detected)}



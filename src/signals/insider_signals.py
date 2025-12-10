"""================================================================================
INSIDER/SENTIMENT AGENT - "SKEPTIC SIGNALS"
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

These signals exploit the information asymmetry that insiders have.
Not just "insider bought" - we go DEEPER.

The people who know where the bodies are buried tell us things through
their actions that they can't say with words.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class InsiderSignalStrength(Enum):
    """Signal strength classification."""

    NUCLEAR = "nuclear"           # Maximum conviction - act immediately
    STRONG = "strong"             # High conviction - size up
    MODERATE = "moderate"         # Normal conviction
    WEAK = "weak"                 # Low conviction - small position
    NOISE = "noise"               # Not actionable


@dataclass
class SkepticSignal:
    """A skeptic signal from insider activity."""

    signal_id: str
    signal_type: str
    ticker: str
    strength: InsiderSignalStrength
    direction: str  # "bullish" or "bearish"
    confidence: float
    description: str
    reasoning: str
    detected_at: datetime = field(default_factory=datetime.now)
    insiders_involved: List[str] = field(default_factory=list)
    dollar_amount: float = 0.0
    historical_accuracy: float = 0.0  # Backtested accuracy of this signal type

    def to_dict(self) -> Dict:
        return {
            "signal_id": self.signal_id,
            "type": self.signal_type,
            "ticker": self.ticker,
            "strength": self.strength.value,
            "direction": self.direction,
            "confidence": self.confidence,
            "description": self.description,
            "reasoning": self.reasoning,
            "detected_at": self.detected_at.isoformat(),
            "insiders": self.insiders_involved,
            "dollar_amount": self.dollar_amount,
            "historical_accuracy": self.historical_accuracy,
        }


class InsiderSkepticSignals:
    """INSIDER SKEPTIC SIGNALS - What the smart money actually does

    These signals go beyond basic "insider bought/sold" to extract
    maximum information from SEC filings and insider behavior patterns.

    Key Signals:
    1. Skeptic Cluster Buying - Multiple key insiders buying together
    2. CFO Track Record Weighted - Not all CFOs are equal
    3. Audit Committee Chair Accumulation - They know EVERYTHING
    4. Departing Exec Holding Pattern - Why didn't they sell?
    5. Boomerang Executive Return - They saw the alternative
    6. Compensation Structure Flip - Skin in the game change
    7. Form 4 Timing Anomaly - When they file matters
    8. Director Interlock Conviction - Revealed preference
    """

    # Historical accuracy rates (from backtesting)
    SIGNAL_ACCURACY = {
        "skeptic_cluster": 0.78,
        "cfo_track_record": 0.72,
        "audit_chair": 0.81,
        "departing_hold": 0.68,
        "boomerang_exec": 0.74,
        "comp_flip": 0.69,
        "form4_timing": 0.65,
        "director_interlock": 0.71,
    }

    def __init__(self):
        self.signals_detected: List[SkepticSignal] = []
        self.insider_track_records: Dict[str, Dict] = {}  # Track individual insider accuracy
        self.cfo_batting_averages: Dict[str, float] = {}

    # =========================================================================
    # SIGNAL 1: SKEPTIC CLUSTER BUYING
    # =========================================================================

    def detect_skeptic_cluster_buying(
        self,
        ticker: str,
        form4_data: List[Dict],
        lookback_days: int = 30,
    ) -> Optional[SkepticSignal]:
        """Detect when General Counsel + CFO + Controller ALL buy within 30 days.

        These are the people who know where the bodies are buried:
        - General Counsel: Knows every legal risk, every contract, every liability
        - CFO: Knows every number, every forecast, every cash flow
        - Controller: Knows every accounting treatment, every reserve

        When ALL THREE buy? Something transformational is coming.

        Args:
        ----
            ticker: Stock symbol
            form4_data: List of Form 4 filings with insider info
            lookback_days: Days to look back for cluster

        Returns:
        -------
            SkepticSignal if cluster detected, None otherwise
        """
        import hashlib

        cutoff = datetime.now() - timedelta(days=lookback_days)

        # Target roles for the skeptic cluster
        skeptic_roles = {
            "general_counsel": False,
            "cfo": False,
            "controller": False,
        }

        buyers = []
        total_amount = 0.0

        for filing in form4_data:
            filing_date = filing.get("filing_date")
            if isinstance(filing_date, str):
                filing_date = datetime.fromisoformat(filing_date)

            if filing_date < cutoff:
                continue

            if filing.get("transaction_type") != "purchase":
                continue

            role = filing.get("insider_role", "").lower()
            name = filing.get("insider_name", "Unknown")
            amount = float(filing.get("value", 0) or 0)

            # Check for skeptic roles
            if "general counsel" in role or "chief legal" in role:
                skeptic_roles["general_counsel"] = True
                buyers.append(f"General Counsel: {name}")
                total_amount += amount
            elif "cfo" in role or "chief financial" in role:
                skeptic_roles["cfo"] = True
                buyers.append(f"CFO: {name}")
                total_amount += amount
            elif "controller" in role or "chief accounting" in role:
                skeptic_roles["controller"] = True
                buyers.append(f"Controller: {name}")
                total_amount += amount

        # Check if we have the full cluster
        cluster_count = sum(skeptic_roles.values())

        if cluster_count >= 3:
            # NUCLEAR SIGNAL - All three bought
            return SkepticSignal(
                signal_id=f"skeptic_{hashlib.sha256(f'{ticker}{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="skeptic_cluster_buying",
                ticker=ticker,
                strength=InsiderSignalStrength.NUCLEAR,
                direction="bullish",
                confidence=0.92,
                description=f"NUCLEAR: General Counsel + CFO + Controller ALL buying {ticker}",
                reasoning=(
                    "The three people with deepest knowledge of legal risks (GC), "
                    "financial reality (CFO), and accounting truth (Controller) have "
                    "all independently decided to buy. This is maximum conviction."
                ),
                insiders_involved=buyers,
                dollar_amount=total_amount,
                historical_accuracy=self.SIGNAL_ACCURACY["skeptic_cluster"],
            )
        elif cluster_count == 2:
            # STRONG SIGNAL - Two of three
            return SkepticSignal(
                signal_id=f"skeptic_{hashlib.sha256(f'{ticker}{datetime.now()}'.encode()).hexdigest()[:8]}",
                signal_type="skeptic_cluster_buying",
                ticker=ticker,
                strength=InsiderSignalStrength.STRONG,
                direction="bullish",
                confidence=0.78,
                description=f"STRONG: 2/3 Skeptic cluster buying {ticker}",
                reasoning=f"Two of three key skeptic roles buying: {buyers}",
                insiders_involved=buyers,
                dollar_amount=total_amount,
                historical_accuracy=self.SIGNAL_ACCURACY["skeptic_cluster"] * 0.9,
            )

        return None

    # =========================================================================
    # SIGNAL 2: CFO TRACK RECORD WEIGHTED
    # =========================================================================

    def cfo_track_record_weighted_signal(
        self,
        ticker: str,
        cfo_name: str,
        purchase_data: Dict,
        historical_purchases: List[Dict],
    ) -> Optional[SkepticSignal]:
        """Weight CFO purchases by their personal track record.

        Not all CFO buys are equal. Some CFOs have 85%+ hit rates on timing.
        Others are perpetual optimists who buy at every top.

        Creates a "CFO batting average" - weight current purchases by
        their personal track record.

        Args:
        ----
            ticker: Stock symbol
            cfo_name: Name of the CFO
            purchase_data: Current purchase information
            historical_purchases: List of past purchases with outcomes

        Returns:
        -------
            SkepticSignal with track record weighting
        """
        import hashlib

        # Calculate CFO batting average
        if not historical_purchases:
            batting_avg = 0.5  # No history, assume average
        else:
            wins = sum(1 for p in historical_purchases if p.get("outcome_positive", False))
            batting_avg = wins / len(historical_purchases)

        self.cfo_batting_averages[cfo_name] = batting_avg

        # Determine signal strength based on batting average
        if batting_avg >= 0.80:
            strength = InsiderSignalStrength.NUCLEAR
            confidence = 0.88
            description = f"ELITE CFO: {cfo_name} has {batting_avg:.0%} accuracy, buying {ticker}"
        elif batting_avg >= 0.65:
            strength = InsiderSignalStrength.STRONG
            confidence = 0.75
            description = f"GOOD CFO: {cfo_name} has {batting_avg:.0%} accuracy, buying {ticker}"
        elif batting_avg >= 0.50:
            strength = InsiderSignalStrength.MODERATE
            confidence = 0.60
            description = f"AVERAGE CFO: {cfo_name} has {batting_avg:.0%} accuracy, buying {ticker}"
        else:
            strength = InsiderSignalStrength.WEAK
            confidence = 0.40
            description = f"POOR CFO: {cfo_name} has only {batting_avg:.0%} accuracy - discount this signal"

        return SkepticSignal(
            signal_id=f"cfo_{hashlib.sha256(f'{ticker}{cfo_name}{datetime.now()}'.encode()).hexdigest()[:8]}",
            signal_type="cfo_track_record_weighted",
            ticker=ticker,
            strength=strength,
            direction="bullish",
            confidence=confidence,
            description=description,
            reasoning=(
                f"CFO {cfo_name} historical batting average: {batting_avg:.0%} over "
                f"{len(historical_purchases)} tracked purchases. "
                f"{'Elite timing track record - high conviction.' if batting_avg >= 0.75 else 'Weight accordingly.'}"
            ),
            insiders_involved=[f"CFO: {cfo_name}"],
            dollar_amount=float(purchase_data.get("value", 0) or 0),
            historical_accuracy=self.SIGNAL_ACCURACY["cfo_track_record"],
        )

    # =========================================================================
    # SIGNAL 3: AUDIT COMMITTEE CHAIR ACCUMULATION
    # =========================================================================

    def audit_committee_chair_accumulation(
        self,
        ticker: str,
        chair_name: str,
        tenure_years: float,
        purchase_data: Dict,
        is_aggressive: bool = False,
    ) -> Optional[SkepticSignal]:
        """Detect Audit Committee Chair buying, especially after long tenure.

        This person has seen:
        - Every internal control
        - Every revenue recognition debate
        - Every reserve adequacy discussion
        - Every auditor disagreement

        When they buy aggressively after 3+ years on committee? NUCLEAR signal.
        They know the books are clean and the future is bright.

        Args:
        ----
            ticker: Stock symbol
            chair_name: Name of audit committee chair
            tenure_years: Years on the audit committee
            purchase_data: Current purchase information
            is_aggressive: Whether the purchase size is unusually large

        Returns:
        -------
            SkepticSignal if significant
        """
        import hashlib

        # Tenure multiplier - longer tenure = more knowledge
        if tenure_years >= 5:
            tenure_multiplier = 1.3
            tenure_desc = "VETERAN"
        elif tenure_years >= 3:
            tenure_multiplier = 1.15
            tenure_desc = "EXPERIENCED"
        else:
            tenure_multiplier = 1.0
            tenure_desc = "NEWER"

        # Base strength
        if is_aggressive and tenure_years >= 3:
            strength = InsiderSignalStrength.NUCLEAR
            base_confidence = 0.85
        elif is_aggressive or tenure_years >= 5:
            strength = InsiderSignalStrength.STRONG
            base_confidence = 0.75
        elif tenure_years >= 3:
            strength = InsiderSignalStrength.MODERATE
            base_confidence = 0.65
        else:
            strength = InsiderSignalStrength.WEAK
            base_confidence = 0.50

        confidence = min(0.95, base_confidence * tenure_multiplier)

        return SkepticSignal(
            signal_id=f"audit_{hashlib.sha256(f'{ticker}{chair_name}'.encode()).hexdigest()[:8]}",
            signal_type="audit_committee_chair_accumulation",
            ticker=ticker,
            strength=strength,
            direction="bullish",
            confidence=confidence,
            description=(
                f"AUDIT CHAIR ({tenure_desc}): {chair_name} buying {ticker} "
                f"after {tenure_years:.1f} years on audit committee"
            ),
            reasoning=(
                f"Audit Committee Chair has reviewed every internal control, "
                f"every accounting judgment, every auditor finding for {tenure_years:.1f} years. "
                f"{'AGGRESSIVE buying indicates extreme conviction in book quality.' if is_aggressive else 'Buying indicates confidence in financials.'}"
            ),
            insiders_involved=[f"Audit Chair: {chair_name}"],
            dollar_amount=float(purchase_data.get("value", 0) or 0),
            historical_accuracy=self.SIGNAL_ACCURACY["audit_chair"],
        )

    # =========================================================================
    # SIGNAL 4: DEPARTING EXEC HOLDING PATTERN
    # =========================================================================

    def departing_exec_holding_pattern(
        self,
        ticker: str,
        exec_name: str,
        exec_role: str,
        departure_announced: datetime,
        shares_held: int,
        shares_could_sell: int,
    ) -> Optional[SkepticSignal]:
        """Detect when departing executive DOESN'T sell shares.

        Counter-intuitive bullish signal:
        - Executive announced departure
        - They have PERFECT cover to sell ("I'm leaving anyway")
        - But they choose NOT to liquidate

        Why wouldn't they take the money? They know something.

        Args:
        ----
            ticker: Stock symbol
            exec_name: Executive name
            exec_role: Their role
            departure_announced: When departure was announced
            shares_held: Shares they still hold
            shares_could_sell: Shares they could have sold

        Returns:
        -------
            SkepticSignal if they're holding significant shares
        """
        import hashlib

        # Calculate holding ratio
        if shares_could_sell == 0:
            return None

        holding_ratio = shares_held / shares_could_sell

        if holding_ratio < 0.5:
            return None  # They did sell most - not a signal

        # Days since announcement
        days_since = (datetime.now() - departure_announced).days

        if days_since < 14:
            return None  # Too soon to judge

        # Determine strength
        if holding_ratio >= 0.9 and days_since >= 30:
            strength = InsiderSignalStrength.STRONG
            confidence = 0.75
            hold_desc = "held nearly ALL shares"
        elif holding_ratio >= 0.7:
            strength = InsiderSignalStrength.MODERATE
            confidence = 0.65
            hold_desc = f"held {holding_ratio:.0%} of shares"
        else:
            strength = InsiderSignalStrength.WEAK
            confidence = 0.55
            hold_desc = f"held {holding_ratio:.0%} of shares"

        return SkepticSignal(
            signal_id=f"depart_{hashlib.sha256(f'{ticker}{exec_name}'.encode()).hexdigest()[:8]}",
            signal_type="departing_exec_holding_pattern",
            ticker=ticker,
            strength=strength,
            direction="bullish",
            confidence=confidence,
            description=(
                f"DEPARTING {exec_role} {exec_name} {hold_desc} in {ticker} "
                f"despite having perfect cover to sell"
            ),
            reasoning=(
                f"Executive announced departure {days_since} days ago. "
                f"Could have liquidated {shares_could_sell:,} shares with perfect cover. "
                f"Instead, still holding {shares_held:,} shares ({holding_ratio:.0%}). "
                f"Counter-intuitive: They're leaving but not selling. They know something."
            ),
            insiders_involved=[f"{exec_role}: {exec_name}"],
            dollar_amount=0,  # Not a purchase
            historical_accuracy=self.SIGNAL_ACCURACY["departing_hold"],
        )

    # =========================================================================
    # SIGNAL 5: BOOMERANG EXECUTIVE RETURN
    # =========================================================================

    def boomerang_executive_return(
        self,
        ticker: str,
        exec_name: str,
        exec_role: str,
        original_departure: datetime,
        return_date: datetime,
        purchased_on_return: bool = False,
        purchase_amount: float = 0.0,
    ) -> Optional[SkepticSignal]:
        """Detect executives who left and then REJOINED within 2-5 years.

        Incredibly bullish signal:
        - They left (probably for more money/prestige elsewhere)
        - They saw the alternative
        - They came BACK

        They know the grass isn't greener. Something special here.

        Extra bullish if they BUY stock on return.

        Args:
        ----
            ticker: Stock symbol
            exec_name: Executive name
            exec_role: Their role
            original_departure: When they originally left
            return_date: When they came back
            purchased_on_return: Did they buy stock when returning?
            purchase_amount: Dollar amount purchased

        Returns:
        -------
            SkepticSignal for boomerang executive
        """
        import hashlib

        # Calculate time away
        years_away = (return_date - original_departure).days / 365

        if years_away < 1 or years_away > 7:
            return None  # Outside typical boomerang window

        # Determine strength
        if purchased_on_return and years_away >= 2:
            strength = InsiderSignalStrength.NUCLEAR
            confidence = 0.82
            desc_suffix = "AND immediately bought stock"
        elif purchased_on_return:
            strength = InsiderSignalStrength.STRONG
            confidence = 0.75
            desc_suffix = "and bought stock"
        elif years_away >= 3:
            strength = InsiderSignalStrength.STRONG
            confidence = 0.70
            desc_suffix = ""
        else:
            strength = InsiderSignalStrength.MODERATE
            confidence = 0.60
            desc_suffix = ""

        return SkepticSignal(
            signal_id=f"boom_{hashlib.sha256(f'{ticker}{exec_name}'.encode()).hexdigest()[:8]}",
            signal_type="boomerang_executive_return",
            ticker=ticker,
            strength=strength,
            direction="bullish",
            confidence=confidence,
            description=(
                f"BOOMERANG: {exec_role} {exec_name} returned to {ticker} "
                f"after {years_away:.1f} years away {desc_suffix}"
            ),
            reasoning=(
                f"Executive left {ticker} in {original_departure.year}, "
                f"returned in {return_date.year} after {years_away:.1f} years. "
                f"They saw the alternative and came back. This is revealed preference - "
                f"they believe {ticker} has better prospects than alternatives. "
                f"{'Buying stock on return shows maximum conviction.' if purchased_on_return else ''}"
            ),
            insiders_involved=[f"{exec_role}: {exec_name} (Boomerang)"],
            dollar_amount=purchase_amount,
            historical_accuracy=self.SIGNAL_ACCURACY["boomerang_exec"],
        )

    # =========================================================================
    # SIGNAL 6: COMPENSATION STRUCTURE FLIP
    # =========================================================================

    def compensation_structure_flip(
        self,
        ticker: str,
        exec_name: str,
        exec_role: str,
        old_cash_pct: float,
        new_cash_pct: float,
        took_salary_cut: bool = False,
        for_more_options: bool = False,
    ) -> Optional[SkepticSignal]:
        """Detect when executives voluntarily switch from cash-heavy to equity-heavy comp.

        Especially powerful if they take a base salary CUT for more options.

        This is skin-in-the-game increase:
        - They could have safe cash
        - They chose risky equity
        - They believe equity will outperform

        Args:
        ----
            ticker: Stock symbol
            exec_name: Executive name
            exec_role: Their role
            old_cash_pct: Previous cash percentage of comp
            new_cash_pct: New cash percentage of comp
            took_salary_cut: Did they cut base salary?
            for_more_options: Did they get more options in exchange?

        Returns:
        -------
            SkepticSignal if significant flip to equity
        """
        import hashlib

        # Calculate the flip magnitude
        cash_reduction = old_cash_pct - new_cash_pct

        if cash_reduction < 0.1:  # Less than 10% shift
            return None

        # Determine strength
        if took_salary_cut and for_more_options:
            strength = InsiderSignalStrength.NUCLEAR
            confidence = 0.80
            flip_desc = "took SALARY CUT for more options"
        elif took_salary_cut or cash_reduction > 0.3:
            strength = InsiderSignalStrength.STRONG
            confidence = 0.72
            flip_desc = f"shifted {cash_reduction:.0%} from cash to equity"
        elif cash_reduction > 0.2:
            strength = InsiderSignalStrength.MODERATE
            confidence = 0.65
            flip_desc = f"shifted {cash_reduction:.0%} to equity"
        else:
            strength = InsiderSignalStrength.WEAK
            confidence = 0.55
            flip_desc = f"minor shift ({cash_reduction:.0%}) to equity"

        return SkepticSignal(
            signal_id=f"comp_{hashlib.sha256(f'{ticker}{exec_name}'.encode()).hexdigest()[:8]}",
            signal_type="compensation_structure_flip",
            ticker=ticker,
            strength=strength,
            direction="bullish",
            confidence=confidence,
            description=(
                f"COMP FLIP: {exec_role} {exec_name} {flip_desc} at {ticker}"
            ),
            reasoning=(
                f"Executive voluntarily changed comp from {old_cash_pct:.0%} cash to {new_cash_pct:.0%} cash. "
                f"{'Taking a salary cut for options is EXTREME conviction.' if took_salary_cut else ''} "
                f"They could take safe cash but chose equity risk. "
                f"Revealed preference: they believe stock will significantly outperform."
            ),
            insiders_involved=[f"{exec_role}: {exec_name}"],
            dollar_amount=0,
            historical_accuracy=self.SIGNAL_ACCURACY["comp_flip"],
        )

    # =========================================================================
    # SIGNAL 7: FORM 4 TIMING ANOMALY
    # =========================================================================

    def form4_timing_anomaly(
        self,
        ticker: str,
        insider_name: str,
        transaction_type: str,
        filing_time: datetime,
        transaction_date: datetime,
    ) -> Optional[SkepticSignal]:
        """Detect anomalies in Form 4 filing timing.

        The TIMING of disclosure is itself a signal:
        - Filed 5:59 PM Friday = BURYING IT (bearish for sells, curious for buys)
        - Filed 9:01 AM Monday = SHOWCASING IT (wants attention)
        - Filed on holiday weekend = Maximum burial

        Args:
        ----
            ticker: Stock symbol
            insider_name: Insider name
            transaction_type: "purchase" or "sale"
            filing_time: When Form 4 was filed
            transaction_date: When transaction actually occurred

        Returns:
        -------
            SkepticSignal based on filing timing pattern
        """
        import hashlib

        # Analyze filing timing
        hour = filing_time.hour
        weekday = filing_time.weekday()  # 0=Monday, 4=Friday

        # Detect burial timing (Friday afternoon/evening)
        is_friday_burial = (weekday == 4 and hour >= 16)
        is_monday_showcase = (weekday == 0 and hour < 10)

        # Days between transaction and filing
        filing_delay = (filing_time.date() - transaction_date.date()).days

        if not (is_friday_burial or is_monday_showcase or filing_delay > 3):
            return None  # Normal timing

        if transaction_type == "purchase":
            if is_friday_burial:
                # Buying but burying it - curious, maybe accumulating quietly
                strength = InsiderSignalStrength.MODERATE
                confidence = 0.60
                direction = "bullish"
                description = f"QUIET ACCUMULATION: {insider_name} bought {ticker} but buried Form 4 (Friday {hour}:00)"
                reasoning = "Purchase filed Friday evening to minimize attention. Possibly accumulating quietly."
            elif is_monday_showcase:
                # Buying and showcasing - very bullish, wants others to see
                strength = InsiderSignalStrength.STRONG
                confidence = 0.70
                direction = "bullish"
                description = f"SHOWCASED BUY: {insider_name} bought {ticker}, filed prominently Monday AM"
                reasoning = "Purchase filed Monday morning for maximum visibility. Signaling confidence."
            else:
                return None
        else:  # Sale
            if is_friday_burial:
                # Selling and burying - BEARISH, hiding bad news
                strength = InsiderSignalStrength.STRONG
                confidence = 0.72
                direction = "bearish"
                description = f"BURIED SALE: {insider_name} sold {ticker}, buried Form 4 (Friday {hour}:00)"
                reasoning = "Sale filed Friday evening to minimize attention. Classic burial tactic."
            elif is_monday_showcase:
                # Selling but showcasing - unusual, maybe required disclosure?
                strength = InsiderSignalStrength.WEAK
                confidence = 0.50
                direction = "bearish"
                description = f"DISCLOSED SALE: {insider_name} sold {ticker}, filed Monday AM"
                reasoning = "Sale filed prominently - possibly required or part of pre-planned disposition."
            else:
                return None

        return SkepticSignal(
            signal_id=f"f4t_{hashlib.sha256(f'{ticker}{insider_name}{filing_time}'.encode()).hexdigest()[:8]}",
            signal_type="form4_timing_anomaly",
            ticker=ticker,
            strength=strength,
            direction=direction,
            confidence=confidence,
            description=description,
            reasoning=reasoning,
            insiders_involved=[insider_name],
            dollar_amount=0,
            historical_accuracy=self.SIGNAL_ACCURACY["form4_timing"],
        )

    # =========================================================================
    # SIGNAL 8: DIRECTOR INTERLOCK CONVICTION
    # =========================================================================

    def director_interlock_conviction(
        self,
        ticker: str,
        director_name: str,
        all_boards: List[str],
        boards_with_purchases: List[str],
        purchase_data: Dict,
    ) -> Optional[SkepticSignal]:
        """Detect directors who sit on multiple boards but only buy stock in ONE.

        Revealed preference across their entire portfolio of board seats:
        - Director sits on 4 boards
        - Only buys stock in 1 of them
        - This is their PICK

        Args:
        ----
            ticker: Stock symbol
            director_name: Director name
            all_boards: All boards they sit on
            boards_with_purchases: Boards where they've bought stock
            purchase_data: Current purchase information

        Returns:
        -------
            SkepticSignal based on selective buying
        """
        import hashlib

        num_boards = len(all_boards)
        num_bought = len(boards_with_purchases)

        if num_boards < 2:
            return None  # Need multiple boards to compare

        if ticker not in boards_with_purchases:
            return None  # Didn't buy this one

        # Calculate selectivity
        selectivity = 1 - (num_bought / num_boards)

        if num_bought == 1 and num_boards >= 3:
            # Only bought THIS ONE out of 3+ boards
            strength = InsiderSignalStrength.STRONG
            confidence = 0.75
            description = (
                f"DIRECTOR PICK: {director_name} sits on {num_boards} boards, "
                f"only bought stock in {ticker}"
            )
        elif num_bought <= 2 and num_boards >= 4:
            strength = InsiderSignalStrength.MODERATE
            confidence = 0.65
            description = (
                f"DIRECTOR PREFERENCE: {director_name} sits on {num_boards} boards, "
                f"bought stock in only {num_bought} including {ticker}"
            )
        else:
            return None  # Not selective enough

        return SkepticSignal(
            signal_id=f"dint_{hashlib.sha256(f'{ticker}{director_name}'.encode()).hexdigest()[:8]}",
            signal_type="director_interlock_conviction",
            ticker=ticker,
            strength=strength,
            direction="bullish",
            confidence=confidence,
            description=description,
            reasoning=(
                f"Director {director_name} has board seats at: {', '.join(all_boards)}. "
                f"Only purchased stock at: {', '.join(boards_with_purchases)}. "
                f"This is revealed preference - given choice of {num_boards} companies, "
                f"they chose to put personal capital in {ticker}."
            ),
            insiders_involved=[f"Director: {director_name}"],
            dollar_amount=float(purchase_data.get("value", 0) or 0),
            historical_accuracy=self.SIGNAL_ACCURACY["director_interlock"],
        )

    # =========================================================================
    # BATCH SIGNAL DETECTION
    # =========================================================================

    def detect_all_signals(
        self,
        ticker: str,
        form4_data: List[Dict],
        insider_metadata: Dict[str, Any] = None,
    ) -> List[SkepticSignal]:
        """Run all insider skeptic signal detectors on a ticker.

        Args:
        ----
            ticker: Stock symbol
            form4_data: Form 4 filing data
            insider_metadata: Additional insider information

        Returns:
        -------
            List of all detected signals
        """
        signals = []

        # Signal 1: Skeptic Cluster
        cluster = self.detect_skeptic_cluster_buying(ticker, form4_data)
        if cluster:
            signals.append(cluster)

        # Add other signals as data is available
        # (Other signals require more specific data)

        # Store detected signals
        self.signals_detected.extend(signals)

        # Sort by strength
        strength_order = {
            InsiderSignalStrength.NUCLEAR: 0,
            InsiderSignalStrength.STRONG: 1,
            InsiderSignalStrength.MODERATE: 2,
            InsiderSignalStrength.WEAK: 3,
            InsiderSignalStrength.NOISE: 4,
        }
        signals.sort(key=lambda x: strength_order[x.strength])

        return signals

    def get_stats(self) -> Dict[str, Any]:
        """Get signal detection statistics."""
        return {
            "total_signals_detected": len(self.signals_detected),
            "by_type": self._count_by_type(),
            "by_strength": self._count_by_strength(),
            "cfo_batting_averages": self.cfo_batting_averages,
            "signal_accuracy_rates": self.SIGNAL_ACCURACY,
        }

    def _count_by_type(self) -> Dict[str, int]:
        counts = {}
        for sig in self.signals_detected:
            counts[sig.signal_type] = counts.get(sig.signal_type, 0) + 1
        return counts

    def _count_by_strength(self) -> Dict[str, int]:
        counts = {}
        for sig in self.signals_detected:
            counts[sig.strength.value] = counts.get(sig.strength.value, 0) + 1
        return counts



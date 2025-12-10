"""================================================================================
RESEARCH PROCESS AGENT - "ANTI-SIGNALS & NEGATIVE SPACE"
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

The dog that didn't bark. What SHOULD have happened but DIDN'T?

Most analysts look at what happened. We look at what DIDN'T happen.
The absence of expected behavior is often more informative than the
behavior itself.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AbsenceType(Enum):
    """Types of absence signals."""

    MISSING_REACTION = "missing_reaction"
    UNEXPECTED_SILENCE = "unexpected_silence"
    DISCLOSURE_ANOMALY = "disclosure_anomaly"
    BEHAVIOR_GAP = "behavior_gap"


@dataclass
class AntiSignal:
    """A signal from what DIDN'T happen."""

    signal_id: str
    signal_type: str
    ticker: str
    direction: str
    confidence: float
    description: str
    expected_behavior: str
    actual_behavior: str
    gap_significance: float  # How significant is this gap?
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "signal_id": self.signal_id,
            "type": self.signal_type,
            "ticker": self.ticker,
            "direction": self.direction,
            "confidence": self.confidence,
            "description": self.description,
            "expected": self.expected_behavior,
            "actual": self.actual_behavior,
            "gap_significance": self.gap_significance,
            "detected_at": self.detected_at.isoformat(),
        }


class ResearchAntiSignals:
    """RESEARCH ANTI-SIGNALS - Finding edge in what DIDN'T happen

    Key Signals:
    1. Absent Signal Detection - What should have happened but didn't?
    2. Footnote Expansion Detector - Track footnote length changes
    3. MD&A Hedging Language Delta - NLP on uncertainty language
    4. Auditor Tenure Cliff - Long-tenured auditors leaving
    5. Restatement Ripple Detector - Peer group restatement risk
    6. SEC Comment Response Velocity - Speed of SEC responses
    7. Guidance Withdrawal Without Event - Pulling guidance mysteriously
    """

    # Hedge words to track in MD&A
    HEDGE_WORDS = [
        "may", "could", "might", "potentially", "possibly",
        "uncertain", "approximately", "estimated", "expected",
        "anticipated", "projected", "believed", "subject to",
        "contingent", "depending", "if", "unless", "assuming",
        "risk", "volatility", "fluctuation", "variable",
    ]

    # Warning phrases that signal problems
    WARNING_PHRASES = [
        "material weakness",
        "going concern",
        "restatement",
        "delayed filing",
        "unable to determine",
        "significant deficiency",
        "control deficiency",
        "remediation",
        "forensic",
        "investigation",
    ]

    def __init__(self):
        self.signals_detected: List[AntiSignal] = []
        self.footnote_history: Dict[str, Dict] = {}  # Track footnote lengths by company
        self.mda_history: Dict[str, Dict] = {}  # Track MD&A language history

    # =========================================================================
    # SIGNAL 1: ABSENT SIGNAL DETECTION
    # =========================================================================

    def detect_absent_signal(
        self,
        ticker: str,
        event_type: str,
        event_data: Dict,
        expected_reactions: List[str],
        actual_reactions: List[str],
    ) -> Optional[AntiSignal]:
        """Detect when expected reaction DIDN'T happen.

        What SHOULD have happened but DIDN'T?
        - Company hit record revenue but CEO didn't buy? Why not?
        - Competitor raised prices but you didn't follow? What do you know?
        - Great earnings but no guidance raise? What's coming?

        Args:
        ----
            ticker: Stock symbol
            event_type: Type of event (earnings, competitor_action, etc.)
            event_data: Details of the event
            expected_reactions: What we expected to happen
            actual_reactions: What actually happened

        Returns:
        -------
            AntiSignal if significant absence detected
        """
        import hashlib

        # Find missing reactions
        missing = set(expected_reactions) - set(actual_reactions)

        if not missing:
            return None  # Everything expected happened

        # Calculate significance based on what's missing
        critical_missing = [
            m for m in missing if any(
                word in m.lower() for word in ["insider", "guidance", "buyback", "dividend"]
            )
        ]

        if not critical_missing and len(missing) < 2:
            return None  # Not significant enough

        # Determine direction and confidence
        if event_type == "positive_earnings" and "insider_buying" in missing:
            direction = "bearish"
            confidence = 0.72
            gap_desc = "Record results but insiders NOT buying"
        elif event_type == "competitor_price_increase" and "price_follow" in missing:
            direction = "bearish"
            confidence = 0.68
            gap_desc = "Competitor raised prices but you didn't follow"
        elif event_type == "strong_quarter" and "guidance_raise" in missing:
            direction = "bearish"
            confidence = 0.70
            gap_desc = "Strong quarter but NO guidance raise"
        else:
            direction = "cautious"
            confidence = 0.55
            gap_desc = f"Missing expected reactions: {', '.join(list(missing)[:3])}"

        return AntiSignal(
            signal_id=f"abs_{hashlib.sha256(f'{ticker}{event_type}'.encode()).hexdigest()[:8]}",
            signal_type="absent_signal",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=f"ABSENCE DETECTED: {gap_desc} for {ticker}",
            expected_behavior=", ".join(expected_reactions),
            actual_behavior=", ".join(actual_reactions) if actual_reactions else "None",
            gap_significance=len(critical_missing) / max(len(expected_reactions), 1),
        )

    # =========================================================================
    # SIGNAL 2: FOOTNOTE EXPANSION DETECTOR
    # =========================================================================

    def footnote_expansion_detector(
        self,
        ticker: str,
        current_10k: Dict[str, int],  # footnote_name -> word_count
        previous_10k: Dict[str, int],
    ) -> List[AntiSignal]:
        """Track footnote length by category across 10-Ks.

        Revenue recognition footnote grew 340%? Something's being explained away.
        Legal contingency section shrunk dramatically? Settlement incoming.

        Args:
        ----
            ticker: Stock symbol
            current_10k: Current year footnote word counts by category
            previous_10k: Previous year footnote word counts by category

        Returns:
        -------
            List of signals for significant footnote changes
        """
        import hashlib

        signals = []

        # Categories to watch closely
        critical_categories = [
            "revenue_recognition",
            "legal_contingencies",
            "related_party",
            "going_concern",
            "segment_reporting",
            "goodwill_impairment",
            "stock_compensation",
            "debt_covenants",
        ]

        for category in critical_categories:
            current = current_10k.get(category, 0)
            previous = previous_10k.get(category, 0)

            if previous == 0:
                continue

            change_pct = (current - previous) / previous

            # Significant expansion
            if change_pct > 1.0:  # More than doubled
                direction = "bearish"
                confidence = 0.70 + min(change_pct / 10, 0.2)

                signals.append(AntiSignal(
                    signal_id=f"fn_{hashlib.sha256(f'{ticker}{category}'.encode()).hexdigest()[:8]}",
                    signal_type="footnote_expansion",
                    ticker=ticker,
                    direction=direction,
                    confidence=confidence,
                    description=(
                        f"FOOTNOTE EXPLOSION: {category.replace('_', ' ').title()} "
                        f"grew {change_pct:.0%} in {ticker} 10-K"
                    ),
                    expected_behavior="Stable or minor changes to footnote",
                    actual_behavior=f"Footnote expanded from {previous} to {current} words (+{change_pct:.0%})",
                    gap_significance=min(change_pct / 5, 1.0),
                ))

            # Significant shrinkage (also suspicious)
            elif change_pct < -0.5:  # More than halved
                if category in ["legal_contingencies", "related_party"]:
                    direction = "uncertain"  # Could be settlement (good) or hiding (bad)
                    confidence = 0.60

                    signals.append(AntiSignal(
                        signal_id=f"fn_{hashlib.sha256(f'{ticker}{category}shrink'.encode()).hexdigest()[:8]}",
                        signal_type="footnote_shrinkage",
                        ticker=ticker,
                        direction=direction,
                        confidence=confidence,
                        description=(
                            f"FOOTNOTE SHRINKAGE: {category.replace('_', ' ').title()} "
                            f"shrunk {abs(change_pct):.0%} in {ticker} 10-K"
                        ),
                        expected_behavior="Stable disclosure",
                        actual_behavior=f"Footnote shrunk from {previous} to {current} words ({change_pct:.0%})",
                        gap_significance=min(abs(change_pct) / 3, 1.0),
                    ))

        # Store history
        self.footnote_history[ticker] = {
            "current": current_10k,
            "previous": previous_10k,
            "analyzed_at": datetime.now().isoformat(),
        }

        self.signals_detected.extend(signals)
        return signals

    # =========================================================================
    # SIGNAL 3: MD&A HEDGING LANGUAGE DELTA
    # =========================================================================

    def mda_hedging_language_delta(
        self,
        ticker: str,
        current_mda: str,
        previous_mda: str,
    ) -> Optional[AntiSignal]:
        """NLP on Management Discussion: track hedge words.

        Quarter-over-quarter increase in uncertainty language = leading indicator.

        Hedge words: may, could, potentially, uncertain, approximately, etc.

        Args:
        ----
            ticker: Stock symbol
            current_mda: Current MD&A text
            previous_mda: Previous period MD&A text

        Returns:
        -------
            AntiSignal if significant hedging increase
        """
        import hashlib

        def count_hedge_words(text: str) -> Tuple[int, int]:
            """Count hedge words and total words."""
            text_lower = text.lower()
            words = text_lower.split()
            total = len(words)
            hedge_count = sum(
                text_lower.count(f" {word} ") + text_lower.count(f" {word}.") + text_lower.count(f" {word},")
                for word in self.HEDGE_WORDS
            )
            return hedge_count, total

        current_hedge, current_total = count_hedge_words(current_mda)
        previous_hedge, previous_total = count_hedge_words(previous_mda)

        if previous_total == 0 or current_total == 0:
            return None

        current_density = current_hedge / current_total
        previous_density = previous_hedge / previous_total

        # Calculate change
        if previous_density == 0:
            density_change = 1.0 if current_density > 0 else 0
        else:
            density_change = (current_density - previous_density) / previous_density

        # Store history
        self.mda_history[ticker] = {
            "current_density": current_density,
            "previous_density": previous_density,
            "change": density_change,
            "analyzed_at": datetime.now().isoformat(),
        }

        # Only signal on significant increase
        if density_change < 0.2:  # Less than 20% increase
            return None

        # Determine severity
        if density_change > 0.5:
            confidence = 0.75
            severity = "MAJOR"
        elif density_change > 0.3:
            confidence = 0.65
            severity = "MODERATE"
        else:
            confidence = 0.55
            severity = "MINOR"

        return AntiSignal(
            signal_id=f"mda_{hashlib.sha256(f'{ticker}hedge'.encode()).hexdigest()[:8]}",
            signal_type="mda_hedging_increase",
            ticker=ticker,
            direction="bearish",
            confidence=confidence,
            description=(
                f"{severity} HEDGING INCREASE: {ticker} MD&A uncertainty language "
                f"up {density_change:.0%}"
            ),
            expected_behavior=f"Stable hedge word density (~{previous_density:.2%})",
            actual_behavior=f"Hedge word density increased to {current_density:.2%} (+{density_change:.0%})",
            gap_significance=min(density_change, 1.0),
        )

    # =========================================================================
    # SIGNAL 4: AUDITOR TENURE CLIFF
    # =========================================================================

    def auditor_tenure_cliff(
        self,
        ticker: str,
        auditor_name: str,
        tenure_years: int,
        departure_reason: str = "rotation",
    ) -> Optional[AntiSignal]:
        """Detect when long-tenured auditor suddenly leaves.

        Auditor of 15+ years suddenly "rotates"?
        Long-tenured auditors don't leave clean clients.

        Args:
        ----
            ticker: Stock symbol
            auditor_name: Departing auditor
            tenure_years: How long they audited this company
            departure_reason: Stated reason for departure

        Returns:
        -------
            AntiSignal if suspicious departure
        """
        import hashlib

        if tenure_years < 10:
            return None  # Not long enough tenure to be suspicious

        # Determine suspicion level
        if tenure_years >= 20:
            confidence = 0.80
            severity = "CRITICAL"
        elif tenure_years >= 15:
            confidence = 0.72
            severity = "HIGH"
        else:
            confidence = 0.62
            severity = "ELEVATED"

        # Check for euphemistic language
        euphemisms = ["mutual decision", "rotation", "transition", "new direction", "fresh perspective"]
        is_euphemistic = any(e in departure_reason.lower() for e in euphemisms)

        if is_euphemistic:
            confidence += 0.05

        return AntiSignal(
            signal_id=f"aud_{hashlib.sha256(f'{ticker}{auditor_name}'.encode()).hexdigest()[:8]}",
            signal_type="auditor_tenure_cliff",
            ticker=ticker,
            direction="bearish",
            confidence=confidence,
            description=(
                f"{severity} AUDITOR DEPARTURE: {auditor_name} leaving {ticker} "
                f"after {tenure_years} years"
            ),
            expected_behavior="Long-tenured auditors stay with clean clients",
            actual_behavior=f"Auditor of {tenure_years} years departing. Reason: '{departure_reason}'",
            gap_significance=min(tenure_years / 25, 1.0),
        )

    # =========================================================================
    # SIGNAL 5: RESTATEMENT RIPPLE DETECTOR
    # =========================================================================

    def restatement_ripple_detector(
        self,
        restating_company: str,
        restating_issue: str,
        peer_companies: List[str],
        peer_accounting_practices: Dict[str, str],
    ) -> List[AntiSignal]:
        """When Company A restates, scan peer group for similar accounting.

        First restater is canary in coal mine.
        If peers use same accounting treatment, they may be next.

        Args:
        ----
            restating_company: Company that announced restatement
            restating_issue: What they're restating (rev rec, lease, etc.)
            peer_companies: List of peer company tickers
            peer_accounting_practices: Dict of peer -> their accounting practice

        Returns:
        -------
            List of signals for at-risk peers
        """
        import hashlib

        signals = []

        for peer in peer_companies:
            peer_practice = peer_accounting_practices.get(peer, "unknown")

            # Check if peer uses similar accounting
            if restating_issue.lower() in peer_practice.lower():
                confidence = 0.68
                risk_level = "HIGH"
            elif "aggressive" in peer_practice.lower():
                confidence = 0.55
                risk_level = "MODERATE"
            else:
                continue  # No clear risk

            signals.append(AntiSignal(
                signal_id=f"rst_{hashlib.sha256(f'{peer}{restating_issue}'.encode()).hexdigest()[:8]}",
                signal_type="restatement_ripple",
                ticker=peer,
                direction="bearish",
                confidence=confidence,
                description=(
                    f"RESTATEMENT RIPPLE ({risk_level}): {restating_company} restated "
                    f"'{restating_issue}' - {peer} uses similar accounting"
                ),
                expected_behavior="Peer should review similar accounting treatments",
                actual_behavior=f"Peer uses: '{peer_practice}' - potential similar issues",
                gap_significance=0.7 if risk_level == "HIGH" else 0.5,
            ))

        self.signals_detected.extend(signals)
        return signals

    # =========================================================================
    # SIGNAL 6: SEC COMMENT RESPONSE VELOCITY
    # =========================================================================

    def sec_comment_response_velocity(
        self,
        ticker: str,
        comment_date: datetime,
        response_date: datetime,
        topic: str,
        historical_avg_days: float = 10.0,
    ) -> Optional[AntiSignal]:
        """Track days-to-respond to SEC comment letters.

        Slow responses = difficult questions internally
        Fast responses = clean books or well-prepared

        Args:
        ----
            ticker: Stock symbol
            comment_date: When SEC sent comment
            response_date: When company responded
            topic: Topic of the comment
            historical_avg_days: Company's historical average response time

        Returns:
        -------
            AntiSignal based on response velocity
        """
        import hashlib

        days_to_respond = (response_date - comment_date).days

        # Compare to historical average
        velocity_ratio = days_to_respond / historical_avg_days if historical_avg_days > 0 else 1.0

        # Detect anomalies
        if velocity_ratio > 2.0:  # Took more than 2x normal time
            direction = "bearish"
            confidence = 0.65 + min((velocity_ratio - 2) / 10, 0.2)
            desc = f"SLOW SEC RESPONSE: {ticker} took {days_to_respond} days (vs {historical_avg_days:.0f} avg)"
            significance = min(velocity_ratio / 5, 1.0)
        elif velocity_ratio < 0.3:  # Responded super fast
            direction = "bullish"
            confidence = 0.60
            desc = f"FAST SEC RESPONSE: {ticker} responded in {days_to_respond} days (vs {historical_avg_days:.0f} avg)"
            significance = 0.4
        else:
            return None  # Normal response time

        return AntiSignal(
            signal_id=f"sec_{hashlib.sha256(f'{ticker}{topic}'.encode()).hexdigest()[:8]}",
            signal_type="sec_comment_velocity",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            expected_behavior=f"Response in ~{historical_avg_days:.0f} days",
            actual_behavior=f"Responded in {days_to_respond} days on topic: {topic}",
            gap_significance=significance,
        )

    # =========================================================================
    # SIGNAL 7: GUIDANCE WITHDRAWAL WITHOUT EVENT
    # =========================================================================

    def guidance_withdrawal_without_event(
        self,
        ticker: str,
        withdrawal_date: datetime,
        had_guidance: bool,
        macro_event: bool = False,
        company_event: bool = False,
        peer_withdrawals: int = 0,
    ) -> Optional[AntiSignal]:
        """Detect when company pulls guidance without obvious reason.

        Company pulls guidance but there's no obvious macro/company event?
        They know something is breaking before it shows in numbers.

        Args:
        ----
            ticker: Stock symbol
            withdrawal_date: When guidance was pulled
            had_guidance: Did they have guidance before?
            macro_event: Is there an obvious macro event?
            company_event: Is there an obvious company event?
            peer_withdrawals: How many peers also pulled guidance?

        Returns:
        -------
            AntiSignal if mysterious withdrawal
        """
        import hashlib

        if not had_guidance:
            return None  # Can't withdraw what you didn't have

        if macro_event and peer_withdrawals >= 3:
            return None  # Macro-driven, everyone's doing it

        if company_event:
            return None  # Explained by company event

        # Mysterious withdrawal - no obvious explanation
        if peer_withdrawals == 0:
            confidence = 0.78
            severity = "CRITICAL"
        elif peer_withdrawals < 2:
            confidence = 0.70
            severity = "HIGH"
        else:
            confidence = 0.60
            severity = "MODERATE"

        return AntiSignal(
            signal_id=f"guid_{hashlib.sha256(f'{ticker}withdrawal'.encode()).hexdigest()[:8]}",
            signal_type="guidance_withdrawal_no_event",
            ticker=ticker,
            direction="bearish",
            confidence=confidence,
            description=(
                f"{severity} MYSTERIOUS GUIDANCE PULL: {ticker} withdrew guidance "
                f"with no obvious event"
            ),
            expected_behavior="Guidance maintained unless material change",
            actual_behavior=(
                f"Guidance withdrawn on {withdrawal_date.strftime('%Y-%m-%d')}. "
                f"No macro event, no company event. Only {peer_withdrawals} peers did same."
            ),
            gap_significance=0.85 if severity == "CRITICAL" else 0.65,
        )

    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get signal detection statistics."""
        return {
            "total_signals": len(self.signals_detected),
            "by_type": self._count_by_type(),
            "companies_with_footnote_history": len(self.footnote_history),
            "companies_with_mda_history": len(self.mda_history),
        }

    def _count_by_type(self) -> Dict[str, int]:
        counts = {}
        for sig in self.signals_detected:
            counts[sig.signal_type] = counts.get(sig.signal_type, 0) + 1
        return counts



"""================================================================================
ALTERNATIVE DATA AGENT - "PHYSICAL WORLD SIGNALS"
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Track the physical world - freight, construction, satellites, jets.
The real economy tells us things before financial statements do.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PhysicalSignal:
    """A signal from physical world data."""

    signal_id: str
    signal_type: str
    ticker: str
    direction: str
    confidence: float
    description: str
    physical_evidence: str
    financial_implication: str
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class AlternativeDataSignals:
    """PHYSICAL WORLD SIGNALS - See the real economy

    1. Freight Invoice Mismatch - Inventory vs logistics data
    2. Construction Permit Leading Indicator - Permits vs capex
    3. Satellite Parking Lot Derivative - Velocity of change
    4. Equipment Auction Liquidation - Distress before filings
    5. Corporate Jet Pattern Anomaly - Track N-numbers
    6. Patent Citation Graph Shift - IP value changes
    7. Job Posting Urgency Analysis - NLP on hiring language
    8. Glassdoor Second Derivative - Rate of rating change
    9. LinkedIn Profile Update Surge - Pre-layoff signals
    10. Domain Registration Stealth - M&A clues
    """

    def __init__(self):
        self.signals_detected: List[PhysicalSignal] = []

    def freight_invoice_mismatch(
        self,
        ticker: str,
        reported_inventory_change: float,
        freight_volume_change: float,
        logistics_data: Dict,
    ) -> Optional[PhysicalSignal]:
        """Cross-reference reported inventory with freight/logistics.
        If inventory UP but freight DOWN? Channel stuffing alert.
        """
        import hashlib

        mismatch = reported_inventory_change - freight_volume_change

        if abs(mismatch) < 0.15:
            return None

        if mismatch > 0.2:  # Inventory up, freight not matching
            return PhysicalSignal(
                signal_id=f"frt_{hashlib.sha256(f'{ticker}freight'.encode()).hexdigest()[:8]}",
                signal_type="freight_invoice_mismatch",
                ticker=ticker,
                direction="bearish",
                confidence=0.72,
                description=f"CHANNEL STUFFING ALERT: {ticker} inventory +{reported_inventory_change:.0%} but freight only +{freight_volume_change:.0%}",
                physical_evidence=f"Freight volume change: {freight_volume_change:.0%}",
                financial_implication="Inventory may not be real sales - potential stuffing",
            )

        return None

    def construction_permit_leading_indicator(
        self,
        ticker: str,
        announced_capex: float,
        permits_filed: int,
        permits_expected: int,
    ) -> Optional[PhysicalSignal]:
        """Track building permits vs announced capex.
        Permits filed but nothing announced = stealth expansion.
        Capex announced but no permits = vapor announcement.
        """
        import hashlib

        if permits_filed > permits_expected * 1.5 and announced_capex == 0:
            return PhysicalSignal(
                signal_id=f"prm_{hashlib.sha256(f'{ticker}permit'.encode()).hexdigest()[:8]}",
                signal_type="construction_permit",
                ticker=ticker,
                direction="bullish",
                confidence=0.68,
                description=f"STEALTH EXPANSION: {ticker} filed {permits_filed} permits but announced no capex",
                physical_evidence=f"{permits_filed} permits filed vs {permits_expected} expected",
                financial_implication="Expansion coming before announcement",
            )
        elif announced_capex > 0 and permits_filed < permits_expected * 0.3:
            return PhysicalSignal(
                signal_id=f"prm_{hashlib.sha256(f'{ticker}vapor'.encode()).hexdigest()[:8]}",
                signal_type="construction_permit",
                ticker=ticker,
                direction="bearish",
                confidence=0.65,
                description=f"VAPOR CAPEX: {ticker} announced ${announced_capex:,.0f} capex but only {permits_filed} permits",
                physical_evidence=f"Only {permits_filed} permits filed for major capex",
                financial_implication="Announced capex may not materialize",
            )

        return None

    def satellite_parking_lot_derivative(
        self,
        ticker: str,
        current_occupancy: float,
        previous_occupancy: float,
        rate_of_change: float,
    ) -> Optional[PhysicalSignal]:
        """Not just parking lot fullness - track VELOCITY of change.
        Rapidly emptying = deteriorating faster than consensus.
        """
        import hashlib

        if abs(rate_of_change) < 0.1:
            return None

        direction = "bearish" if rate_of_change < 0 else "bullish"
        severity = "RAPID" if abs(rate_of_change) > 0.25 else "MODERATE"

        return PhysicalSignal(
            signal_id=f"sat_{hashlib.sha256(f'{ticker}parking'.encode()).hexdigest()[:8]}",
            signal_type="satellite_parking_velocity",
            ticker=ticker,
            direction=direction,
            confidence=0.65 + min(abs(rate_of_change), 0.2),
            description=f"{severity} PARKING CHANGE: {ticker} lots {'emptying' if rate_of_change < 0 else 'filling'} at {abs(rate_of_change):.0%} rate",
            physical_evidence=f"Occupancy: {previous_occupancy:.0%} → {current_occupancy:.0%} (Δ {rate_of_change:+.0%})",
            financial_implication="Traffic trends diverging from consensus" if abs(rate_of_change) > 0.15 else "Monitor closely",
        )

    def corporate_jet_pattern_anomaly(
        self,
        ticker: str,
        n_number: str,
        destination_city: str,
        is_unusual: bool,
        potential_significance: str,
    ) -> Optional[PhysicalSignal]:
        """Track N-number flights for unusual destinations.
        CEO jet to Omaha for no disclosed reason?
        Competitor HQ? PE firm city?
        """
        import hashlib

        if not is_unusual:
            return None

        return PhysicalSignal(
            signal_id=f"jet_{hashlib.sha256(f'{ticker}{destination_city}'.encode()).hexdigest()[:8]}",
            signal_type="corporate_jet_anomaly",
            ticker=ticker,
            direction="uncertain",
            confidence=0.55,
            description=f"JET ANOMALY: {ticker} aircraft ({n_number}) flew to {destination_city}",
            physical_evidence=f"Flight to {destination_city} - {potential_significance}",
            financial_implication=potential_significance,
        )

    def glassdoor_second_derivative(
        self,
        ticker: str,
        current_rating: float,
        previous_rating: float,
        rate_of_change: float,
        engineering_rating_change: float = None,
    ) -> Optional[PhysicalSignal]:
        """Track RATE OF CHANGE of Glassdoor ratings.
        Accelerating negative = culture implosion before turnover shows.
        Focus on engineering team for tech companies.
        """
        import hashlib

        if abs(rate_of_change) < 0.05:
            return None

        # Engineering ratings extra weight for tech
        effective_change = rate_of_change
        if engineering_rating_change is not None:
            effective_change = rate_of_change * 0.6 + engineering_rating_change * 0.4

        direction = "bearish" if effective_change < 0 else "bullish"

        return PhysicalSignal(
            signal_id=f"gls_{hashlib.sha256(f'{ticker}glassdoor'.encode()).hexdigest()[:8]}",
            signal_type="glassdoor_velocity",
            ticker=ticker,
            direction=direction,
            confidence=0.62 + min(abs(effective_change) * 2, 0.2),
            description=f"CULTURE {'IMPLOSION' if effective_change < -0.1 else 'IMPROVEMENT' if effective_change > 0.1 else 'SHIFT'}: {ticker} Glassdoor trending {effective_change:+.0%}",
            physical_evidence=f"Rating: {previous_rating:.1f} → {current_rating:.1f} (Δ {rate_of_change:+.0%})",
            financial_implication="Employee morale leading indicator for turnover and productivity",
        )

    def linkedin_profile_update_surge(
        self,
        ticker: str,
        pct_employees_updated: float,
        timeframe_days: int,
        baseline_pct: float,
    ) -> Optional[PhysicalSignal]:
        """When 15%+ of employees update profiles same week.
        Pre-layoff signal or acquisition prep.
        """
        import hashlib

        surge_ratio = pct_employees_updated / baseline_pct if baseline_pct > 0 else pct_employees_updated * 10

        if surge_ratio < 2.0:
            return None

        return PhysicalSignal(
            signal_id=f"lnk_{hashlib.sha256(f'{ticker}linkedin'.encode()).hexdigest()[:8]}",
            signal_type="linkedin_update_surge",
            ticker=ticker,
            direction="uncertain",  # Could be layoffs (bad) or acquisition (could be good)
            confidence=0.65,
            description=f"LINKEDIN SURGE: {pct_employees_updated:.0%} of {ticker} employees updated profiles in {timeframe_days} days",
            physical_evidence=f"Update rate {surge_ratio:.1f}x normal baseline",
            financial_implication="Potential layoffs or M&A activity incoming",
        )

    def domain_registration_stealth_signal(
        self,
        ticker: str,
        new_domains: List[str],
        domain_type: str,
    ) -> Optional[PhysicalSignal]:
        """Monitor WHOIS for new domain registrations.
        New product domains before announcement.
        Defensive domains = M&A target awareness.
        """
        import hashlib

        if not new_domains:
            return None

        if domain_type == "product":
            direction = "bullish"
            desc = f"NEW PRODUCT DOMAINS: {ticker} registered {len(new_domains)} product-related domains"
            implication = "New product launch likely before announcement"
        elif domain_type == "defensive":
            direction = "uncertain"
            desc = f"DEFENSIVE DOMAINS: {ticker} registered {len(new_domains)} defensive domains"
            implication = "May be aware of M&A interest or brand protection"
        else:
            return None

        return PhysicalSignal(
            signal_id=f"dom_{hashlib.sha256(f'{ticker}domain'.encode()).hexdigest()[:8]}",
            signal_type="domain_registration",
            ticker=ticker,
            direction=direction,
            confidence=0.58,
            description=desc,
            physical_evidence=f"Domains: {', '.join(new_domains[:3])}{'...' if len(new_domains) > 3 else ''}",
            financial_implication=implication,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_signals": len(self.signals_detected),
            "by_type": {s.signal_type: sum(1 for x in self.signals_detected if x.signal_type == s.signal_type) for s in self.signals_detected},
        }



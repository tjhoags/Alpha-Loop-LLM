"""MODULE 4: LIQUIDITY DISTORTION DETECTOR
========================================
Alpha Loop Capital - Consequence Engine

Purpose: Identify when price is artificially supported/suppressed
         Time shorts when artificial support expires
         Avoid shorts when technical flows overwhelm fundamentals

Core Edge: See through liquidity distortions to real price
           Time entries when artificial support ends
           Avoid fighting non-fundamental flows

Author: Tom Hogan
Version: 1.0

TOM'S GAP:
"I lack insight into quantitative liquidity provisions that may
artificially distract price from a real story."

THIS MODULE SOLVES THAT:
- Gamma positioning (market maker hedging)
- Buyback activity (corporate bid)
- Index rebalancing flows (passive forced buying/selling)
- Short squeeze mechanics
- Dark pool activity
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistortionType(Enum):
    """Types of liquidity distortion affecting price"""

    GAMMA_SUPPORT = "gamma_support"
    GAMMA_RESISTANCE = "gamma_resistance"
    BUYBACK_BID = "buyback_bid"
    INDEX_INCLUSION = "index_inclusion"
    INDEX_EXCLUSION = "index_exclusion"
    SHORT_SQUEEZE = "short_squeeze"
    DARK_POOL_ACCUMULATION = "dark_pool_accumulation"
    DARK_POOL_DISTRIBUTION = "dark_pool_distribution"
    OPTIONS_EXPIRY = "options_expiry"
    QUARTER_END = "quarter_end"


class DistortionStrength(Enum):
    """How strong is the artificial support/resistance"""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    DOMINANT = "dominant"


class DistortionDirection(Enum):
    """Direction of the distortion"""

    UPWARD = "upward"
    DOWNWARD = "downward"
    PINNING = "pinning"


@dataclass
class GammaExposure:
    """Market maker gamma exposure data"""

    ticker: str
    date: str
    net_gamma_mm: float  # Net gamma in millions of shares
    gamma_flip_price: float
    call_wall: float
    put_wall: float
    max_pain: float
    gamma_by_strike: Dict[float, float] = field(default_factory=dict)
    dealer_position: str = "neutral"
    days_to_expiry: int = 0

    @property
    def is_supportive(self) -> bool:
        """Is current gamma positioning supportive of price?"""
        return self.net_gamma_mm > 0

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "net_gamma_mm": self.net_gamma_mm,
            "gamma_flip_price": self.gamma_flip_price,
            "call_wall": self.call_wall,
            "put_wall": self.put_wall,
            "max_pain": self.max_pain,
            "dealer_position": self.dealer_position,
            "days_to_expiry": self.days_to_expiry,
        }


@dataclass
class BuybackProgram:
    """Corporate buyback program data"""

    ticker: str
    authorized_mm: float  # Authorized amount in millions
    remaining_mm: float
    daily_limit_pct: float  # 25% of ADV typically
    avg_daily_volume_mm: float
    blackout_start: Optional[str] = None
    blackout_end: Optional[str] = None
    execution_style: str = "regular"

    @property
    def daily_bid_mm(self) -> float:
        """Estimated daily buying power"""
        return self.avg_daily_volume_mm * (self.daily_limit_pct / 100)

    @property
    def days_remaining(self) -> int:
        """Estimated days until program exhausted"""
        if self.daily_bid_mm <= 0:
            return 0
        return int(self.remaining_mm / self.daily_bid_mm)

    @property
    def in_blackout(self) -> bool:
        """Is company in earnings blackout?"""
        if not self.blackout_start or not self.blackout_end:
            return False
        today = datetime.now().strftime("%Y-%m-%d")
        return self.blackout_start <= today <= self.blackout_end


@dataclass
class ShortInterest:
    """Short interest and squeeze risk data"""

    ticker: str
    date: str
    short_interest_shares_mm: float
    short_interest_pct: float
    days_to_cover: float
    borrow_rate_pct: float
    shares_available_mm: float
    utilization_pct: float

    @property
    def squeeze_risk(self) -> str:
        """Assess short squeeze risk"""
        if self.days_to_cover > 5 and self.utilization_pct > 90:
            return "HIGH"
        elif self.days_to_cover > 3 and self.utilization_pct > 70:
            return "MODERATE"
        elif self.days_to_cover > 2:
            return "LOW"
        return "MINIMAL"


@dataclass
class Distortion:
    """A single identified liquidity distortion"""

    ticker: str
    distortion_type: DistortionType
    direction: DistortionDirection
    strength: DistortionStrength
    start_date: str
    expected_end_date: Optional[str]
    price_impact_estimate: float
    description: str
    confidence: float = 0.5

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "type": self.distortion_type.value,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "start_date": self.start_date,
            "expected_end_date": self.expected_end_date,
            "price_impact_estimate": self.price_impact_estimate,
            "description": self.description,
            "confidence": self.confidence,
        }


class LiquidityDistortionDetector:
    """LIQUIDITY DISTORTION DETECTOR

    Identifies artificial price support/suppression.

    Use cases:
    1. Short timing: Wait for support to expire
    2. Long entry: Identify accumulation
    3. Risk management: Know when floor removed

    Data sources:
    - Options flow / gamma positioning
    - Corporate buyback filings
    - Index rebalancing announcements
    - Short interest data
    - Dark pool prints
    """

    def __init__(self):
        self.distortions: Dict[str, List[Distortion]] = {}
        self.gamma_data: Dict[str, GammaExposure] = {}
        self.buyback_programs: Dict[str, BuybackProgram] = {}
        self.short_interest: Dict[str, ShortInterest] = {}

    def add_gamma_exposure(self, gamma: GammaExposure) -> None:
        """Add gamma exposure data for a ticker"""
        self.gamma_data[gamma.ticker] = gamma
        self._detect_gamma_distortion(gamma)

    def add_buyback_program(self, buyback: BuybackProgram) -> None:
        """Add buyback program data"""
        self.buyback_programs[buyback.ticker] = buyback
        self._detect_buyback_distortion(buyback)

    def add_short_interest(self, si: ShortInterest) -> None:
        """Add short interest data"""
        self.short_interest[si.ticker] = si
        self._detect_squeeze_risk(si)

    def _detect_gamma_distortion(self, gamma: GammaExposure) -> None:
        """Detect gamma-based price distortion"""
        if abs(gamma.net_gamma_mm) < 0.5:
            return  # Not significant

        if gamma.is_supportive:
            distortion = Distortion(
                ticker=gamma.ticker,
                distortion_type=DistortionType.GAMMA_SUPPORT,
                direction=DistortionDirection.UPWARD,
                strength=self._gamma_strength(gamma.net_gamma_mm),
                start_date=gamma.date,
                expected_end_date=self._get_next_opex(),
                price_impact_estimate=abs(gamma.net_gamma_mm) * 0.5,
                description=f"Dealer long {gamma.net_gamma_mm:.1f}M gamma - will buy dips",
                confidence=0.7,
            )
        else:
            distortion = Distortion(
                ticker=gamma.ticker,
                distortion_type=DistortionType.GAMMA_RESISTANCE,
                direction=DistortionDirection.DOWNWARD,
                strength=self._gamma_strength(abs(gamma.net_gamma_mm)),
                start_date=gamma.date,
                expected_end_date=self._get_next_opex(),
                price_impact_estimate=abs(gamma.net_gamma_mm) * 0.5,
                description=f"Dealer short {abs(gamma.net_gamma_mm):.1f}M gamma - will sell rips",
                confidence=0.7,
            )

        self._add_distortion(distortion)

    def _detect_buyback_distortion(self, buyback: BuybackProgram) -> None:
        """Detect buyback-based price support"""
        if buyback.in_blackout or buyback.remaining_mm < 10:
            return

        strength = DistortionStrength.WEAK
        if buyback.daily_bid_mm > 5:
            strength = DistortionStrength.MODERATE
        if buyback.daily_bid_mm > 20:
            strength = DistortionStrength.STRONG

        distortion = Distortion(
            ticker=buyback.ticker,
            distortion_type=DistortionType.BUYBACK_BID,
            direction=DistortionDirection.UPWARD,
            strength=strength,
            start_date=datetime.now().strftime("%Y-%m-%d"),
            expected_end_date=buyback.blackout_start,
            price_impact_estimate=buyback.daily_bid_mm * 5,
            description=f"Buyback: ${buyback.daily_bid_mm:.1f}M daily, {buyback.days_remaining} days remaining",
            confidence=0.8,
        )

        self._add_distortion(distortion)

    def _detect_squeeze_risk(self, si: ShortInterest) -> None:
        """Detect short squeeze risk"""
        if si.squeeze_risk not in ["HIGH", "MODERATE"]:
            return

        strength = DistortionStrength.MODERATE if si.squeeze_risk == "MODERATE" else DistortionStrength.STRONG

        distortion = Distortion(
            ticker=si.ticker,
            distortion_type=DistortionType.SHORT_SQUEEZE,
            direction=DistortionDirection.UPWARD,
            strength=strength,
            start_date=si.date,
            expected_end_date=None,
            price_impact_estimate=si.short_interest_pct * 2,
            description=f"Squeeze risk {si.squeeze_risk}: {si.days_to_cover:.1f} DTC, {si.utilization_pct:.0f}% utilization",
            confidence=0.6,
        )

        self._add_distortion(distortion)

    def _add_distortion(self, distortion: Distortion) -> None:
        """Add distortion to tracking"""
        ticker = distortion.ticker
        if ticker not in self.distortions:
            self.distortions[ticker] = []
        self.distortions[ticker].append(distortion)
        logger.info(f"Detected {distortion.distortion_type.value} for {ticker}")

    def _gamma_strength(self, gamma_mm: float) -> DistortionStrength:
        """Classify gamma strength"""
        if gamma_mm > 5:
            return DistortionStrength.DOMINANT
        elif gamma_mm > 2:
            return DistortionStrength.STRONG
        elif gamma_mm > 0.5:
            return DistortionStrength.MODERATE
        return DistortionStrength.WEAK

    def _get_next_opex(self) -> str:
        """Get next monthly options expiry date"""
        today = datetime.now()
        # Third Friday of current/next month
        year = today.year
        month = today.month

        # Find third Friday
        first_day = datetime(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)

        if third_friday <= today:
            # Move to next month
            month = month + 1 if month < 12 else 1
            year = year if month > 1 else year + 1
            first_day = datetime(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)

        return third_friday.strftime("%Y-%m-%d")

    def get_ticker_analysis(self, ticker: str) -> Dict:
        """Get complete distortion analysis for a ticker.
        """
        distortions = self.distortions.get(ticker, [])
        gamma = self.gamma_data.get(ticker)
        buyback = self.buyback_programs.get(ticker)
        si = self.short_interest.get(ticker)

        # Calculate composite scores
        upward_score = sum(
            d.price_impact_estimate for d in distortions
            if d.direction == DistortionDirection.UPWARD
        )
        downward_score = sum(
            d.price_impact_estimate for d in distortions
            if d.direction == DistortionDirection.DOWNWARD
        )

        # Net distortion
        net_distortion = upward_score - downward_score

        # Short timing signal
        if si and upward_score > 5:
            short_timing = "wait"  # Artificial support active
            short_rationale = f"Upward distortion score {upward_score:.1f} - wait for support to expire"
        elif si and si.squeeze_risk in ["HIGH", "MODERATE"]:
            short_timing = "avoid"
            short_rationale = f"Squeeze risk {si.squeeze_risk} - avoid short"
        elif downward_score > upward_score:
            short_timing = "consider"
            short_rationale = "Net downward pressure - short setup may be forming"
        else:
            short_timing = "neutral"
            short_rationale = "No strong distortion signal"

        return {
            "ticker": ticker,
            "distortion_count": len(distortions),
            "distortions": [d.to_dict() for d in distortions],
            "upward_distortion_score": round(upward_score, 1),
            "downward_distortion_score": round(downward_score, 1),
            "net_distortion": round(net_distortion, 1),
            "gamma_data": gamma.to_dict() if gamma else None,
            "buyback_active": buyback is not None and not buyback.in_blackout,
            "squeeze_risk": si.squeeze_risk if si else "UNKNOWN",
            "short_timing": short_timing,
            "short_rationale": short_rationale,
            "key_dates": self._get_key_dates(ticker),
        }

    def _get_key_dates(self, ticker: str) -> List[Dict]:
        """Get key dates when distortions may change"""
        dates = []

        # Next OPEX
        dates.append({
            "date": self._get_next_opex(),
            "event": "Monthly OPEX",
            "impact": "Gamma reset",
        })

        # Buyback blackout
        buyback = self.buyback_programs.get(ticker)
        if buyback and buyback.blackout_start:
            dates.append({
                "date": buyback.blackout_start,
                "event": "Buyback blackout starts",
                "impact": "Support removed",
            })

        return sorted(dates, key=lambda x: x["date"])

    def generate_report(self, ticker: str) -> str:
        """Generate human-readable distortion report"""
        analysis = self.get_ticker_analysis(ticker)

        lines = [
            "=" * 60,
            f"LIQUIDITY DISTORTION ANALYSIS: {ticker}",
            f"Date: {datetime.now().strftime('%Y-%m-%d')}",
            "=" * 60,
            "",
            f"📊 NET DISTORTION: {analysis['net_distortion']:+.1f}",
            f"   ↑ Upward: {analysis['upward_distortion_score']:.1f}",
            f"   ↓ Downward: {analysis['downward_distortion_score']:.1f}",
            "",
            f"⚠️ SQUEEZE RISK: {analysis['squeeze_risk']}",
            f"🛒 BUYBACK ACTIVE: {'Yes' if analysis['buyback_active'] else 'No'}",
            "",
            f"📌 SHORT TIMING: {analysis['short_timing'].upper()}",
            f"   {analysis['short_rationale']}",
            "",
        ]

        if analysis["distortions"]:
            lines.append("ACTIVE DISTORTIONS:")
            for d in analysis["distortions"]:
                lines.append(f"  • {d['type']}: {d['description']}")

        if analysis["key_dates"]:
            lines.extend(["", "KEY DATES:"])
            for kd in analysis["key_dates"]:
                lines.append(f"  • {kd['date']}: {kd['event']} ({kd['impact']})")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    detector = LiquidityDistortionDetector()

    # Add gamma data
    detector.add_gamma_exposure(GammaExposure(
        ticker="NVDA",
        date="2024-12-08",
        net_gamma_mm=3.5,
        gamma_flip_price=130,
        call_wall=145,
        put_wall=125,
        max_pain=135,
        dealer_position="long_gamma",
    ))

    # Add buyback
    detector.add_buyback_program(BuybackProgram(
        ticker="NVDA",
        authorized_mm=25000,
        remaining_mm=15000,
        daily_limit_pct=25,
        avg_daily_volume_mm=50,
        blackout_start="2024-12-15",
        blackout_end="2024-12-22",
    ))

    # Add short interest
    detector.add_short_interest(ShortInterest(
        ticker="NVDA",
        date="2024-12-08",
        short_interest_shares_mm=45,
        short_interest_pct=1.8,
        days_to_cover=1.2,
        borrow_rate_pct=0.5,
        shares_available_mm=50,
        utilization_pct=25,
    ))

    # Print report
    print(detector.generate_report("NVDA"))


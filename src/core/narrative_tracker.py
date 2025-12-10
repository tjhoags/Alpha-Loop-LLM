"""MODULE 3: NARRATIVE EMERGENCE TRACKER
=====================================
Alpha Loop Capital - Consequence Engine

Purpose: Find investment narratives before they become consensus
         Track story lifecycle from emergence to saturation
         Front-run retail capital flows driven by narrative adoption

Core Edge: Identify stories in Substack/podcasts/niche communities
           Position before mainstream (CNBC/Bloomberg) coverage
           Exit when narrative reaches saturation (contrarian signal)

Author: Tom Hogan
Version: 1.0
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NarrativeStage(Enum):
    """Lifecycle stage of an investment narrative.
    Each stage has different trading implications.
    """

    EMERGENCE = "emergence"           # First mentions, <100 social mentions/week
    EARLY_ADOPTION = "early_adoption" # Growing discussion, 100-500 mentions/week
    ACCELERATION = "acceleration"     # Reddit DD posts, 500-2000 mentions/week
    MAINSTREAM = "mainstream"         # WSB discovers, 2000-10000 mentions/week
    SATURATION = "saturation"         # CNBC segments, >10000 mentions/week
    DECLINE = "decline"               # Fading interest, mentions dropping


class SourceTier(Enum):
    """Source credibility and lead time tiers.
    Higher tier = earlier signal but fewer sources.
    """

    TIER_1 = "tier_1"  # Substack, independent researchers (3-6 month lead)
    TIER_2 = "tier_2"  # Niche podcasts, specialized forums (2-4 month lead)
    TIER_3 = "tier_3"  # Reddit sector subs (not WSB) (1-3 month lead)
    TIER_4 = "tier_4"  # FinTwit early adopters (2-6 week lead)
    TIER_5 = "tier_5"  # WSB, mainstream Reddit (0-2 week lead - EXIT signal)
    TIER_6 = "tier_6"  # CNBC, Bloomberg (0 days - CONTRARIAN signal)


class TradingSignal(Enum):
    """Trading signal based on narrative stage"""

    ENTRY = "entry"               # Enter position (Stage 1-2)
    HOLD_ADD = "hold_add"         # Hold, add on pullbacks (Stage 3)
    REDUCE = "reduce"             # Take profits, tighten stops (Stage 4)
    EXIT = "exit"                 # Exit position (Stage 5)
    CONTRARIAN_SHORT = "short"    # Consider shorting (Stage 5-6)


@dataclass
class NarrativeMention:
    """Single mention of a narrative in a source"""

    source: str
    source_tier: SourceTier
    date: str
    title: str
    url: str
    author: str
    sentiment: str  # "bullish", "bearish", "neutral"
    engagement_score: int = 0
    ticker_mentions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "source_tier": self.source_tier.value,
            "date": self.date,
            "title": self.title,
            "url": self.url,
            "author": self.author,
            "sentiment": self.sentiment,
            "engagement_score": self.engagement_score,
            "ticker_mentions": self.ticker_mentions,
        }


@dataclass
class Narrative:
    """Complete narrative with tracking data.
    """

    name: str
    description: str
    keywords: List[str]
    created_date: str

    # Related tickers
    primary_tickers: List[str] = field(default_factory=list)
    secondary_tickers: List[str] = field(default_factory=list)

    # Mentions history
    mentions: List[NarrativeMention] = field(default_factory=list)

    # Metrics
    weekly_mention_count: int = 0
    total_mentions: int = 0
    avg_sentiment_score: float = 0.0

    # Stage tracking
    current_stage: NarrativeStage = NarrativeStage.EMERGENCE
    stage_history: List[Dict] = field(default_factory=list)

    def add_mention(self, mention: NarrativeMention) -> None:
        """Add a mention and update metrics"""
        self.mentions.append(mention)
        self.total_mentions += 1
        self._recalculate_metrics()

    def _recalculate_metrics(self) -> None:
        """Recalculate aggregate metrics"""
        if not self.mentions:
            return

        # Weekly count (last 7 days)
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        recent = [m for m in self.mentions if m.date >= cutoff]
        self.weekly_mention_count = len(recent)

        # Sentiment
        sentiment_scores = {"bullish": 1, "neutral": 0, "bearish": -1}
        scores = [sentiment_scores.get(m.sentiment, 0) for m in self.mentions[-50:]]
        self.avg_sentiment_score = sum(scores) / len(scores) if scores else 0

    def get_tier_distribution(self) -> Dict[int, float]:
        """Get percentage of mentions by source tier"""
        if not self.mentions:
            return {}

        tier_counts = Counter(m.source_tier.value for m in self.mentions[-100:])
        total = sum(tier_counts.values())

        return {
            i: (tier_counts.get(f"tier_{i}", 0) / total) * 100
            for i in range(1, 7)
        }

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "primary_tickers": self.primary_tickers,
            "current_stage": self.current_stage.value,
            "weekly_mentions": self.weekly_mention_count,
            "total_mentions": self.total_mentions,
            "avg_sentiment": round(self.avg_sentiment_score, 2),
            "tier_distribution": self.get_tier_distribution(),
        }


class NarrativeTracker:
    """NARRATIVE EMERGENCE TRACKER

    Monitors investment narratives across source tiers.

    Workflow:
    1. Define narratives to track (keywords, tickers)
    2. Collect mentions from various sources
    3. Classify stage based on mention velocity and source mix
    4. Generate trading signals

    Key insight: CNBC coverage = EXIT signal, not entry.
    """

    def __init__(self):
        self.narratives: Dict[str, Narrative] = {}

        # Stage thresholds (weekly mentions)
        self.stage_thresholds = {
            NarrativeStage.EMERGENCE: (0, 100),
            NarrativeStage.EARLY_ADOPTION: (100, 500),
            NarrativeStage.ACCELERATION: (500, 2000),
            NarrativeStage.MAINSTREAM: (2000, 10000),
            NarrativeStage.SATURATION: (10000, float("inf")),
        }

        # Source configurations
        self.source_configs = {
            # Tier 1: Early alpha
            "substack": SourceTier.TIER_1,
            "seeking_alpha_premium": SourceTier.TIER_1,
            "independent_research": SourceTier.TIER_1,

            # Tier 2: Niche communities
            "podcast_niche": SourceTier.TIER_2,
            "discord_private": SourceTier.TIER_2,
            "specialty_forum": SourceTier.TIER_2,

            # Tier 3: Reddit sectors
            "r_vitards": SourceTier.TIER_3,
            "r_uranium": SourceTier.TIER_3,
            "r_pennystocks": SourceTier.TIER_3,
            "r_spacs": SourceTier.TIER_3,

            # Tier 4: FinTwit
            "twitter_fintwit": SourceTier.TIER_4,
            "stocktwits": SourceTier.TIER_4,

            # Tier 5: Mainstream social (EXIT signal)
            "r_wallstreetbets": SourceTier.TIER_5,
            "r_stocks": SourceTier.TIER_5,
            "tiktok_finance": SourceTier.TIER_5,

            # Tier 6: MSM (CONTRARIAN signal)
            "cnbc": SourceTier.TIER_6,
            "bloomberg": SourceTier.TIER_6,
            "yahoo_finance": SourceTier.TIER_6,
            "motley_fool": SourceTier.TIER_6,
        }

    def create_narrative(
        self,
        name: str,
        description: str,
        keywords: List[str],
        primary_tickers: List[str],
        secondary_tickers: List[str] = None,
    ) -> Narrative:
        """Create a new narrative to track"""
        narrative = Narrative(
            name=name,
            description=description,
            keywords=keywords,
            created_date=datetime.now().strftime("%Y-%m-%d"),
            primary_tickers=primary_tickers,
            secondary_tickers=secondary_tickers or [],
        )
        self.narratives[name] = narrative
        logger.info(f"Created narrative: {name}")
        return narrative

    def add_mention(
        self,
        narrative_name: str,
        source: str,
        title: str,
        url: str,
        author: str,
        sentiment: str,
        engagement_score: int = 0,
        ticker_mentions: List[str] = None,
        date: str = None,
    ) -> None:
        """Add a mention to a narrative"""
        if narrative_name not in self.narratives:
            raise ValueError(f"Narrative {narrative_name} not found")

        source_tier = self.source_configs.get(source, SourceTier.TIER_4)

        mention = NarrativeMention(
            source=source,
            source_tier=source_tier,
            date=date or datetime.now().strftime("%Y-%m-%d"),
            title=title,
            url=url,
            author=author,
            sentiment=sentiment,
            engagement_score=engagement_score,
            ticker_mentions=ticker_mentions or [],
        )

        self.narratives[narrative_name].add_mention(mention)
        self._update_stage(narrative_name)

    def _update_stage(self, narrative_name: str) -> None:
        """Update narrative stage based on current metrics"""
        narrative = self.narratives[narrative_name]
        weekly = narrative.weekly_mention_count
        tier_dist = narrative.get_tier_distribution()

        # Determine stage by weekly mentions
        new_stage = NarrativeStage.EMERGENCE
        for stage, (low, high) in self.stage_thresholds.items():
            if low <= weekly < high:
                new_stage = stage
                break

        # Tier 5-6 presence accelerates stage
        tier_5_6_pct = tier_dist.get(5, 0) + tier_dist.get(6, 0)
        if tier_5_6_pct > 30:
            if new_stage == NarrativeStage.ACCELERATION:
                new_stage = NarrativeStage.MAINSTREAM
            elif new_stage == NarrativeStage.MAINSTREAM:
                new_stage = NarrativeStage.SATURATION

        # Track stage changes
        if new_stage != narrative.current_stage:
            narrative.stage_history.append({
                "from": narrative.current_stage.value,
                "to": new_stage.value,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "weekly_mentions": weekly,
            })
            narrative.current_stage = new_stage
            logger.info(f"Narrative '{narrative_name}' moved to {new_stage.value}")

    def get_trading_signal(self, narrative_name: str) -> Dict:
        """Generate trading signal for a narrative.

        Key principle: Early in = high alpha, late in = negative alpha
        """
        if narrative_name not in self.narratives:
            return {"signal": "NO_DATA", "tickers": []}

        narrative = self.narratives[narrative_name]
        stage = narrative.current_stage
        tier_dist = narrative.get_tier_distribution()
        sentiment = narrative.avg_sentiment_score

        # Stage-based signals
        stage_signals = {
            NarrativeStage.EMERGENCE: (TradingSignal.ENTRY, 90, "Early - maximum alpha opportunity"),
            NarrativeStage.EARLY_ADOPTION: (TradingSignal.ENTRY, 80, "Accumulation phase"),
            NarrativeStage.ACCELERATION: (TradingSignal.HOLD_ADD, 70, "Momentum building"),
            NarrativeStage.MAINSTREAM: (TradingSignal.REDUCE, 60, "Take profits, tighten stops"),
            NarrativeStage.SATURATION: (TradingSignal.EXIT, 85, "EXIT - narrative exhausted"),
            NarrativeStage.DECLINE: (TradingSignal.CONTRARIAN_SHORT, 50, "Consider short if overvalued"),
        }

        base_signal, confidence, rationale = stage_signals[stage]

        # Adjust for Tier 6 presence (CNBC = EXIT)
        tier_6_pct = tier_dist.get(6, 0)
        if tier_6_pct > 10:
            base_signal = TradingSignal.EXIT
            confidence = 90
            rationale = "CNBC COVERAGE DETECTED - EXIT SIGNAL"

        # Adjust for sentiment extremes
        if sentiment > 0.8 and stage in [NarrativeStage.MAINSTREAM, NarrativeStage.SATURATION]:
            rationale += " | WARNING: Extreme bullish sentiment"

        return {
            "signal": base_signal.value,
            "confidence": confidence,
            "stage": stage.value,
            "weekly_mentions": narrative.weekly_mention_count,
            "rationale": rationale,
            "primary_tickers": narrative.primary_tickers,
            "secondary_tickers": narrative.secondary_tickers,
            "tier_distribution": tier_dist,
            "sentiment": round(sentiment, 2),
            "actions": self._get_actions(base_signal, narrative),
        }

    def _get_actions(self, signal: TradingSignal, narrative: Narrative) -> List[str]:
        """Get specific actions based on signal"""
        actions = {
            TradingSignal.ENTRY: [
                f"Build positions in {', '.join(narrative.primary_tickers[:3])}",
                "Use limit orders - no urgency",
                "Size: 2-3% per name",
                "Set alerts for Tier 5-6 coverage",
            ],
            TradingSignal.HOLD_ADD: [
                "Hold existing positions",
                "Add on 5%+ pullbacks",
                "Set trailing stops at -15%",
                "Monitor weekly mention velocity",
            ],
            TradingSignal.REDUCE: [
                "Trim winners by 25-50%",
                "Tighten stops to -8%",
                "No new positions",
                "Watch for Tier 6 coverage",
            ],
            TradingSignal.EXIT: [
                "SELL positions within 1-2 weeks",
                "Don't wait for perfect exit",
                "Narrative is exhausted",
                "Next buyer is retail - not you",
            ],
            TradingSignal.CONTRARIAN_SHORT: [
                "Evaluate for short if 50%+ from fair value",
                "Wait for momentum break",
                "Small size - narratives can persist",
            ],
        }
        return actions.get(signal, [])

    def get_all_signals(self) -> List[Dict]:
        """Get trading signals for all narratives"""
        signals = []
        for name in self.narratives:
            signal = self.get_trading_signal(name)
            signal["narrative_name"] = name
            signals.append(signal)

        # Sort by stage (earlier = better opportunity)
        stage_order = {s.value: i for i, s in enumerate(NarrativeStage)}
        signals.sort(key=lambda x: stage_order.get(x["stage"], 99))

        return signals

    def generate_report(self) -> str:
        """Generate human-readable narrative report"""
        lines = [
            "=" * 60,
            "NARRATIVE TRACKER - STATUS REPORT",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
        ]

        for signal in self.get_all_signals():
            emoji = {
                "entry": "🟢",
                "hold_add": "🟡",
                "reduce": "🟠",
                "exit": "🔴",
                "short": "⚫",
            }.get(signal["signal"], "⚪")

            lines.extend([
                f"{emoji} {signal['narrative_name'].upper()}",
                f"   Stage: {signal['stage']} | Signal: {signal['signal'].upper()}",
                f"   Weekly Mentions: {signal['weekly_mentions']} | Sentiment: {signal['sentiment']}",
                f"   Tickers: {', '.join(signal['primary_tickers'][:5])}",
                f"   {signal['rationale']}",
                "",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    tracker = NarrativeTracker()

    # Create uranium narrative
    uranium = tracker.create_narrative(
        name="Uranium Renaissance",
        description="Nuclear energy revival driven by AI/data centers and clean energy",
        keywords=["uranium", "nuclear", "SMR", "data center", "AI power"],
        primary_tickers=["CCJ", "UEC", "DNN", "UUUU"],
        secondary_tickers=["SII", "NXE", "BWXT"],
    )

    # Simulate mentions
    tracker.add_mention(
        "Uranium Renaissance",
        source="substack",
        title="Why Uranium is the Trade of the Decade",
        url="https://example.com/1",
        author="ResearchGuy",
        sentiment="bullish",
        engagement_score=500,
    )

    tracker.add_mention(
        "Uranium Renaissance",
        source="r_uranium",
        title="CCJ DD: Nuclear is Back",
        url="https://reddit.com/1",
        author="UraniumBull",
        sentiment="bullish",
        engagement_score=250,
    )

    # Print report
    print(tracker.generate_report())

    # Get signals
    signals = tracker.get_all_signals()
    for s in signals:
        print(f"\n{s['narrative_name']}: {s['signal']} (Confidence: {s['confidence']}%)")


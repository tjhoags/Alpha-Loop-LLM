"""================================================================================
NLP/SENTIMENT SIGNALS - Language Analysis for Edge
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

What people SAY and HOW they say it reveals more than numbers.
================================================================================
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NLPSignal:
    """Signal from NLP/sentiment analysis."""

    signal_id: str
    signal_type: str
    ticker: str
    direction: str
    confidence: float
    description: str
    text_evidence: str
    linguistic_markers: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class NLPSentimentSignals:
    """NLP/SENTIMENT SIGNALS

    1. Earnings Call Tone Shift - Compare management tone QoQ
    2. Reddit/WSB Sentiment Extremes - Retail positioning
    3. CEO Confidence Language - Word choice reveals mindset
    4. Analyst Report Hedging - When analysts hedge, pay attention
    5. Press Release Verb Tense Shift - Future vs past tense
    6. Customer Review Sentiment Velocity - Real-time product signal
    7. Twitter Executive Silence - When they stop tweeting
    8. 8-K Filing Urgency Language - How they describe events
    9. Competitor Mention Frequency - Who's on their mind?
    10. Pronoun Usage Analysis - "I" vs "We" vs deflection
    """

    # Language patterns
    CONFIDENCE_WORDS = ["confident", "certain", "definitely", "absolutely", "strong", "robust", "excellent", "outstanding"]
    HEDGE_WORDS = ["may", "might", "could", "potentially", "possibly", "uncertain", "believe", "expect", "anticipate"]
    URGENCY_WORDS = ["immediately", "urgent", "critical", "emergency", "essential", "paramount"]
    DEFLECTION_PATTERNS = ["the market", "macro conditions", "industry-wide", "external factors", "headwinds"]

    def __init__(self):
        self.signals_detected: List[NLPSignal] = []

    def earnings_call_tone_shift(
        self,
        ticker: str,
        current_transcript: str,
        previous_transcript: str,
        speaker: str = "CEO",
    ) -> Optional[NLPSignal]:
        """Compare management tone quarter-over-quarter.

        Sudden shift from confident to hedging = warning
        Shift from hedging to confident = positive turn
        """
        current_confidence = self._score_confidence(current_transcript)
        previous_confidence = self._score_confidence(previous_transcript)

        tone_shift = current_confidence - previous_confidence

        if abs(tone_shift) < 0.15:
            return None

        direction = "bullish" if tone_shift > 0 else "bearish"
        shift_type = "MORE CONFIDENT" if tone_shift > 0 else "MORE HEDGING"

        return NLPSignal(
            signal_id=f"tone_{hashlib.sha256(f'{ticker}tone'.encode()).hexdigest()[:8]}",
            signal_type="earnings_call_tone_shift",
            ticker=ticker,
            direction=direction,
            confidence=0.65 + min(abs(tone_shift), 0.2),
            description=f"TONE SHIFT: {ticker} {speaker} {shift_type} ({tone_shift:+.0%} change)",
            text_evidence=f"Confidence score: {previous_confidence:.0%} → {current_confidence:.0%}",
            linguistic_markers=self._extract_tone_markers(current_transcript),
        )

    def reddit_wsb_sentiment_extreme(
        self,
        ticker: str,
        sentiment_score: float,  # -1 to 1
        mention_velocity: float,  # mentions per hour
        historical_avg_mentions: float,
    ) -> Optional[NLPSignal]:
        """Reddit/WSB sentiment extremes = retail positioning.

        EXTREME bullish sentiment often marks local tops.
        EXTREME bearish often marks bottoms (capitulation).
        """
        mention_ratio = mention_velocity / historical_avg_mentions if historical_avg_mentions > 0 else 1

        if abs(sentiment_score) < 0.6 or mention_ratio < 3:
            return None

        # Contrarian signal - extreme sentiment usually wrong
        if sentiment_score > 0.7 and mention_ratio > 5:
            direction = "bearish"  # Contrarian - too bullish
            desc = f"WSB EUPHORIA: {ticker} extreme bullish sentiment ({sentiment_score:.0%}) - contrarian bearish"
        elif sentiment_score < -0.7 and mention_ratio > 5:
            direction = "bullish"  # Contrarian - too bearish
            desc = f"WSB CAPITULATION: {ticker} extreme bearish sentiment ({sentiment_score:.0%}) - contrarian bullish"
        else:
            return None

        return NLPSignal(
            signal_id=f"wsb_{hashlib.sha256(f'{ticker}wsb'.encode()).hexdigest()[:8]}",
            signal_type="reddit_wsb_extreme",
            ticker=ticker,
            direction=direction,
            confidence=0.62,
            description=desc,
            text_evidence=f"Sentiment: {sentiment_score:.0%}, Mentions: {mention_ratio:.1f}x normal",
            linguistic_markers=["extreme_sentiment", "high_velocity"],
        )

    def ceo_confidence_language(
        self,
        ticker: str,
        ceo_statements: List[str],
        ceo_name: str,
    ) -> Optional[NLPSignal]:
        """CEO word choice reveals their true mindset.

        "I am confident" vs "We believe" vs "Management expects"
        Personal ownership of statements = higher conviction
        """
        combined_text = " ".join(ceo_statements).lower()

        # Count ownership patterns
        i_statements = len(re.findall(r"\bi\s+(am|will|have|believe|know|think)\b", combined_text))
        we_statements = len(re.findall(r"\bwe\s+(are|will|have|believe|expect)\b", combined_text))
        passive_statements = len(re.findall(r"\b(management|the company|it is)\s+(believes?|expects?|anticipates?)\b", combined_text))

        total = i_statements + we_statements + passive_statements
        if total < 5:
            return None

        ownership_ratio = (i_statements + we_statements * 0.7) / total

        if ownership_ratio > 0.7:
            direction = "bullish"
            confidence = 0.65
            desc = f"CEO OWNERSHIP: {ceo_name} using high-ownership language at {ticker}"
        elif ownership_ratio < 0.3:
            direction = "bearish"
            confidence = 0.60
            desc = f"CEO DEFLECTION: {ceo_name} using passive/deflecting language at {ticker}"
        else:
            return None

        return NLPSignal(
            signal_id=f"ceo_{hashlib.sha256(f'{ticker}ceo'.encode()).hexdigest()[:8]}",
            signal_type="ceo_confidence_language",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            text_evidence=f"I:{i_statements} We:{we_statements} Passive:{passive_statements}",
            linguistic_markers=[f"ownership_ratio:{ownership_ratio:.0%}"],
        )

    def analyst_report_hedging(
        self,
        ticker: str,
        analyst_firm: str,
        report_text: str,
        rating: str,
        previous_hedge_level: float = None,
    ) -> Optional[NLPSignal]:
        """When analysts hedge more than usual, pay attention.

        BUY rating with heavy hedging = not really bullish
        SELL rating with hedging = might not be that bad
        """
        hedge_score = self._count_hedging(report_text)
        confidence_score = self._score_confidence(report_text)

        contradiction = False

        if rating.upper() in ["BUY", "STRONG BUY", "OUTPERFORM"] and hedge_score > 0.3:
            contradiction = True
            direction = "bearish"
            desc = f"ANALYST CONTRADICTION: {analyst_firm} says BUY {ticker} but hedges heavily ({hedge_score:.0%})"
        elif rating.upper() in ["SELL", "UNDERPERFORM"] and hedge_score > 0.35:
            contradiction = True
            direction = "bullish"
            desc = f"ANALYST SOFT SELL: {analyst_firm} says SELL {ticker} but hedges ({hedge_score:.0%}) - not fully convicted"
        else:
            return None

        return NLPSignal(
            signal_id=f"anly_{hashlib.sha256(f'{ticker}{analyst_firm}'.encode()).hexdigest()[:8]}",
            signal_type="analyst_hedging_contradiction",
            ticker=ticker,
            direction=direction,
            confidence=0.58,
            description=desc,
            text_evidence=f"Rating: {rating}, Hedge score: {hedge_score:.0%}, Confidence: {confidence_score:.0%}",
            linguistic_markers=["rating_hedge_contradiction"],
        )

    def press_release_verb_tense_shift(
        self,
        ticker: str,
        current_release: str,
        previous_release: str,
    ) -> Optional[NLPSignal]:
        """Shift from future tense to past tense = defensive.
        Shift from past to future = optimistic.
        """
        current_future = self._count_future_tense(current_release)
        current_past = self._count_past_tense(current_release)
        previous_future = self._count_future_tense(previous_release)
        previous_past = self._count_past_tense(previous_release)

        current_ratio = current_future / (current_past + 1)
        previous_ratio = previous_future / (previous_past + 1)

        ratio_change = current_ratio - previous_ratio

        if abs(ratio_change) < 0.3:
            return None

        direction = "bullish" if ratio_change > 0 else "bearish"

        return NLPSignal(
            signal_id=f"tense_{hashlib.sha256(f'{ticker}tense'.encode()).hexdigest()[:8]}",
            signal_type="verb_tense_shift",
            ticker=ticker,
            direction=direction,
            confidence=0.55,
            description=f"TENSE SHIFT: {ticker} PR now {'more future-focused' if ratio_change > 0 else 'more past-focused'}",
            text_evidence=f"Future/Past ratio: {previous_ratio:.2f} → {current_ratio:.2f}",
            linguistic_markers=["verb_tense_analysis"],
        )

    def twitter_executive_silence(
        self,
        ticker: str,
        exec_name: str,
        days_since_last_tweet: int,
        avg_tweets_per_week: float,
        is_typically_active: bool,
    ) -> Optional[NLPSignal]:
        """When normally active executives go silent.

        Silence before earnings = they know something
        Silence after controversy = damage control mode
        """
        if not is_typically_active or avg_tweets_per_week < 2:
            return None

        expected_tweets = (days_since_last_tweet / 7) * avg_tweets_per_week

        if expected_tweets < 5:
            return None

        return NLPSignal(
            signal_id=f"twt_{hashlib.sha256(f'{ticker}{exec_name}'.encode()).hexdigest()[:8]}",
            signal_type="executive_twitter_silence",
            ticker=ticker,
            direction="uncertain",
            confidence=0.52,
            description=f"EXEC SILENCE: {exec_name} hasn't tweeted in {days_since_last_tweet} days (normally {avg_tweets_per_week:.1f}/week)",
            text_evidence=f"Expected ~{expected_tweets:.0f} tweets, got 0",
            linguistic_markers=["social_media_silence"],
        )

    def filing_8k_urgency_language(
        self,
        ticker: str,
        filing_text: str,
        filing_item: str,
    ) -> Optional[NLPSignal]:
        """8-K filing language urgency reveals importance.

        Urgent language + weekend filing = bad news coming
        Casual language + Friday PM = burying routine stuff
        """
        urgency_score = sum(1 for word in self.URGENCY_WORDS if word in filing_text.lower())
        deflection_score = sum(1 for pattern in self.DEFLECTION_PATTERNS if pattern in filing_text.lower())

        word_count = len(filing_text.split())
        urgency_density = urgency_score / (word_count / 100) if word_count > 0 else 0

        if urgency_density > 0.5:
            return NLPSignal(
                signal_id=f"8k_{hashlib.sha256(f'{ticker}8k'.encode()).hexdigest()[:8]}",
                signal_type="8k_urgency_language",
                ticker=ticker,
                direction="bearish",
                confidence=0.60,
                description=f"URGENT 8-K: {ticker} filing has high urgency language density ({urgency_density:.1f})",
                text_evidence=f"Item: {filing_item}, Urgency words: {urgency_score}",
                linguistic_markers=["high_urgency_filing"],
            )
        elif deflection_score > 3:
            return NLPSignal(
                signal_id=f"8k_{hashlib.sha256(f'{ticker}deflect'.encode()).hexdigest()[:8]}",
                signal_type="8k_deflection_language",
                ticker=ticker,
                direction="bearish",
                confidence=0.55,
                description=f"DEFLECTING 8-K: {ticker} filing blames external factors heavily",
                text_evidence=f"Deflection patterns: {deflection_score}",
                linguistic_markers=["blame_external"],
            )

        return None

    def pronoun_usage_analysis(
        self,
        ticker: str,
        speaker: str,
        text: str,
        context: str,
    ) -> Optional[NLPSignal]:
        """Pronoun usage reveals psychological state.

        "I" during success, "we" during struggle = self-serving
        "We" consistently = team player
        "They" or "the team" = distancing
        """
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)

        if total_words < 100:
            return None

        i_count = text_lower.count(" i ")
        we_count = text_lower.count(" we ")
        they_count = text_lower.count(" they ") + text_lower.count(" the team ")

        i_ratio = i_count / total_words * 100
        we_ratio = we_count / total_words * 100

        if i_ratio > 2.0 and context == "bad_news":
            return NLPSignal(
                signal_id=f"pro_{hashlib.sha256(f'{ticker}pronoun'.encode()).hexdigest()[:8]}",
                signal_type="pronoun_analysis",
                ticker=ticker,
                direction="bearish",
                confidence=0.55,
                description=f"PRONOUN WARNING: {speaker} using excessive 'I' ({i_ratio:.1f}%) during {context}",
                text_evidence=f"I:{i_count} We:{we_count} They:{they_count}",
                linguistic_markers=["self_serving_language"],
            )

        return None

    # Helper methods
    def _score_confidence(self, text: str) -> float:
        """Score text confidence level 0-1."""
        text_lower = text.lower()
        words = text_lower.split()
        if not words:
            return 0.5

        confidence_hits = sum(1 for word in self.CONFIDENCE_WORDS if word in text_lower)
        hedge_hits = sum(1 for word in self.HEDGE_WORDS if word in text_lower)

        score = 0.5 + (confidence_hits - hedge_hits) / (len(words) / 50)
        return max(0, min(1, score))

    def _count_hedging(self, text: str) -> float:
        """Count hedging language density."""
        text_lower = text.lower()
        words = text_lower.split()
        if not words:
            return 0

        hedge_hits = sum(1 for word in self.HEDGE_WORDS if word in text_lower)
        return hedge_hits / (len(words) / 100)

    def _extract_tone_markers(self, text: str) -> List[str]:
        """Extract notable tone markers."""
        markers = []
        text_lower = text.lower()

        for word in self.CONFIDENCE_WORDS:
            if word in text_lower:
                markers.append(f"confident:{word}")
        for word in self.HEDGE_WORDS[:3]:
            if word in text_lower:
                markers.append(f"hedge:{word}")

        return markers[:5]

    def _count_future_tense(self, text: str) -> int:
        """Count future tense verbs."""
        return len(re.findall(r"\b(will|shall|going to|expect to|plan to)\b", text.lower()))

    def _count_past_tense(self, text: str) -> int:
        """Count past tense verbs."""
        return len(re.findall(r"\b(was|were|had|did|achieved|completed|delivered)\b", text.lower()))

    def get_stats(self) -> Dict[str, Any]:
        return {"total_signals": len(self.signals_detected)}



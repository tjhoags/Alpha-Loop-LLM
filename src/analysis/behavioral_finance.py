"""
================================================================================
BEHAVIORAL FINANCE & MARKET PSYCHOLOGY MODULE
================================================================================
Quantified behavioral factors for algorithmic trading:

SOCIAL SENTIMENT:
- Twitter/X sentiment analysis
- Reddit (WSB, stocks) sentiment
- StockTwits sentiment
- News sentiment aggregation
- Influencer tracking

MARKET PSYCHOLOGY:
- Fear & Greed Index components
- Put/Call ratio psychology
- VIX term structure
- Retail vs Institutional flow
- FOMO/Panic indicators

GAME THEORY:
- Nash equilibrium detection
- Coordination games (momentum)
- Prisoner's dilemma (holding vs selling)
- Information asymmetry signals
- Market maker positioning

CROWD BEHAVIOR / SOCIOLOGY:
- Herding indicators
- Social proof signals
- Bandwagon effects
- Contrarian indicators
- Information cascade detection

COGNITIVE BIASES (Exploitable):
- Anchoring bias signals
- Recency bias detection
- Loss aversion indicators
- Confirmation bias patterns
- Overconfidence signals

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SentimentSource(Enum):
    """Social media and news sources."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    NEWS = "news"
    ANALYST = "analyst"
    INSIDER = "insider"


class MarketRegime(Enum):
    """Market psychological regime."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


class CrowdBehavior(Enum):
    """Crowd behavior classification."""
    HERDING_BULLISH = "herding_bullish"
    HERDING_BEARISH = "herding_bearish"
    DIVERGENT = "divergent"
    NEUTRAL = "neutral"


@dataclass
class SentimentData:
    """Aggregated sentiment data point."""
    timestamp: datetime
    symbol: str
    source: SentimentSource
    raw_score: float  # -1 to 1
    volume: int  # Number of mentions
    bullish_pct: float
    bearish_pct: float
    neutral_pct: float
    
    @property
    def bull_bear_ratio(self) -> float:
        """Bullish / Bearish ratio (>1 = bullish)."""
        return self.bullish_pct / self.bearish_pct if self.bearish_pct > 0 else float('inf')
    
    @property
    def sentiment_strength(self) -> float:
        """How strong is the consensus (0 to 1)."""
        return max(self.bullish_pct, self.bearish_pct) - self.neutral_pct


# =============================================================================
# SOCIAL SENTIMENT ANALYSIS
# =============================================================================

class SocialSentimentAnalyzer:
    """
    Aggregates and analyzes social media sentiment.
    
    KEY INSIGHT: Social sentiment is a LEADING indicator for retail-driven
    moves but a LAGGING indicator for institutional moves. Use accordingly.
    """
    
    @staticmethod
    def aggregate_sentiment(
        sentiment_data: List[SentimentData],
        weights: Optional[Dict[SentimentSource, float]] = None
    ) -> Dict[str, float]:
        """
        Aggregate sentiment across sources with volume weighting.
        
        INTERPRETATION:
        - score > 0.3: Strong bullish sentiment
        - score < -0.3: Strong bearish sentiment
        - |score| < 0.1: Neutral/mixed
        """
        if not sentiment_data:
            return {"aggregate_score": 0, "total_volume": 0, "consensus": 0}
        
        # Default weights (Twitter and Reddit most important for retail)
        if weights is None:
            weights = {
                SentimentSource.TWITTER: 0.30,
                SentimentSource.REDDIT: 0.30,
                SentimentSource.STOCKTWITS: 0.20,
                SentimentSource.NEWS: 0.15,
                SentimentSource.ANALYST: 0.05,
            }
        
        weighted_score = 0
        total_weight = 0
        total_volume = 0
        
        for data in sentiment_data:
            source_weight = weights.get(data.source, 0.1)
            # Volume-adjusted weighting
            volume_factor = np.log1p(data.volume) / 10  # Normalize
            adj_weight = source_weight * volume_factor
            
            weighted_score += data.raw_score * adj_weight
            total_weight += adj_weight
            total_volume += data.volume
        
        if total_weight == 0:
            return {"aggregate_score": 0, "total_volume": 0, "consensus": 0}
        
        aggregate_score = weighted_score / total_weight
        
        # Calculate consensus (how aligned are sources)
        scores = [d.raw_score for d in sentiment_data]
        consensus = 1 - np.std(scores) if len(scores) > 1 else 1
        
        return {
            "aggregate_score": aggregate_score,
            "total_volume": total_volume,
            "consensus": consensus,
            "bull_bear_ratio": np.mean([d.bull_bear_ratio for d in sentiment_data if d.bull_bear_ratio < 100])
        }
    
    @staticmethod
    def sentiment_momentum(
        sentiment_history: pd.DataFrame,
        window: int = 5
    ) -> pd.Series:
        """
        Calculate sentiment momentum (rate of change).
        
        TRADING SIGNAL:
        - Rising sentiment momentum + Rising price = Trend continuation
        - Rising sentiment + Falling price = Potential reversal (bullish)
        - Falling sentiment + Rising price = Potential reversal (bearish)
        """
        return sentiment_history["score"].diff(window) / window
    
    @staticmethod
    def sentiment_divergence(
        sentiment_score: float,
        price_return: float,
        threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Detect sentiment-price divergence.
        
        TRADING SIGNAL:
        - Bullish divergence: Sentiment improving but price falling
        - Bearish divergence: Sentiment deteriorating but price rising
        """
        divergence_score = sentiment_score - price_return
        
        if sentiment_score > threshold and price_return < -threshold:
            signal = "bullish_divergence"
            description = "Sentiment improving while price falls - potential bottom"
        elif sentiment_score < -threshold and price_return > threshold:
            signal = "bearish_divergence"
            description = "Sentiment deteriorating while price rises - potential top"
        else:
            signal = "no_divergence"
            description = "Sentiment and price aligned"
        
        return {
            "signal": signal,
            "divergence_score": divergence_score,
            "description": description,
            "actionable": abs(divergence_score) > threshold * 2
        }
    
    @staticmethod
    def retail_flow_indicator(
        stocktwits_sentiment: float,
        reddit_mentions: int,
        robinhood_rank: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Retail investor flow indicator.
        
        USAGE: High retail interest + negative institutional flow = 
               potential for squeeze OR dump (context dependent)
        """
        # Normalize Reddit mentions (log scale)
        reddit_score = min(1.0, np.log1p(reddit_mentions) / 10)
        
        # Robinhood popularity (lower rank = more popular)
        rh_score = 1 - (robinhood_rank / 100) if robinhood_rank else 0.5
        
        # Combined retail interest
        retail_score = (
            0.40 * abs(stocktwits_sentiment) +
            0.35 * reddit_score +
            0.25 * rh_score
        )
        
        return {
            "retail_interest_score": retail_score,
            "is_meme_candidate": retail_score > 0.7,
            "squeeze_potential": retail_score > 0.8 and stocktwits_sentiment > 0.5
        }


# =============================================================================
# MARKET PSYCHOLOGY
# =============================================================================

class MarketPsychology:
    """
    Market-wide psychological indicators.
    
    These measure overall market fear/greed which affects ALL stocks.
    """
    
    @staticmethod
    def fear_greed_index(
        vix: float,
        vix_ma_50: float,
        put_call_ratio: float,
        put_call_ma_50: float,
        market_momentum: float,  # S&P 500 return vs 125-day MA
        safe_haven_demand: float,  # Bond vs Stock returns
        junk_bond_demand: float,  # High yield spread
        market_breadth: float  # Advance-decline ratio
    ) -> Dict[str, Any]:
        """
        Calculate Fear & Greed Index (0-100).
        
        INTERPRETATION:
        0-25: Extreme Fear - BUY signal (contrarian)
        25-45: Fear
        45-55: Neutral
        55-75: Greed
        75-100: Extreme Greed - SELL signal (contrarian)
        
        WARREN BUFFETT: "Be fearful when others are greedy, 
                        greedy when others are fearful"
        """
        scores = []
        
        # 1. VIX vs MA (fear when high)
        vix_score = 100 * (1 - min(vix / vix_ma_50, 2) / 2)
        scores.append(("vix", vix_score))
        
        # 2. Put/Call Ratio (fear when high)
        pc_score = 100 * (1 - min(put_call_ratio / put_call_ma_50, 2) / 2)
        scores.append(("put_call", pc_score))
        
        # 3. Market Momentum
        mom_score = 50 + (market_momentum * 100)  # Normalize around 50
        mom_score = max(0, min(100, mom_score))
        scores.append(("momentum", mom_score))
        
        # 4. Safe Haven Demand (greed when low)
        safe_score = 100 * (1 - max(-0.5, min(0.5, safe_haven_demand)) / 0.5) / 2
        scores.append(("safe_haven", safe_score))
        
        # 5. Junk Bond Demand (greed when tight spreads)
        junk_score = 100 * (1 - min(junk_bond_demand / 5, 1))  # 5% spread = max fear
        scores.append(("junk_demand", junk_score))
        
        # 6. Market Breadth
        breadth_score = 50 + (market_breadth * 50)
        breadth_score = max(0, min(100, breadth_score))
        scores.append(("breadth", breadth_score))
        
        # Equal weight average
        index_value = np.mean([s[1] for s in scores])
        
        # Determine regime
        if index_value <= 25:
            regime = MarketRegime.EXTREME_FEAR
            signal = "STRONG BUY (contrarian)"
        elif index_value <= 45:
            regime = MarketRegime.FEAR
            signal = "BUY (contrarian)"
        elif index_value <= 55:
            regime = MarketRegime.NEUTRAL
            signal = "HOLD"
        elif index_value <= 75:
            regime = MarketRegime.GREED
            signal = "SELL (contrarian)"
        else:
            regime = MarketRegime.EXTREME_GREED
            signal = "STRONG SELL (contrarian)"
        
        return {
            "index_value": index_value,
            "regime": regime.value,
            "signal": signal,
            "components": dict(scores),
            "contrarian_opportunity": index_value < 25 or index_value > 75
        }
    
    @staticmethod
    def vix_term_structure_signal(
        vix_spot: float,
        vix_1m: float,
        vix_3m: float
    ) -> Dict[str, Any]:
        """
        VIX term structure analysis.
        
        CONTANGO (normal): Spot < Front < Back
        - Market complacent, potential for vol spike
        
        BACKWARDATION (inverted): Spot > Front > Back
        - Market fearful, potential for vol crush
        
        TRADING SIGNAL:
        - Deep backwardation = Buy equities (fear overdone)
        - Steep contango = Consider hedges (complacency)
        """
        contango_1m = (vix_1m / vix_spot - 1) * 100  # Percent
        contango_3m = (vix_3m / vix_spot - 1) * 100
        
        if contango_1m < -5:
            structure = "backwardation"
            signal = "Fear elevated - potential buying opportunity"
            score = -1 * contango_1m / 20  # Positive = bullish
        elif contango_1m > 10:
            structure = "steep_contango"
            signal = "Complacency high - consider hedging"
            score = -1 * contango_1m / 30  # Negative = bearish
        else:
            structure = "normal_contango"
            signal = "Normal conditions"
            score = 0
        
        return {
            "structure": structure,
            "contango_1m_pct": contango_1m,
            "contango_3m_pct": contango_3m,
            "signal": signal,
            "directional_score": max(-1, min(1, score))
        }


# =============================================================================
# GAME THEORY
# =============================================================================

class GameTheorySignals:
    """
    Game theory concepts applied to market dynamics.
    
    KEY INSIGHT: Markets are coordination games where participants try to
    predict what others will do. Understanding these dynamics creates alpha.
    """
    
    @staticmethod
    def keynesian_beauty_contest(
        your_estimate: float,
        consensus_estimate: float,
        price: float
    ) -> Dict[str, Any]:
        """
        Keynesian Beauty Contest - You win by predicting what OTHERS think.
        
        In markets, the "correct" value matters less than what others believe.
        
        STRATEGY:
        - If consensus differs from your estimate, trade toward consensus
        - Your private valuation only matters at extremes
        """
        consensus_vs_price = (consensus_estimate - price) / price
        your_vs_price = (your_estimate - price) / price
        your_vs_consensus = (your_estimate - consensus_estimate) / consensus_estimate
        
        # If consensus thinks it's undervalued, momentum will push it up
        # regardless of your view (in short term)
        
        if abs(consensus_vs_price) > 0.10:  # Consensus sees 10%+ mispricing
            if abs(your_vs_consensus) < 0.05:  # You agree with consensus
                signal = "follow_consensus"
                confidence = "high"
            else:  # You disagree
                signal = "wait_for_convergence"
                confidence = "medium"
        else:
            signal = "no_clear_play"
            confidence = "low"
        
        return {
            "signal": signal,
            "confidence": confidence,
            "consensus_vs_price_pct": consensus_vs_price * 100,
            "your_vs_consensus_pct": your_vs_consensus * 100,
            "recommendation": "Trade in direction of consensus when strong (>10% divergence)"
        }
    
    @staticmethod
    def information_asymmetry_score(
        insider_buying: float,
        institutional_flow: float,
        retail_sentiment: float,
        unusual_options_activity: float
    ) -> Dict[str, Any]:
        """
        Detect potential information asymmetry.
        
        When "smart money" (insiders, institutions) diverges from 
        "dumb money" (retail), information asymmetry likely exists.
        
        TRADING SIGNAL:
        - Insiders buying + Institutions buying + Retail bearish = BUY
        - Insiders selling + Institutions selling + Retail bullish = SELL
        """
        # Smart money composite
        smart_money = (insider_buying * 0.5 + institutional_flow * 0.5)
        
        # Divergence from retail
        divergence = smart_money - retail_sentiment
        
        # Options activity confirms
        options_confirmation = unusual_options_activity > 0.5
        
        if divergence > 0.3 and smart_money > 0.5:
            signal = "smart_money_accumulating"
            action = "BUY - Follow smart money"
            confidence = "high" if options_confirmation else "medium"
        elif divergence < -0.3 and smart_money < -0.5:
            signal = "smart_money_distributing"
            action = "SELL - Smart money exiting"
            confidence = "high" if options_confirmation else "medium"
        else:
            signal = "no_clear_asymmetry"
            action = "No clear signal"
            confidence = "low"
        
        return {
            "signal": signal,
            "action": action,
            "confidence": confidence,
            "smart_money_score": smart_money,
            "retail_score": retail_sentiment,
            "divergence": divergence,
            "options_confirm": options_confirmation
        }
    
    @staticmethod
    def prisoners_dilemma_sell_pressure(
        days_since_peak: int,
        drawdown_pct: float,
        volume_trend: float,
        holder_concentration: float
    ) -> Dict[str, Any]:
        """
        Prisoner's Dilemma in selling.
        
        When price drops, each holder faces a dilemma:
        - If I hold and others hold: Price may recover
        - If I hold and others sell: I lose more
        - If I sell and others hold: I miss recovery
        - If I sell and others sell: At least I minimized loss
        
        This creates cascading sell pressure.
        
        DETECTION: Identify when dilemma is resolving (capitulation)
        """
        # Capitulation indicators
        is_extended_decline = days_since_peak > 20
        is_severe_drawdown = drawdown_pct > 0.20
        is_volume_spike = volume_trend > 2.0  # 2x normal
        is_concentrated = holder_concentration > 0.5  # Few large holders
        
        capitulation_score = (
            (1 if is_extended_decline else 0) * 0.25 +
            (1 if is_severe_drawdown else 0) * 0.30 +
            (1 if is_volume_spike else 0) * 0.30 +
            (1 if is_concentrated else 0) * 0.15
        )
        
        if capitulation_score > 0.7:
            phase = "capitulation"
            signal = "BUY - Dilemma resolving, forced sellers done"
        elif capitulation_score > 0.4:
            phase = "distribution"
            signal = "WAIT - Dilemma ongoing, more selling likely"
        else:
            phase = "normal"
            signal = "No pressure signal"
        
        return {
            "phase": phase,
            "signal": signal,
            "capitulation_score": capitulation_score,
            "indicators": {
                "extended_decline": is_extended_decline,
                "severe_drawdown": is_severe_drawdown,
                "volume_spike": is_volume_spike,
                "concentrated_holders": is_concentrated
            }
        }
    
    @staticmethod
    def short_squeeze_game(
        short_interest_pct: float,
        days_to_cover: float,
        borrow_rate: float,
        social_sentiment: float,
        recent_return_5d: float
    ) -> Dict[str, Any]:
        """
        Short Squeeze Game Theory.
        
        Shorts face a coordination problem:
        - If shorts hold and stock rises: Unlimited loss potential
        - If shorts cover early: May miss profit if stock falls
        - If all shorts try to cover: Price spikes (squeeze)
        
        DETECTION: High short interest + positive momentum = 
                   game theory cascade imminent
        """
        # Base squeeze potential
        squeeze_score = 0
        
        # High short interest
        if short_interest_pct > 20:
            squeeze_score += 0.30
        elif short_interest_pct > 10:
            squeeze_score += 0.15
        
        # Days to cover (higher = harder to exit)
        if days_to_cover > 5:
            squeeze_score += 0.25
        elif days_to_cover > 3:
            squeeze_score += 0.15
        
        # Borrow rate (expensive = pressure to close)
        if borrow_rate > 50:  # 50% annualized
            squeeze_score += 0.20
        elif borrow_rate > 20:
            squeeze_score += 0.10
        
        # Social coordination (retail piling in)
        if social_sentiment > 0.7:
            squeeze_score += 0.15
        
        # Already moving up (cascade starting)
        if recent_return_5d > 0.10:  # Up 10%+
            squeeze_score += 0.10
        
        if squeeze_score > 0.7:
            phase = "squeeze_imminent"
            signal = "HIGH RISK/REWARD - Squeeze likely"
        elif squeeze_score > 0.4:
            phase = "squeeze_possible"
            signal = "MONITOR - Conditions developing"
        else:
            phase = "normal"
            signal = "No squeeze setup"
        
        return {
            "phase": phase,
            "signal": signal,
            "squeeze_score": squeeze_score,
            "metrics": {
                "short_interest": short_interest_pct,
                "days_to_cover": days_to_cover,
                "borrow_rate": borrow_rate,
                "social_sentiment": social_sentiment
            }
        }


# =============================================================================
# CROWD BEHAVIOR / SOCIOLOGY
# =============================================================================

class CrowdBehaviorAnalysis:
    """
    Sociological analysis of market crowd behavior.
    
    Markets are social systems where information cascades, herding,
    and groupthink create exploitable patterns.
    """
    
    @staticmethod
    def herding_indicator(
        stock_correlation_to_market: float,
        sector_correlation: float,
        return_dispersion: float,
        trading_volume_ratio: float
    ) -> Dict[str, Any]:
        """
        Detect herding behavior.
        
        HERDING: When investors follow others rather than 
                 their own analysis.
        
        HIGH HERDING:
        - All stocks moving together
        - Low dispersion in returns
        - High correlation to market
        
        TRADING SIGNAL:
        - High herding = Mean reversion opportunity coming
        - Low herding = Momentum strategies work better
        """
        # High correlation = herding
        correlation_score = (stock_correlation_to_market + sector_correlation) / 2
        
        # Low dispersion = herding (everyone doing same thing)
        dispersion_score = 1 - min(return_dispersion / 0.05, 1)  # 5% dispersion = normal
        
        # High volume with high correlation = panic herding
        panic_score = correlation_score * (trading_volume_ratio / 2)
        
        herding_index = (
            0.40 * correlation_score +
            0.35 * dispersion_score +
            0.25 * panic_score
        )
        
        if herding_index > 0.7:
            behavior = CrowdBehavior.HERDING_BULLISH if stock_correlation_to_market > 0 else CrowdBehavior.HERDING_BEARISH
            signal = "CONTRARIAN opportunity - Herding extreme"
        elif herding_index > 0.4:
            behavior = CrowdBehavior.HERDING_BULLISH if stock_correlation_to_market > 0 else CrowdBehavior.HERDING_BEARISH
            signal = "Herding present - Use momentum"
        else:
            behavior = CrowdBehavior.DIVERGENT
            signal = "Low herding - Stock-specific factors dominate"
        
        return {
            "herding_index": herding_index,
            "behavior": behavior.value,
            "signal": signal,
            "strategy_recommendation": "Contrarian" if herding_index > 0.6 else "Momentum"
        }
    
    @staticmethod
    def information_cascade_detection(
        sequential_same_direction_trades: int,
        price_impact_decreasing: bool,
        analyst_revision_cluster: bool,
        news_volume_spike: bool
    ) -> Dict[str, Any]:
        """
        Detect information cascades.
        
        INFORMATION CASCADE: When people follow others' actions
        rather than their own information.
        
        Cascades are FRAGILE - can reverse suddenly when 
        contradicting information arrives.
        
        TRADING SIGNAL:
        - Early cascade = Ride momentum
        - Late cascade = Prepare for reversal
        """
        cascade_indicators = 0
        
        # Many sequential same-direction trades
        if sequential_same_direction_trades > 10:
            cascade_indicators += 1
        
        # Price impact decreasing (early trades moved price, late ones don't)
        if price_impact_decreasing:
            cascade_indicators += 1
        
        # Analysts revising in cluster (following each other)
        if analyst_revision_cluster:
            cascade_indicators += 1
        
        # News volume spike (everyone talking about it)
        if news_volume_spike:
            cascade_indicators += 1
        
        if cascade_indicators >= 3:
            phase = "mature_cascade"
            signal = "CAUTION - Cascade mature, reversal risk high"
            fragility = "high"
        elif cascade_indicators >= 2:
            phase = "developing_cascade"
            signal = "MOMENTUM - Ride the cascade"
            fragility = "medium"
        else:
            phase = "no_cascade"
            signal = "Normal information processing"
            fragility = "low"
        
        return {
            "phase": phase,
            "signal": signal,
            "cascade_indicators": cascade_indicators,
            "fragility": fragility,
            "reversal_risk": "HIGH" if cascade_indicators >= 3 else "LOW"
        }
    
    @staticmethod
    def social_proof_score(
        influencer_mentions: int,
        media_coverage_score: float,
        peer_adoption_rate: float,
        celebrity_endorsement: bool
    ) -> Dict[str, Any]:
        """
        Social proof indicator.
        
        SOCIAL PROOF: People assume others' actions reflect correct behavior.
        
        In markets: "If everyone's buying, it must be good"
        
        DANGER: Social proof creates bubbles
        OPPORTUNITY: Early social proof can identify trends
        """
        social_score = (
            min(influencer_mentions / 50, 1) * 0.30 +  # Cap at 50 mentions
            media_coverage_score * 0.30 +
            peer_adoption_rate * 0.25 +
            (0.15 if celebrity_endorsement else 0)
        )
        
        if social_score > 0.8:
            level = "extreme"
            warning = "BUBBLE WARNING - Social proof at extreme"
        elif social_score > 0.5:
            level = "elevated"
            warning = "Monitor for overcrowding"
        else:
            level = "normal"
            warning = None
        
        return {
            "social_proof_score": social_score,
            "level": level,
            "warning": warning,
            "bubble_risk": social_score > 0.7
        }


# =============================================================================
# COGNITIVE BIASES (Exploitable)
# =============================================================================

class CognitiveBiasSignals:
    """
    Detect and exploit cognitive biases in other market participants.
    
    KEY INSIGHT: Markets are made of humans (and algorithms trained on 
    human data). Systematic biases create systematic alpha.
    """
    
    @staticmethod
    def anchoring_bias_signal(
        current_price: float,
        recent_high_52w: float,
        recent_low_52w: float,
        ipo_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Anchoring bias - People anchor to specific price levels.
        
        COMMON ANCHORS:
        - 52-week high/low
        - IPO price
        - Round numbers ($100, $50, etc.)
        - Previous support/resistance
        
        TRADING SIGNAL:
        - Price near anchor = Expect resistance/support
        - Price breaking anchor = Momentum acceleration
        """
        # Distance from anchors
        dist_from_high = (recent_high_52w - current_price) / recent_high_52w
        dist_from_low = (current_price - recent_low_52w) / recent_low_52w
        
        signals = []
        
        # Near 52w high
        if dist_from_high < 0.05:  # Within 5% of high
            signals.append({
                "anchor": "52w_high",
                "proximity": 1 - dist_from_high,
                "expectation": "Resistance likely, sellers anchored to previous high"
            })
        
        # Near 52w low
        if dist_from_low < 0.10:  # Within 10% of low
            signals.append({
                "anchor": "52w_low",
                "proximity": 1 - dist_from_low,
                "expectation": "Support likely, buyers remember low"
            })
        
        # Near IPO price
        if ipo_price:
            dist_from_ipo = abs(current_price - ipo_price) / ipo_price
            if dist_from_ipo < 0.10:
                signals.append({
                    "anchor": "ipo_price",
                    "proximity": 1 - dist_from_ipo,
                    "expectation": "IPO price acts as magnet"
                })
        
        # Round number detection
        round_number = round(current_price, -1)  # Nearest $10
        dist_from_round = abs(current_price - round_number) / current_price
        if dist_from_round < 0.03:  # Within 3%
            signals.append({
                "anchor": f"round_${int(round_number)}",
                "proximity": 1 - dist_from_round,
                "expectation": "Round number resistance/support"
            })
        
        return {
            "anchor_signals": signals,
            "num_active_anchors": len(signals),
            "trading_implication": "Expect price action around anchors" if signals else "No strong anchors nearby"
        }
    
    @staticmethod
    def recency_bias_score(
        return_last_week: float,
        return_last_month: float,
        return_last_year: float,
        sentiment_change_1w: float
    ) -> Dict[str, Any]:
        """
        Recency bias - People overweight recent events.
        
        PATTERN: After big moves, people extrapolate recent trend
        
        TRADING SIGNAL:
        - Extreme recent returns + extreme sentiment change = 
          Overreaction, mean reversion likely
        """
        # Detect overreaction to recent events
        recency_weight = abs(return_last_week) / (abs(return_last_year) / 52 + 0.001)
        
        # Sentiment moving with recent price = recency bias active
        sentiment_price_correlation = return_last_week * sentiment_change_1w
        
        overreaction_score = (
            min(recency_weight / 5, 1) * 0.50 +  # Recent moves weighted 5x normal
            (1 if sentiment_price_correlation > 0.3 else 0) * 0.50
        )
        
        if overreaction_score > 0.7:
            signal = "CONTRARIAN - Recency bias extreme, expect mean reversion"
            opportunity = "high"
        elif overreaction_score > 0.4:
            signal = "Moderate recency bias present"
            opportunity = "medium"
        else:
            signal = "Normal conditions"
            opportunity = "low"
        
        return {
            "recency_bias_score": overreaction_score,
            "signal": signal,
            "opportunity": opportunity,
            "recent_vs_historical_weight": recency_weight
        }
    
    @staticmethod
    def loss_aversion_indicator(
        pct_holders_underwater: float,
        avg_loss_pct_underwater: float,
        volume_on_down_days: float,
        volume_on_up_days: float
    ) -> Dict[str, Any]:
        """
        Loss aversion - Losses hurt 2x as much as gains feel good.
        
        IMPLICATIONS:
        - People hold losers too long (disposition effect)
        - Selling pressure builds as price approaches break-even
        
        TRADING SIGNAL:
        - High underwater % + price rising toward break-even = 
          Resistance as holders sell to "get out even"
        """
        # Volume asymmetry (selling on down days = fear)
        volume_fear_ratio = volume_on_down_days / (volume_on_up_days + 0.001)
        
        # Break-even selling pressure
        breakeven_pressure = pct_holders_underwater * avg_loss_pct_underwater
        
        loss_aversion_score = (
            pct_holders_underwater * 0.35 +
            min(volume_fear_ratio / 2, 1) * 0.35 +
            min(breakeven_pressure, 1) * 0.30
        )
        
        if loss_aversion_score > 0.6:
            signal = "High loss aversion - Expect resistance at break-even levels"
            behavior = "Holders will sell on any bounce"
        else:
            signal = "Normal loss aversion levels"
            behavior = "No unusual selling pressure expected"
        
        return {
            "loss_aversion_score": loss_aversion_score,
            "pct_underwater": pct_holders_underwater,
            "volume_fear_ratio": volume_fear_ratio,
            "signal": signal,
            "expected_behavior": behavior
        }


# =============================================================================
# AGGREGATE BEHAVIORAL SCORE
# =============================================================================

def calculate_behavioral_alpha_score(
    sentiment_data: Dict[str, float],
    psychology_data: Dict[str, float],
    crowd_data: Dict[str, float],
    bias_data: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculate aggregate behavioral alpha score.
    
    Combines all behavioral signals into actionable score.
    
    SCORE INTERPRETATION:
    > 0.6: Strong behavioral signal (high conviction)
    0.3 - 0.6: Moderate signal (use with other factors)
    < 0.3: Weak signal (behavioral factors neutral)
    """
    # Weight different behavioral factors
    sentiment_weight = 0.30
    psychology_weight = 0.25
    crowd_weight = 0.25
    bias_weight = 0.20
    
    # Normalize scores to 0-1
    sentiment_score = sentiment_data.get("aggregate_score", 0) / 2 + 0.5  # -1 to 1 -> 0 to 1
    psychology_score = psychology_data.get("index_value", 50) / 100
    crowd_score = crowd_data.get("herding_index", 0.5)
    bias_score = bias_data.get("recency_bias_score", 0.5)
    
    # Composite score
    composite = (
        sentiment_score * sentiment_weight +
        psychology_score * psychology_weight +
        crowd_score * crowd_weight +
        bias_score * bias_weight
    )
    
    # Direction (bullish/bearish)
    if sentiment_data.get("aggregate_score", 0) > 0.3:
        direction = "bullish"
    elif sentiment_data.get("aggregate_score", 0) < -0.3:
        direction = "bearish"
    else:
        direction = "neutral"
    
    return {
        "behavioral_alpha_score": composite,
        "direction": direction,
        "confidence": "high" if abs(composite - 0.5) > 0.2 else "medium" if abs(composite - 0.5) > 0.1 else "low",
        "components": {
            "sentiment": sentiment_score,
            "psychology": psychology_score,
            "crowd": crowd_score,
            "bias": bias_score
        },
        "trading_recommendation": f"{'BUY' if direction == 'bullish' else 'SELL' if direction == 'bearish' else 'HOLD'} with {abs(composite - 0.5) * 200:.0f}% behavioral confidence"
    }


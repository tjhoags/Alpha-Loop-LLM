"""================================================================================
BEHAVIORAL SIGNALS - Human Psychology Creates Edge
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Markets are made by humans. Human psychology is exploitable.
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BehavioralSignal:
    """Signal from behavioral analysis."""

    signal_id: str
    signal_type: str
    ticker: str
    direction: str
    confidence: float
    description: str
    behavioral_pattern: str
    psychology_explanation: str
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class BehavioralSignals:
    """BEHAVIORAL SIGNALS

    1. Anchoring Bias Exploitation - Price anchor effects
    2. Recency Bias Detector - Overweighting recent events
    3. Herding Indicator - Crowded trades
    4. Disposition Effect Trap - Tax-loss selling windows
    5. Overconfidence Meter - Analyst estimate dispersion
    6. Loss Aversion Levels - Support from psychology
    7. Round Number Magnetism - $50, $100 price effects
    8. Calendar Anomaly Scanner - Day/Month effects
    9. Attention Cascade - News/Social media virality
    10. Mean Reversion After Extremes - Post-event drift
    """

    def __init__(self):
        self.signals_detected: List[BehavioralSignal] = []

    def anchoring_bias_exploitation(
        self,
        ticker: str,
        ipo_price: float,
        all_time_high: float,
        _52_week_high: float,
        current_price: float,
    ) -> Optional[BehavioralSignal]:
        """Anchoring bias - investors anchor to reference points.

        Below IPO = psychological support
        Near ATH = resistance from profit-taking
        """
        pct_from_ipo = (current_price - ipo_price) / ipo_price if ipo_price > 0 else 0
        pct_from_ath = (current_price - all_time_high) / all_time_high if all_time_high > 0 else 0
        pct_from_52wk = (current_price - _52_week_high) / _52_week_high if _52_week_high > 0 else 0

        if -0.05 < pct_from_ipo < 0.05 and pct_from_ipo != 0:
            direction = "bullish" if pct_from_ipo < 0 else "uncertain"
            return BehavioralSignal(
                signal_id=f"anch_{hashlib.sha256(f'{ticker}ipo'.encode()).hexdigest()[:8]}",
                signal_type="ipo_anchor",
                ticker=ticker,
                direction=direction,
                confidence=0.58,
                description=f"IPO ANCHOR: {ticker} near IPO price ${ipo_price:.2f} ({pct_from_ipo:+.1%})",
                behavioral_pattern="anchoring_to_ipo",
                psychology_explanation="Investors anchor to IPO price, creating support/resistance",
            )

        if -0.03 < pct_from_52wk < 0 and abs(pct_from_ath) > 0.20:
            return BehavioralSignal(
                signal_id=f"anch_{hashlib.sha256(f'{ticker}52w'.encode()).hexdigest()[:8]}",
                signal_type="52wk_anchor",
                ticker=ticker,
                direction="uncertain",
                confidence=0.55,
                description=f"52WK HIGH ANCHOR: {ticker} testing ${_52_week_high:.2f} resistance",
                behavioral_pattern="anchoring_to_52wk_high",
                psychology_explanation="Investors who bought at 52wk high may sell to break even",
            )

        return None

    def recency_bias_detector(
        self,
        ticker: str,
        last_earnings_surprise: float,
        estimate_revisions_direction: str,
        estimate_revision_magnitude: float,
        stock_move_since_earnings: float,
    ) -> Optional[BehavioralSignal]:
        """Recency bias - overweighting recent events.

        Strong earnings + excessive estimate revisions = overdone
        """
        if estimate_revisions_direction == "up" and last_earnings_surprise > 0.10:
            if estimate_revision_magnitude > 0.15 and stock_move_since_earnings > 0.15:
                return BehavioralSignal(
                    signal_id=f"rec_{hashlib.sha256(f'{ticker}recency'.encode()).hexdigest()[:8]}",
                    signal_type="recency_bias_up",
                    ticker=ticker,
                    direction="bearish",
                    confidence=0.60,
                    description=f"RECENCY OVERDONE: {ticker} estimates up {estimate_revision_magnitude:.0%} after one beat - may be extrapolated",
                    behavioral_pattern="recency_bias_extrapolation",
                    psychology_explanation="Analysts overweight recent beat, extrapolate linearly",
                )
        elif estimate_revisions_direction == "down" and last_earnings_surprise < -0.10:
            if estimate_revision_magnitude > 0.15 and stock_move_since_earnings < -0.15:
                return BehavioralSignal(
                    signal_id=f"rec_{hashlib.sha256(f'{ticker}recdown'.encode()).hexdigest()[:8]}",
                    signal_type="recency_bias_down",
                    ticker=ticker,
                    direction="bullish",
                    confidence=0.60,
                    description=f"RECENCY OVERDONE: {ticker} estimates down {estimate_revision_magnitude:.0%} after one miss - may be overdone",
                    behavioral_pattern="recency_bias_extrapolation",
                    psychology_explanation="Analysts overweight recent miss, extrapolate negatively",
                )

        return None

    def herding_indicator(
        self,
        ticker: str,
        analyst_buy_pct: float,
        institutional_ownership_change: float,
        short_interest_change: float,
        social_sentiment_score: float,
    ) -> Optional[BehavioralSignal]:
        """Herding behavior - everyone on same side = crowded.

        Extreme consensus often marks turning points.
        """
        # Calculate herding score
        consensus_score = (
            (analyst_buy_pct - 0.5) * 2 +  # 100% buy = +1, 50% = 0
            institutional_ownership_change * 5 +  # 20% increase = +1
            -short_interest_change * 10 +  # SI decrease = bullish herding
            (social_sentiment_score - 0.5) * 2  # Extreme positive = +1
        )

        if abs(consensus_score) < 1.5:
            return None

        if consensus_score > 2.0:
            direction = "bearish"  # Contrarian
            desc = f"HERD BULLISH: {ticker} extreme bullish consensus (score {consensus_score:.1f}) - contrarian bearish"
        elif consensus_score < -2.0:
            direction = "bullish"  # Contrarian
            desc = f"HERD BEARISH: {ticker} extreme bearish consensus (score {consensus_score:.1f}) - contrarian bullish"
        else:
            return None

        return BehavioralSignal(
            signal_id=f"herd_{hashlib.sha256(f'{ticker}herd'.encode()).hexdigest()[:8]}",
            signal_type="herding_extreme",
            ticker=ticker,
            direction=direction,
            confidence=0.58,
            description=desc,
            behavioral_pattern="herding_behavior",
            psychology_explanation="Extreme consensus = crowded trade, mean reversion likely",
        )

    def disposition_effect_trap(
        self,
        ticker: str,
        ytd_return: float,
        days_to_year_end: int,
        historical_december_volume: float,
        avg_daily_volume: float,
    ) -> Optional[BehavioralSignal]:
        """Disposition effect + tax-loss selling creates opportunities.

        Losers sold in Dec for tax harvesting, often overdone.
        """
        if days_to_year_end > 45 or days_to_year_end < 1:
            return None

        if ytd_return < -0.15:
            # Tax loss candidate
            direction = "bullish"
            confidence = 0.62
            desc = f"TAX LOSS CANDIDATE: {ticker} down {ytd_return:.0%} YTD, {days_to_year_end}d to year-end"
            pattern = "tax_loss_selling"
            explanation = "Tax-loss selling creates temporary pressure, often rebounds in January"
        elif ytd_return > 0.30 and days_to_year_end < 20:
            # Winners held for long-term gains (disposition effect)
            direction = "uncertain"
            confidence = 0.50
            desc = f"DISPOSITION HOLD: {ticker} up {ytd_return:.0%} YTD - may sell after Jan 1 for long-term gains"
            pattern = "disposition_effect"
            explanation = "Winners held past year-end for long-term cap gains treatment"
        else:
            return None

        return BehavioralSignal(
            signal_id=f"disp_{hashlib.sha256(f'{ticker}tax'.encode()).hexdigest()[:8]}",
            signal_type="disposition_effect",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            behavioral_pattern=pattern,
            psychology_explanation=explanation,
        )

    def overconfidence_meter(
        self,
        ticker: str,
        analyst_estimate_high: float,
        analyst_estimate_low: float,
        analyst_estimate_mean: float,
        stock_implied_eps: float,  # From current valuation
    ) -> Optional[BehavioralSignal]:
        """Analyst estimate dispersion reveals overconfidence.

        Narrow range = false precision = overconfidence
        """
        if analyst_estimate_mean == 0:
            return None

        dispersion = (analyst_estimate_high - analyst_estimate_low) / analyst_estimate_mean

        if dispersion < 0.10:
            # Very narrow range = overconfidence
            return BehavioralSignal(
                signal_id=f"conf_{hashlib.sha256(f'{ticker}conf'.encode()).hexdigest()[:8]}",
                signal_type="analyst_overconfidence",
                ticker=ticker,
                direction="uncertain",
                confidence=0.55,
                description=f"OVERCONFIDENCE: {ticker} analyst range only {dispersion:.0%} - false precision likely",
                behavioral_pattern="overconfidence_bias",
                psychology_explanation="Narrow analyst range indicates overconfidence, actual outcome often outside",
            )
        elif dispersion > 0.50:
            # High uncertainty = potential for large move
            return BehavioralSignal(
                signal_id=f"conf_{hashlib.sha256(f'{ticker}uncert'.encode()).hexdigest()[:8]}",
                signal_type="high_uncertainty",
                ticker=ticker,
                direction="uncertain",
                confidence=0.52,
                description=f"HIGH UNCERTAINTY: {ticker} analyst range {dispersion:.0%} - big move possible either way",
                behavioral_pattern="uncertainty_premium",
                psychology_explanation="Wide analyst dispersion means high uncertainty, stock should price in risk",
            )

        return None

    def loss_aversion_levels(
        self,
        ticker: str,
        volume_profile: Dict[float, float],  # price -> volume
        current_price: float,
    ) -> Optional[BehavioralSignal]:
        """Volume-weighted price levels create psychological support/resistance.

        High volume at price = many holders, loss aversion creates support.
        """
        if not volume_profile:
            return None

        # Find high-volume price levels near current
        total_volume = sum(volume_profile.values())
        nearby_levels = {
            p: v for p, v in volume_profile.items()
            if abs(p - current_price) / current_price < 0.10
        }

        if not nearby_levels:
            return None

        max_vol_price = max(nearby_levels, key=nearby_levels.get)
        max_vol = nearby_levels[max_vol_price]
        vol_concentration = max_vol / total_volume if total_volume > 0 else 0

        if vol_concentration < 0.05:
            return None

        distance = (current_price - max_vol_price) / max_vol_price

        if distance < 0 and abs(distance) < 0.05:
            direction = "bullish"
            level_type = "support"
            desc = f"LOSS AVERSION SUPPORT: {ticker} near ${max_vol_price:.2f} high-volume level ({vol_concentration:.0%} of volume)"
        elif distance > 0 and distance < 0.05:
            direction = "bearish"
            level_type = "resistance"
            desc = f"LOSS AVERSION RESISTANCE: {ticker} near ${max_vol_price:.2f} high-volume level (break-even selling)"
        else:
            return None

        return BehavioralSignal(
            signal_id=f"loss_{hashlib.sha256(f'{ticker}loss'.encode()).hexdigest()[:8]}",
            signal_type=f"loss_aversion_{level_type}",
            ticker=ticker,
            direction=direction,
            confidence=0.58,
            description=desc,
            behavioral_pattern="loss_aversion",
            psychology_explanation="High volume at price = many holders, loss aversion creates support/resistance",
        )

    def round_number_magnetism(
        self,
        ticker: str,
        current_price: float,
        round_numbers: List[float] = None,
    ) -> Optional[BehavioralSignal]:
        """Round numbers ($50, $100, $500) act as magnets.

        Price tends to gravitate toward and stall at round numbers.
        """
        if round_numbers is None:
            round_numbers = [10, 20, 25, 50, 75, 100, 150, 200, 250, 500, 1000]

        nearest_round = min(round_numbers, key=lambda x: abs(x - current_price))
        distance_pct = (current_price - nearest_round) / nearest_round

        if abs(distance_pct) > 0.03:
            return None

        if distance_pct < 0:
            direction = "bullish"
            desc = f"ROUND NUMBER SUPPORT: {ticker} ${current_price:.2f} approaching ${nearest_round} psychological level"
        else:
            direction = "uncertain"
            desc = f"ROUND NUMBER RESISTANCE: {ticker} ${current_price:.2f} near ${nearest_round} psychological level"

        return BehavioralSignal(
            signal_id=f"round_{hashlib.sha256(f'{ticker}round'.encode()).hexdigest()[:8]}",
            signal_type="round_number",
            ticker=ticker,
            direction=direction,
            confidence=0.52,
            description=desc,
            behavioral_pattern="round_number_effect",
            psychology_explanation="Round numbers create psychological significance, act as magnets",
        )

    def calendar_anomaly_scanner(
        self,
        ticker: str,
        historical_monthly_returns: Dict[int, float],  # month (1-12) -> avg return
        current_month: int,
    ) -> Optional[BehavioralSignal]:
        """Calendar anomalies - seasonality patterns.

        January effect, September swoon, etc.
        """
        current_month_avg = historical_monthly_returns.get(current_month, 0)
        all_months_avg = sum(historical_monthly_returns.values()) / 12 if historical_monthly_returns else 0

        deviation = current_month_avg - all_months_avg

        if abs(deviation) < 0.02:
            return None

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_name = month_names[current_month - 1]

        direction = "bullish" if deviation > 0 else "bearish"

        return BehavioralSignal(
            signal_id=f"cal_{hashlib.sha256(f'{ticker}cal'.encode()).hexdigest()[:8]}",
            signal_type="calendar_anomaly",
            ticker=ticker,
            direction=direction,
            confidence=0.52,
            description=f"SEASONALITY: {ticker} historically {'strong' if deviation > 0 else 'weak'} in {month_name} ({current_month_avg:+.1%} vs {all_months_avg:+.1%} avg)",
            behavioral_pattern="calendar_effect",
            psychology_explanation="Historical seasonality patterns persist due to institutional behavior",
        )

    def attention_cascade(
        self,
        ticker: str,
        news_mention_velocity: float,  # mentions per hour
        news_sentiment: float,
        social_mention_velocity: float,
        google_trends_score: float,
    ) -> Optional[BehavioralSignal]:
        """Attention cascades create momentum then reversal.

        Viral attention = initial momentum, then exhaustion.
        """
        attention_score = (
            news_mention_velocity / 10 * 0.3 +
            social_mention_velocity / 100 * 0.3 +
            google_trends_score / 100 * 0.4
        )

        if attention_score < 0.5:
            return None

        if attention_score > 0.8:
            # Extreme attention = exhaustion coming
            direction = "bearish" if news_sentiment > 0.5 else "bullish"
            confidence = 0.60
            desc = f"ATTENTION PEAK: {ticker} attention score {attention_score:.0%} - exhaustion likely"
            pattern = "attention_cascade_peak"
        else:
            # Rising attention = momentum
            direction = "bullish" if news_sentiment > 0.5 else "bearish"
            confidence = 0.55
            desc = f"ATTENTION RISING: {ticker} gaining attention (score {attention_score:.0%})"
            pattern = "attention_cascade_rising"

        return BehavioralSignal(
            signal_id=f"attn_{hashlib.sha256(f'{ticker}attn'.encode()).hexdigest()[:8]}",
            signal_type="attention_cascade",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            behavioral_pattern=pattern,
            psychology_explanation="Attention drives initial move, then creates exhaustion and reversal",
        )

    def mean_reversion_after_extreme(
        self,
        ticker: str,
        move_1d: float,
        move_5d: float,
        historical_volatility: float,
        is_earnings_related: bool,
    ) -> Optional[BehavioralSignal]:
        """Extreme moves tend to mean revert.

        3+ sigma moves often overshoot, especially non-earnings.
        """
        sigma_1d = move_1d / (historical_volatility / 16) if historical_volatility > 0 else 0  # ~16 trading days vol adjustment
        sigma_5d = move_5d / (historical_volatility / 16 * 2.24) if historical_volatility > 0 else 0  # sqrt(5) adjustment

        # Non-earnings extremes revert more
        reversion_threshold = 3.0 if not is_earnings_related else 4.0

        if abs(sigma_1d) > reversion_threshold:
            direction = "bearish" if sigma_1d > 0 else "bullish"
            confidence = 0.58 if is_earnings_related else 0.65

            return BehavioralSignal(
                signal_id=f"mean_{hashlib.sha256(f'{ticker}mean'.encode()).hexdigest()[:8]}",
                signal_type="mean_reversion",
                ticker=ticker,
                direction=direction,
                confidence=confidence,
                description=f"EXTREME MOVE: {ticker} {sigma_1d:.1f} sigma move ({move_1d:+.1%}) - mean reversion expected",
                behavioral_pattern="extreme_mean_reversion",
                psychology_explanation="Extreme moves (3+ sigma) typically overshoot and revert",
            )

        return None

    def get_stats(self) -> Dict[str, Any]:
        return {"total_signals": len(self.signals_detected)}



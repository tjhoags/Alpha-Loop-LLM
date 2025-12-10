"""
Short Squeeze Detector
Find stocks with high probability of short squeezes

Historical Winners:
- GME: +1,500% (Jan 2021)
- AMC: +2,800% (May-Jun 2021)
- BBBY: +400% (Aug 2022)
- DDS: +300% (Aug 2023)

Key Metrics:
- Short Interest % of Float (SI%)
- Days to Cover (DTC)
- Cost to Borrow (CTB)
- Social Sentiment
- Recent Price Action

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class SqueezeCandidate:
    """Short squeeze candidate"""
    symbol: str
    short_interest_pct: float  # % of float
    days_to_cover: float
    borrow_rate: Optional[float]  # Annual %
    utilization: Optional[float]  # % of shares available to borrow
    recent_price_change: float  # Last 5 days
    volume_surge: float  # vs 30-day avg
    social_sentiment: Optional[float]  # 0-1 scale
    market_cap: float
    squeeze_score: float


class ShortSqueezeDetector:
    """
    Short Squeeze Detection Agent

    Squeeze Criteria:
    1. High SI% (>20% ideal, >15% minimum)
    2. High Days to Cover (>5 days ideal)
    3. High borrow rate (>20% APR)
    4. High utilization (>80%)
    5. Price starting to move up (breaking resistance)
    6. Volume surge (>2x average)
    7. Social media buzz (Reddit, Twitter)

    Entry Timing:
    - Enter early: SI% rising, price consolidating
    - Enter on breakout: Price breaks resistance with volume
    - DO NOT chase: Avoid stocks up >50% in a week

    Exit Strategy:
    - Trail stop loss (20-30%)
    - Take profits in tranches (25%, 50%, 75%, 100%)
    - Watch for SI% decrease (squeeze over)
    """

    def __init__(
        self,
        min_si_pct: float = 0.15,  # 15% minimum short interest
        ideal_si_pct: float = 0.25,  # 25% ideal
        min_dtc: float = 3.0,  # Min days to cover
        min_borrow_rate: float = 0.10,  # 10% APR minimum
        min_volume_surge: float = 1.5,  # 1.5x average volume
        max_recent_gain: float = 0.30,  # Don't chase >30% movers
        max_positions: int = 3,  # Limit squeeze plays
        position_size: float = 0.05,  # 5% per position (risky)
        trailing_stop: float = 0.25  # 25% trailing stop
    ):
        self.min_si_pct = min_si_pct
        self.ideal_si_pct = ideal_si_pct
        self.min_dtc = min_dtc
        self.min_borrow_rate = min_borrow_rate
        self.min_volume_surge = min_volume_surge
        self.max_recent_gain = max_recent_gain
        self.max_positions = max_positions
        self.position_size = position_size
        self.trailing_stop = trailing_stop

        self.positions = {}

        logger.info("Short Squeeze Detector initialized")
        logger.info(f"  Min SI%: {min_si_pct:.0%}, Ideal: {ideal_si_pct:.0%}")
        logger.info(f"  Min DTC: {min_dtc:.1f} days")
        logger.info(f"  Position size: {position_size:.1%} (high risk)")
        logger.info(f"  Trailing stop: {trailing_stop:.0%}")

    def calculate_squeeze_score(self, candidate: SqueezeCandidate) -> float:
        """
        Calculate squeeze probability score (0-1 scale)

        Factors:
        1. Short Interest % (30%)
        2. Days to Cover (20%)
        3. Borrow Rate / Utilization (20%)
        4. Price Action (15%)
        5. Volume Surge (10%)
        6. Social Sentiment (5%)
        """

        score = 0.0

        # 1. Short Interest % (30% weight)
        si_score = min(candidate.short_interest_pct / 0.40, 1.0)  # Max at 40% SI
        score += si_score * 0.3

        # 2. Days to Cover (20% weight)
        dtc_score = min(candidate.days_to_cover / 10.0, 1.0)  # Max at 10 DTC
        score += dtc_score * 0.2

        # 3. Borrow Rate & Utilization (20% weight)
        borrow_score = 0
        if candidate.borrow_rate is not None:
            borrow_score = min(candidate.borrow_rate / 1.0, 1.0)  # Max at 100% APR
        if candidate.utilization is not None:
            util_score = candidate.utilization
            borrow_score = (borrow_score + util_score) / 2

        score += borrow_score * 0.2

        # 4. Recent Price Action (15% weight)
        # Best if up 5-15% (starting to move but not run away)
        price_score = 0
        if 0.05 <= candidate.recent_price_change <= 0.15:
            price_score = 1.0
        elif 0 <= candidate.recent_price_change < 0.05:
            price_score = 0.7  # Consolidating, good
        elif 0.15 < candidate.recent_price_change <= 0.30:
            price_score = 0.5  # Running, risky
        else:
            price_score = 0.2  # Either down or way up

        score += price_score * 0.15

        # 5. Volume Surge (10% weight)
        vol_score = min(candidate.volume_surge / 3.0, 1.0)  # Max at 3x volume
        score += vol_score * 0.1

        # 6. Social Sentiment (5% weight)
        if candidate.social_sentiment is not None:
            score += candidate.social_sentiment * 0.05

        return np.clip(score, 0, 1)

    def detect_squeezes(
        self,
        short_interest_data: pd.DataFrame,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None
    ) -> List[SqueezeCandidate]:
        """
        Scan universe for squeeze candidates

        short_interest_data columns:
        - symbol, short_interest_pct, days_to_cover, borrow_rate,
          utilization, shares_available, update_date

        price_data: Recent price history
        volume_data: Recent volume history
        sentiment_data: Social sentiment scores (optional)
        """

        candidates = []

        logger.info(f"Scanning {len(short_interest_data)} stocks for squeezes...")

        for _, row in short_interest_data.iterrows():
            symbol = row['symbol']

            try:
                # Filter minimum criteria
                si_pct = row['short_interest_pct']
                if si_pct < self.min_si_pct:
                    continue

                dtc = row['days_to_cover']
                if dtc < self.min_dtc:
                    continue

                borrow_rate = row.get('borrow_rate')
                if borrow_rate is not None and borrow_rate < self.min_borrow_rate:
                    continue

                # Get price action
                if symbol not in price_data.columns:
                    continue

                prices = price_data[symbol].dropna()
                if len(prices) < 10:
                    continue

                recent_change = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0

                # Don't chase huge movers
                if recent_change > self.max_recent_gain:
                    continue

                # Get volume surge
                if symbol not in volume_data.columns:
                    continue

                volumes = volume_data[symbol].dropna()
                if len(volumes) < 30:
                    continue

                recent_vol = volumes.iloc[-5:].mean()
                avg_vol = volumes.iloc[-30:].mean()
                volume_surge = recent_vol / avg_vol if avg_vol > 0 else 1.0

                if volume_surge < self.min_volume_surge:
                    continue

                # Get sentiment if available
                sentiment = None
                if sentiment_data is not None and symbol in sentiment_data['symbol'].values:
                    sentiment_row = sentiment_data[sentiment_data['symbol'] == symbol].iloc[0]
                    sentiment = sentiment_row.get('sentiment_score')

                # Create candidate
                candidate = SqueezeCandidate(
                    symbol=symbol,
                    short_interest_pct=si_pct,
                    days_to_cover=dtc,
                    borrow_rate=borrow_rate,
                    utilization=row.get('utilization'),
                    recent_price_change=recent_change,
                    volume_surge=volume_surge,
                    social_sentiment=sentiment,
                    market_cap=row.get('market_cap', 0),
                    squeeze_score=0  # Calculate next
                )

                # Calculate score
                candidate.squeeze_score = self.calculate_squeeze_score(candidate)

                # Threshold
                if candidate.squeeze_score >= 0.6:
                    candidates.append(candidate)

                    logger.info(
                        f"  {symbol}: SI={si_pct:.0%}, DTC={dtc:.1f}, "
                        f"Price={recent_change:+.1%}, Vol={volume_surge:.1f}x, "
                        f"Score={candidate.squeeze_score:.2f}"
                    )

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Sort by score
        candidates.sort(key=lambda x: x.squeeze_score, reverse=True)

        logger.info(f"\nFound {len(candidates)} squeeze candidates")

        return candidates

    def generate_signals(
        self,
        short_interest_data: pd.DataFrame,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        current_positions: Dict[str, Dict],
        sentiment_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Generate trading signals for squeeze plays

        Returns:
            Dict[symbol, target_weight]
        """

        signals = {}

        # Detect squeezes
        candidates = self.detect_squeezes(
            short_interest_data,
            price_data,
            volume_data,
            sentiment_data
        )

        if not candidates:
            return signals

        # Calculate available slots
        current_count = len(current_positions)
        available_slots = self.max_positions - current_count

        if available_slots <= 0:
            logger.info(f"Already at max squeeze positions ({self.max_positions})")
            return signals

        # Take top candidates
        selected = candidates[:available_slots]

        logger.info(f"\nSelected {len(selected)} squeeze plays:")

        for candidate in selected:
            # Position size - smaller for riskier setups
            risk_adjustment = 1.0

            # Reduce size if already moved a lot
            if candidate.recent_price_change > 0.15:
                risk_adjustment *= 0.7

            # Reduce size for lower market cap (more volatile)
            if candidate.market_cap < 500_000_000:  # < $500M
                risk_adjustment *= 0.7

            position_size = self.position_size * risk_adjustment

            signals[candidate.symbol] = position_size

            logger.info(
                f"  {candidate.symbol}: {position_size:.1%} | "
                f"SI={candidate.short_interest_pct:.0%} | "
                f"DTC={candidate.days_to_cover:.1f} | "
                f"Score={candidate.squeeze_score:.2f}"
            )

        return signals

    def check_exit(
        self,
        symbol: str,
        entry_price: float,
        highest_price: float,
        current_price: float,
        current_si_pct: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check exit conditions for squeeze play

        Uses trailing stop loss + SI% decrease detection
        """

        # Current P&L
        pnl_pct = (current_price - entry_price) / entry_price

        # Trailing stop from peak
        drawdown_from_peak = (current_price - highest_price) / highest_price

        if drawdown_from_peak <= -self.trailing_stop:
            return True, f"trailing_stop from peak ({pnl_pct:+.1%})"

        # Hard stop loss (50% from entry)
        if pnl_pct <= -0.50:
            return True, f"hard_stop_loss ({pnl_pct:.1%})"

        # SI% decreased significantly (squeeze may be over)
        if current_si_pct is not None and symbol in self.positions:
            entry_si = self.positions[symbol].get('entry_si_pct', 100)
            si_decrease = (current_si_pct - entry_si) / entry_si

            if si_decrease < -0.30 and pnl_pct > 0.20:  # SI down 30%+ and up 20%+
                return True, f"squeeze_complete (SI down, profit {pnl_pct:+.1%})"

        return False, ""

    def get_profit_taking_levels(self) -> List[Tuple[float, float]]:
        """
        Get profit taking levels for squeeze plays

        Returns:
            List of (price_level, % to sell)
        """

        return [
            (0.25, 0.25),  # 25% up -> sell 25%
            (0.50, 0.25),  # 50% up -> sell another 25%
            (1.00, 0.25),  # 100% up -> sell another 25%
            (2.00, 0.25),  # 200% up -> sell final 25%
        ]


def find_historical_squeezes(
    short_interest_history: pd.DataFrame,
    price_history: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze historical data to find past squeezes

    Useful for backtesting and validating squeeze detection
    """

    squeezes = []

    for symbol in short_interest_history['symbol'].unique():
        try:
            symbol_si = short_interest_history[
                short_interest_history['symbol'] == symbol
            ].sort_values('date')

            if len(symbol_si) < 10:
                continue

            symbol_prices = price_history[symbol].dropna()

            if len(symbol_prices) < 30:
                continue

            # Look for squeeze patterns:
            # 1. High SI% sustained
            # 2. Followed by rapid price increase
            # 3. Then SI% decrease

            for i in range(10, len(symbol_si) - 10):
                si_before = symbol_si.iloc[i-10:i]['short_interest_pct'].mean()
                si_during = symbol_si.iloc[i]['short_interest_pct']
                si_after = symbol_si.iloc[i+10]['short_interest_pct']

                # High SI before
                if si_before < 0.15:
                    continue

                # SI stayed high or increased
                if si_during < si_before * 0.9:
                    continue

                # SI decreased after
                if si_after > si_during * 0.7:
                    continue

                # Check price action
                date = symbol_si.iloc[i]['date']

                price_before = symbol_prices.loc[:date].iloc[-20:].mean()
                price_peak = symbol_prices.loc[date:].iloc[:20].max()

                price_gain = (price_peak - price_before) / price_before

                if price_gain > 0.30:  # 30%+ gain
                    squeezes.append({
                        'symbol': symbol,
                        'date': date,
                        'si_before': si_before,
                        'si_peak': si_during,
                        'si_after': si_after,
                        'price_gain': price_gain
                    })

        except Exception as e:
            continue

    return pd.DataFrame(squeezes)

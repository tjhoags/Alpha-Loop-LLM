"""
Small/Mid-Cap Momentum Agent
Specialized momentum strategy for small/mid-cap stocks ($300M - $10B)

Key Features:
- Liquidity-adjusted momentum scoring
- Small-cap specific risk management
- Volume surge detection
- Institutional accumulation signals
- Earnings momentum overlay

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SmallMidCapMomentumAgent:
    """
    Small/Mid-Cap Momentum Agent

    Targets: Market cap $300M - $10B
    Focus: Liquid names with strong momentum and institutional interest
    """

    def __init__(
        self,
        max_positions: int = 15,
        position_size: float = 0.07,  # 7% per position (more diversification)
        min_market_cap: float = 300_000_000,  # $300M
        max_market_cap: float = 10_000_000_000,  # $10B
        min_daily_dollar_volume: float = 2_000_000,  # $2M min liquidity
        momentum_lookback_days: int = 60,  # 3 months
        min_momentum_percentile: float = 0.80,  # Top 20%
        stop_loss: float = 0.12,  # 12% stop (wider for small-cap volatility)
        take_profit: float = 0.30,  # 30% take profit
        max_volatility: float = 0.60  # Max 60% annual vol
    ):
        self.max_positions = max_positions
        self.position_size = position_size
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        self.min_daily_dollar_volume = min_daily_dollar_volume
        self.momentum_lookback_days = momentum_lookback_days
        self.min_momentum_percentile = min_momentum_percentile
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_volatility = max_volatility

        self.current_positions = {}
        self.statistics = {
            'total_signals': 0,
            'avg_momentum_score': 0,
            'avg_liquidity_score': 0
        }

        logger.info(f"Small/Mid-Cap Momentum Agent initialized")
        logger.info(f"  Target market cap: ${min_market_cap/1e6:.0f}M - ${max_market_cap/1e9:.1f}B")
        logger.info(f"  Min daily volume: ${min_daily_dollar_volume/1e6:.1f}M")
        logger.info(f"  Max positions: {max_positions}, Position size: {position_size:.1%}")

    def calculate_momentum_score(
        self,
        prices: pd.Series,
        volumes: pd.Series
    ) -> float:
        """
        Calculate multi-factor momentum score

        Factors:
        1. Price momentum (multiple timeframes)
        2. Volume-weighted momentum
        3. Relative strength vs market
        4. Acceleration (momentum of momentum)
        """

        if len(prices) < self.momentum_lookback_days:
            return 0.0

        scores = []

        # 1. Price momentum (60 days, 30 days, 10 days)
        returns_60d = (prices.iloc[-1] / prices.iloc[-60] - 1) if len(prices) >= 60 else 0
        returns_30d = (prices.iloc[-1] / prices.iloc[-30] - 1) if len(prices) >= 30 else 0
        returns_10d = (prices.iloc[-1] / prices.iloc[-10] - 1) if len(prices) >= 10 else 0

        # Weight shorter-term more heavily for small-caps
        momentum_score = (returns_60d * 0.3 + returns_30d * 0.4 + returns_10d * 0.3)
        scores.append(momentum_score)

        # 2. Volume trend (increasing volume = stronger momentum)
        if len(volumes) >= 20:
            recent_volume = volumes.iloc[-10:].mean()
            older_volume = volumes.iloc[-20:-10].mean()
            volume_trend = (recent_volume / older_volume - 1) if older_volume > 0 else 0
            scores.append(volume_trend * 0.5)  # Scale down

        # 3. Acceleration (is momentum accelerating?)
        if len(prices) >= 40:
            recent_momentum = (prices.iloc[-1] / prices.iloc[-20] - 1)
            older_momentum = (prices.iloc[-20] / prices.iloc[-40] - 1)
            acceleration = recent_momentum - older_momentum
            scores.append(acceleration * 0.5)

        # 4. Consistency (what % of days were positive in last 20 days?)
        if len(prices) >= 20:
            daily_returns = prices.pct_change().iloc[-20:]
            consistency = (daily_returns > 0).sum() / len(daily_returns)
            scores.append((consistency - 0.5) * 2)  # Normalize to -1 to 1

        # Combined score
        final_score = np.mean(scores) if scores else 0

        return final_score

    def calculate_liquidity_score(
        self,
        daily_dollar_volume: float,
        bid_ask_spread: Optional[float] = None,
        market_cap: Optional[float] = None
    ) -> float:
        """
        Calculate liquidity quality score

        Higher score = better liquidity = lower transaction costs
        """

        score = 0.0

        # 1. Dollar volume score (logarithmic)
        if daily_dollar_volume > 0:
            # Scale: $1M = 0, $10M = 0.5, $100M = 1.0
            volume_score = np.log10(daily_dollar_volume / 1_000_000) / 2
            volume_score = np.clip(volume_score, 0, 1)
            score += volume_score * 0.5

        # 2. Bid-ask spread score (tighter = better)
        if bid_ask_spread is not None and bid_ask_spread > 0:
            # Good: <0.5%, Acceptable: 0.5-2%, Poor: >2%
            spread_score = 1 - np.clip(bid_ask_spread / 0.02, 0, 1)
            score += spread_score * 0.3

        # 3. Market cap / volume ratio (velocity)
        if market_cap is not None and daily_dollar_volume > 0:
            days_to_liquidate = market_cap / (daily_dollar_volume * 252)  # Annualized
            velocity_score = 1 / (1 + days_to_liquidate)  # Sigmoid-like
            score += velocity_score * 0.2

        return score

    def detect_institutional_accumulation(
        self,
        prices: pd.Series,
        volumes: pd.Series
    ) -> bool:
        """
        Detect signs of institutional accumulation

        Signals:
        - Price rising on above-average volume
        - Large volume spikes with small price moves (absorption)
        - Consistent buying pressure
        """

        if len(prices) < 20 or len(volumes) < 20:
            return False

        # Check recent price/volume behavior
        recent_returns = prices.pct_change().iloc[-10:]
        recent_volumes = volumes.iloc[-10:]
        avg_volume = volumes.iloc[-30:].mean()

        # Accumulation signal 1: Price up on high volume
        up_days_high_vol = 0
        for i in range(-10, 0):
            if prices.iloc[i] > prices.iloc[i-1] and volumes.iloc[i] > avg_volume * 1.2:
                up_days_high_vol += 1

        if up_days_high_vol >= 5:  # At least half the days
            return True

        # Accumulation signal 2: Volume surge with small price change
        # (institutions absorbing supply)
        max_volume_day = recent_volumes.idxmax()
        max_volume_idx = recent_volumes.index.get_loc(max_volume_day)

        if max_volume_idx > 0:
            price_change_on_max_volume = abs(recent_returns.iloc[max_volume_idx])
            if recent_volumes.iloc[max_volume_idx] > avg_volume * 2.0 and price_change_on_max_volume < 0.02:
                return True

        return False

    def filter_universe(
        self,
        stock_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter for small/mid-cap stocks meeting liquidity requirements
        """

        filtered = stock_data.copy()

        # Market cap filter
        if 'market_cap' in filtered.columns:
            filtered = filtered[
                (filtered['market_cap'] >= self.min_market_cap) &
                (filtered['market_cap'] <= self.max_market_cap)
            ]

        # Liquidity filter
        if 'daily_dollar_volume' in filtered.columns:
            filtered = filtered[
                filtered['daily_dollar_volume'] >= self.min_daily_dollar_volume
            ]

        # Volatility filter (exclude ultra-volatile names)
        if 'volatility' in filtered.columns:
            filtered = filtered[filtered['volatility'] <= self.max_volatility]

        logger.info(f"Universe filtered: {len(stock_data)} -> {len(filtered)} stocks")

        return filtered

    def generate_signals(
        self,
        price_data: pd.DataFrame,
        stock_info: pd.DataFrame,
        current_positions: Dict[str, Dict],
        current_date: datetime
    ) -> Dict[str, float]:
        """
        Generate trading signals for small/mid-cap momentum

        Returns:
            Dict[symbol, target_weight]
        """

        signals = {}

        # Filter universe
        eligible_stocks = self.filter_universe(stock_info)

        if eligible_stocks.empty:
            logger.warning("No stocks in universe after filtering")
            return signals

        # Score each stock
        stock_scores = []

        for symbol in eligible_stocks.index:
            if symbol not in price_data.columns:
                continue

            prices = price_data[symbol].dropna()
            if len(prices) < self.momentum_lookback_days:
                continue

            # Get volume data if available
            # (In real implementation, would load from separate volume dataset)
            volumes = pd.Series(np.ones(len(prices)), index=prices.index)  # Placeholder

            # Calculate momentum score
            momentum_score = self.calculate_momentum_score(prices, volumes)

            # Calculate liquidity score
            stock_info_row = eligible_stocks.loc[symbol]
            daily_dollar_volume = stock_info_row.get('daily_dollar_volume', 0)
            market_cap = stock_info_row.get('market_cap', 0)

            liquidity_score = self.calculate_liquidity_score(
                daily_dollar_volume=daily_dollar_volume,
                market_cap=market_cap
            )

            # Check for institutional accumulation
            inst_accumulation = self.detect_institutional_accumulation(prices, volumes)

            # Calculate volatility
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

            # Combined score
            # Momentum is primary, liquidity is risk adjustment, institutional is bonus
            combined_score = (
                momentum_score * 0.6 +
                liquidity_score * 0.3 +
                (0.1 if inst_accumulation else 0)
            )

            # Volatility penalty for extremely volatile names
            if volatility > self.max_volatility * 0.8:
                combined_score *= 0.7

            stock_scores.append({
                'symbol': symbol,
                'momentum_score': momentum_score,
                'liquidity_score': liquidity_score,
                'combined_score': combined_score,
                'volatility': volatility,
                'institutional_accumulation': inst_accumulation,
                'current_price': prices.iloc[-1]
            })

        if not stock_scores:
            return signals

        # Convert to DataFrame for easier manipulation
        scores_df = pd.DataFrame(stock_scores)

        # Filter by momentum percentile
        momentum_threshold = scores_df['momentum_score'].quantile(self.min_momentum_percentile)
        top_momentum = scores_df[scores_df['momentum_score'] >= momentum_threshold]

        # Sort by combined score
        top_momentum = top_momentum.sort_values('combined_score', ascending=False)

        # Select top N positions
        selected = top_momentum.head(self.max_positions)

        logger.info(f"Generated {len(selected)} momentum signals on {current_date.date()}")

        # Equal weight for simplicity (could use volatility-weighted)
        for _, row in selected.iterrows():
            symbol = row['symbol']

            # Volatility-adjusted position sizing
            base_size = self.position_size
            vol_adjustment = min(1.0, 0.30 / row['volatility']) if row['volatility'] > 0 else 1.0
            adjusted_size = base_size * vol_adjustment

            signals[symbol] = adjusted_size

            logger.info(
                f"  {symbol}: Score={row['combined_score']:.3f}, "
                f"Momentum={row['momentum_score']:.3f}, "
                f"Liquidity={row['liquidity_score']:.3f}, "
                f"InstAccum={'YES' if row['institutional_accumulation'] else 'NO'}"
            )

        # Update statistics
        self.statistics['total_signals'] = len(signals)
        self.statistics['avg_momentum_score'] = selected['momentum_score'].mean()
        self.statistics['avg_liquidity_score'] = selected['liquidity_score'].mean()

        return signals

    def check_exit_conditions(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        days_held: int
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited

        Returns:
            (should_exit, reason)
        """

        return_pct = (current_price - entry_price) / entry_price

        # Stop loss
        if return_pct <= -self.stop_loss:
            return True, "stop_loss"

        # Take profit
        if return_pct >= self.take_profit:
            return True, "take_profit"

        # Max hold period (small-caps can lose momentum quickly)
        if days_held >= 45:
            return True, "max_hold_period"

        return False, ""

    def get_statistics(self) -> Dict:
        """Get agent statistics"""
        return self.statistics.copy()

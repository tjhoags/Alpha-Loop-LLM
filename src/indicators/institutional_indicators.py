"""
Institutional-Grade Indicators
Author: Tom Hogan | Alpha Loop Capital, LLC

Implementations of institutional indicators inspired by TradingView:
- Volume Profile
- Order Flow / Delta
- Market Profile
- Anchored VWAP
- Smart Money Concepts
- Liquidity Levels

These are Python implementations of common institutional indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class VolumeProfileLevel:
    """Volume Profile price level"""
    price: float
    volume: float
    percentage: float


@dataclass
class MarketProfileNode:
    """Market Profile TPO node"""
    price: float
    time_periods: List[str]
    tpo_count: int


class VolumeProfile:
    """
    Volume Profile Indicator

    Shows volume distribution across price levels.
    Identifies:
    - POC (Point of Control): Price with most volume
    - Value Area: Range containing 70% of volume
    - High/Low Volume Nodes
    """

    def __init__(self, num_bins: int = 50):
        self.num_bins = num_bins

    def calculate(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        value_area_pct: float = 0.70
    ) -> Dict:
        """
        Calculate Volume Profile

        Returns:
            Dict with POC, Value Area High/Low, and profile data
        """
        # Create price bins
        price_min = prices.min()
        price_max = prices.max()
        bins = np.linspace(price_min, price_max, self.num_bins)

        # Assign each price to a bin
        price_bins = np.digitize(prices, bins)

        # Aggregate volume by bin
        volume_by_bin = defaultdict(float)
        for bin_idx, volume in zip(price_bins, volumes):
            volume_by_bin[bin_idx] += volume

        # Calculate profile
        total_volume = sum(volume_by_bin.values())
        profile = []

        for bin_idx in range(1, len(bins)):
            volume = volume_by_bin.get(bin_idx, 0)
            price = (bins[bin_idx - 1] + bins[bin_idx]) / 2
            percentage = volume / total_volume if total_volume > 0 else 0

            profile.append(VolumeProfileLevel(
                price=price,
                volume=volume,
                percentage=percentage
            ))

        # Sort by volume to find POC
        profile_sorted = sorted(profile, key=lambda x: x.volume, reverse=True)
        poc = profile_sorted[0] if profile_sorted else None

        # Calculate Value Area (70% of volume)
        profile_sorted_by_price = sorted(profile, key=lambda x: x.price)
        cumulative_volume = 0
        value_area_start_idx = None
        value_area_end_idx = None

        # Find POC index
        poc_idx = next((i for i, p in enumerate(profile_sorted_by_price) if p.price == poc.price), 0)

        # Expand from POC until we reach 70% volume
        va_indices = {poc_idx}
        va_volume = poc.volume

        left_idx = poc_idx - 1
        right_idx = poc_idx + 1

        while va_volume < total_volume * value_area_pct:
            left_vol = profile_sorted_by_price[left_idx].volume if left_idx >= 0 else 0
            right_vol = profile_sorted_by_price[right_idx].volume if right_idx < len(profile_sorted_by_price) else 0

            if left_vol == 0 and right_vol == 0:
                break

            if left_vol >= right_vol and left_idx >= 0:
                va_indices.add(left_idx)
                va_volume += left_vol
                left_idx -= 1
            elif right_idx < len(profile_sorted_by_price):
                va_indices.add(right_idx)
                va_volume += right_vol
                right_idx += 1
            else:
                break

        va_prices = [profile_sorted_by_price[i].price for i in sorted(va_indices)]
        value_area_high = max(va_prices) if va_prices else price_max
        value_area_low = min(va_prices) if va_prices else price_min

        return {
            'poc': poc.price if poc else None,
            'poc_volume': poc.volume if poc else None,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'profile': profile,
            'total_volume': total_volume
        }


class OrderFlowDelta:
    """
    Order Flow / Delta Indicator

    Measures buying vs selling pressure using tick data.
    Positive delta = more buying, negative = more selling.
    """

    def calculate(
        self,
        prices: pd.Series,
        volumes: pd.Series
    ) -> pd.Series:
        """
        Calculate Order Flow Delta

        Approximates buy/sell volume based on price movement.
        If price goes up, assume volume is buying.
        If price goes down, assume volume is selling.
        """
        price_change = prices.diff()

        # Positive change = buying, negative = selling
        delta = np.where(price_change > 0, volumes, -volumes)
        delta = pd.Series(delta, index=prices.index)

        return delta

    def cumulative_delta(
        self,
        prices: pd.Series,
        volumes: pd.Series
    ) -> pd.Series:
        """Cumulative Volume Delta (CVD)"""
        delta = self.calculate(prices, volumes)
        cvd = delta.cumsum()
        return cvd


class AnchoredVWAP:
    """
    Anchored VWAP

    Volume-Weighted Average Price anchored to a specific point:
    - Session start
    - Significant high/low
    - Earnings release
    - Major event
    """

    def calculate(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        anchor_date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Calculate Anchored VWAP

        Args:
            prices: Price series (typically close or typical price)
            volumes: Volume series
            anchor_date: Date to anchor from (default: start of series)
        """
        if anchor_date is None:
            anchor_date = prices.index[0]

        # Filter data from anchor point
        mask = prices.index >= anchor_date
        prices_anchored = prices[mask]
        volumes_anchored = volumes[mask]

        # Calculate VWAP
        pv = (prices_anchored * volumes_anchored).cumsum()
        v = volumes_anchored.cumsum()

        vwap = pv / v
        vwap = vwap.fillna(method='ffill')

        return vwap

    def calculate_bands(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        anchor_date: Optional[pd.Timestamp] = None,
        std_dev: float = 1.0
    ) -> Dict[str, pd.Series]:
        """
        Calculate VWAP with standard deviation bands

        Returns:
            Dict with vwap, upper_band, lower_band
        """
        vwap = self.calculate(prices, volumes, anchor_date)

        # Calculate standard deviation
        if anchor_date is None:
            anchor_date = prices.index[0]

        mask = prices.index >= anchor_date
        prices_anchored = prices[mask]

        # Weighted variance
        variance = ((prices_anchored - vwap) ** 2).cumsum() / len(prices_anchored)
        std = np.sqrt(variance)

        upper_band = vwap + (std_dev * std)
        lower_band = vwap - (std_dev * std)

        return {
            'vwap': vwap,
            'upper_band': upper_band,
            'lower_band': lower_band
        }


class SmartMoneyConcepts:
    """
    Smart Money Concepts (SMC) / Inner Circle Trader (ICT)

    Identifies:
    - Order Blocks (OB)
    - Fair Value Gaps (FVG)
    - Liquidity Sweeps
    - Break of Structure (BOS)
    - Change of Character (CHoCH)
    """

    def find_order_blocks(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lookback: int = 20
    ) -> List[Dict]:
        """
        Find Order Blocks

        Order blocks are the last up/down candle before a strong move.
        Institutional orders likely placed here.
        """
        order_blocks = []

        for i in range(lookback, len(close) - 1):
            # Bullish Order Block: Last red candle before strong up move
            if close.iloc[i] < close.iloc[i-1]:  # Red candle
                # Check if followed by strong up move
                next_move = close.iloc[i+1:i+4].max() - close.iloc[i]
                if next_move > (high.iloc[i] - low.iloc[i]) * 2:
                    order_blocks.append({
                        'type': 'bullish',
                        'date': close.index[i],
                        'high': high.iloc[i],
                        'low': low.iloc[i],
                        'price': close.iloc[i]
                    })

            # Bearish Order Block: Last green candle before strong down move
            if close.iloc[i] > close.iloc[i-1]:  # Green candle
                # Check if followed by strong down move
                next_move = close.iloc[i] - close.iloc[i+1:i+4].min()
                if next_move > (high.iloc[i] - low.iloc[i]) * 2:
                    order_blocks.append({
                        'type': 'bearish',
                        'date': close.index[i],
                        'high': high.iloc[i],
                        'low': low.iloc[i],
                        'price': close.iloc[i]
                    })

        return order_blocks

    def find_fair_value_gaps(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> List[Dict]:
        """
        Find Fair Value Gaps (FVG)

        FVG = imbalance/inefficiency in price.
        3-candle pattern where there's a gap between candles 1 and 3.
        """
        fvgs = []

        for i in range(2, len(close)):
            # Bullish FVG: Gap between candle 1 high and candle 3 low
            if low.iloc[i] > high.iloc[i-2]:
                fvgs.append({
                    'type': 'bullish',
                    'date': close.index[i],
                    'gap_high': low.iloc[i],
                    'gap_low': high.iloc[i-2],
                    'size': low.iloc[i] - high.iloc[i-2]
                })

            # Bearish FVG: Gap between candle 1 low and candle 3 high
            if high.iloc[i] < low.iloc[i-2]:
                fvgs.append({
                    'type': 'bearish',
                    'date': close.index[i],
                    'gap_high': low.iloc[i-2],
                    'gap_low': high.iloc[i],
                    'size': low.iloc[i-2] - high.iloc[i]
                })

        return fvgs

    def detect_liquidity_sweeps(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lookback: int = 50
    ) -> List[Dict]:
        """
        Detect Liquidity Sweeps

        When price briefly takes out a high/low (liquidity grab)
        then reverses sharply.
        """
        sweeps = []

        for i in range(lookback, len(close) - 2):
            # Recent high/low
            recent_high = high.iloc[i-lookback:i].max()
            recent_low = low.iloc[i-lookback:i].min()

            # Bullish sweep: Take out recent low, then rally
            if low.iloc[i] < recent_low:
                if close.iloc[i+1] > high.iloc[i]:  # Sharp reversal
                    sweeps.append({
                        'type': 'bullish',
                        'date': close.index[i],
                        'liquidity_level': recent_low,
                        'sweep_low': low.iloc[i],
                        'reversal_close': close.iloc[i+1]
                    })

            # Bearish sweep: Take out recent high, then drop
            if high.iloc[i] > recent_high:
                if close.iloc[i+1] < low.iloc[i]:  # Sharp reversal
                    sweeps.append({
                        'type': 'bearish',
                        'date': close.index[i],
                        'liquidity_level': recent_high,
                        'sweep_high': high.iloc[i],
                        'reversal_close': close.iloc[i+1]
                    })

        return sweeps


class MarketProfile:
    """
    Market Profile (TPO - Time Price Opportunity)

    Shows price levels where market spent the most time.
    Used by institutional traders to identify:
    - Value area
    - POC (Point of Control)
    - Initial balance
    """

    def calculate(
        self,
        prices: pd.Series,
        period: str = '30min'
    ) -> Dict:
        """
        Calculate Market Profile

        Args:
            prices: Price series
            period: Time period for TPO letters (default 30min)

        Returns:
            Dict with profile data
        """
        # Resample to time periods
        resampled = prices.resample(period).agg(['first', 'last', 'min', 'max'])

        # Build TPO profile (counts per price level)
        price_bins = np.linspace(prices.min(), prices.max(), 50)
        tpo_counts = np.zeros(len(price_bins) - 1)

        for idx, row in resampled.iterrows():
            # For each time period, count which bins it touched
            bin_range = np.digitize([row['min'], row['max']], price_bins)
            for bin_idx in range(bin_range[0], bin_range[1] + 1):
                if 0 <= bin_idx < len(tpo_counts):
                    tpo_counts[bin_idx] += 1

        # Find POC
        poc_idx = np.argmax(tpo_counts)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2

        # Calculate value area (70% of TPOs)
        total_tpos = tpo_counts.sum()
        target_tpos = total_tpos * 0.70

        # Expand from POC
        va_indices = {poc_idx}
        va_tpos = tpo_counts[poc_idx]

        left = poc_idx - 1
        right = poc_idx + 1

        while va_tpos < target_tpos and (left >= 0 or right < len(tpo_counts)):
            left_val = tpo_counts[left] if left >= 0 else 0
            right_val = tpo_counts[right] if right < len(tpo_counts) else 0

            if left_val >= right_val and left >= 0:
                va_indices.add(left)
                va_tpos += left_val
                left -= 1
            elif right < len(tpo_counts):
                va_indices.add(right)
                va_tpos += right_val
                right += 1
            else:
                break

        va_prices = [(price_bins[i] + price_bins[i+1])/2 for i in sorted(va_indices)]

        return {
            'poc': poc_price,
            'value_area_high': max(va_prices),
            'value_area_low': min(va_prices),
            'tpo_profile': list(zip(price_bins[:-1], tpo_counts))
        }


def example_usage():
    """Example usage of institutional indicators"""
    # Generate sample data
    dates = pd.date_range('2025-01-01', periods=100, freq='1h')
    prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5), index=dates)
    volumes = pd.Series(np.random.randint(1000, 10000, 100), index=dates)
    high = prices + np.random.rand(100) * 0.5
    low = prices - np.random.rand(100) * 0.5

    # Volume Profile
    vp = VolumeProfile(num_bins=20)
    vp_result = vp.calculate(prices, volumes)
    print("Volume Profile:")
    print(f"  POC: ${vp_result['poc']:.2f}")
    print(f"  Value Area: ${vp_result['value_area_low']:.2f} - ${vp_result['value_area_high']:.2f}")

    # Order Flow Delta
    ofd = OrderFlowDelta()
    cvd = ofd.cumulative_delta(prices, volumes)
    print(f"\nCumulative Volume Delta (latest): {cvd.iloc[-1]:.0f}")

    # Anchored VWAP
    avwap = AnchoredVWAP()
    vwap_bands = avwap.calculate_bands(prices, volumes)
    print(f"\nAnchored VWAP (latest): ${vwap_bands['vwap'].iloc[-1]:.2f}")

    # Smart Money Concepts
    smc = SmartMoneyConcepts()
    order_blocks = smc.find_order_blocks(high, low, prices)
    fvgs = smc.find_fair_value_gaps(high, low, prices)
    print(f"\nSmart Money Concepts:")
    print(f"  Order Blocks: {len(order_blocks)}")
    print(f"  Fair Value Gaps: {len(fvgs)}")


if __name__ == "__main__":
    example_usage()

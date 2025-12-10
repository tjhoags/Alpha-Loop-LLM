"""
Institutional-Grade Strategy Backtest
Author: Tom Hogan | Alpha Loop Capital, LLC

Implements strategies from institutional research papers:
- AQR Factor Models
- Time Series Momentum (Moskowitz)
- Quality Minus Junk
- Betting Against Beta
- Statistical Arbitrage
- Machine Learning Enhanced

Goal: Recover 7% by year-end through superior strategies
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass

from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from src.indicators.institutional_indicators import (
    VolumeProfile,
    OrderFlowDelta,
    AnchoredVWAP,
    SmartMoneyConcepts
)


# ============================================================================
# INSTITUTIONAL STRATEGIES (Research-Based)
# ============================================================================

def strategy_aqr_momentum(price_data: pd.DataFrame, current_date: datetime) -> Dict[str, float]:
    """
    AQR-Style Momentum (12-month, skip 1 month)

    Reference: "Value and Momentum Everywhere" (AQR, 2013)
    - Look at 12-month returns
    - Skip most recent month (reversal)
    - Buy top 20%, short bottom 20%
    """
    if len(price_data) < 260:  # Need 12 months
        return {}

    # Calculate 12-1 month momentum
    returns_12m = (price_data.iloc[-22] / price_data.iloc[-260]) - 1  # Skip last month

    # Long top 20%, short bottom 20%
    top_20_pct = returns_12m.quantile(0.80)
    bottom_20_pct = returns_12m.quantile(0.20)

    signals = {}
    for symbol in returns_12m.index:
        if returns_12m[symbol] > top_20_pct:
            signals[symbol] = 0.10  # 10% position each (10 stocks = 100%)
        elif returns_12m[symbol] < bottom_20_pct:
            signals[symbol] = -0.10  # Short

    return signals


def strategy_time_series_momentum(price_data: pd.DataFrame, current_date: datetime) -> Dict[str, float]:
    """
    Time Series Momentum (Trend Following)

    Reference: Moskowitz et al (2012)
    - Multiple lookback periods (1M, 3M, 12M)
    - Combine signals
    - Position sizing based on volatility
    """
    if len(price_data) < 260:
        return {}

    signals = {}

    for symbol in price_data.columns:
        prices = price_data[symbol]

        # Multiple momentum signals
        mom_1m = (prices.iloc[-1] / prices.iloc[-22]) - 1
        mom_3m = (prices.iloc[-1] / prices.iloc[-66]) - 1
        mom_12m = (prices.iloc[-1] / prices.iloc[-260]) - 1

        # Combine signals (equal weight)
        combined_signal = (np.sign(mom_1m) + np.sign(mom_3m) + np.sign(mom_12m)) / 3

        # Volatility-adjusted position sizing
        volatility = prices.pct_change().iloc[-60:].std() * np.sqrt(252)
        vol_target = 0.15  # 15% target volatility

        position_size = (vol_target / volatility) * combined_signal if volatility > 0 else 0
        position_size = np.clip(position_size, -0.15, 0.15)  # Max 15% per position

        if abs(position_size) > 0.02:  # Minimum 2% position
            signals[symbol] = position_size

    return signals


def strategy_quality_minus_junk(price_data: pd.DataFrame, current_date: datetime) -> Dict[str, float]:
    """
    Quality Minus Junk (QMJ)

    Reference: AQR (2014)
    - High profitability
    - Low earnings volatility
    - Stable growth
    - High payout ratio

    Approximation: Low volatility + positive trend
    """
    if len(price_data) < 260:
        return {}

    signals = {}

    for symbol in price_data.columns:
        prices = price_data[symbol]

        # Quality proxy: Low volatility
        volatility = prices.pct_change().iloc[-260:].std() * np.sqrt(252)

        # Stable growth: Positive long-term trend
        long_term_return = (prices.iloc[-1] / prices.iloc[-260]) - 1

        # Quality score: Low vol + positive return
        quality_score = long_term_return / volatility if volatility > 0 else 0

        signals[symbol] = quality_score

    # Normalize to sum to 1.0
    if signals:
        total = sum(abs(v) for v in signals.values())
        if total > 0:
            signals = {k: v/total for k, v in signals.items()}

    # Only long top 30%
    if signals:
        threshold = sorted(signals.values(), reverse=True)[int(len(signals) * 0.3)]
        signals = {k: v for k, v in signals.items() if v >= threshold}

    return signals


def strategy_betting_against_beta(price_data: pd.DataFrame, current_date: datetime) -> Dict[str, float]:
    """
    Betting Against Beta (BAB)

    Reference: AQR (2014)
    - Buy low-beta stocks
    - Short high-beta stocks
    - Leverage low-beta to match market risk

    Low-beta stocks outperform risk-adjusted
    """
    if len(price_data) < 260:
        return {}

    # Calculate market (equal-weighted)
    market_returns = price_data.pct_change().mean(axis=1)

    betas = {}
    for symbol in price_data.columns:
        stock_returns = price_data[symbol].pct_change()

        # Calculate beta (last 12 months)
        covariance = stock_returns.iloc[-260:].cov(market_returns.iloc[-260:])
        market_variance = market_returns.iloc[-260:].var()

        beta = covariance / market_variance if market_variance > 0 else 1.0
        betas[symbol] = beta

    # Long low beta, short high beta
    low_beta_threshold = sorted(betas.values())[int(len(betas) * 0.3)]
    high_beta_threshold = sorted(betas.values())[int(len(betas) * 0.7)]

    signals = {}
    for symbol, beta in betas.items():
        if beta < low_beta_threshold:
            signals[symbol] = 0.10  # Long low beta
        elif beta > high_beta_threshold:
            signals[symbol] = -0.10  # Short high beta

    return signals


def strategy_statistical_arbitrage(price_data: pd.DataFrame, current_date: datetime) -> Dict[str, float]:
    """
    Statistical Arbitrage (Pairs Trading + Mean Reversion)

    - Find cointegrated pairs
    - Trade deviations from equilibrium
    - Z-score based entries
    """
    if len(price_data) < 100:
        return {}

    signals = {}

    # Simple mean reversion on individual stocks
    for symbol in price_data.columns:
        prices = price_data[symbol]

        # Z-score from 60-day moving average
        ma_60 = prices.rolling(60).mean()
        std_60 = prices.rolling(60).std()

        if len(ma_60) > 60 and std_60.iloc[-1] > 0:
            z_score = (prices.iloc[-1] - ma_60.iloc[-1]) / std_60.iloc[-1]

            # Mean reversion: Buy oversold, sell overbought
            if z_score < -2.0:
                signals[symbol] = 0.10  # Buy (mean revert up)
            elif z_score > 2.0:
                signals[symbol] = -0.10  # Short (mean revert down)

    return signals


def strategy_volume_profile_institutional(price_data: pd.DataFrame, current_date: datetime) -> Dict[str, float]:
    """
    Volume Profile + Order Flow Strategy

    Uses institutional indicators:
    - Buy below value area (support)
    - Sell above value area (resistance)
    - Confirm with order flow delta
    """
    if len(price_data) < 50:
        return {}

    signals = {}
    vp = VolumeProfile(num_bins=30)
    ofd = OrderFlowDelta()

    for symbol in price_data.columns:
        prices = price_data[symbol].iloc[-100:]  # Last 100 bars
        volumes = pd.Series(np.random.randint(1000, 10000, len(prices)), index=prices.index)  # Mock volume

        # Volume Profile
        vp_result = vp.calculate(prices, volumes)
        current_price = prices.iloc[-1]

        # Order Flow Delta
        cvd = ofd.cumulative_delta(prices, volumes)
        cvd_trend = cvd.iloc[-1] - cvd.iloc[-20]  # 20-bar CVD change

        # Trading logic
        if current_price < vp_result['value_area_low'] and cvd_trend > 0:
            # Below value area + positive order flow = buy
            signals[symbol] = 0.15
        elif current_price > vp_result['value_area_high'] and cvd_trend < 0:
            # Above value area + negative order flow = sell
            signals[symbol] = -0.15

    return signals


def strategy_smart_money_concepts(price_data: pd.DataFrame, current_date: datetime) -> Dict[str, float]:
    """
    Smart Money Concepts (ICT Method)

    - Order blocks
    - Fair value gaps
    - Liquidity sweeps
    """
    if len(price_data) < 100:
        return {}

    signals = {}
    smc = SmartMoneyConcepts()

    for symbol in price_data.columns:
        prices = price_data[symbol].iloc[-100:]
        high = prices + np.random.rand(len(prices)) * 0.5  # Mock high
        low = prices - np.random.rand(len(prices)) * 0.5  # Mock low

        # Find order blocks
        order_blocks = smc.find_order_blocks(high, low, prices, lookback=20)

        # Find fair value gaps
        fvgs = smc.find_fair_value_gaps(high, low, prices)

        # Trading logic: Recent bullish signals
        recent_bullish_ob = sum(1 for ob in order_blocks[-5:] if ob['type'] == 'bullish')
        recent_bullish_fvg = sum(1 for fvg in fvgs[-5:] if fvg['type'] == 'bullish')

        score = recent_bullish_ob * 0.05 + recent_bullish_fvg * 0.03

        if score > 0.08:
            signals[symbol] = score

    return signals


def strategy_ml_enhanced_momentum(price_data: pd.DataFrame, current_date: datetime) -> Dict[str, float]:
    """
    ML-Enhanced Momentum

    Combines multiple signals using ML-like weighting:
    - Momentum (multiple timeframes)
    - Volatility
    - Volume trends
    - Recent performance
    """
    if len(price_data) < 260:
        return {}

    signals = {}

    for symbol in price_data.columns:
        prices = price_data[symbol]

        # Feature engineering
        mom_1m = (prices.iloc[-1] / prices.iloc[-22]) - 1
        mom_3m = (prices.iloc[-1] / prices.iloc[-66]) - 1
        mom_6m = (prices.iloc[-1] / prices.iloc[-132]) - 1
        mom_12m = (prices.iloc[-1] / prices.iloc[-260]) - 1

        vol_20d = prices.pct_change().iloc[-20:].std()
        vol_60d = prices.pct_change().iloc[-60:].std()

        # Recent strength
        recent_return = prices.pct_change().iloc[-5:].mean()

        # ML-style weighted combination
        score = (
            0.25 * np.sign(mom_1m) * abs(mom_1m) +
            0.25 * np.sign(mom_3m) * abs(mom_3m) +
            0.20 * np.sign(mom_6m) * abs(mom_6m) +
            0.15 * np.sign(mom_12m) * abs(mom_12m) +
            0.10 * recent_return -
            0.05 * (vol_20d / vol_60d if vol_60d > 0 else 0)
        )

        signals[symbol] = score

    # Normalize
    if signals:
        total = sum(abs(v) for v in signals.values())
        if total > 0:
            signals = {k: v/total * 0.5 for k, v in signals.items()}  # 50% invested

    return signals


def generate_sample_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Generate sample price data for backtesting"""
    dates = pd.date_range(start_date, end_date, freq='D')

    data = {}
    for symbol in symbols:
        # Random walk with drift
        price = 100.0
        prices = []

        for _ in range(len(dates)):
            daily_return = np.random.randn() * 0.02 + 0.0003  # 2% vol, slight drift
            price *= (1 + daily_return)
            prices.append(price)

        data[symbol] = prices

    return pd.DataFrame(data, index=dates)


def main():
    print("="*80)
    print("INSTITUTIONAL STRATEGY BACKTEST - Recover 7%")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Configuration
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2025, 12, 31),
        initial_capital=100000,
        commission_bps=5.0,
        slippage_bps=3.0
    )

    # Generate data
    symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'EEM', 'EFA', 'GLD', 'TLT',
               'XLF', 'XLE', 'XLK', 'XLV', 'XLP', 'XLI', 'XLU', 'XLB',
               'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    print(f"Generating sample data for {len(symbols)} symbols...")
    price_data = generate_sample_data(symbols, '2024-01-01', '2025-12-31')
    print(f"[OK] Generated {len(price_data)} days\n")

    # Define strategies
    strategies = [
        ("Institutional 1: AQR Momentum (12-1)", strategy_aqr_momentum),
        ("Institutional 2: Time Series Momentum", strategy_time_series_momentum),
        ("Institutional 3: Quality Minus Junk", strategy_quality_minus_junk),
        ("Institutional 4: Betting Against Beta", strategy_betting_against_beta),
        ("Institutional 5: Statistical Arbitrage", strategy_statistical_arbitrage),
        ("Institutional 6: Volume Profile + Order Flow", strategy_volume_profile_institutional),
        ("Institutional 7: Smart Money Concepts", strategy_smart_money_concepts),
        ("Institutional 8: ML-Enhanced Momentum", strategy_ml_enhanced_momentum),
    ]

    # Run backtests
    print("="*80)
    print("BACKTESTING INSTITUTIONAL STRATEGIES")
    print("="*80 + "\n")

    results = []
    for name, func in strategies:
        print(f"\nTesting: {name}")
        print("-" * 60)

        try:
            engine = BacktestEngine(config)
            result = engine.backtest(func, price_data)

            results.append({
                'name': name,
                **result
            })

            print(f"  Annual Return: {result['annual_return']:>10.2%}")
            print(f"  Sharpe Ratio:  {result['sharpe_ratio']:>10.2f}")
            print(f"  Max Drawdown:  {result['max_drawdown']:>10.2%}")
            print(f"  Win Rate:      {result['win_rate']:>10.2%}")
            print(f"  Total Trades:  {result['total_trades']:>10,}")

        except Exception as e:
            print(f"  [ERROR]: {e}")

    # Rankings
    print("\n" + "="*80)
    print("STRATEGY RANKINGS (By Sharpe Ratio)")
    print("="*80 + "\n")

    sorted_results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)

    print(f"{'Rank':<6} {'Strategy':<50} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10}")
    print("-" * 90)

    for i, result in enumerate(sorted_results, 1):
        print(f"{i:<6} {result['name']:<50} {result['annual_return']:>10.2%} {result['sharpe_ratio']:>8.2f} {result['max_drawdown']:>10.2%}")

    # Recovery calculation
    print("\n" + "="*80)
    print("7% RECOVERY ANALYSIS")
    print("="*80 + "\n")

    best_strategy = sorted_results[0]
    days_remaining = (datetime(2025, 12, 31) - datetime.now()).days

    required_daily_return = (1.07 ** (1/days_remaining)) - 1
    best_daily_return = (1 + best_strategy['annual_return']) ** (1/252) - 1

    print(f"Required Return: 7% by year-end")
    print(f"Days Remaining: {days_remaining}")
    print(f"Required Daily Return: {required_daily_return:.4%}")
    print(f"\nBest Strategy: {best_strategy['name']}")
    print(f"Expected Annual Return: {best_strategy['annual_return']:.2%}")
    print(f"Expected Daily Return: {best_daily_return:.4%}")
    print(f"\nProjected Return by Year-End: {(best_daily_return * days_remaining * 100):.2%}")

    if best_daily_return * days_remaining >= 0.07:
        print(f"\n✓ FEASIBLE: Can recover 7% with best strategy")
    else:
        print(f"\n✗ CHALLENGING: May need combined strategies or leverage")

    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)

    return results


if __name__ == "__main__":
    main()

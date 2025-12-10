#!/usr/bin/env python3
"""
Blind Strategy Backtesting for 2025
Author: Tom Hogan | Alpha Loop Capital, LLC

Tests 10 diverse strategies blindly on 2025 data.
No optimization, no curve-fitting - pure forward testing.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtesting.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestMode,
    FillModel,
)

# ============================================================================
# STRATEGY DEFINITIONS (10 Diverse Strategies)
# ============================================================================

def strategy_1_momentum_daily(price_data: pd.DataFrame, current_date: datetime) -> dict:
    """
    Strategy 1: Daily Momentum (20-day)
    Buy top 3 momentum stocks, hold 5 days
    """
    if len(price_data) < 20:
        return {}

    # Calculate 20-day momentum
    returns_20d = (price_data.iloc[-1] / price_data.iloc[-20]) - 1

    # Top 3 by momentum
    top_3 = returns_20d.nlargest(3)

    # Equal weight
    signals = {symbol: 0.33 for symbol in top_3.index}

    return signals


def strategy_2_mean_reversion_intraday(price_data: pd.DataFrame, current_date: datetime) -> dict:
    """
    Strategy 2: Mean Reversion Intraday (RSI)
    Buy oversold (RSI < 30), sell overbought (RSI > 70)
    Hold until reversal
    """
    if len(price_data) < 14:
        return {}

    signals = {}

    for symbol in price_data.columns:
        prices = price_data[symbol].dropna()
        if len(prices) < 14:
            continue

        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        # Oversold = buy
        if current_rsi < 30:
            signals[symbol] = 0.25
        # Overbought = sell
        elif current_rsi > 70:
            signals[symbol] = -0.25

    return signals


def strategy_3_breakout_weekly(price_data: pd.DataFrame, current_date: datetime) -> dict:
    """
    Strategy 3: Weekly Breakout (52-week high)
    Buy stocks breaking 52-week highs
    Hold for 4 weeks
    """
    if len(price_data) < 252:
        return {}

    signals = {}

    for symbol in price_data.columns:
        prices = price_data[symbol].dropna()
        if len(prices) < 252:
            continue

        # 52-week high
        high_52w = prices.iloc[-252:].max()
        current_price = prices.iloc[-1]

        # Breakout (within 2% of 52w high)
        if current_price >= high_52w * 0.98:
            signals[symbol] = 0.20

    return signals


def strategy_4_trend_following_daily(price_data: pd.DataFrame, current_date: datetime) -> dict:
    """
    Strategy 4: Trend Following (50/200 SMA Golden Cross)
    Buy when 50 SMA crosses above 200 SMA
    Hold until death cross
    """
    if len(price_data) < 200:
        return {}

    signals = {}

    for symbol in price_data.columns:
        prices = price_data[symbol].dropna()
        if len(prices) < 200:
            continue

        sma_50 = prices.rolling(50).mean().iloc[-1]
        sma_200 = prices.rolling(200).mean().iloc[-1]

        # Golden cross = bullish
        if sma_50 > sma_200:
            signals[symbol] = 0.25

    return signals


def strategy_5_pairs_trading_daily(price_data: pd.DataFrame, current_date: datetime) -> dict:
    """
    Strategy 5: Pairs Trading (Cointegrated pairs)
    Find pairs with correlation > 0.8
    Trade the spread
    """
    if len(price_data) < 60:
        return {}

    signals = {}

    # Calculate correlations
    corr_matrix = price_data.iloc[-60:].corr()

    # Find highly correlated pairs (> 0.8)
    for i, symbol1 in enumerate(price_data.columns):
        for symbol2 in price_data.columns[i+1:]:
            if corr_matrix.loc[symbol1, symbol2] > 0.8:
                # Calculate spread
                ratio = price_data[symbol1].iloc[-1] / price_data[symbol2].iloc[-1]
                mean_ratio = (price_data[symbol1].iloc[-60:] / price_data[symbol2].iloc[-60:]).mean()

                # Trade the spread
                if ratio > mean_ratio * 1.05:  # symbol1 overvalued
                    signals[symbol1] = -0.10
                    signals[symbol2] = 0.10
                elif ratio < mean_ratio * 0.95:  # symbol1 undervalued
                    signals[symbol1] = 0.10
                    signals[symbol2] = -0.10

    return signals


def strategy_6_volatility_breakout_intraday(price_data: pd.DataFrame, current_date: datetime) -> dict:
    """
    Strategy 6: Volatility Breakout (Bollinger Bands)
    Buy on lower band, sell on upper band
    """
    if len(price_data) < 20:
        return {}

    signals = {}

    for symbol in price_data.columns:
        prices = price_data[symbol].dropna()
        if len(prices) < 20:
            continue

        # Bollinger Bands
        sma = prices.rolling(20).mean()
        std = prices.rolling(20).std()

        upper = sma + 2 * std
        lower = sma - 2 * std

        current_price = prices.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]

        # Touch lower band = buy
        if current_price <= current_lower:
            signals[symbol] = 0.20
        # Touch upper band = sell
        elif current_price >= current_upper:
            signals[symbol] = -0.20

    return signals


def strategy_7_dividend_growth_weekly(price_data: pd.DataFrame, current_date: datetime) -> dict:
    """
    Strategy 7: Dividend Growth
    Buy stocks with consistent uptrend (proxy for dividend growers)
    Hold long-term
    """
    if len(price_data) < 252:
        return {}

    signals = {}

    for symbol in price_data.columns:
        prices = price_data[symbol].dropna()
        if len(prices) < 252:
            continue

        # Check for consistent uptrend (positive 1Y return + low volatility)
        annual_return = (prices.iloc[-1] / prices.iloc[-252]) - 1
        volatility = prices.pct_change().iloc[-252:].std() * np.sqrt(252)

        # Low vol + positive return = dividend grower proxy
        if annual_return > 0.05 and volatility < 0.25:
            signals[symbol] = 0.15

    return signals


def strategy_8_gap_trading_intraday(price_data: pd.DataFrame, current_date: datetime) -> dict:
    """
    Strategy 8: Gap Trading
    Buy gaps up, sell gaps down (momentum continuation)
    """
    if len(price_data) < 5:
        return {}

    signals = {}

    for symbol in price_data.columns:
        prices = price_data[symbol].dropna()
        if len(prices) < 5:
            continue

        # Gap = today's open vs yesterday's close
        yesterday_close = prices.iloc[-2]
        today_open = prices.iloc[-1]

        gap = (today_open - yesterday_close) / yesterday_close

        # Gap up > 2% = buy (momentum)
        if gap > 0.02:
            signals[symbol] = 0.25
        # Gap down > 2% = sell
        elif gap < -0.02:
            signals[symbol] = -0.25

    return signals


def strategy_9_sector_rotation_weekly(price_data: pd.DataFrame, current_date: datetime) -> dict:
    """
    Strategy 9: Sector Rotation
    Buy top performing sector (via ETFs)
    Rotate monthly
    """
    if len(price_data) < 20:
        return {}

    # Calculate 20-day performance
    returns_20d = (price_data.iloc[-1] / price_data.iloc[-20]) - 1

    # Top sector
    if len(returns_20d) > 0:
        top_sector = returns_20d.idxmax()
        signals = {top_sector: 1.0}  # 100% in best sector
    else:
        signals = {}

    return signals


def strategy_10_contrarian_daily(price_data: pd.DataFrame, current_date: datetime) -> dict:
    """
    Strategy 10: Contrarian
    Buy the worst performers (assuming mean reversion)
    Hold until recovery
    """
    if len(price_data) < 20:
        return {}

    # Calculate 20-day performance
    returns_20d = (price_data.iloc[-1] / price_data.iloc[-20]) - 1

    # Bottom 3 performers
    bottom_3 = returns_20d.nsmallest(3)

    # Equal weight (contrarian bet)
    signals = {symbol: 0.33 for symbol in bottom_3.index}

    return signals


# ============================================================================
# MAIN BACKTEST RUNNER
# ============================================================================

def generate_sample_data(symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate realistic sample data for backtesting.
    In production, replace with real market data from Polygon.io/Alpha Vantage.
    """
    print(f"Generating sample data for {len(symbols)} symbols...")

    dates = pd.date_range(start_date, end_date, freq='D')

    # Simulate realistic price movements
    np.random.seed(42)

    data = {}
    for symbol in symbols:
        # Start at $100
        price = 100.0
        prices = []

        for _ in dates:
            # Random walk with drift
            daily_return = np.random.randn() * 0.02 + 0.0003  # 2% vol, slight upward drift
            price *= (1 + daily_return)
            prices.append(price)

        data[symbol] = prices

    df = pd.DataFrame(data, index=dates)

    print(f"[OK] Generated {len(df)} days of data")
    return df


def run_backtest_for_strategy(
    strategy_name: str,
    strategy_func,
    price_data: pd.DataFrame,
    config: BacktestConfig
) -> dict:
    """Run backtest for a single strategy"""
    print(f"\n{'='*60}")
    print(f"BACKTESTING: {strategy_name}")
    print(f"{'='*60}")

    engine = BacktestEngine(config)
    result = engine.run_backtest(strategy_func, price_data)

    return {
        'name': strategy_name,
        'total_return': result.total_return,
        'annual_return': result.annual_return,
        'sharpe_ratio': result.sharpe_ratio,
        'sortino_ratio': result.sortino_ratio,
        'calmar_ratio': result.calmar_ratio,
        'max_drawdown': result.max_drawdown,
        'volatility': result.volatility,
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'total_trades': result.total_trades,
        'total_commission': result.total_commission,
        'total_slippage': result.total_slippage,
    }


def calculate_confidence_intervals(results: list) -> dict:
    """Calculate 98% confidence intervals using bootstrap"""
    returns = [r['annual_return'] for r in results]
    sharpes = [r['sharpe_ratio'] for r in results]

    # Bootstrap resampling (1000 iterations)
    n_bootstrap = 1000
    bootstrap_returns = []
    bootstrap_sharpes = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        bootstrap_returns.append(np.mean(sample))

        sample_sharpe = np.random.choice(sharpes, size=len(sharpes), replace=True)
        bootstrap_sharpes.append(np.mean(sample_sharpe))

    # 98% confidence interval (1st and 99th percentile)
    return_ci = (np.percentile(bootstrap_returns, 1), np.percentile(bootstrap_returns, 99))
    sharpe_ci = (np.percentile(bootstrap_sharpes, 1), np.percentile(bootstrap_sharpes, 99))

    return {
        'return_ci_98': return_ci,
        'sharpe_ci_98': sharpe_ci,
        'mean_return': np.mean(returns),
        'mean_sharpe': np.mean(sharpes),
    }


def main():
    """Main backtesting execution"""
    print("="*60)
    print("BLIND STRATEGY BACKTESTING - 2025")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Define strategies
    strategies = [
        ("Strategy 1: Daily Momentum (20-day)", strategy_1_momentum_daily),
        ("Strategy 2: Mean Reversion Intraday (RSI)", strategy_2_mean_reversion_intraday),
        ("Strategy 3: Weekly Breakout (52-week high)", strategy_3_breakout_weekly),
        ("Strategy 4: Trend Following (Golden Cross)", strategy_4_trend_following_daily),
        ("Strategy 5: Pairs Trading", strategy_5_pairs_trading_daily),
        ("Strategy 6: Volatility Breakout (Bollinger)", strategy_6_volatility_breakout_intraday),
        ("Strategy 7: Dividend Growth Weekly", strategy_7_dividend_growth_weekly),
        ("Strategy 8: Gap Trading Intraday", strategy_8_gap_trading_intraday),
        ("Strategy 9: Sector Rotation Weekly", strategy_9_sector_rotation_weekly),
        ("Strategy 10: Contrarian Daily", strategy_10_contrarian_daily),
    ]

    # Test universe (diversified)
    symbols = [
        # Large cap tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        # Finance
        'JPM', 'BAC', 'GS',
        # Healthcare
        'JNJ', 'UNH', 'PFE',
        # Energy
        'XOM', 'CVX',
        # Consumer
        'WMT', 'KO', 'PG',
        # Industrial
        'CAT', 'BA',
        # ETFs (for sector rotation)
        'SPY', 'QQQ', 'IWM'
    ]

    # Generate data for 2025
    price_data = generate_sample_data(symbols, '2024-01-01', '2025-12-31')

    # Backtest config
    config = BacktestConfig(
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 12, 31),
        initial_capital=100000,
        commission_bps=5.0,
        slippage_bps=3.0,
        mode=BacktestMode.OUT_OF_SAMPLE,
    )

    # Run backtests
    results = []
    for name, func in strategies:
        try:
            result = run_backtest_for_strategy(name, func, price_data, config)
            results.append(result)

            print(f"\n[OK] {name}")
            print(f"  Return: {result['annual_return']:>8.2%}")
            print(f"  Sharpe: {result['sharpe_ratio']:>8.2f}")
            print(f"  Max DD: {result['max_drawdown']:>8.2%}")
            print(f"  Trades: {result['total_trades']:>8,}")

        except Exception as e:
            print(f"\n[ERROR] {name} - ERROR: {e}")
            continue

    # Calculate confidence intervals
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS (98% Confidence Intervals)")
    print(f"{'='*60}\n")

    ci_results = calculate_confidence_intervals(results)

    print(f"Mean Annual Return: {ci_results['mean_return']:>8.2%}")
    print(f"98% CI: [{ci_results['return_ci_98'][0]:>7.2%}, {ci_results['return_ci_98'][1]:>7.2%}]")
    print()
    print(f"Mean Sharpe Ratio: {ci_results['mean_sharpe']:>8.2f}")
    print(f"98% CI: [{ci_results['sharpe_ci_98'][0]:>7.2f}, {ci_results['sharpe_ci_98'][1]:>7.2f}]")
    print()

    # Rankings
    print(f"\n{'='*60}")
    print("STRATEGY RANKINGS")
    print(f"{'='*60}\n")

    # Sort by Sharpe ratio
    sorted_results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)

    print(f"{'Rank':<6} {'Strategy':<40} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 80)

    for i, r in enumerate(sorted_results, 1):
        print(f"{i:<6} {r['name']:<40} {r['annual_return']:>9.2%} {r['sharpe_ratio']:>8.2f} {r['max_drawdown']:>8.2%}")

    # Save detailed results
    output_file = 'backtest_results_2025.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'start_date': '2025-01-01',
                'end_date': '2025-12-31',
                'initial_capital': config.initial_capital,
                'commission_bps': config.commission_bps,
                'slippage_bps': config.slippage_bps,
            },
            'strategies': results,
            'confidence_intervals': {
                'return_ci_98_lower': ci_results['return_ci_98'][0],
                'return_ci_98_upper': ci_results['return_ci_98'][1],
                'sharpe_ci_98_lower': ci_results['sharpe_ci_98'][0],
                'sharpe_ci_98_upper': ci_results['sharpe_ci_98'][1],
                'mean_return': ci_results['mean_return'],
                'mean_sharpe': ci_results['mean_sharpe'],
            }
        }, f, indent=2)

    print(f"\n[OK] Detailed results saved to: {output_file}")

    print(f"\n{'='*60}")
    print("BACKTEST COMPLETE")
    print(f"{'='*60}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Strategies Tested: {len(results)}/10")
    print(f"Best Strategy: {sorted_results[0]['name']}")
    print(f"Best Sharpe: {sorted_results[0]['sharpe_ratio']:.2f}")
    print()


if __name__ == "__main__":
    main()

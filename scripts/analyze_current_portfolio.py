"""
Portfolio Analysis & Optimization for 17% Recovery
Author: Tom Hogan | Alpha Loop Capital, LLC

Analyzes your current portfolio and recommends:
1. What to hold vs sell
2. Position sizing optimization
3. Additions for 17% target
4. Risk management

Run this BEFORE deploying new strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

print("="*80)
print("PORTFOLIO ANALYZER - 17% RECOVERY PLAN")
print("="*80)
print()

# Since IBKR isn't connected, let's create a manual input system
print("STEP 1: ENTER YOUR CURRENT POSITIONS")
print("-" * 80)
print()
print("Please enter your current holdings manually:")
print("(Press Enter with blank symbol to finish)")
print()

positions = []

while True:
    print()
    symbol = input("Ticker symbol (or Enter to finish): ").strip().upper()

    if not symbol:
        break

    quantity_str = input(f"  {symbol} - Quantity: ").strip()
    avg_cost_str = input(f"  {symbol} - Average cost: $").strip()

    try:
        quantity = float(quantity_str)
        avg_cost = float(avg_cost_str)

        positions.append({
            'symbol': symbol,
            'quantity': quantity,
            'avg_cost': avg_cost
        })

        print(f"  [OK] Added {symbol}: {quantity:,.0f} shares @ ${avg_cost:.2f}")

    except ValueError:
        print("  [ERROR] Invalid input, skipped")

if not positions:
    print("\n[INFO] No positions entered.")
    print("\nYou are 100% CASH - this is actually GOOD for 17% recovery!")
    print("\nRecommendation:")
    print("  Start fresh with optimized allocation:")
    print("  1. Download full stock universe (running)")
    print("  2. Run momentum scanner tomorrow")
    print("  3. Deploy capital to top 50 signals")
    print("  4. Use 1.5x leverage from day 1")
    print("\n  With full capital flexibility, 17% is MORE achievable.")
    exit(0)

# Calculate portfolio metrics
print("\n" + "="*80)
print("PORTFOLIO ANALYSIS")
print("="*80)

# Get current prices (using yfinance)
try:
    import yfinance as yf

    for pos in positions:
        ticker = yf.Ticker(pos['symbol'])
        try:
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            pos['current_price'] = current_price
        except:
            pos['current_price'] = pos['avg_cost']  # Fallback
            print(f"  [WARNING] Could not fetch price for {pos['symbol']}, using avg cost")

except ImportError:
    print("\n[WARNING] yfinance not installed, using average costs as current prices")
    for pos in positions:
        pos['current_price'] = pos['avg_cost']

# Calculate metrics
total_cost = sum(pos['quantity'] * pos['avg_cost'] for pos in positions)
total_market_value = sum(pos['quantity'] * pos['current_price'] for pos in positions)
total_pnl = total_market_value - total_cost
total_pnl_pct = (total_pnl / total_cost) if total_cost > 0 else 0

print(f"\nTotal Cost Basis: ${total_cost:,.2f}")
print(f"Current Market Value: ${total_market_value:,.2f}")
print(f"Total P&L: ${total_pnl:,.2f} ({total_pnl_pct:+.2%})")

# Individual positions
print("\n" + "-" * 80)
print("POSITION DETAILS")
print("-" * 80)

for pos in positions:
    market_value = pos['quantity'] * pos['current_price']
    cost_basis = pos['quantity'] * pos['avg_cost']
    pnl = market_value - cost_basis
    pnl_pct = (pnl / cost_basis) if cost_basis > 0 else 0
    position_pct = (market_value / total_market_value) * 100 if total_market_value > 0 else 0

    print(f"\n{pos['symbol']}")
    print(f"  Quantity: {pos['quantity']:,.0f}")
    print(f"  Avg Cost: ${pos['avg_cost']:.2f}")
    print(f"  Current: ${pos['current_price']:.2f}")
    print(f"  Market Value: ${market_value:,.2f}")
    print(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2%})")
    print(f"  Portfolio %: {position_pct:.1f}%")

    # Recommendations
    if pnl_pct < -0.15:
        print(f"  [RECOMMENDATION] DOWN 15%+ → Consider cutting loss")
    elif pnl_pct > 0.30:
        print(f"  [RECOMMENDATION] UP 30%+ → Consider taking profits")
    elif position_pct > 25:
        print(f"  [WARNING] Position >25% → HIGH CONCENTRATION RISK")

# 17% Target Analysis
print("\n" + "="*80)
print("17% RECOVERY ANALYSIS")
print("="*80)

target_value = total_market_value * 1.17
needed_gain = target_value - total_market_value
needed_gain_pct = 0.17

print(f"\nCurrent Portfolio Value: ${total_market_value:,.2f}")
print(f"Target Value (17% gain): ${target_value:,.2f}")
print(f"Needed Gain: ${needed_gain:,.2f}")
print(f"\nDays Remaining: 22")
print(f"Required Daily Return: {(1.17 ** (1/22) - 1) * 100:.2f}%")

# Portfolio composition
print("\n" + "-" * 80)
print("CURRENT ALLOCATION")
print("-" * 80)

for pos in positions:
    market_value = pos['quantity'] * pos['current_price']
    pct = (market_value / total_market_value) * 100
    print(f"{pos['symbol']:6} {pct:>6.1f}%  |" + "█" * int(pct/2))

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS FOR 17% RECOVERY")
print("="*80)

print("\n1. POSITION MANAGEMENT:")

losers = [p for p in positions if ((p['current_price'] - p['avg_cost']) / p['avg_cost']) < -0.10]
winners = [p for p in positions if ((p['current_price'] - p['avg_cost']) / p['avg_cost']) > 0.20]

if losers:
    print(f"\n  CUT LOSERS (Down >10%):")
    for pos in losers:
        pnl_pct = ((pos['current_price'] - pos['avg_cost']) / pos['avg_cost'])
        print(f"    - {pos['symbol']}: {pnl_pct:+.1%} → SELL and redeploy capital")

if winners:
    print(f"\n  TAKE PROFITS (Up >20%):")
    for pos in winners:
        pnl_pct = ((pos['current_price'] - pos['avg_cost']) / pos['avg_cost'])
        print(f"    - {pos['symbol']}: {pnl_pct:+.1%} → Consider selling 50%")

print("\n2. CAPITAL ALLOCATION:")
freed_capital = sum(p['quantity'] * p['current_price'] for p in losers)
profit_taking = sum(p['quantity'] * p['current_price'] * 0.5 for p in winners)
available = freed_capital + profit_taking

print(f"  From cutting losers: ${freed_capital:,.2f}")
print(f"  From profit taking: ${profit_taking:,.2f}")
print(f"  Total available for redeployment: ${available:,.2f}")

print("\n3. NEW ALLOCATIONS (Using alternative data):")
print("  Deploy freed capital to:")
print(f"    - 40% Reddit momentum plays (from WSB data)")
print(f"    - 30% Insider buying signals (from SEC filings)")
print(f"    - 20% Short squeeze candidates (from short interest data)")
print(f"    - 10% 13F copycat trades (follow smart money)")

print("\n4. LEVERAGE STRATEGY:")
current_deployment = (total_market_value / total_market_value)  # Will be 1.0
recommended_leverage = 1.5 if needed_gain > total_market_value * 0.10 else 1.2

print(f"  Current leverage: {current_deployment:.1f}x")
print(f"  Recommended: {recommended_leverage:.1f}x")
print(f"  This would increase buying power to: ${total_market_value * recommended_leverage:,.2f}")

print("\n5. RISK MANAGEMENT:")
print("  - Set stop-loss at -5% portfolio level")
print("  - Position size: Max 5% per stock (unleveraged)")
print("  - Daily P&L check: Required")
print("  - Circuit breaker: -10% cumulative → reduce leverage")

# Save analysis
output = {
    'timestamp': datetime.now().isoformat(),
    'portfolio_value': total_market_value,
    'target_value': target_value,
    'needed_gain': needed_gain,
    'positions': positions,
    'losers': [p['symbol'] for p in losers],
    'winners': [p['symbol'] for p in winners]
}

output_file = Path("data/portfolio_history/current_analysis.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[OK] Analysis saved to: {output_file}")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Review recommendations above")
print("2. Execute cuts and profit-taking")
print("3. Wait for stock universe download to complete")
print("4. Run momentum scanner tomorrow")
print("5. Deploy to top signals with 1.5x leverage")
print("\n✓ Analysis complete")

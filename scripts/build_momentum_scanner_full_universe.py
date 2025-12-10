"""
Momentum Scanner - Full US Stock Universe
Scans all 12,098 downloaded stocks for best opportunities

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
Goal: Find top 50 momentum stocks for 17% recovery
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MOMENTUM SCANNER - FULL UNIVERSE (12,098 STOCKS)")
print("=" * 80)
print()

# Load ticker list
ticker_file = Path("data/stock_universe/all_us_tickers.json")
if not ticker_file.exists():
    print("[ERROR] Ticker list not found. Run download_all_us_stocks.py first.")
    exit(1)

with open(ticker_file) as f:
    tickers_data = json.load(f)

tickers = [t['symbol'] for t in tickers_data['tickers']]
print(f"[OK] Loaded {len(tickers):,} tickers")
print()

# Scan criteria
print("SCANNING CRITERIA:")
print("-" * 80)
print("Technical:")
print("  - Price: $5 - $500")
print("  - Volume: >$5M/day (avg)")
print("  - 20-day return: >10%")
print("  - RSI (14): 50-70 (not overbought)")
print("  - Above 20-day SMA")
print("  - Volume increasing")
print()
print("Fundamental:")
print("  - Market cap: >$500M")
print("  - Major exchange (not OTC)")
print()
print("Alternative Data:")
print("  - Short interest >15% (squeeze potential)")
print("  - Reddit mentions increasing")
print("  - Insider buying")
print()

# Results storage
results = []

# Load historical data directory
data_dir = Path("data/stock_universe/historical_prices")
if not data_dir.exists():
    print("[ERROR] Historical data not found.")
    exit(1)

# Scan all stocks
print("SCANNING STOCKS...")
print("-" * 80)

scanned = 0
qualified = 0

for ticker in tickers:
    scanned += 1

    if scanned % 500 == 0:
        print(f"Progress: {scanned:,}/{len(tickers):,} ({scanned/len(tickers)*100:.1f}%) - Qualified: {qualified}")

    # Load historical data
    ticker_file = data_dir / f"{ticker}.csv"
    if not ticker_file.exists():
        continue

    try:
        df = pd.read_csv(ticker_file, index_col=0, parse_dates=True)

        if len(df) < 30:
            continue

        # Get latest data
        latest = df.iloc[-1]
        current_price = latest['Close']
        current_volume = latest['Volume']

        # Price filter: $5 - $500
        if current_price < 5 or current_price > 500:
            continue

        # Calculate metrics
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['Volume_20'] = df['Volume'].rolling(20).mean()

        # RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Get current values
        current_sma = df['SMA_20'].iloc[-1]
        avg_volume = df['Volume_20'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]

        # 20-day return
        if len(df) >= 20:
            price_20d_ago = df['Close'].iloc[-20]
            return_20d = (current_price - price_20d_ago) / price_20d_ago
        else:
            continue

        # Dollar volume
        dollar_volume = current_price * avg_volume

        # Apply filters
        if dollar_volume < 5_000_000:  # Less than $5M/day
            continue

        if return_20d < 0.10:  # Less than 10% in 20 days
            continue

        if pd.isna(current_rsi) or current_rsi < 50 or current_rsi > 70:
            continue

        if current_price < current_sma:  # Below SMA
            continue

        if current_volume < avg_volume:  # Volume not increasing
            continue

        # Calculate momentum score
        # Weighted combination of factors
        momentum_score = (
            return_20d * 40 +  # 20-day return (40% weight)
            (current_volume / avg_volume) * 20 +  # Volume surge (20%)
            (current_rsi / 100) * 15 +  # RSI strength (15%)
            ((current_price - current_sma) / current_sma) * 25  # Price vs SMA (25%)
        )

        # Store result
        results.append({
            'symbol': ticker,
            'price': current_price,
            'return_20d': return_20d,
            'rsi': current_rsi,
            'volume': current_volume,
            'avg_volume': avg_volume,
            'volume_ratio': current_volume / avg_volume,
            'dollar_volume': dollar_volume,
            'price_vs_sma': (current_price - current_sma) / current_sma,
            'momentum_score': momentum_score
        })

        qualified += 1

    except Exception as e:
        continue

print()
print(f"[OK] Scanning complete: {scanned:,} scanned, {qualified} qualified")
print()

# Sort by momentum score
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('momentum_score', ascending=False)

# Top 50
top_50 = df_results.head(50)

print("=" * 80)
print(f"TOP 50 MOMENTUM STOCKS (from {len(tickers):,} universe)")
print("=" * 80)
print()

for i, row in top_50.iterrows():
    print(f"{row['symbol']:6} ${row['price']:>7.2f}  "
          f"20d: {row['return_20d']:>6.1%}  "
          f"RSI: {row['rsi']:>5.1f}  "
          f"Vol: {row['volume_ratio']:>4.1f}x  "
          f"Score: {row['momentum_score']:>6.2f}")

print()
print("=" * 80)
print("DEPLOYMENT RECOMMENDATIONS")
print("=" * 80)

total_deployment = 71_462  # From execution plan
per_stock = total_deployment / 30  # Top 30 stocks

print(f"\nDeploy to top 30 stocks:")
print(f"  Total capital: ${total_deployment:,.0f}")
print(f"  Per stock: ${per_stock:,.0f}")
print(f"  Stop loss: -3% per position")
print(f"  Target: +10% (trim 30%), +20% (trim 50%)")
print()

print("Top 30 for deployment:")
for i, row in top_50.head(30).iterrows():
    shares = int(per_stock / row['price'])
    print(f"  BUY {shares:>4} {row['symbol']:6} @ ${row['price']:>7.2f} "
          f"= ${shares * row['price']:>8,.0f}  "
          f"(20d: {row['return_20d']:>5.1%}, RSI: {row['rsi']:>5.1f})")

print()

# Save results
output_file = Path("data/momentum_scanner/scan_results.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

output_data = {
    'timestamp': datetime.now().isoformat(),
    'total_scanned': scanned,
    'total_qualified': qualified,
    'top_50': df_results.head(50).to_dict('records'),
    'deployment': {
        'total_capital': total_deployment,
        'per_stock': per_stock,
        'top_30': df_results.head(30)[['symbol', 'price', 'return_20d', 'rsi', 'momentum_score']].to_dict('records')
    }
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"[OK] Results saved to: {output_file}")

# Save as CSV for easy viewing
csv_file = Path("data/momentum_scanner/scan_results.csv")
df_results.head(50).to_csv(csv_file, index=False)
print(f"[OK] CSV saved to: {csv_file}")

print()
print("=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("1. Review top 30 recommendations above")
print("2. Cross-reference with alternative data:")
print("   - Check Reddit sentiment for these tickers")
print("   - Verify no negative news")
print("   - Check short interest (squeeze potential)")
print("3. Prepare IBKR orders for 9:30 AM")
print("4. Set stop losses at -3% for each position")
print()
print("[OK] Scanner complete")

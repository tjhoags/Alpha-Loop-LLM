#!/usr/bin/env python
"""
Get Current IBKR Positions
Connects to Interactive Brokers and fetches portfolio

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

print("\n" + "="*80)
print("IBKR PORTFOLIO ANALYZER")
print("="*80)
print()

try:
    from ib_insync import IB, Stock, util

    # Connect to IBKR
    ib = IB()

    host = os.getenv('IBKR_HOST', '127.0.0.1')
    port = int(os.getenv('IBKR_PORT', '7497'))
    client_id = int(os.getenv('IBKR_CLIENT_ID', '1'))

    print(f"Connecting to TWS/Gateway at {host}:{port}...")

    try:
        ib.connect(host, port, clientId=client_id, timeout=10)
        print("[OK] Connected to IBKR\n")
    except Exception as e:
        print(f"[ERROR] Could not connect to IBKR: {e}")
        print("\nMake sure:")
        print("1. TWS or IB Gateway is running")
        print("2. API is enabled in settings")
        print("3. Port is correct (7497 for paper, 7496 for live)")
        print("4. Socket client connections are allowed")
        sys.exit(1)

    # Get account info
    account = ib.managedAccounts()[0]
    print(f"Account: {account}\n")

    # Get account summary
    summary = ib.accountSummary(account)

    net_liq = 0
    cash = 0

    for item in summary:
        if item.tag == 'NetLiquidation':
            net_liq = float(item.value)
        elif item.tag == 'TotalCashValue':
            cash = float(item.value)

    print(f"Net Liquidation Value: ${net_liq:,.2f}")
    print(f"Cash: ${cash:,.2f}")
    print()

    # Get positions
    positions = ib.positions()

    if not positions:
        print("No open positions.")
        print("\n[ACTION NEEDED]: You're 100% cash. Need to deploy capital!")
        print("Recommended allocation for 17% target:")
        print("  - 30% Short squeeze plays")
        print("  - 25% Earnings momentum")
        print("  - 20% Options flow")
        print("  - 15% Small-cap breakouts")
        print("  - 10% Leveraged ETFs")
    else:
        print("="*80)
        print("CURRENT POSITIONS")
        print("="*80)

        total_market_value = 0

        for pos in positions:
            symbol = pos.contract.symbol
            quantity = pos.position
            avg_cost = pos.avgCost

            # Get current price
            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            ticker = ib.reqMktData(contract, '', False, False)
            ib.sleep(0.5)  # Wait for data

            current_price = ticker.last if ticker.last else ticker.close

            if current_price == 0 or current_price is None:
                current_price = avg_cost

            market_value = quantity * current_price
            total_market_value += abs(market_value)

            pnl = (current_price - avg_cost) * quantity
            pnl_pct = (pnl / (avg_cost * abs(quantity))) if avg_cost > 0 else 0

            position_pct = (abs(market_value) / net_liq) * 100 if net_liq > 0 else 0

            print(f"\n{symbol}")
            print(f"  Quantity: {quantity:,.0f}")
            print(f"  Avg Cost: ${avg_cost:.2f}")
            print(f"  Current Price: ${current_price:.2f}")
            print(f"  Market Value: ${market_value:,.2f}")
            print(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2%})")
            print(f"  Position Size: {position_pct:.1f}% of portfolio")

            # Alerts
            if abs(position_pct) > 25:
                print(f"  [WARNING] Position >25% of portfolio - HIGH RISK")
            if pnl_pct < -0.10:
                print(f"  [ALERT] Down >10% - consider stop loss")
            if pnl_pct > 0.20:
                print(f"  [PROFIT] Up >20% - consider taking profits")

        print("\n" + "="*80)
        print("PORTFOLIO SUMMARY")
        print("="*80)
        print(f"Total Market Value: ${total_market_value:,.2f}")
        print(f"Cash: ${cash:,.2f}")
        print(f"Net Liquidation: ${net_liq:,.2f}")
        print(f"Deployed: {(total_market_value/net_liq)*100:.1f}%")
        print()

        # Calculate what you need for 17%
        target_value = net_liq * 1.17
        needed = target_value - net_liq

        print(f"Current Value: ${net_liq:,.2f}")
        print(f"Target Value (17%): ${target_value:,.2f}")
        print(f"Needed Gain: ${needed:,.2f}")
        print()

        # Trading plan
        print("RECOMMENDED ACTIONS:")

        if total_market_value / net_liq < 0.5:
            print("  [ACTION] Deploy more capital - currently <50% invested")
            print("  [ACTION] Allocate to squeeze candidates: CVNA, BYND, W")
            print("  [ACTION] Set up earnings plays for this week")

        if total_market_value / net_liq > 0.90:
            print("  [CAUTION] >90% deployed - little dry powder")
            print("  [CONSIDER] Take some profits to free up capital")

    # Disconnect
    ib.disconnect()
    print("\n" + "="*80)
    print("Analysis complete")
    print("="*80 + "\n")

except ImportError:
    print("[ERROR] ib_insync not installed")
    print("\nInstall with: pip install ib_insync")
    print("Then re-run this script")
    sys.exit(1)

except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

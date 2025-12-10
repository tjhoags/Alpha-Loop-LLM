"""
Get Specific IBKR Account Portfolio
For account U20266921
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import os
load_dotenv()

try:
    from ib_insync import IB, Stock

    ib = IB()
    ib.connect('127.0.0.1', 7496, clientId=2, timeout=10)

    TARGET_ACCOUNT = 'U20266921'

    print("\n" + "="*80)
    print(f"PORTFOLIO ANALYSIS - Account {TARGET_ACCOUNT}")
    print("="*80 + "\n")

    # Get account summary
    summary = ib.accountSummary(TARGET_ACCOUNT)

    net_liq = 0
    cash = 0
    gross_position_value = 0

    for item in summary:
        if item.tag == 'NetLiquidation':
            net_liq = float(item.value)
        elif item.tag == 'TotalCashValue':
            cash = float(item.value)
        elif item.tag == 'GrossPositionValue':
            gross_position_value = float(item.value)

    print(f"Net Liquidation Value: ${net_liq:,.2f}")
    print(f"Cash: ${cash:,.2f}")
    print(f"Gross Position Value: ${gross_position_value:,.2f}")
    print(f"Invested: {(gross_position_value/net_liq)*100:.1f}%" if net_liq > 0 else "N/A")
    print()

    # Get positions for this account
    all_positions = ib.positions()
    positions = [p for p in all_positions if p.account == TARGET_ACCOUNT]

    if not positions:
        print("No positions in this account.")
        print("\nYou're 100% cash - ready to deploy!")
        print(f"Available capital: ${cash:,.2f}")
        print("\nFor 17% gain: Need ${(net_liq * 0.17):,.2f}")
        print("With full universe data, can deploy to top momentum stocks tomorrow.")
    else:
        print("="*80)
        print(f"POSITIONS ({len(positions)} total)")
        print("="*80 + "\n")

        total_pnl = 0

        for pos in positions:
            symbol = pos.contract.symbol
            quantity = pos.position
            avg_cost = pos.avgCost

            # Get current price
            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            ticker = ib.reqMktData(contract, '', False, False)
            ib.sleep(1)

            current_price = ticker.last if ticker.last and ticker.last > 0 else ticker.close
            if not current_price or current_price == 0:
                current_price = avg_cost

            market_value = quantity * current_price
            pnl = (current_price - avg_cost) * quantity
            pnl_pct = (pnl / (avg_cost * abs(quantity))) if avg_cost > 0 else 0
            position_pct = (abs(market_value) / net_liq) * 100 if net_liq > 0 else 0

            total_pnl += pnl

            print(f"{symbol}")
            print(f"  Quantity: {quantity:,.0f}")
            print(f"  Avg Cost: ${avg_cost:.2f}")
            print(f"  Current: ${current_price:.2f}")
            print(f"  Market Value: ${market_value:,.2f}")
            print(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2%})")
            print(f"  % of Portfolio: {position_pct:.1f}%")

            # Alerts
            if pnl_pct < -0.15:
                print(f"  [CUT] Down {pnl_pct:.1%} - Consider selling")
            elif pnl_pct > 0.25:
                print(f"  [PROFIT] Up {pnl_pct:.1%} - Consider taking profits")
            elif position_pct > 25:
                print(f"  [WARNING] HIGH CONCENTRATION - {position_pct:.1f}% of portfolio")

            print()

        print("="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Return: {(total_pnl/net_liq)*100:.2f}%" if net_liq > 0 else "N/A")
        print()

    # 17% target
    target = net_liq * 1.17
    needed = target - net_liq

    print("="*80)
    print("17% RECOVERY TARGET")
    print("="*80)
    print(f"Current Value: ${net_liq:,.2f}")
    print(f"Target Value: ${target:,.2f}")
    print(f"Needed Gain: ${needed:,.2f}")
    print(f"Days Remaining: 22")
    print(f"Required Daily: ${(needed/22):,.2f} ({((1.17**(1/22))-1)*100:.2f}%)")
    print()

    ib.disconnect()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

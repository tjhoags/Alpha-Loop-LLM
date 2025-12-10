"""
Get YTD Performance from IBKR
Pulls actual year-to-date returns for account U20266921

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import os
load_dotenv()

try:
    from ib_insync import IB
    from datetime import datetime, timedelta

    ib = IB()
    ib.connect('127.0.0.1', 7496, clientId=3, timeout=10)

    TARGET_ACCOUNT = 'U20266921'

    print("=" * 80)
    print(f"YTD PERFORMANCE ANALYSIS - Account {TARGET_ACCOUNT}")
    print("=" * 80)
    print()

    # Get current NAV
    summary = ib.accountSummary(TARGET_ACCOUNT)

    current_nav = 0
    for item in summary:
        if item.tag == 'NetLiquidation':
            current_nav = float(item.value)
            break

    print(f"Current NAV: ${current_nav:,.2f}")
    print()

    # Try to get account value history
    # IBKR API doesn't directly provide historical NAV, so we'll ask user
    print("To calculate YTD return, I need your starting NAV for 2025.")
    print()
    print("Please enter one of the following:")
    print("1. Your NAV on January 1, 2025")
    print("2. Or press Enter to use common starting amounts")
    print()

    user_input = input("Starting NAV (or Enter for menu): ").strip()

    if user_input:
        starting_nav = float(user_input.replace(',', '').replace('$', ''))
    else:
        print()
        print("Common options:")
        print("1. $305,000 (from CURRENT_PORTFOLIO_STATUS.md)")
        print("2. $250,000")
        print("3. $200,000")
        print("4. Other")
        choice = input("Select (1-4): ").strip()

        if choice == '1':
            starting_nav = 305000
        elif choice == '2':
            starting_nav = 250000
        elif choice == '3':
            starting_nav = 200000
        else:
            starting_nav = float(input("Enter starting NAV: ").replace(',', '').replace('$', ''))

    print()
    print("=" * 80)
    print("YTD PERFORMANCE")
    print("=" * 80)
    print(f"Starting NAV (Jan 1, 2025): ${starting_nav:,.2f}")
    print(f"Current NAV (Dec 9, 2025):  ${current_nav:,.2f}")
    print()

    ytd_gain = current_nav - starting_nav
    ytd_return = (ytd_gain / starting_nav) * 100

    print(f"YTD Gain: ${ytd_gain:,.2f}")
    print(f"YTD Return: {ytd_return:+.2f}%")
    print()

    # 17% target analysis
    target_nav = starting_nav * 1.17
    vs_target = current_nav - target_nav

    print("=" * 80)
    print("17% TARGET ANALYSIS")
    print("=" * 80)
    print(f"Target NAV (+17%): ${target_nav:,.2f}")
    print(f"Current NAV:       ${current_nav:,.2f}")
    print()

    if vs_target > 0:
        print(f"[SUCCESS] You're ABOVE target by ${vs_target:,.2f} (+{(vs_target/target_nav)*100:.2f}%)")
        print(f"You've achieved {ytd_return:.2f}% vs 17% target")
        print()
        print("RECOMMENDATION: Lock in gains, don't need aggressive recovery")
    else:
        print(f"[RECOVERY NEEDED] You're BELOW target by ${-vs_target:,.2f}")
        print(f"You're at {ytd_return:.2f}% vs 17% target")
        print()
        needed_gain = target_nav - current_nav
        days_left = (datetime(2025, 12, 31) - datetime.now()).days
        daily_needed = needed_gain / days_left if days_left > 0 else 0

        print(f"Needed gain: ${needed_gain:,.2f}")
        print(f"Days remaining: {days_left}")
        print(f"Daily gain needed: ${daily_needed:,.2f} ({(daily_needed/current_nav)*100:.2f}%)")
        print()
        print("RECOMMENDATION: Execute aggressive recovery plan")

    ib.disconnect()

except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

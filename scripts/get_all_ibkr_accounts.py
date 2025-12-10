"""
Get All IBKR Accounts
Shows all connected accounts so you can pick the right one
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import os
load_dotenv()

try:
    from ib_insync import IB

    ib = IB()

    host = os.getenv('IBKR_HOST', '127.0.0.1')
    port = int(os.getenv('IBKR_PORT', '7496'))

    print("\n" + "="*80)
    print("IBKR ACCOUNT DISCOVERY")
    print("="*80)
    print(f"\nConnecting to {host}:{port}...\n")

    ib.connect(host, port, clientId=1, timeout=10)

    # Get all managed accounts
    accounts = ib.managedAccounts()

    print(f"Found {len(accounts)} account(s):\n")

    for i, account in enumerate(accounts, 1):
        print(f"{i}. Account: {account}")

        # Get account summary for each
        summary = ib.accountSummary(account)

        for item in summary:
            if item.tag in ['NetLiquidation', 'TotalCashValue', 'GrossPositionValue']:
                print(f"   {item.tag}: {item.value} {item.currency}")

        # Get positions count
        positions = [p for p in ib.positions() if p.account == account]
        print(f"   Positions: {len(positions)}")
        print()

    print("="*80)
    print("\nTo use a specific account, update your .env file:")
    print("IBKR_ACCOUNT=U1234567  # Replace with your account number")
    print("\nOr run the position script with:")
    print("IBKR_ACCOUNT=U1234567 python scripts/get_ibkr_positions.py")
    print("="*80 + "\n")

    ib.disconnect()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

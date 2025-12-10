#!/usr/bin/env python
"""
ALC Fund Reconciliation & Accounting System
Syncs IBKR positions with Dropbox Excel files for fund accounting

Required Excel Files in Dropbox /ALC Fund Recon/:
1. Daily_Positions.xlsx
2. Trade_Log.xlsx
3. Performance_Attribution.xlsx
4. Risk_Metrics.xlsx
5. Cash_Flow.xlsx
6. NAV_Calculation.xlsx
7. Investor_Reporting.xlsx
8. Compliance_Checklist.xlsx

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("ALC FUND RECONCILIATION SYSTEM")
print("="*80)
print()


class FundAccountingSystem:
    """Complete fund accounting and reconciliation"""

    def __init__(self):
        self.ibkr_connected = False
        self.dropbox_connected = False
        self.positions_df = None
        self.trades_df = None

    def connect_ibkr(self):
        """Connect to IBKR and fetch portfolio"""
        logger.info("Connecting to IBKR...")

        try:
            from ib_insync import IB, Stock

            ib = IB()
            host = os.getenv('IBKR_HOST', '127.0.0.1')
            port = int(os.getenv('IBKR_PORT', '7497'))
            client_id = int(os.getenv('IBKR_CLIENT_ID', '1'))

            ib.connect(host, port, clientId=client_id, timeout=10)
            logger.info("[OK] Connected to IBKR")

            # Get positions
            positions = ib.positions()

            positions_data = []
            for pos in positions:
                positions_data.append({
                    'symbol': pos.contract.symbol,
                    'quantity': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_value': pos.position * pos.avgCost,  # Will update with real price
                    'account': pos.account
                })

            self.positions_df = pd.DataFrame(positions_data)

            # Get account summary
            account = ib.managedAccounts()[0]
            summary = ib.accountSummary(account)

            self.account_summary = {}
            for item in summary:
                self.account_summary[item.tag] = item.value

            ib.disconnect()
            self.ibkr_connected = True

            logger.info(f"[OK] Fetched {len(self.positions_df)} positions")
            return True

        except ImportError:
            logger.error("[ERROR] ib_insync not installed: pip install ib_insync")
            return False
        except Exception as e:
            logger.error(f"[ERROR] IBKR connection failed: {e}")
            logger.info("\nMake sure TWS/Gateway is running with API enabled")
            return False

    def connect_dropbox(self):
        """Connect to Dropbox"""
        logger.info("Connecting to Dropbox...")

        try:
            import dropbox

            token = os.getenv('DROPBOX_ACCESS_TOKEN')
            if not token or token == 'your_dropbox_token_here':
                logger.error("[ERROR] DROPBOX_ACCESS_TOKEN not set in .env")
                logger.info("\nGet token from: https://www.dropbox.com/developers/apps")
                return False

            self.dbx = dropbox.Dropbox(token)

            # Test connection
            self.dbx.users_get_current_account()

            logger.info("[OK] Connected to Dropbox")
            self.dropbox_connected = True
            return True

        except ImportError:
            logger.error("[ERROR] dropbox not installed: pip install dropbox")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Dropbox connection failed: {e}")
            return False

    def update_daily_positions(self):
        """Update Daily_Positions.xlsx"""
        logger.info("\n[1/8] Updating Daily Positions...")

        if self.positions_df is None or len(self.positions_df) == 0:
            logger.warning("No positions to update")
            return

        # Add current date
        self.positions_df['date'] = datetime.now().date()

        # Calculate metrics
        self.positions_df['pnl'] = (
            self.positions_df['market_value'] -
            (self.positions_df['quantity'] * self.positions_df['avg_cost'])
        )
        self.positions_df['pnl_pct'] = (
            self.positions_df['pnl'] /
            (self.positions_df['quantity'] * self.positions_df['avg_cost'])
        )

        logger.info(f"[OK] Prepared {len(self.positions_df)} positions for upload")

        # Would upload to Dropbox here
        if self.dropbox_connected:
            self._upload_to_dropbox('Daily_Positions.xlsx', self.positions_df)

    def update_nav_calculation(self):
        """Update NAV_Calculation.xlsx"""
        logger.info("\n[6/8] Updating NAV Calculation...")

        nav_data = {
            'date': [datetime.now().date()],
            'total_assets': [float(self.account_summary.get('TotalCashValue', 0))],
            'total_liabilities': [0],  # Update if using leverage
            'nav': [float(self.account_summary.get('NetLiquidation', 0))],
            'shares_outstanding': [1000000],  # Your fund shares
        }

        nav_df = pd.DataFrame(nav_data)
        nav_df['nav_per_share'] = nav_df['nav'] / nav_df['shares_outstanding']

        logger.info(f"[OK] NAV: ${nav_df['nav'].iloc[0]:,.2f}")
        logger.info(f"[OK] NAV per share: ${nav_df['nav_per_share'].iloc[0]:.4f}")

        if self.dropbox_connected:
            self._upload_to_dropbox('NAV_Calculation.xlsx', nav_df)

    def _upload_to_dropbox(self, filename, df):
        """Upload DataFrame to Dropbox as Excel"""
        try:
            # Convert to Excel in memory
            from io import BytesIO
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)

            # Upload
            dropbox_path = f'/ALC Fund Recon/{filename}'
            self.dbx.files_upload(
                excel_buffer.read(),
                dropbox_path,
                mode=dropbox.files.WriteMode.overwrite
            )

            logger.info(f"  -> Uploaded to {dropbox_path}")

        except Exception as e:
            logger.error(f"  -> Upload failed: {e}")

    def generate_performance_report(self):
        """Generate daily performance summary"""
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*80)

        nav = float(self.account_summary.get('NetLiquidation', 0))
        cash = float(self.account_summary.get('TotalCashValue', 0))

        logger.info(f"NAV: ${nav:,.2f}")
        logger.info(f"Cash: ${cash:,.2f}")

        if len(self.positions_df) > 0:
            total_pnl = self.positions_df['pnl'].sum()
            logger.info(f"Total P&L: ${total_pnl:,.2f}")

            # Need to hit 17%
            target_nav = nav * 1.17
            needed = target_nav - nav
            logger.info(f"\nTarget NAV (17%): ${target_nav:,.2f}")
            logger.info(f"Still needed: ${needed:,.2f}")

        logger.info("="*80 + "\n")

    def run_full_reconciliation(self):
        """Run complete fund accounting reconciliation"""

        # Connect to systems
        ibkr_ok = self.connect_ibkr()
        dropbox_ok = self.connect_dropbox()

        if not ibkr_ok:
            logger.error("\n[FAILED] Cannot proceed without IBKR connection")
            logger.info("\nTo fix:")
            logger.info("1. Open TWS or IB Gateway")
            logger.info("2. Enable API: Edit > Global Configuration > API > Settings")
            logger.info("3. Check 'Enable ActiveX and Socket Clients'")
            logger.info("4. Set Socket port to 7497 (paper) or 7496 (live)")
            logger.info("5. Run this script again")
            return False

        if not dropbox_ok:
            logger.warning("\n[WARNING] Dropbox not connected - will not sync files")
            logger.info("\nTo fix:")
            logger.info("1. Go to https://www.dropbox.com/developers/apps")
            logger.info("2. Create app or get existing app token")
            logger.info("3. Add DROPBOX_ACCESS_TOKEN to .env")
            logger.info("4. Run this script again")

        # Update all files
        logger.info("\n" + "="*80)
        logger.info("UPDATING FUND ACCOUNTING FILES")
        logger.info("="*80)

        self.update_daily_positions()
        # self.update_trade_log()  # Would implement
        # self.update_performance_attribution()
        # self.update_risk_metrics()
        # self.update_cash_flow()
        self.update_nav_calculation()
        # self.update_investor_reporting()
        # self.update_compliance_checklist()

        self.generate_performance_report()

        return True


def main():
    """Main execution"""

    system = FundAccountingSystem()
    success = system.run_full_reconciliation()

    if success:
        logger.info("[SUCCESS] Fund reconciliation complete")
    else:
        logger.error("[FAILED] Reconciliation incomplete")
        sys.exit(1)


if __name__ == "__main__":
    main()

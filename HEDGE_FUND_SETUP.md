# ALC Hedge Fund - Complete Setup Guide

**Fund**: Alpha Loop Capital, LLC
**Manager**: Tom Hogan
**AUM**: $305,000
**Target**: 17% by Dec 31, 2025 (+$51,850)

---

## System Architecture

### 1. **IBKR Integration** (Live Trading)
- Real-time portfolio data
- Trade execution
- P&L tracking
- Risk monitoring

### 2. **Dropbox Sync** (Fund Accounting)
- 8 Excel files in `/ALC Fund Recon/`
- Daily reconciliation
- Investor reporting
- Compliance tracking

### 3. **Agent Network** (Strategy Execution)
- Earnings Surprise Momentum
- Short Squeeze Detector
- Options Arbitrage
- Small/Mid-Cap Momentum
- Risk Management

---

## Setup Steps

### Step 1: Configure IBKR

**Current Status**: ❓ Need to verify

**Config in `.env`:**
```bash
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # 7497 = paper, 7496 = live
IBKR_CLIENT_ID=1
IBKR_ACCOUNT=your_account_id  # ← UPDATE THIS
```

**Actions:**
1. Open TWS or IB Gateway
2. Go to: Edit → Global Configuration → API → Settings
3. Check: "Enable ActiveX and Socket Clients"
4. Socket Port: 7497 (paper) or 7496 (live)
5. Check: "Read-Only API"  = OFF (to allow trading)
6. Click OK and restart TWS

**Test Connection:**
```bash
python scripts/get_ibkr_positions.py
```

Should show your current positions and NAV.

---

### Step 2: Configure Dropbox

**Current Status**: ❓ Need to verify

**Config in `.env`:**
```bash
DROPBOX_ACCESS_TOKEN=your_dropbox_token_here  # ← UPDATE THIS
```

**Get Token:**
1. Go to: https://www.dropbox.com/developers/apps
2. Click "Create app"
3. Choose "Scoped access"
4. Choose "Full Dropbox"
5. Name: "ALC-Fund-Accounting"
6. Click "Create app"
7. Under "OAuth 2" → Click "Generate" for Access Token
8. Copy token to `.env`

**Dropbox Folder Structure:**
```
/ALC Fund Recon/
  ├── Daily_Positions.xlsx
  ├── Trade_Log.xlsx
  ├── Performance_Attribution.xlsx
  ├── Risk_Metrics.xlsx
  ├── Cash_Flow.xlsx
  ├── NAV_Calculation.xlsx
  ├── Investor_Reporting.xlsx
  └── Compliance_Checklist.xlsx
```

**Test Connection:**
```bash
python scripts/sync_fund_accounting.py
```

Should sync all 8 Excel files.

---

### Step 3: Verify Agent Network

**Test all agents:**
```bash
python scripts/test_imports.py
```

Should show: `6 passed, 0 failed`

**Key Agents:**
- ✅ Enhanced Momentum Agent
- ✅ Mean Reversion Agent
- ✅ Earnings Surprise Momentum
- ✅ Short Squeeze Detector
- ✅ Options Arbitrage Agent
- ✅ Small/Mid-Cap Momentum

---

## Daily Operations

### Pre-Market (6:30-9:00am ET)

```bash
# 1. Sync positions from IBKR
python scripts/sync_fund_accounting.py

# 2. Check earnings calendar
python scripts/scan_earnings_surprises.py

# 3. Scan for squeeze candidates
python scripts/scan_short_squeezes_live.py

# 4. Generate daily watchlist
python scripts/generate_daily_watchlist.py
```

### Market Hours (9:30am-4:00pm ET)

```bash
# Monitor positions (run every 30 min)
python scripts/get_ibkr_positions.py

# Check real-time P&L
python scripts/show_pnl.py
```

### After Hours (4:00pm-8:00pm ET)

```bash
# Daily reconciliation
python scripts/sync_fund_accounting.py

# Performance report
python scripts/daily_performance_report.py

# Update Dropbox
# (automatic via sync_fund_accounting.py)
```

---

## Fund Accounting Files

### 1. Daily_Positions.xlsx
**Updates**: Every market close
**Content**:
- Symbol, Quantity, Avg Cost, Current Price
- Market Value, P&L, P&L %
- Position size % of NAV

### 2. Trade_Log.xlsx
**Updates**: After each trade
**Content**:
- Date, Time, Symbol, Side, Quantity, Price
- Commission, Strategy, P&L

### 3. Performance_Attribution.xlsx
**Updates**: Daily
**Content**:
- Daily return, MTD, YTD
- By strategy (momentum, squeeze, arb)
- Sharpe ratio, volatility

### 4. Risk_Metrics.xlsx
**Updates**: Daily
**Content**:
- Portfolio VaR
- Max drawdown
- Position concentration
- Beta, correlation

### 5. Cash_Flow.xlsx
**Updates**: On transactions
**Content**:
- Deposits, withdrawals
- Dividends, interest
- Fees, commissions

### 6. NAV_Calculation.xlsx
**Updates**: Daily
**Content**:
- Total assets
- Total liabilities
- NAV, NAV per share
- High water mark

### 7. Investor_Reporting.xlsx
**Updates**: Monthly
**Content**:
- Monthly return letter
- Performance vs benchmark
- Top positions
- Outlook

### 8. Compliance_Checklist.xlsx
**Updates**: Daily/Weekly
**Content**:
- Position limits check
- Leverage limits
- Concentration limits
- Regulatory filings

---

## Trading Strategy (17% Target)

### Week 1 (Dec 9-13): +5-8%
**Focus**: Setup + Initial positions

**Actions**:
- Deploy 30% to squeeze candidates (CVNA, BYND, W)
- Enter 3-4 earnings plays
- Start small-cap momentum scanner

**Target**: $15K-$24K gain

### Week 2 (Dec 16-20): +6-10%
**Focus**: Maximum deployment

**Actions**:
- Scale into winners
- Add more earnings plays
- Options flow following
- Aggressive day trading

**Target**: $18K-$30K gain

### Week 3 (Dec 23-27): +4-6%
**Focus**: Lock in gains

**Actions**:
- Take profits on 75% of positions
- Only keep high conviction
- Prepare for year-end

**Target**: $12K-$18K gain

**TOTAL**: $45K-$72K (15-24%)

---

## Risk Management

### Position Limits
- Max single position: 15% of NAV ($45K)
- Max sector exposure: 40% of NAV
- Max squeeze plays: 30% of NAV combined

### Stop Losses
- Hard stop: -5% on all positions
- Trailing stop: 25% on squeeze plays
- Mental stop: -10% daily loss limit

### Profit Taking
- First 25% at +5-7%
- Second 25% at +10-15%
- Third 25% at +20%+
- Final 25% runner

---

## Current Status

**Account Size**: $305,000
**Target**: $356,850 (17% gain = $51,850)
**Days Remaining**: 22 trading days
**Daily Target**: $2,357 (+0.75%)

**Next Actions:**
1. [ ] Verify IBKR connection
2. [ ] Setup Dropbox sync
3. [ ] Run initial reconciliation
4. [ ] Review current positions
5. [ ] Plan tomorrow's trades

---

## Support & Monitoring

**Real-time Dashboard**:
- IBKR TWS or IB Gateway
- Dropbox web interface
- Agent logs in `logs/`

**Alerts**:
- Position > 20% of NAV
- Daily loss > 2%
- Drawdown > 10%
- Failed trades

**Backup**:
- All data synced to Dropbox
- Daily exports to CSV
- Trade logs in database

---

**Ready to execute. Let's hit that 17% target!**

*Last Updated: 2025-12-09*
*Author: Tom Hogan | Alpha Loop Capital, LLC*

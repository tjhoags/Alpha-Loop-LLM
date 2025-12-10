# IBKR API Setup - Quick Guide

## Enable API Access (2 minutes)

### If using TWS (Trader Workstation):
1. Open TWS
2. Go to **File → Global Configuration → API → Settings**
3. Check these boxes:
   - ✅ "Enable ActiveX and Socket Clients"
   - ✅ "Allow connections from localhost only" (for security)
   - ✅ "Read-Only API" (if you want safety)
4. **Socket Port:** Should be **7497** (paper) or **7496** (live)
5. Click **OK**
6. **Restart TWS**

### If using IB Gateway:
1. Open IB Gateway
2. Click **Configure → Settings → API → Settings**
3. Check:
   - ✅ "Enable ActiveX and Socket Clients"
   - ✅ "Allow connections from localhost only"
4. **Socket Port:** **7497** (paper) or **7496** (live)
5. Click **OK**
6. **Restart Gateway**

---

## Test Connection

Once enabled, run:
```bash
python scripts/get_ibkr_positions.py
```

**Expected output:**
```
Connected to IBKR
Account: U1234567

Net Liquidation Value: $XXX,XXX
Cash: $XX,XXX

CURRENT POSITIONS
==================
AAPL
  Quantity: 100
  Avg Cost: $150.00
  Current Price: $175.00
  P&L: $2,500 (+16.67%)
  ...
```

---

## Common Issues

### Error: "Connection Refused"
**Solution:** API not enabled or wrong port
- Check TWS/Gateway settings
- Verify port 7497 (paper) or 7496 (live)
- Restart TWS/Gateway after enabling

### Error: "Not connected"
**Solution:** TWS/Gateway not running
- Make sure TWS or Gateway is open
- Login to your account first

### Error: "Port already in use"
**Solution:** Another program using the port
- Close other trading programs
- Or change port in TWS and .env file

---

## Configuration

Update your `.env` file:
```bash
# IBKR API Settings
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # 7497 for paper, 7496 for live
IBKR_CLIENT_ID=1
```

---

## Once Connected

The script will:
1. ✅ Pull all your current positions
2. ✅ Calculate real-time P&L
3. ✅ Show portfolio allocation
4. ✅ Calculate 17% target
5. ✅ Recommend specific actions:
   - What to sell (losers)
   - What to hold (working)
   - What to add (new signals)

---

## Security Note

**Read-Only API (Recommended):**
- Check "Read-Only API" in settings
- This prevents accidental trades
- Script can only READ positions, not place orders

**For automated trading later:**
- Uncheck "Read-Only API"
- Add IP whitelist
- Test with paper account first

---

## Quick Checklist

- [ ] TWS or Gateway is OPEN and LOGGED IN
- [ ] API is ENABLED in settings
- [ ] Port is 7497 (paper) or 7496 (live)
- [ ] "Enable ActiveX and Socket Clients" is CHECKED
- [ ] TWS/Gateway RESTARTED after changes
- [ ] Run: `python scripts/get_ibkr_positions.py`

---

**After you enable it, we can pull your ACTUAL positions and build the 17% recovery plan around what you ACTUALLY hold.**

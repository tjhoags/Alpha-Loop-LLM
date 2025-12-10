# MacBook Setup Instructions - COMPREHENSIVE DATA INGESTION

## 🚀 FULL THROTTLE: Premium Alpha Vantage + Massive S3 Data Hydration

This setup pulls ALL historical data from:
- **Alpha Vantage Premium**: Stocks, Indices, Currencies, Options, Fundamentals
- **Massive S3**: 5+ years of minute-by-minute data for all asset classes
- **Advanced Valuation Metrics**: Delta-Adjusted VaR, Convexity, EV/EBITDA, etc.

---

## Setup for Overnight Training on MacBook

### Step 1: Open Terminal
- Press `Cmd + Space` → Type "Terminal" → Enter
- Or use Cursor's integrated terminal

### Step 2: Navigate to Project
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
```

**Note:** Adjust path if your project is in a different location.

### Step 3: Set Up Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### Step 4: Install Packages
```bash
pip install -r requirements.txt
```

### Step 5: Copy Environment File
```bash
# Adjust path to your .env file location
cp ~/OneDrive/Alpha\ Loop\ LLM/API\ -\ Dec\ 2025.env .env

# Or if using iCloud/other location:
# cp ~/path/to/your/.env .env
```

### Step 6: Test Database
```bash
python scripts/test_db_connection.py
```

---

## 🎯 START COMPREHENSIVE DATA HYDRATION (MacBook)

### Option A: FULL THROTTLE - All Data Sources (Recommended)

**Terminal 1 - Massive S3 (5 years backfill):**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate
python scripts/hydrate_massive.py
```
This pulls:
- Stocks (equity/minute/) - 5 years of 1-minute bars
- Options (option/minute/) - WITH Greeks (delta, gamma, theta, vega)
- Indices (index/minute/) - S&P 500, NASDAQ, etc.
- Currencies (forex/minute/) - Major FX pairs

**Terminal 2 - Alpha Vantage Premium (continuous):**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate
python scripts/hydrate_all_alpha_vantage.py
```
This pulls:
- Stock intraday (1-minute, full history)
- Stock daily (20+ years)
- Stock fundamentals (P/E, EV/EBITDA, ROIC, Altman Z, etc.)
- Indices (SPX, NDX, DJI)
- Forex pairs (USD/EUR, USD/JPY, etc.)

**Terminal 3 - Model Training (runs continuously):**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate
python src/ml/train_models.py
```

### Option B: Standard Data Collection (if you want simpler)

**Terminal 1 - Data Collection:**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate
python src/data_ingestion/collector.py
```

**Terminal 2 - Model Training:**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate
python src/ml/train_models.py
```

---

## Mac-Specific Notes

- Use `python3` instead of `python` (if needed)
- Use forward slashes: `~/path/to/file`
- Activate script: `source venv/bin/activate`
- Use `tail -f` for viewing logs: `tail -f logs/data_collection.log`

### Keep MacBook Awake Overnight

**Option 1: Prevent Sleep**
```bash
# Prevent sleep (run before starting training)
caffeinate -d
```

**Option 2: Keep Terminal Running**
- System Preferences → Energy Saver → Prevent computer from sleeping
- Or use `caffeinate` command above

### View Logs on Mac
```bash
# Data collection
tail -f logs/data_collection.log

# Model training
tail -f logs/model_training.log

# Check models
ls -la models/*.pkl
```

---

## Running Both Windows and Mac Simultaneously

You can run training on BOTH machines simultaneously:

**Windows:** Run data collection
**MacBook:** Run model training

Or vice versa - both can run the same processes. They'll both write to the same database (if accessible) or work independently.

---

## Mac Troubleshooting

### "python: command not found"
**Solution:** Use `python3` instead:
```bash
python3 -m venv venv
python3 src/data_ingestion/collector.py
```

### "Permission denied"
**Solution:** Make scripts executable:
```bash
chmod +x scripts/*.sh
```

### Database connection fails
**Solution:** 
- Make sure SQL Server is accessible from Mac
- Check firewall settings
- Verify .env file has correct credentials

### Mac goes to sleep
**Solution:** Use caffeinate:
```bash
# Prevent sleep while running training
caffeinate -d python src/ml/train_models.py

# Or prevent sleep for all terminals
caffeinate -d
```

---

## 📊 WHAT DATA IS BEING COLLECTED?

### From Massive S3:
- **Stocks**: 5+ years of 1-minute OHLCV bars
- **Options**: Full options chains with Greeks (delta, gamma, theta, vega, rho)
- **Indices**: S&P 500, NASDAQ, Dow, VIX
- **Forex**: Major currency pairs (USD/EUR, USD/JPY, etc.)

### From Alpha Vantage Premium:
- **Stock Intraday**: 1-minute bars, full history (up to 2 years)
- **Stock Daily**: 20+ years of daily bars
- **Fundamentals**: 
  - Valuation: P/E, PEG, P/B, P/S, EV/EBITDA, EV/Sales
  - Profitability: Profit Margin, Operating Margin, ROE, ROA
  - Growth: Revenue Growth, Earnings Growth
  - Financial Health: Current Ratio, Debt/Equity, Altman Z-Score
- **Indices**: SPX, NDX, DJI daily data
- **Forex**: Major pairs intraday

### Advanced Metrics Calculated:
- **Delta-Adjusted VaR**: Portfolio risk for options
- **Convexity**: Non-linear price sensitivity
- **Graham Number**: Intrinsic value estimate
- **Piotroski F-Score**: Value investing score
- **Altman Z-Score**: Bankruptcy risk predictor

---

## ⚡ PERFORMANCE TIPS

1. **Run hydration scripts overnight** - They pull years of data
2. **Use multiple terminals** - Run Massive + Alpha Vantage + Training simultaneously
3. **Keep Mac awake**: `caffeinate -d` prevents sleep
4. **Monitor logs**: `tail -f logs/massive_ingest.log` and `tail -f logs/alpha_vantage_hydration.log`

---

## 🎯 TOMORROW MORNING (9:15 AM ET) - Start Trading

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate
python src/trading/execution_engine.py
```

**Make sure IBKR TWS/Gateway is running** (paper trading port 7497 by default)


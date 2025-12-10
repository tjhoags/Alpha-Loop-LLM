# MacBook Setup Instructions

> **📚 For complete cross-platform reference, see [`CROSS_PLATFORM_COMMANDS.md`](CROSS_PLATFORM_COMMANDS.md)**

## 🚀 FULL THROTTLE: Premium Data Ingestion

This setup pulls ALL historical data from:
- **Alpha Vantage Premium**: Stocks, Indices, Currencies, Options, Fundamentals
- **Massive S3**: 5+ years of minute-by-minute data for all asset classes
- **Advanced Valuation Metrics**: Delta-Adjusted VaR, Convexity, EV/EBITDA, etc.

---

## Setup for Overnight Training on MacBook

### Step 1: Open Terminal
**In Plain English:** "Open a command window where you can type instructions"

- Press `Cmd + Space` → Type "Terminal" → Press Enter
- Or in Cursor: Press `Cmd + ~` or go to Terminal → New Terminal

### Step 2: Navigate to Project
**In Plain English:** "Go to the folder where all the code lives"

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
```

**Note:** Adjust path if your project is in a different location.

### Step 3: Set Up Environment
**In Plain English:** "Create an isolated Python workspace for this project"

```bash
# Create virtual environment (one-time setup)
python3 -m venv venv

# Activate it (do this every time you open a new terminal)
source venv/bin/activate
```

**Success:** You'll see `(venv)` at the start of your prompt

### Step 4: Install Packages
**In Plain English:** "Install all required Python packages"

```bash
pip install -r requirements.txt
```

### Step 5: Copy Environment File
**In Plain English:** "Copy your API keys and database credentials"

```bash
# From OneDrive
cp ~/OneDrive/Alpha\ Loop\ LLM/API\ -\ Dec\ 2025.env .env

# Or from iCloud
cp ~/Library/Mobile\ Documents/com~apple~CloudDocs/Alpha\ Loop\ LLM/.env .env
```

### Step 6: Test Database
**In Plain English:** "Make sure we can connect to the database"

```bash
python scripts/test_db_connection.py
```

---

## 🎯 START COMPREHENSIVE DATA HYDRATION (MacBook)

### Option A: FULL THROTTLE - All Data Sources (Recommended)

**Terminal 1 - Massive S3 (5 years backfill):**
**In Plain English:** "Pull 5+ years of minute-by-minute data"

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
caffeinate -d python scripts/hydrate_massive.py 2>&1 | tee logs/massive.log
```

This pulls:
- Stocks (equity/minute/) - 5 years of 1-minute bars
- Options (option/minute/) - WITH Greeks (delta, gamma, theta, vega)
- Indices (index/minute/) - S&P 500, NASDAQ, etc.
- Currencies (forex/minute/) - Major FX pairs

**Terminal 2 - Alpha Vantage Premium (continuous):**
**In Plain English:** "Pull fundamentals and premium intraday data"

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
caffeinate -d python scripts/hydrate_all_alpha_vantage.py 2>&1 | tee logs/alpha_vantage.log
```

This pulls:
- Stock intraday (1-minute, full history)
- Stock daily (20+ years)
- Stock fundamentals (P/E, EV/EBITDA, ROIC, Altman Z, etc.)
- Indices (SPX, NDX, DJI)
- Forex pairs (USD/EUR, USD/JPY, etc.)

**Terminal 3 - Model Training (runs continuously):**
**In Plain English:** "Train machine learning models on collected data"

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
caffeinate -d python src/ml/train_models.py 2>&1 | tee logs/training.log
```

### Option B: Standard Data Collection (Simpler)

**Terminal 1 - Data Collection:**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
caffeinate -d python src/data_ingestion/collector.py
```

**Terminal 2 - Model Training:**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
caffeinate -d python src/ml/train_models.py
```

---

## Mac-Specific Notes

| Topic | Details |
|-------|---------|
| Python | Use `python3` instead of `python` |
| Paths | Use forward slashes: `~/path/to/file` |
| Activate | `source venv/bin/activate` |
| View logs | `tail -f file` |
| Prevent sleep | `caffeinate -d` |

### Keep MacBook Awake Overnight

**Option 1: Prevent Sleep (Simple)**
```bash
# Run before starting any long process
caffeinate -d
```

**Option 2: Prevent Sleep for Specific Duration**
```bash
# Keep awake for 8 hours (28800 seconds)
caffeinate -t 28800
```

**Option 3: System Preferences**
- System Preferences → Battery → Power Adapter
- Turn off "Put hard disks to sleep when possible"
- Set "Turn display off after" to a longer time

### View Logs on Mac
```bash
# Data collection (live updates)
tail -f logs/data_collection.log

# Model training (live updates)
tail -f logs/model_training.log

# Check how many models
ls -la models/*.pkl | wc -l
```

---

## Running Both Windows and Mac Simultaneously

You can run training on BOTH machines simultaneously:

| Machine | Recommended Task |
|---------|-----------------|
| Windows | Data collection |
| MacBook | Model training |

Or vice versa - both can run the same processes. They'll both write to the same database.

---

## 🎯 MORNING (9:15 AM ET) - Start Trading

**In Plain English:** "Start the trading engine before market opens"

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
python src/trading/execution_engine.py
```

**Prerequisites:** IBKR TWS/Gateway running (Paper: port 7497, Live: port 7496)

---

## 📊 WHAT DATA IS BEING COLLECTED?

### From Massive S3:
| Data Type | Details |
|-----------|---------|
| **Stocks** | 5+ years of 1-minute OHLCV bars |
| **Options** | Full chains with Greeks (delta, gamma, theta, vega, rho) |
| **Indices** | S&P 500, NASDAQ, Dow, VIX |
| **Forex** | Major pairs (USD/EUR, USD/JPY, etc.) |

### From Alpha Vantage Premium:
| Data Type | Details |
|-----------|---------|
| **Intraday** | 1-minute bars, up to 2 years |
| **Daily** | 20+ years of daily bars |
| **Fundamentals** | P/E, PEG, P/B, P/S, EV/EBITDA, EV/Sales |
| **Profitability** | Margins, ROE, ROA, ROIC |
| **Health** | Current Ratio, Debt/Equity, Altman Z |

### Advanced Metrics Calculated:
- **Delta-Adjusted VaR**: Portfolio risk for options
- **Convexity**: Non-linear price sensitivity
- **Graham Number**: Intrinsic value estimate
- **Piotroski F-Score**: Value investing score
- **Altman Z-Score**: Bankruptcy risk predictor

---

## Troubleshooting

### "python: command not found"
Use `python3` instead:
```bash
python3 -m venv venv
python3 src/data_ingestion/collector.py
```

### "Permission denied"
Make scripts executable:
```bash
chmod +x scripts/*.sh
```

### Database connection fails
- Make sure SQL Server is accessible from Mac
- Check firewall settings
- Verify .env file has correct credentials:
```bash
ls -la .env
grep "DB_" .env
```

### Mac goes to sleep
Use caffeinate with any command:
```bash
caffeinate -d python src/ml/train_models.py
```

---

## ⚡ PERFORMANCE TIPS

1. **Run hydration scripts overnight** - They pull years of data
2. **Use multiple terminals** - Run Massive + Alpha Vantage + Training simultaneously
3. **Keep Mac awake**: `caffeinate -d` prevents sleep
4. **Monitor logs**: Check progress with `tail -f logs/*.log`

---

## Quick Reference

```bash
# Complete setup in one block (copy-paste)
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ~/OneDrive/Alpha\ Loop\ LLM/API\ -\ Dec\ 2025.env .env
python scripts/test_db_connection.py
```

**Built for Alpha Loop Capital - Institutional-Grade Long/Short Quant Hedge Fund**

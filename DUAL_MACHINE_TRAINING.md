# DUAL MACHINE TRAINING COORDINATION

## Overview

You have TWO machines - use them both to train FASTER!

| Machine | Best For | Why |
|---------|----------|-----|
| **Lenovo (Windows)** | Model Training | More CPU cores, better ML libraries |
| **MacBook** | Data Collection + Sentiment | Can run 24/7, good for I/O tasks |

---

## RECOMMENDED SPLIT

### TONIGHT'S SETUP (Do This Now)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LENOVO (WINDOWS)                              │
│                                                                  │
│  PRIMARY JOB: ML MODEL TRAINING                                 │
│                                                                  │
│  1. Double-click: START_OVERNIGHT_TRAINING.bat                  │
│  2. Leave running ALL NIGHT                                     │
│                                                                  │
│  Training these symbols:                                        │
│  ALL US EQUITIES, OPTIONS, ETFS, CRYPTO AND ECONOMIC DATA        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    MACBOOK                                       │
│                                                                  │
│  PRIMARY JOB: DATA COLLECTION                                   │
│                                                                  │
│  1. Open Terminal (NOT Cursor)                                  │
│  2. Run: caffeinate -d ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/      │
│          scripts/mac_data_collection.sh                         │
│  3. Leave running (will keep pulling fresh data)                │
│                                                                  │
│  Also good for: Research ingestion, sentiment analysis          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## STEP-BY-STEP TONIGHT

### On LENOVO (Windows) - Do First

1. **Open File Explorer** → Go to `C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1`
2. **Double-click** `START_OVERNIGHT_TRAINING.bat`
3. **DO NOT CLOSE** the black window that opens
4. Go to sleep

### On MACBOOK - Do Second (Optional but Recommended)

1. **Open Terminal** (Cmd+Space, type "Terminal")
2. Run these commands:

```bash
# Navigate to project
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1

# Make scripts executable (one time only)
chmod +x scripts/*.sh

# Start data collection (caffeinate prevents sleep)
caffeinate -d ./scripts/mac_data_collection.sh
```

---

## TOMORROW MORNING (9:15 AM)

### Check Training Results (Either Machine)

```
# Windows - Double-click:
CHECK_MODELS.bat

# Mac - Run:
ls -la ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/models/*.pkl
```

### Start Trading Engine (Pick ONE Machine)

**Recommended: Run trading on Lenovo (where models were trained)**

```
# Windows - Double-click:
START_TRADING_ENGINE.bat
```

OR on Mac:
```bash
./scripts/mac_trading_engine.sh
```

---

## ALTERNATIVE: SPLIT TRAINING BY SYMBOL

If you want BOTH machines training different symbols:

### Lenovo Trains These (Edit settings.py):
```python
target_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "META"]
```

### Mac Trains These (Edit settings.py on Mac):
```python
target_symbols = ["IWM", "DIA", "AMD", "TSLA", "AMZN", "BTC-USD", "ETH-USD"]
```

**IMPORTANT**: Both machines write to SAME SQL database, so models accumulate!

---

## PREVENTING SLEEP

### Windows (Lenovo)
The training script automatically disables sleep. But also:
1. Settings → System → Power & Sleep
2. Set "Sleep" to **Never** when plugged in

### Mac
Use `caffeinate` command (already in scripts). But also:
1. System Preferences → Battery → Power Adapter
2. Check "Prevent your Mac from sleeping automatically"

---

## MONITORING PROGRESS

### From Windows
```powershell
# Live log
Get-Content "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1\logs\overnight_robust.log" -Tail 20 -Wait

# Model count
(Get-ChildItem "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1\models\*.pkl").Count
```

### From Mac
```bash
# Live log
tail -f ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/logs/overnight_mac.log

# Model count
ls ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/models/*.pkl | wc -l
```

---

## IF SOMETHING CRASHES

### Windows
1. Training script auto-retries 5 times
2. If still failing, check `logs/overnight_robust.log`
3. Re-run `START_OVERNIGHT_TRAINING.bat`

### Mac
1. Check `logs/overnight_mac.log`
2. Re-run with: `caffeinate -d ./scripts/mac_overnight_training.sh`

---

## CHECKLIST FOR TONIGHT

### Before Bed:
- [ ] Lenovo plugged in
- [ ] MacBook plugged in
- [ ] Both machines on "Never Sleep" power setting
- [ ] `START_OVERNIGHT_TRAINING.bat` running on Lenovo
- [ ] (Optional) Data collection running on Mac
- [ ] .env file copied to project folder on both machines

### Morning (9:15 AM):
- [ ] Check models folder has .pkl files
- [ ] Start IBKR TWS/Gateway
- [ ] Run `START_TRADING_ENGINE.bat`
- [ ] Monitor first few trades

---

## FILES CREATED

```
scripts/
├── START_OVERNIGHT_TRAINING.bat   ← Windows: Double-click to train
├── START_DATA_COLLECTION.bat      ← Windows: Pull market data
├── START_TRADING_ENGINE.bat       ← Windows: Start trading at 9:15 AM
├── CHECK_MODELS.bat               ← Windows: See trained models
├── overnight_training_robust.ps1  ← Windows: PowerShell training script
├── mac_overnight_training.sh      ← Mac: Training script
├── mac_data_collection.sh         ← Mac: Data collection
└── mac_trading_engine.sh          ← Mac: Trading engine
```

---

## TL;DR FOR TONIGHT

1. **Lenovo**: Double-click `START_OVERNIGHT_TRAINING.bat` → Sleep
2. **Mac**: Run `caffeinate -d ./scripts/mac_data_collection.sh` → Sleep
3. **Morning**: Check `models/` folder → Run `START_TRADING_ENGINE.bat`

**That's it! Go make back that 10%!**


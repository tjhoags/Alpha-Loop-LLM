# Alpha Loop LLM - Algorithmic Trading System

## Mission Critical: Production Trading System

**This is the NEW, REFINED version** (`tjhoags/alpha-loop-llm`) built from lessons learned in:
- `/alc-algo` (original version)
- `/alc-algo-clean` (cleaned up version)
- Multiple iterations and improvements

This is a **sophisticated, institutional-grade algorithmic trading system** designed to run overnight training and execute trades by market open (9:30 AM ET).

---

## QUICK START

### Which Terminal to Use?

**You can use EITHER:**
- **Local Windows PowerShell** (Windows + X → Terminal)
- **Cursor's Integrated Terminal** (Terminal → New Terminal)

Both work the same way! Use whichever you prefer.

---

### Windows Setup

See `SETUP_WINDOWS.md` for complete Windows instructions.

**Quick start:**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\bek"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item "C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env" -Destination ".env"
python scripts/test_db_connection.py
```

### MacBook Setup

See `SETUP_MAC.md` for complete Mac instructions.

**Quick start:**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/bek
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ~/path/to/.env .env
python scripts/test_db_connection.py
```

### Running on Both Machines

See `MULTI_MACHINE_SETUP.md` for running on Windows + MacBook simultaneously.

---

## STARTING OVERNIGHT TRAINING

### Windows - Terminal 1: Data Collection
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\bek"
.\venv\Scripts\Activate.ps1
python src/data_ingestion/collector.py
```

### Windows - Terminal 2: Model Training
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\bek"
.\venv\Scripts\Activate.ps1
python src/ml/train_models.py
```

### MacBook - Terminal 1: Data Collection
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/bek
source venv/bin/activate
python src/data_ingestion/collector.py
```

### MacBook - Terminal 2: Model Training
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/bek
source venv/bin/activate
python src/ml/train_models.py
```

**Leave both running overnight!**

---

## TOMORROW MORNING (9:15 AM) - Start Trading

### Terminal 3 - Trading Engine:
```powershell
# Windows
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\bek"
.\venv\Scripts\Activate.ps1
python src/trading/execution_engine.py
```

```bash
# MacBook
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/bek
source venv/bin/activate
python src/trading/execution_engine.py
```

**Start this at 9:15 AM** - ready for 9:30 AM market open!

---

## CHECKING PROGRESS

### View Logs (Windows):
```powershell
# Data collection
Get-Content logs\data_collection.log -Tail 50

# Model training
Get-Content logs\model_training.log -Tail 50

# Trading engine
Get-Content logs\trading_engine.log -Tail 50
```

### View Logs (MacBook):
```bash
# Data collection
tail -f logs/data_collection.log

# Model training
tail -f logs/model_training.log

# Trading engine
tail -f logs/trading_engine.log
```

### Check Models:
```powershell
# Windows
Get-ChildItem models\*.pkl
```

```bash
# MacBook
ls -la models/*.pkl
```

---

## System Architecture

```
alpha-loop-llm/
├── src/
│   ├── config/          # Configuration management (Pydantic)
│   ├── data_ingestion/  # Multi-source data collection
│   ├── database/        # Azure SQL Server integration
│   ├── ml/              # ML models & feature engineering
│   ├── trading/         # Execution engine & order management
│   ├── risk/            # Risk management & position sizing
│   └── monitoring/      # Logging and alerts
├── scripts/             # Utility scripts & helpers
├── models/              # Trained model files (.pkl)
├── data/                # Market data storage
├── logs/                # System logs
├── .env                 # API keys (DO NOT COMMIT)
└── requirements.txt     # Python dependencies
```

---

## INSTITUTIONAL-GRADE FEATURES

### Advanced ML Pipeline
- **100+ Features**: Price, technical indicators, volume, volatility, momentum, microstructure
- **Ensemble Models**: XGBoost, LightGBM, CatBoost
- **Time-Series CV**: No data leakage
- **Model Versioning**: Timestamped saves

### Risk Management
- **Kelly Criterion**: Position sizing with confidence weighting
- **Daily Loss Limits**: 2% max daily loss
- **Drawdown Protection**: 5% max drawdown
- **Position Limits**: Max 10 positions, 10% per position

### Data Infrastructure
- **Multi-Source**: Alpha Vantage, Polygon, Coinbase
- **Azure SQL**: Centralized data storage
- **Retry Logic**: Exponential backoff
- **Rate Limiting**: Automatic handling

### Trading Execution
- **Interactive Brokers**: Full integration
- **Paper Trading**: Safe testing (IBKR_PORT=7497)
- **Order Management**: Market orders with fill tracking
- **Real-Time Signals**: ML-based with confidence

---

## DOCUMENTATION

- **`SETUP_WINDOWS.md`** - Complete Windows setup guide
- **`SETUP_MAC.md`** - Complete MacBook setup guide
- **`MULTI_MACHINE_SETUP.md`** - Running on both machines
- **`NEXT_STEPS.md`** - Step-by-step action items
- **`TERMINAL_GUIDE.md`** - Detailed terminal commands
- **`QUICK_START.md`** - Quick reference
- **`SETUP_COMPLETE.md`** - Complete setup guide
- **`CONFIRMATION.md`** - Confirms this is the new repo
- **`INSTITUTIONAL_GRADE_CHECKLIST.md`** - Feature checklist

---

## TROUBLESHOOTING

### "Module not found"
→ Activate venv: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (Mac)

### "Execution policy error" (Windows)
→ Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Database connection fails
→ Check `.env` file has correct SQL credentials

### "No models found"
→ Make sure model training completed (check `models/` folder)

### API rate limits
→ System handles automatically with retries

### Mac goes to sleep
→ Use: `caffeinate -d python src/ml/train_models.py`

---

## IMPORTANT NOTES

1. **Never commit `.env` file** - Contains API keys
2. **Test in paper trading first** - IBKR_PORT=7497 is paper trading
3. **Monitor logs** - Check `logs/` folder for errors
4. **Start data collection early** - Needs time to gather historical data
5. **Keep machines awake** - Use caffeinate on Mac, power settings on Windows

---

## VERIFICATION CHECKLIST

Before starting trading:
- [ ] Virtual environment created and activated
- [ ] All packages installed
- [ ] `.env` file copied to project folder
- [ ] Database connection test passed
- [ ] Data collection ran overnight
- [ ] Models trained (check `models/` folder)
- [ ] Trading engine starts without errors

---

## SUCCESS CRITERIA

You'll know everything is working when:

1. Data collection logs show "Collecting data for..." messages
2. Model training logs show "Training XGBoost...", "Training LightGBM..." messages
3. `models/` folder has `.pkl` files after training completes
4. Trading engine logs show "Loaded X models" when starting
5. Trading engine shows "Starting trading engine" at 9:30 AM

---

**Built for Alpha Loop Capital - Institutional Grade Trading System**

**This is the FINAL, PRODUCTION-READY version**


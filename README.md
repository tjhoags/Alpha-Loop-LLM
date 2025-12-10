# Alpha Loop LLM - Algorithmic Trading System

## Mission Critical: Production Trading System

**This is the NEW, REFINED version** (`tjhoags/alpha-loop-llm`) built from lessons learned in:
- `/alc-algo` (original version)
- `/alc-algo-clean` (cleaned up version)
- Multiple iterations and improvements

This is a **sophisticated, institutional-grade algorithmic trading system** designed to run overnight training and execute trades by market open (9:30 AM ET).

---

## 📚 DOCUMENTATION INDEX

| Document | Description |
|----------|-------------|
| **`CROSS_PLATFORM_COMMANDS.md`** | ⭐ Complete natural language guide for ALL commands |
| `SETUP_WINDOWS.md` | Windows-specific setup details |
| `SETUP_MAC.md` | MacBook-specific setup details |
| `MULTI_MACHINE_SETUP.md` | Running on both machines simultaneously |
| `TERMINAL_COMMANDS.md` | Quick terminal command reference |
| `FULL_THROTTLE_SETUP.md` | Maximum data ingestion guide |
| `TRAINING_GUIDE.md` | ML model training details |

---

## 🚀 QUICK START

### Step 1: Open Your Terminal

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

**In Plain English:** "Open a command window where you can type instructions"

1. Press `Windows + X` on your keyboard
2. Click "Terminal" or "Windows PowerShell"
3. A window opens with a prompt like `PS C:\Users\tom>`

**Or in Cursor IDE:** Press `Ctrl + ~` or go to Terminal → New Terminal
</details>

<details>
<summary><b>🍎 MacBook Pro (Terminal)</b></summary>

**In Plain English:** "Open a command window where you can type instructions"

1. Press `Cmd + Space` to open Spotlight
2. Type "Terminal" and press Enter
3. A window opens with a prompt like `tom@MacBook-Pro ~ %`

**Or in Cursor IDE:** Press `Cmd + ~` or go to Terminal → New Terminal
</details>

---

### Step 2: Navigate to Project

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

**In Plain English:** "Go to the folder where all the code lives"

```powershell
# Type this and press Enter:
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"

# Verify you're in the right place:
dir
```
</details>

<details>
<summary><b>🍎 MacBook Pro (Terminal)</b></summary>

**In Plain English:** "Go to the folder where all the code lives"

```bash
# Type this and press Enter:
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii

# Verify you're in the right place:
ls
```
</details>

---

### Step 3: Set Up Virtual Environment

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

**In Plain English:** "Create an isolated Python workspace for this project"

```powershell
# Create the virtual environment (one-time):
python -m venv venv

# Activate it (do this every time you open a new terminal):
.\venv\Scripts\Activate.ps1

# If you get an "execution policy" error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Success:** You'll see `(venv)` at the start of your prompt
</details>

<details>
<summary><b>🍎 MacBook Pro (Terminal)</b></summary>

**In Plain English:** "Create an isolated Python workspace for this project"

```bash
# Create the virtual environment (one-time):
python3 -m venv venv

# Activate it (do this every time you open a new terminal):
source venv/bin/activate
```

**Success:** You'll see `(venv)` at the start of your prompt
</details>

---

### Step 4: Install Dependencies

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

**In Plain English:** "Install all the required Python packages"

```powershell
pip install -r requirements.txt
```
</details>

<details>
<summary><b>🍎 MacBook Pro (Terminal)</b></summary>

**In Plain English:** "Install all the required Python packages"

```bash
pip install -r requirements.txt
```
</details>

---

### Step 5: Configure Environment

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

**In Plain English:** "Copy your API keys and database credentials to the project"

```powershell
Copy-Item "C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env" -Destination ".env"
```
</details>

<details>
<summary><b>🍎 MacBook Pro (Terminal)</b></summary>

**In Plain English:** "Copy your API keys and database credentials to the project"

```bash
cp ~/OneDrive/Alpha\ Loop\ LLM/API\ -\ Dec\ 2025.env .env
```
</details>

---

### Step 6: Test Database Connection

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

**In Plain English:** "Make sure we can connect to the database"

```powershell
python scripts/test_db_connection.py
```
</details>

<details>
<summary><b>🍎 MacBook Pro (Terminal)</b></summary>

**In Plain English:** "Make sure we can connect to the database"

```bash
python scripts/test_db_connection.py
```
</details>

---

## 📊 RUNNING THE SYSTEM

### Full Command Reference: See [`CROSS_PLATFORM_COMMANDS.md`](CROSS_PLATFORM_COMMANDS.md)

### Running on Both Machines

See `MULTI_MACHINE_SETUP.md` for running on Windows + MacBook simultaneously.

---

## 🌙 OVERNIGHT TRAINING

### Terminal 1: Data Collection

**In Plain English:** "Start pulling market data from all sources"

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1
python src/data_ingestion/collector.py
```
</details>

<details>
<summary><b>🍎 MacBook Pro (Terminal)</b></summary>

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
caffeinate -d python src/data_ingestion/collector.py
```
**Note:** `caffeinate -d` prevents your Mac from sleeping
</details>

### Terminal 2: Model Training

**In Plain English:** "Train machine learning models on the collected data"

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1
python src/ml/train_models.py
```
</details>

<details>
<summary><b>🍎 MacBook Pro (Terminal)</b></summary>

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
caffeinate -d python src/ml/train_models.py
```
</details>

**Leave both running overnight!**

---

## ☀️ MORNING (9:15 AM ET) - Start Trading

**In Plain English:** "Start the trading engine that will execute trades at market open"

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1
python src/trading/execution_engine.py
```
</details>

<details>
<summary><b>🍎 MacBook Pro (Terminal)</b></summary>

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
python src/trading/execution_engine.py
```
</details>

**Prerequisites:** IBKR TWS/Gateway running (paper: port 7497, live: port 7496)

---

## 📊 MONITORING

### View Logs

**In Plain English:** "Watch what the system is doing in real-time"

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
# Watch data collection (live updates):
Get-Content logs\data_collection.log -Tail 50 -Wait

# Watch model training (live updates):
Get-Content logs\model_training.log -Tail 50 -Wait

# Watch trading engine (live updates):
Get-Content logs\trading_engine.log -Tail 50 -Wait

# Check how many models have been trained:
(Get-ChildItem models\*.pkl).Count
```
</details>

<details>
<summary><b>🍎 MacBook Pro (Terminal)</b></summary>

```bash
# Watch data collection (live updates):
tail -f logs/data_collection.log

# Watch model training (live updates):
tail -f logs/model_training.log

# Watch trading engine (live updates):
tail -f logs/trading_engine.log

# Check how many models have been trained:
ls models/*.pkl | wc -l
```
</details>

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


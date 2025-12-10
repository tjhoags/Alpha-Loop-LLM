# TERMINAL COMMANDS - WHAT TO RUN WHERE

You have 5 terminals. Here's exactly what to run in each:

---

## CURRENT STATUS

**Data in Azure SQL:** 3,400+ symbols, 1.4M+ rows
**Asset Types Being Hydrated:**
- Stocks: ALL (5,245) - RUNNING
- ETFs: ALL (2,500+) - After stocks complete
- Crypto: Top 100
- Forex: 50 major pairs  
- Options: Top 50 underlyings

---

## LENOVO (WINDOWS) - 3 TERMINALS

### TERMINAL 1: DATA HYDRATION (Already Running)
```powershell
# This should already be running - check with:
Get-Content logs/hydration_azure.log -Tail 5

# If not running, start with:
cd "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
.\venv\Scripts\activate
python scripts/hydrate_full_universe.py 2>&1 | Tee-Object -FilePath logs/hydration_azure.log
```
**Status:** Pulling stocks (batch 35/53), then ETFs, crypto, forex, options

---

### TERMINAL 2: ML TRAINING - STOCKS (Start Now)
```powershell
cd "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
.\venv\Scripts\activate
python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()" 2>&1 | Tee-Object -FilePath logs/training_stocks.log
```
**What it trains:** All 3,400+ stock symbols in database
**Runtime:** 4-8 hours

---

### TERMINAL 3: MONITOR / DASHBOARD
```powershell
cd "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
.\venv\Scripts\activate

# Check hydration progress
Get-Content logs/hydration_azure.log -Tail 10

# Check training progress  
Get-Content logs/training_stocks.log -Tail 10

# Check model count
(Get-ChildItem models\*.pkl -ErrorAction SilentlyContinue).Count

# Check Azure SQL row count
python -c "import pyodbc; conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};Server=alc-sql-server.database.windows.net;Database=alc_market_data;UID=CloudSAb3fcbb35;PWD=ALCadmin27!'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM price_bars'); print(f'Rows: {cursor.fetchone()[0]:,}'); conn.close()"

# Full dashboard
python scripts/model_dashboard.py
```

---

## MAC - 2 TERMINALS  

### TERMINAL 4: ADDITIONAL DATA SOURCES
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate

# Pull research/sentiment data
caffeinate -d python scripts/ingest_research.py 2>&1 | tee logs/research_mac.log
```

---

### TERMINAL 5: BACKUP TRAINING (Optional)
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate

# If you want Mac to help train too:
caffeinate -d python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()" 2>&1 | tee logs/training_mac.log
```

---

## WHAT'S BEING TRAINED

| Asset Type | Count | Status |
|------------|-------|--------|
| **US Stocks** | 3,400+ | In database, training |
| **ETFs** | 2,500+ | Hydrating next |
| **Crypto** | 100 | Hydrating after ETFs |
| **Forex** | 50 | Hydrating after crypto |
| **Options** | Top 50 | Hydrating last |

---

## QUICK STATUS COMMANDS

### Windows (PowerShell):
```powershell
# How many models trained?
(Get-ChildItem C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1\models\*.pkl).Count

# How many rows in SQL?
python -c "import pyodbc; c=pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};Server=alc-sql-server.database.windows.net;Database=alc_market_data;UID=CloudSAb3fcbb35;PWD=ALCadmin27!').cursor(); c.execute('SELECT COUNT(*) FROM price_bars'); print(c.fetchone()[0])"

# What batch is hydration on?
Get-Content logs/hydration_azure.log -Tail 3
```

### Mac (Terminal):
```bash
# How many models?
ls ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/models/*.pkl 2>/dev/null | wc -l

# Training log
tail -f ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/logs/training_mac.log
```

---

## OVERNIGHT TIMELINE

| Time | What's Happening |
|------|------------------|
| Now | Hydration running, start training |
| +1 hour | First models finishing |
| +3 hours | 500+ models trained |
| +6 hours | Most stocks complete |
| 9:00 AM | Check model grades |
| 9:30 AM | Start trading engine |

---

## TL;DR - COPY PASTE NOW

**Terminal 1 (Windows):** Already running hydration - leave it

**Terminal 2 (Windows):** Run this NOW:
```powershell
cd "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"; .\venv\Scripts\activate; python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()" 2>&1 | Tee-Object -FilePath logs/training_stocks.log
```

**Terminal 3 (Windows):** Monitor dashboard

**Terminal 4 (Mac):** Research ingestion

**Terminal 5 (Mac):** Backup training or monitor

---

## YES - HYDRATION INCLUDES EVERYTHING

The hydration script pulls:
- ALL US Stocks (no limit)
- ALL ETFs (no limit)  
- Top 100 Crypto
- 50 Major Forex pairs
- Top 50 Options underlyings

**You're getting institutional-grade data coverage!**



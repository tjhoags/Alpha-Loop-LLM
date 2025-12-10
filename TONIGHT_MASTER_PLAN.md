# MASTER TRAINING PLAN - TONIGHT

## COORDINATION BETWEEN ALL AGENTS

This plan maximizes compute across BOTH machines with multiple terminals.

---

## MACHINE ASSIGNMENTS

```
LENOVO (WINDOWS) - PRIMARY TRAINING MACHINE
============================================
Terminal 1: ML Training (Indices + Large Cap)
Terminal 2: ML Training (Tech + Growth)
Terminal 3: Data Collection (continuous)

MACBOOK - SECONDARY MACHINE  
============================================
Terminal 1: ML Training (Mid Cap + Crypto)
Terminal 2: Sentiment/NLP Processing
Terminal 3: Research Ingestion
```

---

## STEP-BY-STEP INSTRUCTIONS

### LENOVO TERMINAL 1 - Indices & Large Cap Training

Open PowerShell (Win+X -> Terminal), then run:

```powershell
cd "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
.\venv\Scripts\Activate.ps1

# Set symbols for this terminal
$env:TRAINING_SYMBOLS = "SPY,QQQ,IWM,DIA,AAPL,MSFT"

# Prevent sleep
powercfg -change -standby-timeout-ac 0

# Run training
python -c "
import os
os.environ['TRAINING_SYMBOLS'] = 'SPY,QQQ,IWM,DIA,AAPL,MSFT'
from src.ml.advanced_training import AdvancedTrainer, TrainingConfig
config = TrainingConfig(xgb_n_estimators=800, lgb_n_estimators=800, cat_iterations=800)
trainer = AdvancedTrainer(config)
symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT']
trainer.train_all_parallel(symbols, max_workers=2)
"
```

### LENOVO TERMINAL 2 - Tech & Growth Training

Open ANOTHER PowerShell window, then run:

```powershell
cd "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
.\venv\Scripts\Activate.ps1

# Run training for tech stocks
python -c "
from src.ml.advanced_training import AdvancedTrainer, TrainingConfig
config = TrainingConfig(xgb_n_estimators=800, lgb_n_estimators=800, cat_iterations=800)
trainer = AdvancedTrainer(config)
symbols = ['NVDA', 'AMD', 'GOOGL', 'META', 'TSLA', 'AMZN']
trainer.train_all_parallel(symbols, max_workers=2)
"
```

### LENOVO TERMINAL 3 - Data Collection (Background)

Open ANOTHER PowerShell window, then run:

```powershell
cd "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
.\venv\Scripts\Activate.ps1

# Continuous data collection loop
python -c "
import time
from src.data_ingestion.collector import main as collect_data
while True:
    print('Starting data collection cycle...')
    try:
        collect_data()
    except Exception as e:
        print(f'Collection error: {e}')
    print('Sleeping 30 minutes before next collection...')
    time.sleep(1800)
"
```

---

### MACBOOK TERMINAL 1 - Crypto & Additional Training

Open Terminal app (not in Cursor), then run:

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate

# Prevent sleep with caffeinate
caffeinate -d python -c "
from src.ml.advanced_training import AdvancedTrainer, TrainingConfig
config = TrainingConfig(xgb_n_estimators=800, lgb_n_estimators=800, cat_iterations=800)
trainer = AdvancedTrainer(config)
symbols = ['BTC-USD', 'ETH-USD']
trainer.train_all_parallel(symbols, max_workers=2)
"
```

### MACBOOK TERMINAL 2 - Sentiment Processing

Open ANOTHER Terminal window, then run:

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate

caffeinate -d python -c "
from src.nlp.sentiment import _get_pipeline
print('Loading FinBERT sentiment model...')
pipeline = _get_pipeline()
print('Sentiment model loaded and cached!')
print('Ready for sentiment analysis.')
"
```

### MACBOOK TERMINAL 3 - Research Ingestion

Open ANOTHER Terminal window, then run:

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate

caffeinate -d python scripts/ingest_research.py
```

---

## SUMMARY TABLE

| Machine | Terminal | Task | Symbols/Focus |
|---------|----------|------|---------------|
| Lenovo | 1 | ML Training | SPY, QQQ, IWM, DIA, AAPL, MSFT |
| Lenovo | 2 | ML Training | NVDA, AMD, GOOGL, META, TSLA, AMZN |
| Lenovo | 3 | Data Collection | All symbols (continuous) |
| Mac | 1 | ML Training | BTC-USD, ETH-USD |
| Mac | 2 | Sentiment Model | Load and cache FinBERT |
| Mac | 3 | Research Ingestion | Dropbox documents |

---

## TIMELINE

```
NOW (10:00 PM):
  [x] Start Lenovo Terminal 1 (Indices training)
  [x] Start Lenovo Terminal 2 (Tech training)  
  [x] Start Lenovo Terminal 3 (Data collection)
  [x] Start Mac Terminal 1 (Crypto training)
  [x] Start Mac Terminal 2 (Sentiment)
  [x] Start Mac Terminal 3 (Research)

OVERNIGHT:
  - All terminals running
  - Models saving to models/ folder
  - Logs saving to logs/ folder

MORNING (9:00 AM):
  - Check models: dir models\*.pkl
  - Should see 30+ model files
  
MORNING (9:15 AM):
  - Start IBKR TWS/Gateway
  - Run: python src/trading/execution_engine.py

MARKET OPEN (9:30 AM):
  - Trading engine generates signals
  - Executes trades via IBKR
```

---

## QUICK COMMANDS TO CHECK PROGRESS

### On Lenovo (PowerShell):
```powershell
# Count models trained so far
(Get-ChildItem "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1\models\*.pkl").Count

# View recent log entries
Get-Content "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1\logs\overnight_training.log" -Tail 20
```

### On Mac (Terminal):
```bash
# Count models
ls ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/models/*.pkl | wc -l

# View logs
tail -20 ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/logs/overnight_training.log
```

---

## EXPECTED RESULTS BY MORNING

| Metric | Expected |
|--------|----------|
| Total Models | 30-42 (3 models x 14 symbols) |
| Pass Rate | 50-70% |
| Training Time | 5-7 hours |
| Data Points | 100,000+ per symbol |

---

## IF SOMETHING CRASHES

1. Check which terminal crashed
2. Look at the error message
3. Re-run that specific command
4. Other terminals keep running independently

---

## IMPORTANT REMINDERS

1. DO NOT close the terminal windows
2. Keep both machines plugged in
3. Disable sleep on both machines
4. Each terminal runs independently
5. All models save to same models/ folder
6. SQL database is shared between machines


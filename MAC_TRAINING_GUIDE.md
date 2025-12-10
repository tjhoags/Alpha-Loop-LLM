# MAC TRAINING GUIDE - FULL SYSTEM

Since Cursor is stable on Mac, this is your PRIMARY training machine.

---

## QUICK START (Copy-Paste Ready)

Open Terminal on Mac and run these commands in order:

```bash
# SETUP (one time)
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
chmod +x scripts/*.sh
```

---

## TERMINAL LAYOUT - RUN ALL 4 SIMULTANEOUSLY

Open 4 Terminal tabs/windows (Cmd+T for new tab):

```
+---------------------------+---------------------------+
|   TERMINAL 1              |   TERMINAL 2              |
|   DATA HYDRATION          |   ML TRAINING             |
|   (Azure SQL)             |   (Core Models)           |
+---------------------------+---------------------------+
|   TERMINAL 3              |   TERMINAL 4              |
|   MASSIVE TRAINING        |   SENTIMENT/RESEARCH      |
|   (Full Universe)         |   (NLP Pipeline)          |
+---------------------------+---------------------------+
```

---

## TERMINAL 1: DATA HYDRATION (Start First)

```bash
# Navigate and activate
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate

# Prevent Mac from sleeping + start hydration
caffeinate -d python scripts/hydrate_full_universe.py 2>&1 | tee logs/hydration_mac.log
```

**What it does:**
- Pulls 5,245+ stocks from Polygon
- Pulls all ETFs, crypto, forex
- Writes directly to Azure SQL
- Also saves CSV backups locally

**Runtime:** ~30-45 minutes for initial load

---

## TERMINAL 2: CORE ML TRAINING (Start After Hydration Begins)

```bash
# Navigate and activate
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate

# Run overnight training with restart protection
caffeinate -d ./scripts/mac_overnight_training.sh
```

**What it trains:**
- XGBoost ensemble models
- LightGBM ensemble models  
- CatBoost ensemble models
- Time-series cross-validation
- 100+ technical features per symbol

**Runtime:** 4-8 hours depending on universe size

---

## TERMINAL 3: MASSIVE PARALLEL TRAINING (For Full Universe)

```bash
# Navigate and activate
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate

# Run massive trainer (all symbols in parallel)
caffeinate -d python -c "
from src.ml.massive_trainer import MassiveTrainer
trainer = MassiveTrainer(max_workers=4)  # Adjust based on Mac CPU cores
trainer.train_full_universe()
" 2>&1 | tee logs/massive_training_mac.log
```

**What it does:**
- Trains models for EVERY symbol in database
- Uses parallel processing (4 workers)
- Grades models and only keeps good ones
- Spawns agent mutations (ACA)

**Runtime:** 6-12 hours for full universe

---

## TERMINAL 4: SENTIMENT & RESEARCH INGESTION

```bash
# Navigate and activate
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate

# Run research ingestion (builds vector store)
caffeinate -d python scripts/ingest_research.py 2>&1 | tee logs/research_mac.log
```

**What it does:**
- Ingests all research PDFs from Dropbox
- Builds FAISS vector store for RAG
- Runs sentiment analysis on news/filings
- Creates behavioral feature dataset

**Runtime:** 1-2 hours depending on research volume

---

## ALL TRAINING MODULES EXPLAINED

| Module | File | Purpose |
|--------|------|---------|
| **Core Training** | `src/ml/train_models.py` | Single symbol model training |
| **Advanced Training** | `src/ml/advanced_training.py` | Overnight orchestration with grading |
| **Massive Training** | `src/ml/massive_trainer.py` | Full universe parallel training |
| **Feature Engineering** | `src/ml/feature_engineering.py` | 100+ technical indicators |
| **Behavioral Features** | `src/ml/behavioral_features.py` | Psychology/sentiment features |
| **Valuation Metrics** | `src/ml/valuation_metrics.py` | DCF, LBO, M&A features |
| **Ensemble Models** | `src/ml/models.py` | XGBoost, LightGBM, CatBoost |

---

## MODEL GRADING (How to Know if Training is Working)

Models are graded on these metrics:

| Metric | Minimum | Excellent | What It Means |
|--------|---------|-----------|---------------|
| **AUC** | 0.52 | 0.65+ | Model beats random |
| **Accuracy** | 0.52 | 0.58+ | Correct predictions |
| **Sharpe Ratio** | 1.5 | 2.5+ | Risk-adjusted returns |
| **Max Drawdown** | < 5% | < 3% | Worst loss period |

**Only models that pass ALL thresholds get promoted to production!**

---

## CHECKING PROGRESS

### Check Hydration Progress
```bash
# Row count in Azure SQL
python -c "
import pyodbc
conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};Server=alc-sql-server.database.windows.net;Database=alc_market_data;UID=CloudSAb3fcbb35;PWD=ALCadmin27!')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM price_bars')
print(f'Rows in Azure SQL: {cursor.fetchone()[0]:,}')
"
```

### Check Training Progress
```bash
# View training log
tail -f logs/overnight_mac.log

# Count models
ls models/*.pkl 2>/dev/null | wc -l
```

### Check Model Quality
```bash
python -c "
import pickle
from pathlib import Path
for f in Path('models').glob('*.pkl'):
    try:
        model = pickle.load(open(f, 'rb'))
        if hasattr(model, 'grade'):
            print(f'{f.name}: AUC={model.grade.get(\"auc\", \"N/A\")}, Sharpe={model.grade.get(\"sharpe\", \"N/A\")}')
    except:
        pass
"
```

---

## OVERNIGHT CHECKLIST

### Before Bed:
- [ ] Mac plugged in to power
- [ ] System Preferences > Battery > "Prevent automatic sleeping" ON
- [ ] Terminal 1: Hydration running
- [ ] Terminal 2: ML training running  
- [ ] Terminal 3: Massive training running (optional)
- [ ] Terminal 4: Research ingestion running (optional)

### Morning 9:00 AM:
- [ ] Check logs for errors: `tail -100 logs/*.log`
- [ ] Count models: `ls models/*.pkl | wc -l`
- [ ] Check Azure SQL rows
- [ ] Ready for trading at 9:30 AM

---

## ONE-LINER TO START EVERYTHING

If you want ONE command that does it all:

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1 && source venv/bin/activate && caffeinate -d bash -c "
echo 'Starting hydration...'
python scripts/hydrate_full_universe.py &
sleep 60
echo 'Starting training...'
python -c 'from src.ml.advanced_training import run_overnight_training; run_overnight_training()' &
echo 'Starting research ingestion...'
python scripts/ingest_research.py &
wait
echo 'ALL COMPLETE!'
" 2>&1 | tee logs/full_overnight_mac.log
```

---

## IF MAC RUNS OUT OF MEMORY

Reduce parallel workers:

```bash
# Edit and reduce workers
export ML_MAX_WORKERS=2

# Or edit directly in Python
python -c "
from src.ml.massive_trainer import MassiveTrainer
trainer = MassiveTrainer(max_workers=2)  # Reduced from 4
trainer.train_full_universe()
"
```

---

## SYNC WITH LENOVO

Both machines write to the SAME Azure SQL database, so:
- Data collected on either machine is available to both
- Models trained on Mac can be used on Lenovo
- Copy `models/*.pkl` between machines if needed

To sync models via Dropbox:
```bash
# Mac: Copy models to sync folder
cp models/*.pkl ~/Dropbox/ALC_Models/

# Lenovo: Pull models from sync folder
copy "%USERPROFILE%\Dropbox\ALC_Models\*.pkl" "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1\models\"
```

---

## TL;DR

1. Open 4 Terminal tabs on Mac
2. Run commands from Terminals 1-4 above
3. Go to sleep
4. Wake up to trained models
5. Run trading engine at 9:30 AM

**Your Mac is now a hedge fund training cluster!**



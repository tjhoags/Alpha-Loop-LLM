## Mac Setup & Overnight Training Guide

This is for the new `tjhoags/alpha-loop-llm` repo. Follow these steps exactly on your Mac. This pipeline ingests market data into SQL, engineers features, trains ensemble models (XGBoost/LightGBM/CatBoost), and writes models to `models/`.

### 0) Prereqs
- Ensure Python 3.9+ is installed (`python3 --version`).
- Keep the Mac awake overnight (use `caffeinate -d` or Energy Saver settings).

### 1) Open Terminal and go to the project
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
```
Adjust the path if your checkout is elsewhere.

### 2) Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Copy your .env (premium API + SQL credentials)
```bash
cp ~/OneDrive/Alpha\ Loop\ LLM/API\ -\ Dec\ 2025.env .env
```
If your .env lives elsewhere, change the source path accordingly.

### 4) Install dependencies
```bash
pip install -r requirements.txt
```

### 5) Smoke-test SQL connectivity
```bash
python scripts/test_db_connection.py
```
This hits your SQL using the `database_url` or `DB_*` values in `.env` and logs to `logs/test_db_connection.log`.

### 6) Start overnight processes (two terminals)
**Terminal A – Data collection (writes to SQL table `price_bars`):**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate
caffeinate -d python src/data_ingestion/collector.py
```

**Terminal B – Model training (reads `price_bars`, writes `.pkl` models):**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate
caffeinate -d python src/ml/train_models.py
```

Leave both running overnight. Data is continuously written to SQL; training will run through available history and save models under `models/`.

### 7) Morning trading (after models exist)
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate
python src/trading/execution_engine.py
```

### What data is used and where it goes
- **Sources:** Polygon (minute bars), Alpha Vantage (intraday), Coinbase (BTC-USD candles). You have premium keys, so the scripts will pull the fullest range allowed.
- **Destination:** Azure SQL via SQLAlchemy/pyodbc. Table used: `price_bars`.
- **Schema (implicit):** symbol, timestamp, open, high, low, close, volume, source.
- **Feature set:** Technical indicators (returns, volatility, volume z-score, RSI, EMAs, MACD).
- **Labels:** Binary “future return > 0” over the configured horizon.
- **Models trained:** XGBoost, LightGBM, CatBoost (classification). Not RL/agents—supervised ensemble classifiers for directional signals.

### Key environment variables (from `.env`)
- DB connection (`database_url` or `DB_SERVER`, `DB_DATABASE`, `DB_USERNAME`, `DB_PASSWORD`, `DB_ODBC_DRIVER`)
- API keys (`ALPHA_VANTAGE_API_KEY`, `POLYGON_API_KEY`, `COINBASE_API_KEY`)
- IBKR paper/live (`IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`)

### Monitoring
- Logs: `logs/data_collection.log`, `logs/model_training.log`, `logs/test_db_connection.log`.
- Models: `models/*.pkl` (each includes metadata like CV metrics and timestamp).

### Tips for “as much data as possible”
- Keep data collection running continuously; Polygon and Alpha Vantage full/extended intraday pull as much as your plan allows.
- You can rerun `collector.py` multiple times to backfill; it appends to `price_bars` and deduplicates latest-per-timestamp.
- For more symbols, add them to `target_symbols` in `src/config/settings.py` or via environment variables.

### If you need to backfill fast
- Run `collector.py` once per symbol set; consider increasing `lookback_hours` in `polygon.fetch_aggregates` or using Polygon’s full-range endpoints if available on your tier.
- Ensure SQL has enough storage; the ingestion uses append mode.

### Common issues
- **Import/module errors:** Make sure `source venv/bin/activate` is active in each terminal.
- **SQL connect fails:** Re-check `.env` values; confirm network/firewall access to SQL.
- **No models produced:** Ensure `price_bars` has data (check in SQL) and that `train_models.py` finds enough rows (needs a few hundred+).



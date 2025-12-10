## ac etup & vernight raining uide

his is for the new `tjhoags/alpha-loop-llm` repo. ollow these steps exactly on your ac. his pipeline ingests market data into , engineers features, trains ensemble models (oost/ight/atoost), and writes models to `models/`.

### ) rereqs
- nsure ython .+ is installed (`python --version`).
- eep the ac awake overnight (use `caffeinate -d` or nergy aver settings).

### ) pen erminal and go to the project
```bash
cd ~/lpha-oop-/lpha-oop--
```
djust the path if your checkout is elsewhere.

### ) reate and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### ) opy your .env (premium  +  credentials)
```bash
cp ~/lphaloopcapital ropbox/ ech gents/ - ec .env .env
```
f your .env lives elsewhere, change the source path accordingly.

### ) nstall dependencies
```bash
pip install -r requirements.txt
```

### ) moke-test  connectivity
```bash
python scripts/test_db_connection.py
```
his hits your  using the `database_url` or `_*` values in `.env` and logs to `logs/test_db_connection.log`.

### ) tart overnight processes (two terminals)
**erminal  – ata collection (writes to  table `price_bars`)**
```bash
cd ~/lpha-oop-/lpha-oop--
source venv/bin/activate
caffeinate -d python src/data_ingestion/collector.py
```

**erminal  – odel training (reads `price_bars`, writes `.pkl` models)**
```bash
cd ~/lpha-oop-/lpha-oop--
source venv/bin/activate
caffeinate -d python src/ml/train_models.py
```

eave both running overnight. ata is continuously written to  training will run through available history and save models under `models/`.

### ) orning trading (after models exist)
```bash
cd ~/lpha-oop-/lpha-oop--
source venv/bin/activate
python src/trading/execution_engine.py
```

### hat data is used and where it goes
- **ources** olygon (minute bars), lpha antage (intraday), oinbase (- candles). ou have premium keys, so the scripts will pull the fullest range allowed.
- **estination** zure  via lchemy/pyodbc. able used `price_bars`.
- **chema (implicit)** symbol, timestamp, open, high, low, close, volume, source.
- **eature set** echnical indicators (returns, volatility, volume z-score, , s, ).
- **abels** inary “future return  ” over the configured horizon.
- **odels trained** oost, ight, atoost (classification). ot /agents—supervised ensemble classifiers for directional signals.

### ey environment variables (from `.env`)
-  connection (`database_url` or `_`, `_`, `_`, `_`, `__`)
-  keys (`___`, `__`, `__`)
-  paper/live (`_`, `_`, `__`)

### onitoring
- ogs `logs/data_collection.log`, `logs/model_training.log`, `logs/test_db_connection.log`.
- odels `models/*.pkl` (each includes metadata like  metrics and timestamp).

### ips for “as much data as possible”
- eep data collection running continuously olygon and lpha antage full/extended intraday pull as much as your plan allows.
- ou can rerun `collector.py` multiple times to backfill it appends to `price_bars` and deduplicates latest-per-timestamp.
- or more symbols, add them to `target_symbols` in `src/config/settings.py` or via environment variables.

### f you need to backfill fast
- un `collector.py` once per symbol set consider increasing `lookback_hours` in `polygon.fetch_aggregates` or using olygon’s full-range endpoints if available on your tier.
- nsure  has enough storage the ingestion uses append mode.

### ommon issues
- **mport/module errors** ake sure `source venv/bin/activate` is active in each terminal.
- ** connect fails** e-check `.env` values confirm network/firewall access to .
- **o models produced** nsure `price_bars` has data (check in ) and that `train_models.py` finds enough rows (needs a few hundred+).



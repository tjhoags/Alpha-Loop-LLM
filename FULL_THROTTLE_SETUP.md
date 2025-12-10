#     -   

##    (     )

### indows (owerhell)
```powershell
cd "serstomlpha-oop-lpha-oop--"
python -m venv venv
.venvcriptsctivate.ps
pip install -r requirements.txt
opy-tem "serstomlphaloopcapital ropbox ech gents - ec .env" -estination ".env"
python scripts/test_db_connection.py

#    (opens  terminals automatically)
.scriptsstart_full_throttle_training.ps
```

### ac (erminal)
```bash
cd ~/lpha-oop-/lpha-oop--
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ~/lphaloopcapital ropbox/ ech gents/ - ec .env .env
python scripts/test_db_connection.py

#    (opens  terminals automatically)
bash scripts/start_full_throttle_training.sh
```

---

##  '  

### . assive  ( ears ackfill)
- **tocks** `equity/minute/` - + years of -minute  bars
- **ptions** `option/minute/` - ull options chains  reeks
  - elta (price sensitivity)
  - amma (delta sensitivity)
  - heta (time decay)
  - ega (volatility sensitivity)
  - ho (interest rate sensitivity)
- **ndices** `index/minute/` - & , , ow, 
- **orex** `forex/minute/` - ajor currency pairs

### . lpha antage remium (ontinuous)
- **tock ntraday** -minute bars, full history (up to  years)
- **tock aily** + years of daily bars
- **tock undamentals** (dvanced aluation etrics)
  - **aluation atios** /, , /, /, /, /ales
  - **rofitability** rofit argin, perating argin, , , 
  - **rowth** evenue rowth, arnings rowth
  - **inancial ealth** urrent atio, uick atio, ebt/quity
  - **isk etrics** ltman -core (bankruptcy risk), eta
  - **alue etrics** raham umber (intrinsic value), iotroski -core
- **ndices** , ,  daily data
- **orex** ajor pairs (/, /, etc.)

### . dvanced uant etrics alculated
- **elta-djusted a** ortfolio risk for options positions
- **onvexity** on-linear price sensitivity (amma exposure)
- **nterprise alue** arket ap + ebt - ash
- **ree ash low ield**  / arket ap
- **** eturn on nvested apital

---

##   

### odels rained
. **oost** ( trees, depth , learning rate .)
. **ight** ( trees, learning rate .)
. **atoost** ( iterations, learning rate .)

### eatures sed
- **+ echnical ndicators** , , , ollinger ands, , etc.
- **aluation etrics** /, /, , ltman -core, etc.
- **arket icrostructure** olume -score, mihud illiquidity, spread proxies
- **attern etection** oji candles, gaps, higher highs/lower lows

### raining rocess
- **ime-eries ross-alidation**  splits (no data leakage)
- **etrics ogged** ccuracy, , recision, ecall, , harpe-like, ax rawdown
- **ontinuous etraining** odels retrain every hour with fresh data
- **odel ersioning** imestamped saves with metadata

---

##   & 

### odel erformance etrics
- ** ccuracy** ust be  % (beats random)
- ** ** ust be  . (beats random)
- **harpe atio** isk-adjusted returns
- **ax rawdown** aximum loss in validation

### isk anagement
- **aily oss imit** % max daily loss (kill switch)
- **rawdown imit** % max drawdown (kill switch)
- **osition izing** elly-capped (% max), % per position
- **ax ositions**  concurrent positions

---

##    (  )

### tart rading ngine
```powershell
# indows
cd "serstomlpha-oop-lpha-oop--"
.venvcriptsctivate.ps
python src/trading/execution_engine.py
```

```bash
# ac
cd ~/lpha-oop-/lpha-oop--
source venv/bin/activate
python src/trading/execution_engine.py
```

**ake sure  /ateway is running** (paper trading port  by default)

---

##  

### iew ogs
```powershell
# indows
et-ontent logsmassive_ingest.log -ail  -ait
et-ontent logsalpha_vantage_hydration.log -ail  -ait
et-ontent logsmodel_training.log -ail  -ait
et-ontent logstrading_engine.log -ail  -ait
```

```bash
# ac
tail -f logs/massive_ingest.log
tail -f logs/alpha_vantage_hydration.log
tail -f logs/model_training.log
tail -f logs/trading_engine.log
```

### heck odels
```powershell
# indows
et-hildtem models*.pkl | ort-bject astriteime -escending
```

```bash
# ac
ls -lt models/*.pkl
```

---

##  

### "odule not found"
→ ctivate venv `.venvcriptsctivate.ps` (indows) or `source venv/bin/activate` (ac)

### "atabase connection fails"
→ heck `.env` file has correct  credentials (_, _, _, _)

### "o data collected"
→ heck  keys in `.env`
  - `___`
  - `__`
  - `__` and `__`

### ac goes to sleep
→ se `caffeinate -d` (prevents sleep)

### ate limiting (lpha antage)
→ ystem handles automatically with -second delays between calls

---

##   

ou'll know everything is working when

. **ata ollection**
   - `logs/massive_ingest.log` shows " assive backfill   rows ingested"
   - `logs/alpha_vantage_hydration.log` shows " tock hydration complete  rows stored"

. **odel raining**
   - `logs/model_training.log` shows " acc. auc." for each model
   - `models/` folder has `.pkl` files with timestamps

. **rading ngine** (tomorrow morning)
   - `logs/trading_engine.log` shows "ignal generated" messages
   - rders are placed via  (check /ateway)

---

##     mall/id-ap ( $ )

### urrent ymbols
- efault , , , , , , , , , , , 
- **dd your small/mid-cap universe** in `.env` or `src/config/settings.py`
  ```python
  target_symbols  "__", "__", ...]
  ```

### isk djustments for mall/id-ap
- onsider tighter position caps (-% instead of %) for thinner liquidity
- onitor `amihud` (illiquidity) metric in features
- se `volume_z` to avoid low-volume traps

---

##   

. **ever commit `.env` file** - ontains  keys
. **est in paper trading first** - _ is paper trading
. **onitor logs continuously** - heck for errors
. **eep machines awake** - se `caffeinate -d` on ac, power settings on indows
. **tart data collection early** - eeds time to gather historical data
. **isk limits are ** - ystem will pause trading if breached

---

**uilt for lpha oop apital - nstitutional-rade ong/hort uant edge und**

**his is the , - version with comprehensive data ingestion**


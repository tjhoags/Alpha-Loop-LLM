# lpha oop  - lgorithmic rading ystem

## ission ritical roduction rading ystem

**his is the ,  version** (`tjhoags/alpha-loop-llm`) built from lessons learned in
- `/alc-algo` (original version)
- `/alc-algo-clean` (cleaned up version)
- ultiple iterations and improvements

his is a **sophisticated, institutional-grade algorithmic trading system** designed to run overnight training and execute trades by market open (  ).

---

##  

| ocument | escription |
|----------|-------------|
| **`__.md`** |  omplete natural language guide for  commands |
| `_.md` | indows-specific setup details |
| `_.md` | acook-specific setup details |
| `__.md` | unning on both machines simultaneously |
| `_.md` | uick terminal command reference |
| `__.md` | aximum data ingestion guide |
| `_.md` |  model training details |

---

##  

### tep  pen our erminal

details
summarybindows (owerhell)/b/summary

**n lain nglish** "pen a command window where you can type instructions"

. ress `indows + ` on your keyboard
. lick "erminal" or "indows owerhell"
.  window opens with a prompt like ` serstom`

**r in ursor ** ress `trl + ~` or go to erminal → ew erminal
/details

details
summarybacook ro (erminal)/b/summary

**n lain nglish** "pen a command window where you can type instructions"

. ress `md + pace` to open potlight
. ype "erminal" and press nter
.  window opens with a prompt like `tomacook-ro ~ %`

**r in ursor ** ress `md + ~` or go to erminal → ew erminal
/details

---

### tep  avigate to roject

details
summarybindows (owerhell)/b/summary

**n lain nglish** "o to the folder where all the code lives"

```powershell
# ype this and press nter
cd "serstom.cursorworktreeslpha-oop--sii"

# erify you're in the right place
dir
```
/details

details
summarybacook ro (erminal)/b/summary

**n lain nglish** "o to the folder where all the code lives"

```bash
# ype this and press nter
cd ~/lpha-oop-/lpha-oop--/sii

# erify you're in the right place
ls
```
/details

---

### tep  et p irtual nvironment

details
summarybindows (owerhell)/b/summary

**n lain nglish** "reate an isolated ython workspace for this project"

```powershell
# reate the virtual environment (one-time)
python -m venv venv

# ctivate it (do this every time you open a new terminal)
.venvcriptsctivate.ps

# f you get an "execution policy" error, run this first
et-xecutionolicy -xecutionolicy emoteigned -cope urrentser
```

**uccess** ou'll see `(venv)` at the start of your prompt
/details

details
summarybacook ro (erminal)/b/summary

**n lain nglish** "reate an isolated ython workspace for this project"

```bash
# reate the virtual environment (one-time)
python -m venv venv

# ctivate it (do this every time you open a new terminal)
source venv/bin/activate
```

**uccess** ou'll see `(venv)` at the start of your prompt
/details

---

### tep  nstall ependencies

details
summarybindows (owerhell)/b/summary

**n lain nglish** "nstall all the required ython packages"

```powershell
pip install -r requirements.txt
```
/details

details
summarybacook ro (erminal)/b/summary

**n lain nglish** "nstall all the required ython packages"

```bash
pip install -r requirements.txt
```
/details

---

### tep  onfigure nvironment

details
summarybindows (owerhell)/b/summary

**n lain nglish** "opy your  keys and database credentials to the project"

```powershell
opy-tem "serstomlphaloopcapital ropbox ech gents - ec .env" -estination ".env"
```
/details

details
summarybacook ro (erminal)/b/summary

**n lain nglish** "opy your  keys and database credentials to the project"

```bash
cp ~/lphaloopcapital ropbox/ ech gents/ - ec .env .env
```
/details

---

### tep  est atabase onnection

details
summarybindows (owerhell)/b/summary

**n lain nglish** "ake sure we can connect to the database"

```powershell
python scripts/test_db_connection.py
```
/details

details
summarybacook ro (erminal)/b/summary

**n lain nglish** "ake sure we can connect to the database"

```bash
python scripts/test_db_connection.py
```
/details

---

##   

### ull ommand eference ee `__.md`](__.md)

### unning on oth achines

ee `__.md` for running on indows + acook simultaneously.

---

##  

### erminal  ata ollection

**n lain nglish** "tart pulling market data from all sources"

details
summarybindows (owerhell)/b/summary

```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps
python src/data_ingestion/collector.py
```
/details

details
summarybacook ro (erminal)/b/summary

```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
caffeinate -d python src/data_ingestion/collector.py
```
**ote** `caffeinate -d` prevents your ac from sleeping
/details

### erminal  odel raining

**n lain nglish** "rain machine learning models on the collected data"

details
summarybindows (owerhell)/b/summary

```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps
python src/ml/train_models.py
```
/details

details
summarybacook ro (erminal)/b/summary

```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
caffeinate -d python src/ml/train_models.py
```
/details

**eave both running overnight!**

---

##  (  ) - tart rading

**n lain nglish** "tart the trading engine that will execute trades at market open"

details
summarybindows (owerhell)/b/summary

```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps
python src/trading/execution_engine.py
```
/details

details
summarybacook ro (erminal)/b/summary

```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
python src/trading/execution_engine.py
```
/details

**rerequisites**  /ateway running (paper port , live port )

---

##  

### iew ogs

**n lain nglish** "atch what the system is doing in real-time"

details
summarybindows (owerhell)/b/summary

```powershell
# atch data collection (live updates)
et-ontent logsdata_collection.log -ail  -ait

# atch model training (live updates)
et-ontent logsmodel_training.log -ail  -ait

# atch trading engine (live updates)
et-ontent logstrading_engine.log -ail  -ait

# heck how many models have been trained
(et-hildtem models*.pkl).ount
```
/details

details
summarybacook ro (erminal)/b/summary

```bash
# atch data collection (live updates)
tail -f logs/data_collection.log

# atch model training (live updates)
tail -f logs/model_training.log

# atch trading engine (live updates)
tail -f logs/trading_engine.log

# heck how many models have been trained
ls models/*.pkl | wc -l
```
/details

---

## ystem rchitecture

```
alpha-loop-llm/
├── src/
│   ├── config/          # onfiguration management (ydantic)
│   ├── data_ingestion/  # ulti-source data collection
│   ├── database/        # zure  erver integration
│   ├── ml/              #  models & feature engineering
│   ├── trading/         # xecution engine & order management
│   ├── risk/            # isk management & position sizing
│   └── monitoring/      # ogging and alerts
├── scripts/             # tility scripts & helpers
├── models/              # rained model files (.pkl)
├── data/                # arket data storage
├── logs/                # ystem logs
├── .env                 #  keys (  )
└── requirements.txt     # ython dependencies
```

---

## - 

### dvanced  ipeline
- **+ eatures** rice, technical indicators, volume, volatility, momentum, microstructure
- **nsemble odels** oost, ight, atoost
- **ime-eries ** o data leakage
- **odel ersioning** imestamped saves

### isk anagement
- **elly riterion** osition sizing with confidence weighting
- **aily oss imits** % max daily loss
- **rawdown rotection** % max drawdown
- **osition imits** ax  positions, % per position

### ata nfrastructure
- **ulti-ource** lpha antage, olygon, oinbase
- **zure ** entralized data storage
- **etry ogic** xponential backoff
- **ate imiting** utomatic handling

### rading xecution
- **nteractive rokers** ull integration
- **aper rading** afe testing (_)
- **rder anagement** arket orders with fill tracking
- **eal-ime ignals** -based with confidence

---

## 

- **`_.md`** - omplete indows setup guide
- **`_.md`** - omplete acook setup guide
- **`__.md`** - unning on both machines
- **`_.md`** - tep-by-step action items
- **`_.md`** - etailed terminal commands
- **`_.md`** - uick reference
- **`_.md`** - omplete setup guide
- **`.md`** - onfirms this is the new repo
- **`__.md`** - eature checklist

---

## 

### "odule not found"
→ ctivate venv `.venvcriptsctivate.ps` (indows) or `source venv/bin/activate` (ac)

### "xecution policy error" (indows)
→ un `et-xecutionolicy -xecutionolicy emoteigned -cope urrentser`

### atabase connection fails
→ heck `.env` file has correct  credentials

### "o models found"
→ ake sure model training completed (check `models/` folder)

###  rate limits
→ ystem handles automatically with retries

### ac goes to sleep
→ se `caffeinate -d python src/ml/train_models.py`

---

##  

. **ever commit `.env` file** - ontains  keys
. **est in paper trading first** - _ is paper trading
. **onitor logs** - heck `logs/` folder for errors
. **tart data collection early** - eeds time to gather historical data
. **eep machines awake** - se caffeinate on ac, power settings on indows

---

##  

efore starting trading
-  ] irtual environment created and activated
-  ] ll packages installed
-  ] `.env` file copied to project folder
-  ] atabase connection test passed
-  ] ata collection ran overnight
-  ] odels trained (check `models/` folder)
-  ] rading engine starts without errors

---

##  

ou'll know everything is working when

. ata collection logs show "ollecting data for..." messages
. odel training logs show "raining oost...", "raining ight..." messages
. `models/` folder has `.pkl` files after training completes
. rading engine logs show "oaded  models" when starting
. rading engine shows "tarting trading engine" at  

---

**uilt for lpha oop apital - nstitutional rade rading ystem**

**his is the , - version**


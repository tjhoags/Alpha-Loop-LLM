# omplete ommand eference - erminal pen to ction

his document provides step-by-step commands for both indows and acook, from opening the terminal to executing actions.

---

## uick etup (irst ime nly)

### indows (owerhell)

```powershell
# tep  pen owerhell (ress indows+, select "erminal")

# tep  avigate to project
cd "serstom.cursorworktreeslpha-oop--zqp"

# tep  reate virtual environment
python -m venv venv

# tep  ctivate virtual environment
.venvcriptsctivate.ps
# f you get "execution policy" error, run this first
# et-xecutionolicy -xecutionolicy emoteigned -cope urrentser

# tep  nstall dependencies
pip install -r requirements.txt

# tep  opy environment file from ropbox
opy-tem "serstomlphaloopcapital ropbox ech gents - ec .env" -estination ".env"

# tep  erify setup
python scripts/test_db_connection.py
python scripts/test_api_connections.py
```

### ac (erminal)

```bash
# tep  pen erminal (md+pace, type "erminal", press nter)

# tep  avigate to project
cd ~/.cursor/worktrees/lpha-oop--/zqp

# tep  reate virtual environment
python -m venv venv

# tep  ctivate virtual environment
source venv/bin/activate

# tep  nstall dependencies
pip install -r requirements.txt

# tep  opy environment file from ropbox
cp ~/lphaloopcapital ropbox/ ech gents/ - ec .env .env

# tep  erify setup
python scripts/test_db_connection.py
python scripts/test_api_connections.py
```

---

## aily perations

### tart of ay - indows

```powershell
# pen owerhell, then
cd "serstom.cursorworktreeslpha-oop--zqp"
.venvcriptsctivate.ps
python scripts/test_api_connections.py  # erify s working
python scripts/training_status.py        # heck model grades
```

### tart of ay - ac

```bash
# pen erminal, then
cd ~/.cursor/worktrees/lpha-oop--/zqp
source venv/bin/activate
python scripts/test_api_connections.py  # erify s working
python scripts/training_status.py        # heck model grades
```

---

## ata ollection

### indows - tart ata ollection

```powershell
# erminal  ull data hydration
cd "serstom.cursorworktreeslpha-oop--zqp"
.venvcriptsctivate.ps
python src/data_ingestion/collector.py

#  use batch file (double-click)
# scripts__.bat
```

### ac - tart ata ollection

```bash
# erminal  ull data hydration (prevent sleep)
cd ~/.cursor/worktrees/lpha-oop--/zqp
source venv/bin/activate
caffeinate -d python src/data_ingestion/collector.py

#  use shell script
# bash scripts/mac_data_collection.sh
```

### uick ata ydration (- minutes)

**indows**
```powershell
python scripts/hydrate_full_universe.py --quick
#  double-click scripts_.bat
```

**ac**
```bash
python scripts/hydrate_full_universe.py --quick
```

### ull niverse ydration (- hours)

**indows**
```powershell
python scripts/hydrate_full_universe.py
#  double-click scripts__.bat
```

**ac**
```bash
caffeinate -d python scripts/hydrate_full_universe.py
#  bash scripts/__.sh
```

---

## odel raining

### indows - rain ll odels

```powershell
# erminal  (parallel to data collection)
cd "serstom.cursorworktreeslpha-oop--zqp"
.venvcriptsctivate.ps
python src/ml/train_models.py

#  massive parallel training
python src/ml/massive_trainer.py --batch-size 

#  use batch file (double-click)
# scripts_.bat
```

### ac - rain ll odels

```bash
# erminal  (parallel to data collection)
cd ~/.cursor/worktrees/lpha-oop--/zqp
source venv/bin/activate
caffeinate -d python src/ml/train_models.py

#  massive parallel training
caffeinate -d python src/ml/massive_trainer.py --batch-size 

#  use shell script
# bash scripts/_.sh
```

### vernight raining etup

**indows**
```powershell
# tart owerhell script that manages everything
.scriptsstart_full_throttle_training.ps
#  .scriptsovernight_training_robust.ps
```

**ac**
```bash
# tart shell script that manages everything
bash scripts/start_full_throttle_training.sh
#  bash scripts/mac_overnight_training.sh
```

---

## rading perations

### aper rading (afe - o eal oney)

**indows**
```powershell
# nsure  /ateway is running on port 
cd "serstom.cursorworktreeslpha-oop--zqp"
.venvcriptsctivate.ps
python src/trading/production_algo.py --paper

#  double-click scripts__.bat
```

**ac**
```bash
# nsure  /ateway is running on port 
cd ~/.cursor/worktrees/lpha-oop--/zqp
source venv/bin/activate
python src/trading/production_algo.py --paper
```

### ive rading (eal oney - )

**indows**
```powershell
#  his trades with  money!
# nsure   is running on port  (live)
cd "serstom.cursorworktreeslpha-oop--zqp"
.venvcriptsctivate.ps
python src/trading/production_algo.py --live

#  use batch file (requires confirmation)
# scripts__.bat
```

**ac**
```bash
#  his trades with  money!
cd ~/.cursor/worktrees/lpha-oop--/zqp
source venv/bin/activate
python src/trading/production_algo.py --live
```

---

## onitoring and tatus

### heck odel rades

**indows**
```powershell
python scripts/training_status.py
#  double-click scripts__.bat
```

**ac**
```bash
python scripts/training_status.py
```

### heck  onnections

**indows**
```powershell
python scripts/test_api_connections.py
```

**ac**
```bash
python scripts/test_api_connections.py
```

### heck atabase onnection

**indows**
```powershell
python scripts/test_db_connection.py
```

**ac**
```bash
python scripts/test_db_connection.py
```

### iew ogs

**indows**
```powershell
# eal-time log monitoring
et-ontent datalogs*.log -ail  -ait

# r view specific log
et-ontent datalogsalc_algo_--_om-.log -ail 
```

**ac**
```bash
# eal-time log monitoring
tail -f data/logs/*.log

# r view specific log
tail - data/logs/*.log
```

### heck odel ount

**indows**
```powershell
(et-hildtem models*.pkl).ount
```

**ac**
```bash
ls models/*.pkl | wc -l
```

---

## it perations

### ull atest hanges

**indows**
```powershell
git pull origin main
```

**ac**
```bash
git pull origin main
```

### ommit our hanges

**indows**
```powershell
git add .
git commit -m "our message here"
git push origin your-branch-name  #  main
```

**ac**
```bash
git add .
git commit -m "our message here"
git push origin your-branch-name  #  main
```

### reate ew ranch

**indows/ac**
```bash
git checkout -b feature/your-feature-name
```

---

## esearch ngestion

### ngest esearch ocuments

**indows**
```powershell
python scripts/ingest_research.py
#  double-click scripts_.bat
```

**ac**
```bash
python scripts/ingest_research.py
```

---

## atch iles eference (indows nly)

| atch ile | urpose |
|------------|---------|
| `__.bat` | iew current model performance |
| `__.bat` | erify trading prerequisites |
| `__.bat` | ull data collection (- hrs) |
| `_.bat` | uick data sample (- min) |
| `_.bat` | rocess research documents |
| `__.bat` | tart data pipeline |
| `__.bat` | afe paper trading |
| `__.bat` | eal money trading |
| `_.bat` | ull model training |

---

## hell cripts eference (ac nly)

| hell cript | urpose |
|--------------|---------|
| `mac_data_collection.sh` | tart data pipeline with caffeinate |
| `mac_full_training.sh` | omplete training cycle |
| `mac_overnight_training.sh` | vernight unattended training |
| `mac_trading_engine.sh` | tart trading engine |
| `__.sh` | ull data collection |
| `_.sh` | ull model training |
| `start_full_throttle_training.sh` | aximum throughput mode |

---

## roubleshooting ommands

### odule ot ound rror

```powershell
# indows
.venvcriptsctivate.ps
pip install -r requirements.txt
```

```bash
# ac
source venv/bin/activate
pip install -r requirements.txt
```

### xecution olicy rror (indows)

```powershell
et-xecutionolicy -xecutionolicy emoteigned -cope urrentser
```

### atabase onnection ailed

```powershell
# heck  river is installed
# hen verify .env has correct credentials
python scripts/test_db_connection.py
```

### ac leep revention

```bash
# refix any long-running command with
caffeinate -d python your_script.py
```

---

## ne-ine ull etup (opy-aste eady)

### indows

```powershell
cd "serstom.cursorworktreeslpha-oop--zqp" python -m venv venv .venvcriptsctivate.ps pip install -r requirements.txt opy-tem "serstomlphaloopcapital ropbox ech gents - ec .env" -estination ".env" python scripts/test_api_connections.py
```

### ac

```bash
cd ~/.cursor/worktrees/lpha-oop--/zqp && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cp ~/lphaloopcapital ropbox/ ech gents/ - ec .env .env && python scripts/test_api_connections.py
```

---

## mportant otes

. **lways activate venv first** - ou should see `(venv)` in your prompt
. **ever push to main** - se your own branch
. **erify s daily** - un test_api_connections.py
. **onitor logs** - heck for errors in data/logs/
. **ack up models** - opy to ropbox after successful training


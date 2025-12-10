#    - indows & ac

## omplete atural anguage uide for ll perations

his document provides step-by-step instructions in plain nglish for both indows (owerhell) and acook ro (erminal) users.

---

##   

. pening our erminal](#-opening-your-terminal)
. avigating to roject](#-navigating-to-project)
. etting p nvironment](#-setting-up-environment)
. nstalling ependencies](#-installing-dependencies)
. opying onfiguration iles](#-copying-configuration-files)
. esting atabase onnection](#-testing-database-connection)
. tarting ata ollection](#-starting-data-collection)
. unning odel raining](#-running-model-training)
. tarting rading ngine](#-starting-trading-engine)
. onitoring & ogs](#-monitoring--logs)
. roubleshooting](#-troubleshooting)

---

## .   

### indows (owerhell)

**ption  - indows erminal (ecommended)**
. ress `indows + ` on your keyboard
. lick "erminal" or "indows erminal" from the menu
.  blue owerhell window will open

**ption  - earch**
. ress `indows key` and type "owerhell"
. lick "indows owerhell" or "indows erminal"

**ption  - rom ursor **
. pen ursor
. o to menu erminal → ew erminal
. erminal opens at bottom of 

### acook ro (erminal)

**ption  - potlight (astest)**
. ress `md + pace` to open potlight
. ype "erminal"
. ress `nter` when erminal appears

**ption  - inder**
. pen inder
. o to pplications → tilities → erminal
. ouble-click erminal

**ption  - rom ursor **
. pen ursor
. o to menu erminal → ew erminal
. erminal opens at bottom of 

---

## .   

### indows (owerhell)

**tep  hange to project directory**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv"
```

**lain nglish** ype `cd` (change directory), then the full path in quotes, then press nter.

**lternative path (if different location)**
```powershell
cd "serstomlpha-oop-lpha-oop--"
```

**erify you're in the right place**
```powershell
et-ocation
```
his shows your current directory. ou should see the project path.

### acook ro (erminal)

**tep  hange to project directory**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv
```

**lain nglish** ype `cd` (change directory), then `~` (your home folder), then the path, then press nter.

**lternative paths**
```bash
# f project is directly in home folder
cd ~/lpha-oop--/gkv

# f project is in ocuments
cd ~/ocuments/lpha-oop-/gkv
```

**erify you're in the right place**
```bash
pwd
```
his prints working directory. ou should see the project path.

---

## .   

### indows (owerhell)

**tep  reate virtual environment**
```powershell
python -m venv venv
```
**lain nglish** his creates a folder called "venv" that will hold all ython packages separate from your system.

**tep  ctivate the virtual environment**
```powershell
.venvcriptsctivate.ps
```
**lain nglish** his "turns on" the virtual environment. ou'll see `(venv)` appear at the start of your prompt.

**f you get an execution policy error**
```powershell
et-xecutionolicy -xecutionolicy emoteigned -cope urrentser
```
**lain nglish** his allows owerhell to run local scripts. ype `` and press nter when asked.

**hen try activating again**
```powershell
.venvcriptsctivate.ps
```

### acook ro (erminal)

**tep  reate virtual environment**
```bash
python -m venv venv
```
**lain nglish** ame as indows - creates isolated ython environment. ote ac uses `python` not `python`.

**tep  ctivate the virtual environment**
```bash
source venv/bin/activate
```
**lain nglish** his activates the environment. ou'll see `(venv)` at the start of your prompt.

---

## .  

### indows (owerhell)

**ake sure venv is activated first (you should see (venv) in prompt)**

**tep  pgrade pip (optional but recommended)**
```powershell
python -m pip install --upgrade pip
```

**tep  nstall all required packages**
```powershell
pip install -r requirements.txt
```
**lain nglish** his reads the requirements.txt file and installs all listed packages. akes - minutes.

### acook ro (erminal)

**ake sure venv is activated first (you should see (venv) in prompt)**

**tep  pgrade pip (optional but recommended)**
```bash
python -m pip install --upgrade pip
```

**tep  nstall all required packages**
```bash
pip install -r requirements.txt
```
**lain nglish** ame as indows - installs all required ython packages.

---

## .   

### indows (owerhell)

**tep  opy the .env file from ropbox**
```powershell
opy-tem "serstomlphaloopcapital ropbox ech gents - ec .env" -estination ".env"
```
**lain nglish** his copies your  keys file into the project folder as ".env".

**erify the file exists**
```powershell
est-ath ".env"
```
hould return `rue`.

**lternative - if .env is elsewhere**
```powershell
# rom ropbox
opy-tem "serstomropbox eys.env" -estination ".env"

# r create manually
notepad .env
```

### acook ro (erminal)

**tep  opy the .env file from ropbox**
```bash
cp ~/lphaloopcapital ropbox/ ech gents/ - ec .env .env
```
**lain nglish** his copies your  keys file. ote the backslashes before spaces in the path.

**lternative paths**
```bash
# rom ropbox
cp ~/ropbox/ eys/.env .env

# rom iloud
cp ~/ibrary/obile ocuments/com~apple~loudocs/.env .env

# r manually create
nano .env
```

**erify the file exists**
```bash
ls -la .env
```
hould show the file with its size.

---

## .   

### indows (owerhell)

**tep  un the test script**
```powershell
python scripts/test_db_connection.py
```
**lain nglish** his tests if ython can connect to the zure  database.

**hat to look for**
-  "onnection successful"  ood!
-  "onnection failed"  heck your .env file credentials

### acook ro (erminal)

**tep  un the test script**
```bash
python scripts/test_db_connection.py
```
**lain nglish** ame test - verifies database connectivity from ac.

---

## .   

### indows (owerhell)

**ption  uick ata ollection (standard)**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv"
.venvcriptsctivate.ps
python src/data_ingestion/collector.py
```

**ption  ull niverse ydration (comprehensive)**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv"
.venvcriptsctivate.ps
python scripts/hydrate_full_universe.py
```

**ption  lpha antage remium ata**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv"
.venvcriptsctivate.ps
python scripts/hydrate_alpha_vantage.py
```

**ption  assive  eep istorical ata**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv"
.venvcriptsctivate.ps
python scripts/hydrate_massive.py
```

**ave output to log file**
```powershell
python scripts/hydrate_full_universe.py & | ee-bject -ileath logs/hydration.log
```

### acook ro (erminal)

**ption  uick ata ollection (standard)**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv
source venv/bin/activate
python src/data_ingestion/collector.py
```

**ption  ull niverse ydration (comprehensive)**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv
source venv/bin/activate
caffeinate -d python scripts/hydrate_full_universe.py
```
**ote** `caffeinate -d` prevents ac from sleeping during long operations.

**ption  lpha antage remium ata**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv
source venv/bin/activate
caffeinate -d python scripts/hydrate_alpha_vantage.py
```

**ption  assive  eep istorical ata**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv
source venv/bin/activate
caffeinate -d python scripts/hydrate_massive.py
```

**ave output to log file**
```bash
python scripts/hydrate_full_universe.py & | tee logs/hydration.log
```

---

## .   

### indows (owerhell)

**ption  tandard raining**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv"
.venvcriptsctivate.ps
python src/ml/train_models.py
```

**ption  dvanced vernight raining**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv"
.venvcriptsctivate.ps
python -c "from src.ml.advanced_training import run_overnight_training run_overnight_training()"
```

**ption  assive arallel raining (ull niverse)**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv"
.venvcriptsctivate.ps
python src/ml/massive_trainer.py
```

**ption  gent raining**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv"
.venvcriptsctivate.ps
python -m src.training.agent_trainer --all
```

**ave training output to log**
```powershell
python src/ml/train_models.py & | ee-bject -ileath logs/training.log
```

### acook ro (erminal)

**ption  tandard raining**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv
source venv/bin/activate
caffeinate -d python src/ml/train_models.py
```

**ption  dvanced vernight raining**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv
source venv/bin/activate
caffeinate -d python -c "from src.ml.advanced_training import run_overnight_training run_overnight_training()"
```

**ption  assive arallel raining (ull niverse)**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv
source venv/bin/activate
caffeinate -d python src/ml/massive_trainer.py
```

**ption  gent raining**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv
source venv/bin/activate
caffeinate -d python -m src.training.agent_trainer --all
```

**ave training output to log**
```bash
caffeinate -d python src/ml/train_models.py & | tee logs/training.log
```

---

## .   

### indows (owerhell)

**tep  nsure   or ateway is running**
- aper rading ort 
- ive rading ort 

**tep  tart the trading engine**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv"
.venvcriptsctivate.ps
python src/trading/execution_engine.py
```

### acook ro (erminal)

**tep  nsure   or ateway is running**
- aper rading ort 
- ive rading ort 

**tep  tart the trading engine**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv
source venv/bin/activate
python src/trading/execution_engine.py
```

---

## .  & 

### indows (owerhell)

**iew log file (last  lines)**
```powershell
et-ontent logsdata_collection.log -ail 
```

**atch log in real-time**
```powershell
et-ontent logstraining.log -ail  -ait
```

**ount trained models**
```powershell
(et-hildtem models*.pkl).ount
```

**ist all models with timestamps**
```powershell
et-hildtem models*.pkl | ort-bject astriteime -escending | elect-bject ame, astriteime
```

**heck database row count**
```powershell
python -c "from src.database.connection import get_engine from sqlalchemy import text engine  get_engine() result  engine.execute(text(' (*)  price_bars')) print(f'ows {result.fetchone()],}')"
```

**un model dashboard**
```powershell
python scripts/model_dashboard.py
```

### acook ro (erminal)

**iew log file (last  lines)**
```bash
tail - logs/data_collection.log
```

**atch log in real-time**
```bash
tail -f logs/training.log
```

**ount trained models**
```bash
ls models/*.pkl /dev/null | wc -l
```

**ist all models with timestamps**
```bash
ls -lt models/*.pkl
```

**heck database row count**
```bash
python -c "from src.database.connection import get_engine from sqlalchemy import text engine  get_engine() result  engine.execute(text(' (*)  price_bars')) print(f'ows {result.fetchone()],}')"
```

**un model dashboard**
```bash
python scripts/model_dashboard.py
```

---

## . 

### ommon ssues - indows

**ssue "python is not recognized"**
```powershell
# olution  se full path
ythonpython.exe -m venv venv

# olution  dd ython to 
# earch "nvironment ariables" in indows
# dit  to include ython installation folder
```

**ssue "xecution policy" error**
```powershell
et-xecutionolicy -xecutionolicy emoteigned -cope urrentser
```

**ssue "odule not found"**
```powershell
# ake sure venv is activated
.venvcriptsctivate.ps
# einstall requirements
pip install -r requirements.txt
```

**ssue atabase connection fails**
```powershell
# heck .env file exists
est-ath ".env"
# iew .env contents (careful with secrets!)
et-ontent ".env" | elect-tring ""
```

### ommon ssues - ac

**ssue "python command not found"**
```bash
# se python instead
python -m venv venv
python src/ml/train_models.py
```

**ssue "ermission denied" on scripts**
```bash
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

**ssue "odule not found"**
```bash
# ake sure venv is activated
source venv/bin/activate
# einstall requirements
pip install -r requirements.txt
```

**ssue ac goes to sleep during training**
```bash
# revent display sleep
caffeinate -d python src/ml/train_models.py

# revent all sleep
caffeinate -i python src/ml/train_models.py
```

**ssue  river not found on ac**
```bash
# nstall icrosoft  river
brew tap microsoft/mssql-release https//github.com/icrosoft/homebrew-mssql-release
brew update
_ brew install msodbcsql
```

---

##    

### ne-iner etup ommands

**indows - omplete etup**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv" python -m venv venv .venvcriptsctivate.ps pip install -r requirements.txt opy-tem "serstomlphaloopcapital ropbox ech gents - ec .env" -estination ".env" python scripts/test_db_connection.py
```

**ac - omplete etup**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cp ~/lphaloopcapital ropbox/ ech gents/ - ec .env .env && python scripts/test_db_connection.py
```

### ne-iner raining ommands

**indows - ull vernight raining**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv" .venvcriptsctivate.ps python scripts/hydrate_full_universe.py & | ee-bject logs/hydration.log
```

**ac - ull vernight raining**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv && source venv/bin/activate && caffeinate -d python scripts/hydrate_full_universe.py & | tee logs/hydration.log
```

---

## - 

### ecommended etup for vernight raining

**indows  (rimary)**
- erminal  ata ydration
- erminal  odel raining
- erminal  onitoring ashboard

**acook ro (econdary)**
- erminal  esearch ngestion
- erminal  entiment nalysis
- erminal  ackup raining

### indows erminals (pen )

**erminal  - ata ydration**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv"
.venvcriptsctivate.ps
python scripts/hydrate_full_universe.py & | ee-bject logs/hydration.log
```

**erminal  - odel raining**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv"
.venvcriptsctivate.ps
python -c "from src.ml.advanced_training import run_overnight_training run_overnight_training()" & | ee-bject logs/training.log
```

**erminal  - onitoring**
```powershell
cd "serstom.cursorworktreeslpha-oop--gkv"
.venvcriptsctivate.ps
python scripts/model_dashboard.py
```

### ac erminals (pen )

**erminal  - esearch ngestion**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv
source venv/bin/activate
caffeinate -d python scripts/ingest_research.py & | tee logs/research.log
```

**erminal  - ackup raining**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv
source venv/bin/activate
caffeinate -d python src/ml/train_models.py & | tee logs/training_mac.log
```

**erminal  - onitoring**
```bash
cd ~/lpha-oop-/lpha-oop--/gkv
source venv/bin/activate
watch -n  'ls -la models/*.pkl | wc -l'
```

---

##    

### indows

**ia owerhell**
```powershell
# revent sleep when plugged in
powercfg /change standby-timeout-ac 

# heck current settings
powercfg /query
```

**ia ettings**
. indows ettings → ystem → ower & battery
. et "creen timeout" to ever when plugged in
. et "leep" to ever when plugged in

### acook ro

**revent sleep during training**
```bash
caffeinate -d python src/ml/train_models.py
```

**revent sleep indefinitely (background)**
```bash
caffeinate -d &
```

**ia ystem ettings**
. ystem ettings → attery → ptions
. urn  "revent automatic sleeping when display is off"

---

**uilt for lpha oop apital - nstitutional rade rading ystem**

*his document provides complete natural language instructions for operating the system on both indows and ac platforms.*


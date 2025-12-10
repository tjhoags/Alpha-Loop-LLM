# ️ ross-latform ommand eference
## indows (owerhell) & acook ro (erminal) nstructions

 **atural anguage uide** very command explained in plain nglish with step-by-step instructions for both platforms.

---

##  able of ontents
. pening erminal](#-opening-terminal)
. avigate to roject](#-navigate-to-project)
. irtual nvironment etup](#-virtual-environment-setup)
. nstall ependencies](#-install-dependencies)
. nvironment onfiguration](#-environment-configuration)
. atabase perations](#-database-operations)
. ata ollection](#-data-collection)
. odel raining](#-model-training)
. rading ngine](#-trading-engine)
. onitoring & ogs](#-monitoring--logs)
. gent perations](#-agent-operations)
. eview ystem](#-review-system)
. roubleshooting](#-troubleshooting)

---

## . pening erminal

### hat ou're oing
pening a command-line interface where you can type commands to control the system.

### indows (owerhell)
**ption  - uick ccess**
. ress `indows + ` on your keyboard
. lick "erminal" or "indows owerhell" in the menu
.  blue/black window opens - you're ready!

**ption  - earch**
. ress `indows` key
. ype "owerhell"
. lick "indows owerhell" or "erminal"

**ption  - ursor **
. n ursor, press `trl + ~` (backtick)
. r go to erminal menu → ew erminal

```powershell
# ou should see a prompt like
#  serstom
```

### acook ro (erminal)
**ption  - potlight**
. ress `md + pace` to open potlight
. ype "erminal"
. ress nter
.  white/black window opens - you're ready!

**ption  - inder**
. pen inder
. o to pplications → tilities → erminal
. ouble-click erminal

**ption  - ursor **
. n ursor, press `md + ~` (backtick)
. r go to erminal menu → ew erminal

```bash
# ou should see a prompt like
# tomacook-ro ~ %
```

---

## . avigate to roject

### hat ou're oing
hanging to the folder where all the code lives. his is always your first step.

### indows (owerhell)
```powershell
# avigate to the lpha oop  project
cd "serstom.cursorworktreeslpha-oop--sii"

# erify you're in the right place (should show project files)
et-hildtem

# lternative ist files in simplified view
dir
```

**n lain nglish** "o to the folder called sii inside lpha-oop--"

### acook ro (erminal)
```bash
# avigate to the lpha oop  project
cd ~/lpha-oop-/lpha-oop--/sii

# erify you're in the right place (should show project files)
ls -la

# lternative imple list
ls
```

**n lain nglish** "o to the folder called sii inside lpha-oop-- in your home directory"

### nderstanding aths
| latform | ome irectory | roject ath |
|----------|---------------|--------------|
| indows | `serstom` | `serstom.cursorworktreeslpha-oop--sii` |
| ac | `~/` or `/sers/tom/` | `~/lpha-oop-/lpha-oop--/sii` |

---

## . irtual nvironment etup

### hat ou're oing
reating an isolated ython environment so our packages don't conflict with other projects.

### indows (owerhell)

**tep  reate the virtual environment**
```powershell
# irst, navigate to project
cd "serstom.cursorworktreeslpha-oop--sii"

# reate virtual environment (one-time setup)
python -m venv venv
```
**hat his oes** reates a `venv` folder with a fresh ython installation

**tep  ctivate the virtual environment**
```powershell
# ctivate the virtual environment
.venvcriptsctivate.ps
```
**hat his oes** urns on the isolated environment. ou'll see `(venv)` before your prompt.

**f ou et an xecution olicy rror**
```powershell
# llow script execution (one-time setup, admin not required)
et-xecutionolicy -xecutionolicy emoteigned -cope urrentser

# hen try activating again
.venvcriptsctivate.ps
```

### acook ro (erminal)

**tep  reate the virtual environment**
```bash
# irst, navigate to project
cd ~/lpha-oop-/lpha-oop--/sii

# reate virtual environment (one-time setup)
python -m venv venv
```
**hat his oes** reates a `venv` folder with a fresh ython installation

**tep  ctivate the virtual environment**
```bash
# ctivate the virtual environment
source venv/bin/activate
```
**hat his oes** urns on the isolated environment. ou'll see `(venv)` before your prompt.

### erification
```powershell
# indows - heck ython location (should show venv path)
et-ommand python
```

```bash
# ac - heck ython location (should show venv path)
which python
```

---

## . nstall ependencies

### hat ou're oing
nstalling all the ython packages the system needs to run.

### indows (owerhell)
```powershell
# ake sure you're in the project and venv is active
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps

# nstall all required packages
pip install -r requirements.txt

# pgrade pip first if you get warnings
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### acook ro (erminal)
```bash
# ake sure you're in the project and venv is active
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate

# nstall all required packages
pip install -r requirements.txt

# pgrade pip first if you get warnings
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**hat his oes** eads the `requirements.txt` file and installs oost, ight, atoost, pandas, and all other required packages.

---

## . nvironment onfiguration

### hat ou're oing
etting up your  keys and database credentials securely.

### indows (owerhell)
```powershell
# avigate to project
cd "serstom.cursorworktreeslpha-oop--sii"

# opy the environment file from ropbox
opy-tem "serstomlphaloopcapital ropbox ech gents - ec .env" -estination ".env"

# erify the file was copied
est-ath ".env"  # hould return rue

# iew contents (careful - contains secrets)
et-ontent ".env" | elect-bject -irst 
```

### acook ro (erminal)
```bash
# avigate to project
cd ~/lpha-oop-/lpha-oop--/sii

# opy the environment file from ropbox
cp ~/lphaloopcapital ropbox/ ech gents/ - ec .env .env

# r from iloud
cp ~/ibrary/obile ocuments/com~apple~loudocs/lpha oop /.env .env

# erify the file was copied
ls -la .env

# iew first few lines (careful - contains secrets)
head - .env
```

### equired nvironment ariables
our `.env` file should contain
```env
# atabase
_your-server.database.windows.net
_alc_market_data
_your-username
_your-password

#  eys
___your-key
__your-key
__your-key

# rading
_...
_  # aper trading
```

---

## . atabase perations

### hat ou're oing
esting and managing the zure  database connection.

### indows (owerhell)
```powershell
# avigate and activate
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps

# est database connection
python scripts/test_db_connection.py

# etup database schema (if needed)
python scripts/setup_db_schema.py

# heck row count in price_bars table
python -c "from src.database.connection import get_engine import pandas as pd print(pd.read_sql(' (*) as count  price_bars', get_engine()))"
```

### acook ro (erminal)
```bash
# avigate and activate
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate

# est database connection
python scripts/test_db_connection.py

# etup database schema (if needed)
python scripts/setup_db_schema.py

# heck row count in price_bars table
python -c "from src.database.connection import get_engine import pandas as pd print(pd.read_sql(' (*) as count  price_bars', get_engine()))"
```

---

## . ata ollection

### hat ou're oing
ulling market data from multiple sources into the database.

### indows (owerhell)

**ption  tandard ata ollection**
```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps

# tart data collector
python src/data_ingestion/collector.py
```

**ption  ull hrottle - lpha antage remium**
```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps

# ydrate all lpha antage data
python scripts/hydrate_all_alpha_vantage.py & | ee-bject -ileath logs/alpha_vantage.log
```

**ption  ull niverse ydration**
```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps

# ull data for entire market universe
python scripts/hydrate_full_universe.py & | ee-bject -ileath logs/hydration.log
```

### acook ro (erminal)

**ption  tandard ata ollection**
```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate

# tart data collector
python src/data_ingestion/collector.py
```

**ption  ull hrottle - lpha antage remium (eep ac wake)**
```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate

# ydrate with caffeinate to prevent sleep
caffeinate -d python scripts/hydrate_all_alpha_vantage.py & | tee logs/alpha_vantage.log
```

**ption  ull niverse ydration (eep ac wake)**
```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate

# ull data for entire market universe
caffeinate -d python scripts/hydrate_full_universe.py & | tee logs/hydration.log
```

### ata ources ollected
| ource | ata ype | istory |
|--------|-----------|---------|
| lpha antage | tocks, undamentals | + years daily |
| olygon | -minute bars |  years |
| oinbase | rypto (, ) | ull history |
|  | acro indicators | ull history |

---

## . odel raining

### hat ou're oing
raining machine learning models (oost, ight, atoost) on the collected data.

### indows (owerhell)

**tandard raining**
```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps

# tart model training
python src/ml/train_models.py
```

**dvanced vernight raining**
```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps

# ull overnight training with logging
python -c "from src.ml.advanced_training import run_overnight_training run_overnight_training()" & | ee-bject -ileath logs/training.log
```

**ull hrottle etup ( terminals)**
```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps

# tart automated full throttle training
.scriptsstart_full_throttle_training.ps
```

### acook ro (erminal)

**tandard raining**
```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate

# tart model training
python src/ml/train_models.py
```

**dvanced vernight raining (eep ac wake)**
```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate

# ull overnight training with caffeinate
caffeinate -d python -c "from src.ml.advanced_training import run_overnight_training run_overnight_training()" & | tee logs/training.log
```

**ull hrottle etup ( terminals)**
```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate

# tart automated full throttle training
bash scripts/start_full_throttle_training.sh
```

### heck raining rogress
```powershell
# indows - iew training logs
et-ontent logstraining.log -ail 

# ount trained models
(et-hildtem models*.pkl).ount
```

```bash
# ac - iew training logs
tail -f logs/training.log

# ount trained models
ls models/*.pkl | wc -l
```

---

## . rading ngine

### hat ou're oing
tarting the trading execution system that generates signals and places orders.

### rerequisites
-   or ateway must be running
- aper trading ort 
- ive trading ort 

### indows (owerhell)
```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps

# tart trading engine (run at   )
python src/trading/execution_engine.py

# lternative se batch file
.scripts__.bat
```

### acook ro (erminal)
```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate

# tart trading engine (run at   )
python src/trading/execution_engine.py

# lternative se shell script
bash scripts/mac_trading_engine.sh
```

---

## . onitoring & ogs

### hat ou're oing
atching system progress and checking for errors.

### indows (owerhell)

**iew ogs in eal-ime**
```powershell
cd "serstom.cursorworktreeslpha-oop--sii"

# ata collection log
et-ontent logsdata_collection.log -ail  -ait

# odel training log
et-ontent logstraining.log -ail  -ait

# rading engine log
et-ontent logstrading_engine.log -ail  -ait

# ll logs at once (separate terminals)
et-ontent logs*.log -ail 
```

**heck ystem tatus**
```powershell
# odel dashboard
python scripts/model_dashboard.py

# raining status
python scripts/training_status.py

# heck model grades
.scripts__.bat
```

### acook ro (erminal)

**iew ogs in eal-ime**
```bash
cd ~/lpha-oop-/lpha-oop--/sii

# ata collection log
tail -f logs/data_collection.log

# odel training log
tail -f logs/training.log

# rading engine log
tail -f logs/trading_engine.log

# ll logs at once (use separate terminals or tmux)
tail -f logs/*.log
```

**heck ystem tatus**
```bash
# odel dashboard
python scripts/model_dashboard.py

# raining status
python scripts/training_status.py

# heck model count
ls -la models/*.pkl | wc -l
```

---

## . gent perations

### hat ou're oing
orking with the   agents that power the system.

### indows (owerhell)

**tart gent hat nterface**
```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps

# nteractive agent chat
python src/interfaces/agent_chat.py

# r use batch file
.scripts__.bat
```

**rain ll gents**
```powershell
# rain all  agents
.scripts__.bat
```

### acook ro (erminal)

**tart gent hat nterface**
```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate

# nteractive agent chat
python src/interfaces/agent_chat.py
```

**rain ll gents**
```bash
# rain agents with overnight protection
caffeinate -d python src/training/train_all_agents.py
```

---

## . eview ystem

### hat ou're oing
unning automated code review across the entire project.

### indows (owerhell)
```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps

# un code review
python -m src.review.orchestrator

# r use batch file
.scripts_.bat

# erform specific review
python scripts/perform_review.py
```

### acook ro (erminal)
```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate

# un code review
python -m src.review.orchestrator

# erform specific review
python scripts/perform_review.py
```

---

## . roubleshooting

### ommon ssues & olutions

#### "odule not found" rror

**indows**
```powershell
# ake sure venv is activated (you should see (venv) in prompt)
.venvcriptsctivate.ps

# f still failing, reinstall packages
pip install -r requirements.txt
```

**ac**
```bash
# ake sure venv is activated (you should see (venv) in prompt)
source venv/bin/activate

# f still failing, reinstall packages
pip install -r requirements.txt
```

#### "xecution olicy" rror (indows nly)
```powershell
# un this once to allow scripts
et-xecutionolicy -xecutionolicy emoteigned -cope urrentser

# erify it worked
et-xecutionolicy -ist
```

#### "python command not found" (ac nly)
```bash
# se python instead
python -m venv venv
python src/data_ingestion/collector.py

# r create an alias in ~/.zshrc or ~/.bashrc
echo "alias pythonpython"  ~/.zshrc
source ~/.zshrc
```

#### atabase onnection ails
```powershell
# indows - heck .env file exists
est-ath ".env"

# iew  settings (without passwords)
et-ontent ".env" | elect-tring "_"
```

```bash
# ac - heck .env file exists
ls -la .env

# iew  settings (without passwords)
grep "_" .env
```

#### ac oes to leep uring raining
```bash
# se caffeinate with any long-running command
caffeinate -d python src/ml/train_models.py

# eep running until you manually stop (trl+)
caffeinate -d

# eep running for  hours ( seconds)
caffeinate -t  python src/ml/train_models.py
```

#### heck ython/ackage ersions
```powershell
# indows
python --version
pip list | elect-tring "xgboost|lightgbm|catboost|pandas"
```

```bash
# ac
python --version
pip list | grep - "xgboost|lightgbm|catboost|pandas"
```

---

##  uick eference ards

### indows uick tart
```powershell
# omplete setup in one block
cd "serstom.cursorworktreeslpha-oop--sii"
python -m venv venv
.venvcriptsctivate.ps
pip install -r requirements.txt
opy-tem "serstomlphaloopcapital ropbox ech gents - ec .env" -estination ".env"
python scripts/test_db_connection.py
```

### ac uick tart
```bash
# omplete setup in one block
cd ~/lpha-oop-/lpha-oop--/sii
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ~/lphaloopcapital ropbox/ ech gents/ - ec .env .env
python scripts/test_db_connection.py
```

### ommand ranslation able

| ction | indows (owerhell) | ac (erminal) |
|--------|---------------------|----------------|
| hange directory | `cd "pathtofolder"` | `cd ~/path/to/folder` |
| ist files | `et-hildtem` or `dir` | `ls -la` |
| opy file | `opy-tem src -estination dst` | `cp src dst` |
| iew file | `et-ontent file.txt` | `cat file.txt` |
| iew last  lines | `et-ontent file -ail ` | `tail - file` |
| ollow log | `et-ontent file -ail  -ait` | `tail -f file` |
| ind text | `elect-tring "pattern" file` | `grep "pattern" file` |
| heck if file exists | `est-ath "file"` | `test -f file && echo yes` |
| reate directory | `ew-tem -temype irectory -ath dir` | `mkdir -p dir` |
| elete file | `emove-tem file` | `rm file` |
| ctivate venv | `.venvcriptsctivate.ps` | `source venv/bin/activate` |
| revent sleep | (ower settings) | `caffeinate -d` |

---

**uilt for lpha oop apital - nstitutional-rade rading ystem**
**ross-latform ompatible indows / & mac**


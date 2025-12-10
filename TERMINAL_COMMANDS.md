# ️ erminal ommands eference
## uick eference for indows & ac perations

 **or detailed explanations, see `__.md`](__.md)**

---

##  urrent ystem tatus

| omponent | tatus | otes |
|-----------|--------|-------|
| **ata in zure ** | ,+ symbols, .+ rows | ontinuously updating |
| **odels rained** | heck `models/` folder | etrained hourly |
| **rading ngine** | eady | tart at    |

---

##  uick tart ommands

### nitial etup (ne-ime)

table
tr
th🪟 indows (owerhell)/th
th acook ro (erminal)/th
/tr
tr
td

```powershell
# tep  pen terminal
# ress indows +  → erminal

# tep  avigate to project
cd "serstom.cursorworktreeslpha-oop--sii"

# tep  reate virtual environment
python -m venv venv

# tep  ctivate it
.venvcriptsctivate.ps

# tep  nstall packages
pip install -r requirements.txt

# tep  opy environment file
opy-tem "serstomlphaloopcapital ropbox ech gents - ec .env" -estination ".env"

# tep  est database
python scripts/test_db_connection.py
```

/td
td

```bash
# tep  pen terminal
# ress md + pace → type "erminal"

# tep  avigate to project
cd ~/lpha-oop-/lpha-oop--/sii

# tep  reate virtual environment
python -m venv venv

# tep  ctivate it
source venv/bin/activate

# tep  nstall packages
pip install -r requirements.txt

# tep  opy environment file
cp ~/lphaloopcapital ropbox/ ech gents/ - ec .env .env

# tep  est database
python scripts/test_db_connection.py
```

/td
/tr
/table

---

##  vernight raining (+ erminals)

### erminal onfiguration

| erminal | indows | ac | urpose |
|----------|---------|-----|---------|
| **** | ata ydration | ata ydration | ull market data |
| **** |  raining |  raining | rain models |
| **** | onitor | onitor | ashboard |
| **** | - | esearch | ac-specific data |
| **** | - | ackup raining | edundancy |

### erminal  ata ydration

table
tr
th🪟 indows/th
th ac/th
/tr
tr
td

```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps
python scripts/hydrate_full_universe.py & | ee-bject -ileath logs/hydration.log
```

/td
td

```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
caffeinate -d python scripts/hydrate_full_universe.py & | tee logs/hydration.log
```

/td
/tr
/table

### erminal  odel raining

table
tr
th🪟 indows/th
th ac/th
/tr
tr
td

```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps
python -c "from src.ml.advanced_training import run_overnight_training run_overnight_training()" & | ee-bject -ileath logs/training.log
```

/td
td

```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
caffeinate -d python -c "from src.ml.advanced_training import run_overnight_training run_overnight_training()" & | tee logs/training.log
```

/td
/tr
/table

### erminal  onitor ashboard

table
tr
th🪟 indows/th
th ac/th
/tr
tr
td

```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps

# heck hydration progress
et-ontent logs/hydration.log -ail 

# heck training progress
et-ontent logs/training.log -ail 

# odel count
(et-hildtem models*.pkl -rrorction ilentlyontinue).ount

# ull dashboard
python scripts/model_dashboard.py
```

/td
td

```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate

# heck hydration progress
tail - logs/hydration.log

# heck training progress
tail - logs/training.log

# odel count
ls models/*.pkl /dev/null | wc -l

# ull dashboard
python scripts/model_dashboard.py
```

/td
/tr
/table

### erminal  (ac nly) esearch ngestion

```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
caffeinate -d python scripts/ingest_research.py & | tee logs/research.log
```

### erminal  (ac nly) ackup raining

```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
caffeinate -d python src/ml/train_models.py & | tee logs/training_backup.log
```

---

## ️ orning rading (  )

table
tr
th🪟 indows/th
th ac/th
/tr
tr
td

```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps
python src/trading/execution_engine.py
```

/td
td

```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
python src/trading/execution_engine.py
```

/td
/tr
/table

**rerequisites**  /ateway running (aper , ive )

---

##  tatus heck ommands

### uick tatus

table
tr
th🪟 indows/th
th ac/th
/tr
tr
td

```powershell
# ow many models trained
(et-hildtem models*.pkl).ount

# ast  log entries
et-ontent logstraining.log -ail 

# atabase row count
python -c "from src.database.connection import get_engine import pandas as pd print(pd.read_sql(' (*)  price_bars', get_engine()))"
```

/td
td

```bash
# ow many models trained
ls models/*.pkl | wc -l

# ast  log entries
tail - logs/training.log

# atabase row count
python -c "from src.database.connection import get_engine import pandas as pd print(pd.read_sql(' (*)  price_bars', get_engine()))"
```

/td
/tr
/table

---

##  tility ommands

### ata perations

table
tr
th🪟 indows/th
th ac/th
/tr
tr
td

```powershell
# est  connection
python scripts/test_db_connection.py

# uick data hydration
.scripts_.bat

# ull universe hydration
.scripts__.bat

# ngest research
.scripts_.bat
```

/td
td

```bash
# est  connection
python scripts/test_db_connection.py

# uick data hydration
bash scripts/mac_data_collection.sh

# ull overnight training
bash scripts/mac_overnight_training.sh

# ngest research
python scripts/ingest_research.py
```

/td
/tr
/table

### gent perations

table
tr
th🪟 indows/th
th ac/th
/tr
tr
td

```powershell
# tart agent chat
.scripts__.bat

# rain all agents
.scripts__.bat

# heck model grades
.scripts__.bat
```

/td
td

```bash
# tart agent chat
python src/interfaces/agent_chat.py

# rain all agents
python src/training/train_all_agents.py

# heck model grades
python scripts/training_status.py
```

/td
/tr
/table

### ode eview

table
tr
th🪟 indows/th
th ac/th
/tr
tr
td

```powershell
# un code review
.scripts_.bat

# r directly
python -m src.review.orchestrator

# ssue scanner
python -m src.review.issue_scanner
```

/td
td

```bash
# un code review
python -m src.review.orchestrator

# ssue scanner
python -m src.review.issue_scanner
```

/td
/tr
/table

---

##  raining imeline

| ime | ctivity | tatus |
|------|----------|--------|
| ow | tart hydration + training |  unning |
| + hour | irst models complete |  heck |
| + hours | + models trained |  heck |
| + hours | ost stocks complete |  heck |
|   | heck model grades |  eview |
|   | tart trading engine |  o |
|   | arket open |  rading |

---

## ️ roubleshooting

### ommon ssues

| roblem | indows ix | ac ix |
|---------|-------------|---------|
| "odule not found" | `.venvcriptsctivate.ps` | `source venv/bin/activate` |
| "xecution policy" | `et-xecutionolicy emoteigned -cope urrentser` | / |
| "python not found" | se `python` | se `python` |
| ac sleeps | / | `caffeinate -d` |
|  connection fails | heck `.env` file | heck `.env` file |

### erify nvironment

table
tr
th🪟 indows/th
th ac/th
/tr
tr
td

```powershell
# heck ython version
python --version

# heck venv is active
et-ommand python

# heck .env exists
est-ath ".env"

# iew .env (first  lines)
et-ontent ".env" | elect -irst 
```

/td
td

```bash
# heck ython version
python --version

# heck venv is active
which python

# heck .env exists
ls -la .env

# iew .env (first  lines)
head - .env
```

/td
/tr
/table

---

##  ne-ine ommands (opy-aste eady)

### indows - tart verything

```powershell
cd "serstom.cursorworktreeslpha-oop--sii" .venvcriptsctivate.ps .scriptsstart_full_throttle_training.ps
```

### ac - tart verything

```bash
cd ~/lpha-oop-/lpha-oop--/sii && source venv/bin/activate && bash scripts/start_full_throttle_training.sh
```

---

##  ocumentation inks

| ocument | escription |
|----------|-------------|
| __.md](__.md) | etailed natural language guide |
| _.md](_.md) | indows-specific setup |
| _.md](_.md) | ac-specific setup |
| __.md](__.md) | aximum data ingestion |
| _.md](_.md) |  training details |
| _.md](_.md) | gent system overview |

---

**lpha oop apital - nstitutional-rade rading ystem**

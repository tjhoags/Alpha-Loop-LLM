# indows etup nstructions

 ** or complete cross-platform reference, see `__.md`](__.md)**

## hich erminal to se

**ou can use **
. **ocal indows owerhell** (indows +  → erminal)
. **ursor's ntegrated erminal** (erminal menu → ew erminal)

oth work the same way! se whichever you prefer.

---

## uick etup (indows)

### tep  pen erminal
**n lain nglish** "pen a command window where you can type instructions"

- **ption ** ress `indows + ` → lick "erminal"
- **ption ** n ursor, press `trl + ~` or go to erminal → ew erminal

### tep  avigate to roject
**n lain nglish** "o to the folder where all the code lives"

```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
```

### tep  et p nvironment
**n lain nglish** "reate an isolated ython workspace for this project"

```powershell
# reate virtual environment (one-time setup)
python -m venv venv

# ctivate it (do this every time you open a new terminal)
.venvcriptsctivate.ps

# f you get an execution policy error, run this once
et-xecutionolicy -xecutionolicy emoteigned -cope urrentser
```

**uccess** ou'll see `(venv)` at the start of your prompt

### tep  nstall ackages
**n lain nglish** "nstall all required ython packages"

```powershell
pip install -r requirements.txt
```

### tep  opy nvironment ile
**n lain nglish** "opy your  keys and database credentials"

```powershell
opy-tem "serstomlphaloopcapital ropbox ech gents - ec .env" -estination ".env"
```

### tep  est atabase
**n lain nglish** "ake sure we can connect to the database"

```powershell
python scripts/test_db_connection.py
```

---

## tart raining (indows)

### erminal  - ata ollection
**n lain nglish** "tart pulling market data from all sources"

```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps
python src/data_ingestion/collector.py
```

### erminal  - odel raining
**n lain nglish** "rain machine learning models on the collected data"

```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps
python src/ml/train_models.py
```

### erminal  - onitoring
**n lain nglish** "atch what the system is doing"

```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps

# atch data collection (live updates)
et-ontent logsdata_collection.log -ail  -ait

# heck how many models have been trained
(et-hildtem models*.pkl).ount
```

---

## orning rading (  )

**n lain nglish** "tart the trading engine before market opens"

```powershell
cd "serstom.cursorworktreeslpha-oop--sii"
.venvcriptsctivate.ps
python src/trading/execution_engine.py
```

**rerequisites**  /ateway must be running (aper port )

---

## otes for indows

| opic | etails |
|-------|---------|
| hell | se owerhell (not ) |
| aths | se backslashes `serstom...` |
| ctivate | `.venvcriptsctivate.ps` |
| iew logs | `et-ontent file -ail ` |
| ive tail | `et-ontent file -ail  -ait` |
| ile exists | `est-ath "file"` |

---

## roubleshooting

### "xecution olicy" rror
```powershell
et-xecutionolicy -xecutionolicy emoteigned -cope urrentser
```

### "odule not found" rror
ake sure venv is activated (you should see `(venv)` in prompt)
```powershell
.venvcriptsctivate.ps
```

### atabase onnection ails
heck that .env file exists and has correct credentials
```powershell
est-ath ".env"
et-ontent ".env" | elect-tring "_"
```

---

## uick eference

```powershell
# omplete setup in one block (copy-paste)
cd "serstom.cursorworktreeslpha-oop--sii"
python -m venv venv
.venvcriptsctivate.ps
pip install -r requirements.txt
opy-tem "serstomlphaloopcapital ropbox ech gents - ec .env" -estination ".env"
python scripts/test_db_connection.py
```


# ulti-achine etup - indows + acook

## unning raining on oth achines vernight

ou can run the training system on  your indows  and acook simultaneously to maximize compute power.

---

## ecommended etup

### indows 
- **un** ata ollection (`src/data_ingestion/collector.py`)
- **hy** indows often has better  connectivity
- **eep running** vernight

### acook
- **un** odel raining (`src/ml/train_models.py`)
- **hy** acook can use / for training
- **eep running** vernight

**** run both processes on both machines for redundancy!

---

## etup oth achines

### indows etup
ee `_.md` for complete instructions.

**uick start**
```powershell
cd "serstom.cursorworktreeslpha-oop--bek"
python -m venv venv
.venvcriptsctivate.ps
pip install -r requirements.txt
opy-tem "serstomlphaloopcapital ropbox ech gents - ec .env" -estination ".env"
python scripts/test_db_connection.py
```

### acook etup
ee `_.md` for complete instructions.

**uick start**
```bash
cd ~/lpha-oop-/lpha-oop--/bek
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ~/path/to/.env .env
python scripts/test_db_connection.py
```

---

## hared atabase onfiguration

oth machines will write to the same zure  database. ake sure

. **ame .env file** on both machines (copy it)
. **atabase accessible** from both networks
. **o conflicts** - processes coordinate via database

---

## unning rocesses

### indows - ata ollection
```powershell
python src/data_ingestion/collector.py
```

### acook - odel raining
```bash
python src/ml/train_models.py
```

### r un oth on ach achine

**indows erminal **
```powershell
python src/data_ingestion/collector.py
```

**indows erminal **
```powershell
python src/ml/train_models.py
```

**acook erminal **
```bash
python src/data_ingestion/collector.py
```

**acook erminal **
```bash
python src/ml/train_models.py
```

---

## onitoring oth achines

### indows
```powershell
et-ontent logsdata_collection.log -ail 
et-ontent logsmodel_training.log -ail 
```

### acook
```bash
tail -f logs/data_collection.log
tail -f logs/model_training.log
```

---

## eep achines wake

### indows
- ower ettings → ever sleep when plugged in
- r `powercfg /change standby-timeout-ac `

### acook
```bash
# revent sleep
caffeinate -d

# r run training with caffeinate
caffeinate -d python src/ml/train_models.py
```

---

## enefits of ulti-achine etup

. **aster raining** ore compute power
. **edundancy** f one machine fails, other continues
. **arallel rocessing** ifferent symbols/models on each machine
. **esource ptimization** se each machine's strengths

---

## roubleshooting ulti-achine

### atabase onflicts
- ystem handles concurrent writes automatically
- ach process has unique identifiers

### etwork ssues
- oth machines need internet for s
- atabase must be accessible from both

### ync ssues
- odels saved independently on each machine
- opy models from both machines before trading

---

## ext orning - ollect esults

### rom indows
```powershell
et-hildtem models*.pkl
```

### rom acook
```bash
ls -la models/*.pkl
```

### opy est odels
opy the best performing models from both machines to your trading machine.

---

**ou're ready to run training on both machines simultaneously!**


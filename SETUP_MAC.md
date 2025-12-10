# acook etup nstructions

 ** or complete cross-platform reference, see `__.md`](__.md)**

##    remium ata ngestion

his setup pulls  historical data from
- **lpha antage remium** tocks, ndices, urrencies, ptions, undamentals
- **assive ** + years of minute-by-minute data for all asset classes
- **dvanced aluation etrics** elta-djusted a, onvexity, /, etc.

---

## etup for vernight raining on acook

### tep  pen erminal
**n lain nglish** "pen a command window where you can type instructions"

- ress `md + pace` → ype "erminal" → ress nter
- r in ursor ress `md + ~` or go to erminal → ew erminal

### tep  avigate to roject
**n lain nglish** "o to the folder where all the code lives"

```bash
cd ~/lpha-oop-/lpha-oop--/sii
```

**ote** djust path if your project is in a different location.

### tep  et p nvironment
**n lain nglish** "reate an isolated ython workspace for this project"

```bash
# reate virtual environment (one-time setup)
python -m venv venv

# ctivate it (do this every time you open a new terminal)
source venv/bin/activate
```

**uccess** ou'll see `(venv)` at the start of your prompt

### tep  nstall ackages
**n lain nglish** "nstall all required ython packages"

```bash
pip install -r requirements.txt
```

### tep  opy nvironment ile
**n lain nglish** "opy your  keys and database credentials"

```bash
# rom ropbox
cp ~/lphaloopcapital ropbox/ ech gents/ - ec .env .env

# r from iloud
cp ~/ibrary/obile ocuments/com~apple~loudocs/lpha oop /.env .env
```

### tep  est atabase
**n lain nglish** "ake sure we can connect to the database"

```bash
python scripts/test_db_connection.py
```

---

##      (acook)

### ption    - ll ata ources (ecommended)

**erminal  - assive  ( years backfill)**
**n lain nglish** "ull + years of minute-by-minute data"

```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
caffeinate -d python scripts/hydrate_massive.py & | tee logs/massive.log
```

his pulls
- tocks (equity/minute/) -  years of -minute bars
- ptions (option/minute/) -  reeks (delta, gamma, theta, vega)
- ndices (index/minute/) - & , , etc.
- urrencies (forex/minute/) - ajor  pairs

**erminal  - lpha antage remium (continuous)**
**n lain nglish** "ull fundamentals and premium intraday data"

```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
caffeinate -d python scripts/hydrate_all_alpha_vantage.py & | tee logs/alpha_vantage.log
```

his pulls
- tock intraday (-minute, full history)
- tock daily (+ years)
- tock fundamentals (/, /, , ltman , etc.)
- ndices (, , )
- orex pairs (/, /, etc.)

**erminal  - odel raining (runs continuously)**
**n lain nglish** "rain machine learning models on collected data"

```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
caffeinate -d python src/ml/train_models.py & | tee logs/training.log
```

### ption  tandard ata ollection (impler)

**erminal  - ata ollection**
```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
caffeinate -d python src/data_ingestion/collector.py
```

**erminal  - odel raining**
```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
caffeinate -d python src/ml/train_models.py
```

---

## ac-pecific otes

| opic | etails |
|-------|---------|
| ython | se `python` instead of `python` |
| aths | se forward slashes `~/path/to/file` |
| ctivate | `source venv/bin/activate` |
| iew logs | `tail -f file` |
| revent sleep | `caffeinate -d` |

### eep acook wake vernight

**ption  revent leep (imple)**
```bash
# un before starting any long process
caffeinate -d
```

**ption  revent leep for pecific uration**
```bash
# eep awake for  hours ( seconds)
caffeinate -t 
```

**ption  ystem references**
- ystem references → attery → ower dapter
- urn off "ut hard disks to sleep when possible"
- et "urn display off after" to a longer time

### iew ogs on ac
```bash
# ata collection (live updates)
tail -f logs/data_collection.log

# odel training (live updates)
tail -f logs/model_training.log

# heck how many models
ls -la models/*.pkl | wc -l
```

---

## unning oth indows and ac imultaneously

ou can run training on  machines simultaneously

| achine | ecommended ask |
|---------|-----------------|
| indows | ata collection |
| acook | odel training |

r vice versa - both can run the same processes. hey'll both write to the same database.

---

##   (  ) - tart rading

**n lain nglish** "tart the trading engine before market opens"

```bash
cd ~/lpha-oop-/lpha-oop--/sii
source venv/bin/activate
python src/trading/execution_engine.py
```

**rerequisites**  /ateway running (aper port , ive port )

---

##      

### rom assive 
| ata ype | etails |
|-----------|---------|
| **tocks** | + years of -minute  bars |
| **ptions** | ull chains with reeks (delta, gamma, theta, vega, rho) |
| **ndices** | & , , ow,  |
| **orex** | ajor pairs (/, /, etc.) |

### rom lpha antage remium
| ata ype | etails |
|-----------|---------|
| **ntraday** | -minute bars, up to  years |
| **aily** | + years of daily bars |
| **undamentals** | /, , /, /, /, /ales |
| **rofitability** | argins, , ,  |
| **ealth** | urrent atio, ebt/quity, ltman  |

### dvanced etrics alculated
- **elta-djusted a** ortfolio risk for options
- **onvexity** on-linear price sensitivity
- **raham umber** ntrinsic value estimate
- **iotroski -core** alue investing score
- **ltman -core** ankruptcy risk predictor

---

## roubleshooting

### "python command not found"
se `python` instead
```bash
python -m venv venv
python src/data_ingestion/collector.py
```

### "ermission denied"
ake scripts executable
```bash
chmod +x scripts/*.sh
```

### atabase connection fails
- ake sure  erver is accessible from ac
- heck firewall settings
- erify .env file has correct credentials
```bash
ls -la .env
grep "_" .env
```

### ac goes to sleep
se caffeinate with any command
```bash
caffeinate -d python src/ml/train_models.py
```

---

##   

. **un hydration scripts overnight** - hey pull years of data
. **se multiple terminals** - un assive + lpha antage + raining simultaneously
. **eep ac awake** `caffeinate -d` prevents sleep
. **onitor logs** heck progress with `tail -f logs/*.log`

---

## uick eference

```bash
# omplete setup in one block (copy-paste)
cd ~/lpha-oop-/lpha-oop--/sii
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ~/nerive/lpha oop / - ec .env .env
python scripts/test_db_connection.py
```

**uilt for lpha oop apital - nstitutional-rade ong/hort uant edge und**

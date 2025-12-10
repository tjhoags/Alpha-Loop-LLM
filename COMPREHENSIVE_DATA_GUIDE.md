#    
## lpha oop apital - nstitutional-rade ata ipeline

**ast pdated** ecember   
**tatus**   - ull hrottle ode

---

## 

his system pulls **** historical data from your premium lpha antage  subscription
- **tocks** (quities) -  years of -minute bars
- **ndices** (, , , , ) -  years
- **urrencies** (, , , etc.) -  years
-  **ptions** (with reeks elta, amma, heta, ega, ) -  year
-  **rypto** (-, -) - oinbase
-  **acro** ( ed unds, , nemployment, , iquidity)
-  **dvanced undamentals** (/,  ield, , ltman -core, iotroski -core)

---

##    (indows)

### tep  erminal etup
```powershell
cd "serstomlpha-oop-lpha-oop--"
python -m venv venv
.venvcriptsctivate.ps
pip install -r requirements.txt
opy-tem "serstomlphaloopcapital ropbox ech gents - ec .env" -estination ".env"
python scripts/test_db_connection.py
```

### tep  tart vernight ata ollection
```powershell
# erminal  - ata ollection (  )
python src/data_ingestion/collector.py
```

**his will collect**
- quities from lpha antage  (assive) + olygon +  
- ndices from lpha antage 
- urrencies from lpha antage 
- ptions with reeks from lpha antage 
- rypto from oinbase
- acro from 
- dvanced undamentals from lpha antage

### tep  tart odel raining (arallel erminal)
```powershell
# erminal  - odel raining
python src/ml/train_models.py
```

---

##    (ac)

### tep  erminal etup
```bash
cd ~/lpha-oop-/lpha-oop--
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ~/lphaloopcapital ropbox/ ech gents/ - ec .env .env
python scripts/test_db_connection.py
```

### tep  tart vernight ata ollection
```bash
# erminal  - ata ollection
caffeinate -d python src/data_ingestion/collector.py
```

### tep  tart odel raining
```bash
# erminal  - odel raining
caffeinate -d python src/ml/train_models.py
```

---

##    & 

### .  (tocks)
**ources**
- **lpha antage  (assive)** - rimary source,  years of -minute bars
- **olygon ** -  years of -minute bars
- **lpha antage ** - ecent high-resolution data

**ymbols** onfigured in `src/config/settings.py` → `target_symbols`
- efault , , , , , , , , , , , 

**torage** `price_bars` table in 

---

### . 
**ource** lpha antage  (assive)

**ymbols** , , , , 

**torage** `price_bars` table

---

### . /
**ource** lpha antage  (assive)

**airs** , , , , 

**torage** `price_bars` table

---

### .  (with reeks)
**ource** lpha antage  (assive)

**reeks ncluded**
- **elta** - rice sensitivity to underlying
- **amma** - elta sensitivity (convexity)
- **heta** - ime decay
- **ega** - olatility sensitivity
- **** - mplied olatility

**torage** `options_bars` table

**ote** ptions data is . ystem limits to top  underlying symbols by default.

---

### . 
**ource** oinbase ro 

**ymbols** -, -

**torage** `price_bars` table

---

### .  
**ource**  (ederal eserve conomic ata)

**ndicators**
- **** - ederal unds ate
- **** - onsumer rice ndex
- **** - nemployment ate
- **** -  (olatility ndex)
- **** - -ear reasury ield
- **** - - ield urve
- **** - ed otal ssets (iquidity)
- **** - reasury eneral ccount
- **** - vernight everse epo

**torage** `macro_indicators` table

---

### .  
**ource** lpha antage 

**etrics omputed**

**aluation**
- /, /ales, /
- /, , /, /
-  ield,  argin

**rofitability**
- , , 
- ross argin, et argin, perating argin
-  (ash eturn on apital nvested)

**everage & olvency**
- ebt/quity, ebt/ssets
- nterest overage
- urrent atio, uick atio

**fficiency**
- sset urnover
- nventory urnover
- eceivables urnover

**isk etrics**
- **ltman -core** - ankruptcy risk (  .  high risk)
- **iotroski -core** - inancial strength (-, higher  better)
- **uality core** - omposite quality metric (-)

**perating everage**
- perating everage ratio

**torage** `fundamentals` table

---

##      

he system computes **institutional-grade quant metrics** for  training

### alue at isk (a)
- **a %** - % confidence level
- **a %** - % confidence level
- **a (xpected hortfall)** - verage of tail losses

### ail isk
- **kewness** - symmetry of returns
- **urtosis** - at tails (extreme events)
- **ail isk roxy** - robability of extreme moves (  std dev)

### onvexity
- **onvexity roxy** - econd derivative of returns (acceleration)
- **eturn cceleration** - ate of change of returns

### rawdown etrics
- **rawdown** - urrent drawdown from peak
- **ax rawdown** - aximum drawdown in rolling window

### isk-djusted eturns
- **harpe roxy** - eturn / olatility
- **ortino roxy** - eturn / ownside eviation

---

##    (+ eatures)

### echnical ndicators (+)
- eturns (, , ,  periods)
- og eturns
- olatility (, ,  windows)
-  (, )
- tochastic scillator
- illiams %
- 
- s (, , , )
- 
- 
- ollinger ands
- 
- eltner hannel
- 
- 
- olume -score

### uant etrics (+)
- a (%, %)
- a
- kewness
- urtosis
- ail isk
- onvexity
- rawdown
- ax rawdown
- harpe roxy
- ortino roxy

### arket icrostructure (+)
- pread roxy
- mihud lliquidity
- rice mpact

### attern etection (+)
- oji andles
- ap p/own
- igher igh / ower ow

---

## ️   

### `price_bars` able
tores all price data (equities, indices, currencies, crypto)

**olumns**
- `symbol` ()
- `timestamp` ()
- `open` ()
- `high` ()
- `low` ()
- `close` ()
- `volume` ()
- `source` () - 'alpha_vantage_s', 'polygon', 'coinbase', etc.

### `options_bars` able
tores options data with reeks

**olumns**
- `symbol` () - ption ticker
- `underlying` () - nderlying stock
- `timestamp` ()
- `open`, `high`, `low`, `close`, `volume`
- `delta`, `gamma`, `theta`, `vega` ()
- `iv` () - mplied olatility
- `strike` ()
- `expiry` ()
- `source` ()

### `macro_indicators` able
tores  macro data

**olumns**
- `symbol` () - eries  (e.g., '')
- `timestamp` ()
- `value` ()
- `source` () - 'fred'

### `fundamentals` able
tores advanced fundamental metrics

**olumns**
- `symbol` ()
- `timestamp` ()
- `ev`, `ev_ebitda`, `ev_sales`, `ev_fcf`
- `fcf`, `fcf_yield`, `fcf_margin`
- `roic`, `roe`, `roa`
- `gross_margin`, `net_margin`, `operating_margin`
- `croci`
- `pe_ratio`, `peg_ratio`, `pb_ratio`, `ps_ratio`
- `debt_equity`, `debt_assets`, `interest_coverage`
- `current_ratio`, `quick_ratio`
- `asset_turnover`, `inventory_turnover`
- `altman_z_score`, `bankruptcy_risk`
- `piotroski_f_score`
- `quality_score`
- `operating_leverage`

---

## ️ 

### nvironment ariables (.env)

**atabase**
```
_alc-sql-server.database.windows.net
_alc_market_data
_your_username
_your_password
```

**s**
```
___your_key
__your_key
__your_s_key
__your_s_secret
__your_key
__your_key
```

**rading**
```
_...
_  # aper, ive
__
```

---

##    

he collector runs **endlessly** by default, collecting data every  minutes

```python
# n src/config/settings.py
data_collection_forever bool  rue
data_collection_interval_minutes int  
```

o run once and exit
```python
data_collection_forever bool  alse
```

---

##  

### iew ogs (indows)
```powershell
et-ontent logsdata_collection.log -ail 
et-ontent logsmodel_training.log -ail 
```

### iew ogs (ac)
```bash
tail -f logs/data_collection.log
tail -f logs/model_training.log
```

### heck atabase
```python
python scripts/test_db_connection.py
```

---

##   

ou'll know everything is working when

.  ata collection logs show " ...", " ...", etc.
.  ogs show row counts for each asset class
.   database has data in `price_bars`, `options_bars`, `macro_indicators`, `fundamentals`
.  odel training logs show "raining oost...", "raining ight..."
.  `models/` folder has `.pkl` files after training

---

##  

### "assive  credentials not found"
→ dd `__` and `__` to `.env`

### "o equity files found in "
→ heck  credentials and bucket access

### "atabase connection fails"
→ erify  credentials in `.env` and test with `python scripts/test_db_connection.py`

### "ptions collection failed"
→ ptions data is huge. ystem limits to top  symbols. djust in `collector.py` if needed.

### "ac goes to sleep"
→ se `caffeinate -d` before running scripts

---

##   

. **un data collection overnight** - et it pull  years of history
. **rain models** - odels will use all + features including quant metrics
. **onitor logs** - nsure all asset classes are collecting
. **heck ** - erify data is being stored correctly
. **tart trading** - t  , run `python src/trading/execution_engine.py`

---

**uilt for lpha oop apital - nstitutional-rade rading ystem**

**his is the , - version with  data sources**


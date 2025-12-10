# COMPREHENSIVE DATA COLLECTION GUIDE
## Alpha Loop Capital - Institutional-Grade Data Pipeline

**Last Updated:** December 2024  
**Status:** PRODUCTION READY - Full Throttle Mode

---

## 🎯 OVERVIEW

This system pulls **ALL** historical data from your premium Alpha Vantage S3 subscription:
- ✅ **Stocks** (Equities) - 5 years of 1-minute bars
- ✅ **Indices** (SPY, QQQ, IWM, DIA, VIX) - 5 years
- ✅ **Currencies** (EURUSD, GBPUSD, USDJPY, etc.) - 5 years
- ✅ **Options** (with Greeks: Delta, Gamma, Theta, Vega, IV) - 1 year
- ✅ **Crypto** (BTC-USD, ETH-USD) - Coinbase
- ✅ **Macro** (FRED: Fed Funds, CPI, Unemployment, VIX, Liquidity)
- ✅ **Advanced Fundamentals** (EV/EBITDA, FCF Yield, ROIC, Altman Z-Score, Piotroski F-Score)

---

## 🚀 QUICK START (Windows)

### Step 1: Terminal Setup
```powershell
cd "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item "C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env" -Destination ".env"
python scripts/test_db_connection.py
```

### Step 2: Start Overnight Data Collection
```powershell
# Terminal 1 - Data Collection (ALL ASSET CLASSES)
python src/data_ingestion/collector.py
```

**This will collect:**
- Equities from Alpha Vantage S3 (Massive) + Polygon + AV API
- Indices from Alpha Vantage S3
- Currencies from Alpha Vantage S3
- Options with Greeks from Alpha Vantage S3
- Crypto from Coinbase
- Macro from FRED
- Advanced Fundamentals from Alpha Vantage

### Step 3: Start Model Training (Parallel Terminal)
```powershell
# Terminal 2 - Model Training
python src/ml/train_models.py
```

---

## 🚀 QUICK START (Mac)

### Step 1: Terminal Setup
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ~/OneDrive/Alpha\ Loop\ LLM/API\ -\ Dec\ 2025.env .env
python scripts/test_db_connection.py
```

### Step 2: Start Overnight Data Collection
```bash
# Terminal 1 - Data Collection
caffeinate -d python src/data_ingestion/collector.py
```

### Step 3: Start Model Training
```bash
# Terminal 2 - Model Training
caffeinate -d python src/ml/train_models.py
```

---

## 📊 DATA SOURCES & COVERAGE

### 1. EQUITIES (Stocks)
**Sources:**
- **Alpha Vantage S3 (Massive)** - Primary source, 5 years of 1-minute bars
- **Polygon API** - 2 years of 1-minute bars
- **Alpha Vantage API** - Recent high-resolution data

**Symbols:** Configured in `src/config/settings.py` → `target_symbols`
- Default: SPY, QQQ, IWM, DIA, AAPL, MSFT, NVDA, AMD, GOOGL, META, TSLA, AMZN

**Storage:** `price_bars` table in SQL

---

### 2. INDICES
**Source:** Alpha Vantage S3 (Massive)

**Symbols:** SPY, QQQ, IWM, DIA, VIX

**Storage:** `price_bars` table

---

### 3. CURRENCIES/FOREX
**Source:** Alpha Vantage S3 (Massive)

**Pairs:** EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD

**Storage:** `price_bars` table

---

### 4. OPTIONS (with Greeks)
**Source:** Alpha Vantage S3 (Massive)

**Greeks Included:**
- **Delta** - Price sensitivity to underlying
- **Gamma** - Delta sensitivity (convexity)
- **Theta** - Time decay
- **Vega** - Volatility sensitivity
- **IV** - Implied Volatility

**Storage:** `options_bars` table

**Note:** Options data is HUGE. System limits to top 10 underlying symbols by default.

---

### 5. CRYPTO
**Source:** Coinbase Pro API

**Symbols:** BTC-USD, ETH-USD

**Storage:** `price_bars` table

---

### 6. MACRO INDICATORS
**Source:** FRED (Federal Reserve Economic Data)

**Indicators:**
- **FEDFUNDS** - Federal Funds Rate
- **CPIAUCSL** - Consumer Price Index
- **UNRATE** - Unemployment Rate
- **VIXCLS** - VIX (Volatility Index)
- **DGS10** - 10-Year Treasury Yield
- **T10Y2Y** - 10Y-2Y Yield Curve
- **WALCL** - Fed Total Assets (Liquidity)
- **WTREGEN** - Treasury General Account
- **RRPONTSYD** - Overnight Reverse Repo

**Storage:** `macro_indicators` table

---

### 7. ADVANCED FUNDAMENTALS
**Source:** Alpha Vantage API

**Metrics Computed:**

**Valuation:**
- EV/EBITDA, EV/Sales, EV/FCF
- P/E, PEG, P/B, P/S
- FCF Yield, FCF Margin

**Profitability:**
- ROIC, ROE, ROA
- Gross Margin, Net Margin, Operating Margin
- CROCI (Cash Return on Capital Invested)

**Leverage & Solvency:**
- Debt/Equity, Debt/Assets
- Interest Coverage
- Current Ratio, Quick Ratio

**Efficiency:**
- Asset Turnover
- Inventory Turnover
- Receivables Turnover

**Risk Metrics:**
- **Altman Z-Score** - Bankruptcy risk (Z < 1.8 = high risk)
- **Piotroski F-Score** - Financial strength (0-9, higher = better)
- **Quality Score** - Composite quality metric (0-1)

**Operating Leverage:**
- Operating Leverage ratio

**Storage:** `fundamentals` table

---

## 🔬 QUANT METRICS IN FEATURE ENGINEERING

The system computes **institutional-grade quant metrics** for ML training:

### Value at Risk (VaR)
- **VaR 95%** - 95% confidence level
- **VaR 99%** - 99% confidence level
- **CVaR (Expected Shortfall)** - Average of tail losses

### Tail Risk
- **Skewness** - Asymmetry of returns
- **Kurtosis** - Fat tails (extreme events)
- **Tail Risk Proxy** - Probability of extreme moves (> 2 std dev)

### Convexity
- **Convexity Proxy** - Second derivative of returns (acceleration)
- **Return Acceleration** - Rate of change of returns

### Drawdown Metrics
- **Drawdown** - Current drawdown from peak
- **Max Drawdown** - Maximum drawdown in rolling window

### Risk-Adjusted Returns
- **Sharpe Proxy** - Return / Volatility
- **Sortino Proxy** - Return / Downside Deviation

---

## 📈 FEATURE ENGINEERING (100+ Features)

### Technical Indicators (50+)
- Returns (1, 5, 10, 20 periods)
- Log Returns
- Volatility (10, 20, 50 windows)
- RSI (7, 14)
- Stochastic Oscillator
- Williams %R
- CCI
- EMAs (5, 10, 20, 50)
- MACD
- ADX
- Bollinger Bands
- ATR
- Keltner Channel
- OBV
- VWAP
- Volume Z-score

### Quant Metrics (20+)
- VaR (95%, 99%)
- CVaR
- Skewness
- Kurtosis
- Tail Risk
- Convexity
- Drawdown
- Max Drawdown
- Sharpe Proxy
- Sortino Proxy

### Market Microstructure (10+)
- Spread Proxy
- Amihud Illiquidity
- Price Impact

### Pattern Detection (10+)
- Doji Candles
- Gap Up/Down
- Higher High / Lower Low

---

## 🗄️ SQL DATABASE SCHEMA

### `price_bars` Table
Stores all price data (equities, indices, currencies, crypto)

**Columns:**
- `symbol` (VARCHAR)
- `timestamp` (DATETIME)
- `open` (FLOAT)
- `high` (FLOAT)
- `low` (FLOAT)
- `close` (FLOAT)
- `volume` (FLOAT)
- `source` (VARCHAR) - 'alpha_vantage_s3', 'polygon', 'coinbase', etc.

### `options_bars` Table
Stores options data with Greeks

**Columns:**
- `symbol` (VARCHAR) - Option ticker
- `underlying` (VARCHAR) - Underlying stock
- `timestamp` (DATETIME)
- `open`, `high`, `low`, `close`, `volume`
- `delta`, `gamma`, `theta`, `vega` (FLOAT)
- `iv` (FLOAT) - Implied Volatility
- `strike` (FLOAT)
- `expiry` (DATETIME)
- `source` (VARCHAR)

### `macro_indicators` Table
Stores FRED macro data

**Columns:**
- `symbol` (VARCHAR) - Series ID (e.g., 'FEDFUNDS')
- `timestamp` (DATETIME)
- `value` (FLOAT)
- `source` (VARCHAR) - 'fred'

### `fundamentals` Table
Stores advanced fundamental metrics

**Columns:**
- `symbol` (VARCHAR)
- `timestamp` (DATETIME)
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

## ⚙️ CONFIGURATION

### Environment Variables (.env)

**Database:**
```
SQL_SERVER=alc-sql-server.database.windows.net
SQL_DB=alc_market_data
DB_USERNAME=your_username
DB_PASSWORD=your_password
```

**APIs:**
```
ALPHA_VANTAGE_API_KEY=your_key
POLYGON_API_KEY=your_key
MASSIVE_ACCESS_KEY=your_s3_key
MASSIVE_SECRET_KEY=your_s3_secret
COINBASE_API_KEY=your_key
FRED_API_KEY=your_key
```

**Trading:**
```
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # 7497=Paper, 7496=Live
IBKR_CLIENT_ID=1
```

---

## 🔄 CONTINUOUS DATA COLLECTION

The collector runs **endlessly** by default, collecting data every 5 minutes:

```python
# In src/config/settings.py
data_collection_forever: bool = True
data_collection_interval_minutes: int = 5
```

To run once and exit:
```python
data_collection_forever: bool = False
```

---

## 📊 MONITORING

### View Logs (Windows)
```powershell
Get-Content logs\data_collection.log -Tail 50
Get-Content logs\model_training.log -Tail 50
```

### View Logs (Mac)
```bash
tail -f logs/data_collection.log
tail -f logs/model_training.log
```

### Check Database
```python
python scripts/test_db_connection.py
```

---

## 🎯 SUCCESS CRITERIA

You'll know everything is working when:

1. ✅ Data collection logs show "COLLECTING EQUITIES...", "COLLECTING INDICES...", etc.
2. ✅ Logs show row counts for each asset class
3. ✅ SQL database has data in `price_bars`, `options_bars`, `macro_indicators`, `fundamentals`
4. ✅ Model training logs show "Training XGBoost...", "Training LightGBM..."
5. ✅ `models/` folder has `.pkl` files after training

---

## 🚨 TROUBLESHOOTING

### "Massive S3 credentials not found"
→ Add `MASSIVE_ACCESS_KEY` and `MASSIVE_SECRET_KEY` to `.env`

### "No equity files found in S3"
→ Check S3 credentials and bucket access

### "Database connection fails"
→ Verify SQL credentials in `.env` and test with `python scripts/test_db_connection.py`

### "Options collection failed"
→ Options data is huge. System limits to top 10 symbols. Adjust in `collector.py` if needed.

### "Mac goes to sleep"
→ Use `caffeinate -d` before running scripts

---

## 📚 NEXT STEPS

1. **Run data collection overnight** - Let it pull 5 years of history
2. **Train models** - Models will use all 100+ features including quant metrics
3. **Monitor logs** - Ensure all asset classes are collecting
4. **Check SQL** - Verify data is being stored correctly
5. **Start trading** - At 9:15 AM, run `python src/trading/execution_engine.py`

---

**Built for Alpha Loop Capital - Institutional-Grade Trading System**

**This is the COMPREHENSIVE, PRODUCTION-READY version with ALL data sources**


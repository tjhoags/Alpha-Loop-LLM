# 🚀 FULL THROTTLE SETUP - COMPREHENSIVE DATA INGESTION

## ⚡ QUICK START (7:30 AM START IN 6 HOURS)

### Windows (PowerShell):
```powershell
cd "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item "C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env" -Destination ".env"
python scripts/test_db_connection.py

# START FULL THROTTLE (opens 3 terminals automatically)
.\scripts\start_full_throttle_training.ps1
```

### Mac (Terminal):
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ~/OneDrive/Alpha\ Loop\ LLM/API\ -\ Dec\ 2025.env .env
python scripts/test_db_connection.py

# START FULL THROTTLE (opens 3 terminals automatically)
bash scripts/start_full_throttle_training.sh
```

---

## 📊 WHAT'S BEING COLLECTED

### 1. Massive S3 (5 Years Backfill)
- **Stocks**: `equity/minute/` - 5+ years of 1-minute OHLCV bars
- **Options**: `option/minute/` - Full options chains WITH Greeks:
  - Delta (price sensitivity)
  - Gamma (delta sensitivity)
  - Theta (time decay)
  - Vega (volatility sensitivity)
  - Rho (interest rate sensitivity)
- **Indices**: `index/minute/` - S&P 500, NASDAQ, Dow, VIX
- **Forex**: `forex/minute/` - Major currency pairs

### 2. Alpha Vantage Premium (Continuous)
- **Stock Intraday**: 1-minute bars, full history (up to 2 years)
- **Stock Daily**: 20+ years of daily bars
- **Stock Fundamentals** (Advanced Valuation Metrics):
  - **Valuation Ratios**: P/E, PEG, P/B, P/S, EV/EBITDA, EV/Sales
  - **Profitability**: Profit Margin, Operating Margin, ROE, ROA, ROIC
  - **Growth**: Revenue Growth, Earnings Growth
  - **Financial Health**: Current Ratio, Quick Ratio, Debt/Equity
  - **Risk Metrics**: Altman Z-Score (bankruptcy risk), Beta
  - **Value Metrics**: Graham Number (intrinsic value), Piotroski F-Score
- **Indices**: SPX, NDX, DJI daily data
- **Forex**: Major pairs (USD/EUR, USD/JPY, etc.)

### 3. Advanced Quant Metrics Calculated
- **Delta-Adjusted VaR**: Portfolio risk for options positions
- **Convexity**: Non-linear price sensitivity (Gamma exposure)
- **Enterprise Value**: Market Cap + Debt - Cash
- **Free Cash Flow Yield**: FCF / Market Cap
- **ROIC**: Return on Invested Capital

---

## 🤖 TRAINING PROCESS

### Models Trained:
1. **XGBoost** (300 trees, depth 5, learning rate 0.05)
2. **LightGBM** (400 trees, learning rate 0.05)
3. **CatBoost** (300 iterations, learning rate 0.05)

### Features Used:
- **100+ Technical Indicators**: RSI, MACD, EMA, Bollinger Bands, ATR, etc.
- **Valuation Metrics**: P/E, EV/EBITDA, ROIC, Altman Z-Score, etc.
- **Market Microstructure**: Volume Z-score, Amihud illiquidity, spread proxies
- **Pattern Detection**: Doji candles, gaps, higher highs/lower lows

### Training Process:
- **Time-Series Cross-Validation**: 3 splits (no data leakage)
- **Metrics Logged**: Accuracy, AUC, Precision, Recall, F1, Sharpe-like, Max Drawdown
- **Continuous Retraining**: Models retrain every hour with fresh data
- **Model Versioning**: Timestamped saves with metadata

---

## 📈 GRADING & METRICS

### Model Performance Metrics:
- **CV Accuracy**: Must be > 52% (beats random)
- **CV AUC**: Must be > 0.52 (beats random)
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum loss in validation

### Risk Management:
- **Daily Loss Limit**: 2% max daily loss (kill switch)
- **Drawdown Limit**: 5% max drawdown (kill switch)
- **Position Sizing**: Kelly-capped (25% max), 10% per position
- **Max Positions**: 10 concurrent positions

---

## 🎯 TOMORROW MORNING (9:15 AM ET)

### Start Trading Engine:
```powershell
# Windows
cd "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
.\venv\Scripts\Activate.ps1
python src/trading/execution_engine.py
```

```bash
# Mac
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate
python src/trading/execution_engine.py
```

**Make sure IBKR TWS/Gateway is running** (paper trading port 7497 by default)

---

## 📊 MONITORING

### View Logs:
```powershell
# Windows
Get-Content logs\massive_ingest.log -Tail 50 -Wait
Get-Content logs\alpha_vantage_hydration.log -Tail 50 -Wait
Get-Content logs\model_training.log -Tail 50 -Wait
Get-Content logs\trading_engine.log -Tail 50 -Wait
```

```bash
# Mac
tail -f logs/massive_ingest.log
tail -f logs/alpha_vantage_hydration.log
tail -f logs/model_training.log
tail -f logs/trading_engine.log
```

### Check Models:
```powershell
# Windows
Get-ChildItem models\*.pkl | Sort-Object LastWriteTime -Descending
```

```bash
# Mac
ls -lt models/*.pkl
```

---

## 🔧 TROUBLESHOOTING

### "Module not found"
→ Activate venv: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (Mac)

### "Database connection fails"
→ Check `.env` file has correct SQL credentials (DB_SERVER, DB_DATABASE, DB_USERNAME, DB_PASSWORD)

### "No data collected"
→ Check API keys in `.env`:
  - `ALPHA_VANTAGE_API_KEY`
  - `POLYGON_API_KEY`
  - `MASSIVE_ACCESS_KEY` and `MASSIVE_SECRET_KEY`

### Mac goes to sleep
→ Use: `caffeinate -d` (prevents sleep)

### Rate limiting (Alpha Vantage)
→ System handles automatically with 12-second delays between calls

---

## ✅ SUCCESS CRITERIA

You'll know everything is working when:

1. **Data Collection**:
   - `logs/massive_ingest.log` shows "✅ Massive backfill COMPLETE: X rows ingested"
   - `logs/alpha_vantage_hydration.log` shows "✅ Stock hydration complete: X rows stored"

2. **Model Training**:
   - `logs/model_training.log` shows "CV acc=X.XXX auc=X.XXX" for each model
   - `models/` folder has `.pkl` files with timestamps

3. **Trading Engine** (tomorrow morning):
   - `logs/trading_engine.log` shows "Signal generated" messages
   - Orders are placed via IBKR (check TWS/Gateway)

---

## 🎯 PORTFOLIO FOCUS: US Small/Mid-Cap (< $25B AUM)

### Current Symbols:
- Default: SPY, QQQ, IWM, DIA, AAPL, MSFT, NVDA, AMD, GOOGL, META, TSLA, AMZN
- **Add your small/mid-cap universe** in `.env` or `src/config/settings.py`:
  ```python
  target_symbols = ["YOUR_SYMBOL_1", "YOUR_SYMBOL_2", ...]
  ```

### Risk Adjustments for Small/Mid-Cap:
- Consider tighter position caps (5-7% instead of 10%) for thinner liquidity
- Monitor `amihud` (illiquidity) metric in features
- Use `volume_z` to avoid low-volume traps

---

## 🚨 CRITICAL NOTES

1. **Never commit `.env` file** - Contains API keys
2. **Test in paper trading first** - IBKR_PORT=7497 is paper trading
3. **Monitor logs continuously** - Check for errors
4. **Keep machines awake** - Use `caffeinate -d` on Mac, power settings on Windows
5. **Start data collection early** - Needs time to gather historical data
6. **Risk limits are HARD** - System will pause trading if breached

---

**Built for Alpha Loop Capital - Institutional-Grade Long/Short Quant Hedge Fund**

**This is the FINAL, PRODUCTION-READY version with comprehensive data ingestion**


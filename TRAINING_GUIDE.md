# Training Guide - Full Universe

## QUICK START

### Step 1: Hydrate Data (Choose One)

**Option A - Quick Test (10-20 min)**
```
Double-click: scripts\HYDRATE_QUICK.bat
```
- 500 stocks + 200 ETFs + 20 crypto

**Option B - Full Universe (6-12 hours)**
```
Double-click: scripts\HYDRATE_FULL_UNIVERSE.bat
```
- 8,000+ stocks
- 2,500+ ETFs
- 100+ crypto
- Options chains

### Step 2: Train Models

**Run training:**
```
Double-click: scripts\TRAIN_MASSIVE.bat
```

This trains on EVERYTHING in the database with:
- 100+ features (technical + behavioral)
- 3 model types (XGBoost, LightGBM, CatBoost)
- Auto-checkpoint (can pause and resume)

---

## WHAT'S BEING TRAINED

### Asset Coverage
| Asset Class | Count | Training Features |
|-------------|-------|-------------------|
| US Stocks | 8,000+ | Technical + Behavioral |
| ETFs | 2,500+ | Technical + Behavioral |
| Crypto | 100+ | Technical + Behavioral |
| Options | 100,000+ contracts | Stored for strategy use |

### Feature Categories (100+ Total)

**Technical Features (60+)**
- Price dynamics (returns, volatility, log returns)
- Momentum (RSI, Williams %R, CCI, ROC)
- Trend (EMAs, MACD, ADX, crossovers)
- Volatility (ATR, Bollinger, Keltner)
- Volume (OBV, VWAP, volume ratios)
- Microstructure (spread proxy, Amihud illiquidity)
- Time-based (hour, day of week, month, session)
- Pattern detection (doji, gaps, higher highs)

**Behavioral Features (40+)**
- Emotional: Fear/greed composite, panic score, FOMO
- Crowd: Herding proxy, cascade score, contrarian signal
- Cognitive Bias: Anchoring, recency, loss aversion
- Game Theory: Squeeze setup, coordination, informed trading

### Model Types
1. **XGBoost** - Fast, handles missing data well
2. **LightGBM** - Memory efficient, great for large datasets
3. **CatBoost** - Best with categorical features

---

## GRADING CRITERIA

Models must pass ALL criteria to be saved:

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| AUC | >= 0.52 | Better than random |
| Accuracy | >= 0.51 | Slight edge |
| Sharpe Ratio | >= 0.5 | Risk-adjusted return |
| Max Drawdown | <= 20% | Risk control |

### Why These Thresholds?

**52% accuracy = Profitable** with proper position sizing:
- 1000 trades/year
- 52% win rate
- 1:1 risk/reward
- Expected: +40 net winning trades
- With 1% risk per trade: ~40% annual return before costs

---

## OVERNIGHT TRAINING PLAN

### Windows (Lenovo) - Primary Training Machine

**Terminal 1: Data Hydration**
```powershell
cd "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
.\venv\Scripts\activate
python scripts\hydrate_full_universe.py
```

**Terminal 2: Model Training (after hydration starts)**
```powershell
cd "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
.\venv\Scripts\activate
python src\ml\massive_trainer.py
```

### Mac - Support Tasks

**Terminal 1: Research & Sentiment**
```bash
cd ~/Alpha-Loop-LLM
source venv/bin/activate
python src/nlp/research_ingestion.py
```

**Terminal 2: Alternative Data**
```bash
cd ~/Alpha-Loop-LLM
source venv/bin/activate
python src/data_ingestion/social_sentiment.py
```

---

## CHECKPOINTING & RESUME

Training auto-saves progress every 100 symbols.

**To pause:** Press Ctrl+C
**To resume:** Run the same command again

Progress saved in: `models/training_checkpoint.json`

---

## MONITORING PROGRESS

**Check trained models:**
```powershell
(Get-ChildItem models\*.pkl).Count
```

**Check checkpoint:**
```powershell
Get-Content models\training_checkpoint.json | ConvertFrom-Json
```

**Watch logs:**
```powershell
Get-Content logs\training.log -Tail 50 -Wait
```

---

## ESTIMATED TIMES

| Task | Symbols | Time |
|------|---------|------|
| Quick Hydration | 720 | 10-20 min |
| Full Hydration | 10,000+ | 6-12 hours |
| Quick Training | 720 | 2-4 hours |
| Full Training | 10,000+ | 12-24 hours |

**Speed:** ~500-1000 symbols/hour depending on data quality

---

## TROUBLESHOOTING

**"Insufficient data" for many symbols**
- Normal for thinly traded stocks
- System skips and moves on

**Training seems stuck**
- Check if rate-limited by API
- Reduce batch_size to 20
- Check logs/training.log

**Out of memory**
- Reduce max_workers to 2
- Reduce batch_size to 20

**Want to restart fresh**
```powershell
del models\training_checkpoint.json
python src\ml\massive_trainer.py --fresh
```



# Alpha Loop Capital - Training Guide

## Overview

The training system consists of:
1. **Data Collection** - Pull market data from APIs
2. **Model Training** - Train ML models (XGBoost, LightGBM, CatBoost)
3. **Model Grading** - Validate and grade trained models

---

## Quick Start

### Option A: Quick Test (20-30 min)
```bash
# Windows
python scripts/hydrate_quick.py

# Mac
caffeinate -d python scripts/hydrate_quick.py
```

### Option B: Full Universe (6-12 hours)
```bash
# Windows
python scripts/hydrate_full_universe.py

# Mac
caffeinate -d python scripts/hydrate_full_universe.py
```

---

## Overnight Training Plan

### Terminal 1: Data Collection
```bash
# Windows
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\hfc"
.\venv\Scripts\Activate.ps1
python src/data_ingestion/collector.py

# Mac
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/hfc
source venv/bin/activate
caffeinate -d python src/data_ingestion/collector.py
```

### Terminal 2: Model Training
```bash
# Windows
python src/ml/train_models.py

# Mac
caffeinate -d python src/ml/train_models.py
```

### Terminal 3: Monitoring
```bash
# Windows
Get-Content logs\training.log -Tail 50 -Wait

# Mac
tail -f logs/training.log
```

---

## Model Grading Criteria

Models must pass ALL criteria:

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| AUC | >= 0.52 | Better than random |
| Accuracy | >= 0.51 | Slight edge |
| Sharpe Ratio | >= 0.5 | Risk-adjusted return |
| Max Drawdown | <= 20% | Risk control |

### Why 52% Accuracy = Profitable

- 1,000 trades/year
- 52% win rate, 1:1 risk/reward
- **Expected**: +40 net winning trades
- With 1% risk per trade: ~40% annual return before costs

---

## Feature Categories (100+)

### Technical Features (60+)
- Price dynamics (returns, volatility, log returns)
- Momentum (RSI, Williams %R, CCI, ROC)
- Trend (EMAs, MACD, ADX, crossovers)
- Volatility (ATR, Bollinger, Keltner)
- Volume (OBV, VWAP, volume ratios)
- Microstructure (spread proxy, Amihud illiquidity)

### Behavioral Features (40+)
- Emotional: Fear/greed, panic score, FOMO
- Crowd: Herding proxy, cascade score, contrarian signal
- Cognitive Bias: Anchoring, recency, loss aversion
- Game Theory: Squeeze setup, coordination

---

## Dual-Machine Training

| Machine | Recommended Task |
|---------|------------------|
| Windows (Lenovo) | Data collection |
| Mac (MacBook Pro) | Model training |

Both can run simultaneously - they write to the same Azure SQL database.

---

## Checkpointing & Resume

Training auto-saves every 100 symbols.

- **Pause**: Press `Ctrl+C`
- **Resume**: Run the same command again
- **Progress file**: `models/training_checkpoint.json`

---

## Estimated Times

| Task | Symbols | Time |
|------|---------|------|
| Quick Hydration | ~720 | 20-30 min |
| Full Hydration | 10,000+ | 6-12 hours |
| Quick Training | ~720 | 2-4 hours |
| Full Training | 10,000+ | 12-24 hours |

---

## Monitoring Commands

### Check Trained Models
```bash
# Windows
(Get-ChildItem models\*.pkl).Count

# Mac
ls models/*.pkl | wc -l
```

### Check Checkpoint
```bash
# Windows
Get-Content models\training_checkpoint.json

# Mac
cat models/training_checkpoint.json
```

### Watch Logs
```bash
# Windows
Get-Content logs\training.log -Tail 50 -Wait

# Mac
tail -f logs/training.log
```

---

## Troubleshooting

### "Insufficient data" warnings
Normal for thinly-traded stocks - system skips and continues.

### Training seems stuck
- Check for API rate limits
- Reduce `batch_size` to 20
- Check `logs/training.log`

### Out of memory
- Reduce `max_workers` to 2
- Reduce `batch_size` to 20

### Fresh restart
```bash
# Delete checkpoint
del models\training_checkpoint.json  # Windows
rm models/training_checkpoint.json   # Mac

# Restart training
python src/ml/massive_trainer.py --fresh
```

---

*© 2025 Alpha Loop Capital, LLC*


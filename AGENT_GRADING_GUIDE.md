# AGENT & GRADING SYSTEM GUIDE

## Understanding Your Trading Agents

This guide explains how agents work, how they're graded, and how to interpret training progress.

---

## THE AGENTS

### What is an Agent?
An **Agent** is a trained ML model specialized for a specific symbol with specific hyperparameters. Each agent:
- Has a unique configuration (lookback window, learning rate, etc.)
- Is trained on historical data for ONE symbol
- Gets graded on its predictive ability
- Can be "promoted" to production or "terminated"

### Agent Types

| Agent Type | Strategy | Best For | Key Features |
|------------|----------|----------|--------------|
| **MOMENTUM** | Trend-following | Trending markets | High momentum features, longer lookback |
| **MEAN_REVERSION** | Counter-trend | Range-bound markets | Bollinger bands, RSI extremes |
| **SENTIMENT** | NLP-driven | News-sensitive stocks | Social sentiment, news analysis |
| **MACRO** | Economic indicators | Market-wide moves | FRED data, yield curves |
| **VOLATILITY** | Vol strategies | High-vol periods | VIX, options Greeks |
| **LIQUIDITY** | Microstructure | All conditions | Spread, volume analysis |
| **ENSEMBLE** | Combined | General use | All features |

### Agent Lifecycle

```
CREATED → TRAINING → VALIDATING → [PROMOTED/REJECTED]
                                       ↓
                               ACTIVE (trading)
                                       ↓
                               SUSPENDED/TERMINATED
```

---

## THE GRADING SYSTEM

### Key Metrics Explained

#### 1. **AUC (Area Under ROC Curve)** - Most Important
```
AUC = 0.50 → Random guessing (coin flip)
AUC = 0.52 → Minimum threshold (small edge)
AUC = 0.55 → Good predictive power
AUC = 0.60 → Excellent (rare for financial data)
AUC > 0.65 → Suspicious (possible overfitting)
```

**What it means**: AUC measures how well the model separates "price goes up" from "price goes down" predictions. Higher = better discrimination.

#### 2. **Accuracy** - Directional Correctness
```
Accuracy = 50% → Random
Accuracy = 52% → Minimum threshold
Accuracy = 55% → Very good for finance
Accuracy > 60% → Check for data leakage
```

**What it means**: Percentage of correct directional predictions.

#### 3. **Precision** - When We Say "Buy", How Often Right?
```
Precision = 50% → Half our buy signals are wrong
Precision = 55% → Acceptable
Precision = 60% → Good
```

**What it means**: When the model says "BUY", what percentage actually goes up?

#### 4. **Recall** - How Many Ups Did We Catch?
```
Recall = 50% → We miss half the up moves
Recall = 70% → We catch most up moves
```

**What it means**: Of all the times price went up, how many did we predict?

#### 5. **F1 Score** - Balance of Precision & Recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## GRADING THRESHOLDS

### Minimum Requirements for Production

| Metric | Threshold | Why |
|--------|-----------|-----|
| AUC | > 0.52 | Must beat random |
| Accuracy | > 52% | Must be right more than wrong |
| Precision | > 50% | Buy signals must be profitable |
| Max Drawdown | < 10% | Risk control |

### What Happens When Training

```
START OF TRAINING:
┌─────────────────────────────────────────────────────────┐
│ All metrics start at ~0.50 (random)                     │
│ Early epochs: Metrics fluctuate wildly                  │
│ This is NORMAL - model is learning patterns             │
└─────────────────────────────────────────────────────────┘

DURING TRAINING (1-3 hours):
┌─────────────────────────────────────────────────────────┐
│ AUC should climb from 0.50 → 0.51 → 0.52 → 0.53       │
│ Accuracy should stabilize above 0.52                    │
│ If AUC plateaus at 0.51 = Model struggling              │
│ If AUC drops after rising = Overfitting starting        │
└─────────────────────────────────────────────────────────┘

END OF TRAINING PHASE → CONTINUOUS IMPROVEMENT:
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⚠️  TRAINING NEVER STOPS - Only the weak stop at "good enough"              │
│                                                                             │
│ PHASE 1 (Baseline): AUC > 0.52, Accuracy > 52% → Model enters production    │
│ PHASE 2 (Competitive): AUC > 0.55, Accuracy > 55% → Wall Street parity      │
│ PHASE 3 (Elite): AUC > 0.58, Accuracy > 58% → Beating Wall Street           │
│ PHASE 4 (Dominant): AUC > 0.62, Accuracy > 62% → Alpha Loop Standard        │
│                                                                             │
│ 🔄 ACA FEEDBACK LOOP:                                                       │
│    - Underperforming models → ACA spawns improvement agents automatically   │
│    - Successful patterns → ACA clones and mutates for exploration           │
│    - Speed bottlenecks → ACA spawns parallel processing agents              │
│                                                                             │
│ 📊 LOGGING CONFIRMED: All metrics, decisions, spawns logged to audit trail  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## INTERPRETING TRAINING PROGRESS

### Good Signs (Training Working)

✅ **AUC steadily increasing** (0.50 → 0.52 → 0.53)
✅ **CV scores consistent across folds** (not varying wildly)
✅ **Precision and Recall both improving**
✅ **Training loss decreasing**

### Warning Signs

⚠️ **AUC stuck at 0.50-0.51** → Not enough signal in data
⚠️ **High variance between CV folds** → Model unstable
⚠️ **Train accuracy much higher than test** → Overfitting
⚠️ **AUC > 0.65** → Likely data leakage, too good to be true

### Bad Signs (Abort/Investigate)

❌ **AUC decreasing over time** → Model degrading
❌ **All models for symbol failing** → Data quality issue
❌ **Memory errors** → Reduce batch size
❌ **NaN values in metrics** → Feature engineering bug

---

## UNDERSTANDING FEASIBILITY

### How Grading Shows Feasibility

```
If MOST models pass (>70%):
→ Data has predictable patterns
→ Strategy is FEASIBLE
→ Confidence: HIGH

If SOME models pass (40-70%):
→ Some patterns exist, some noise
→ Strategy is FEASIBLE with caveats
→ Confidence: MEDIUM

If FEW models pass (<40%):
→ Weak patterns or noisy data
→ Strategy has LOW feasibility
→ Consider: Different features, more data, different symbols

If NO models pass (0%):
→ No detectable patterns
→ Strategy NOT FEASIBLE as configured
→ Action: Check data quality, feature engineering, or symbol selection
```

### Symbol-Specific Feasibility

```
High Feasibility Symbols (typically easier to predict):
- SPY, QQQ (liquid, lots of data)
- Large-cap tech (AAPL, MSFT, GOOGL)

Medium Feasibility:
- Mid-cap stocks (your target universe)
- Sector ETFs

Low Feasibility (harder):
- Small-cap (less data, more noise)
- Crypto (high volatility, regime changes)
- Meme stocks (social-driven, unpredictable)
```

---

## AGENT SKILLS (What Each Component Does)

### Feature Engineering Skills
| Skill | What It Does | Helps With |
|-------|--------------|------------|
| **Momentum Features** | RSI, MACD, ROC | Trend detection |
| **Mean Reversion** | Bollinger %B, Z-scores | Overbought/oversold |
| **Volume Analysis** | OBV, Volume Z | Confirmation signals |
| **Volatility** | ATR, BB Width | Risk sizing |
| **Microstructure** | Spread proxy, Amihud | Liquidity assessment |

### Behavioral Finance Skills
| Skill | What It Does | Helps With |
|-------|--------------|------------|
| **Sentiment Analysis** | Social media scoring | Retail flow prediction |
| **Fear/Greed Index** | Market psychology | Contrarian timing |
| **Herding Detection** | Crowd behavior | Mean reversion signals |
| **Anchor Detection** | Price level psychology | Support/resistance |
| **Cascade Detection** | Information flow | Momentum vs reversal |

### Valuation Skills
| Skill | What It Does | Helps With |
|-------|--------------|------------|
| **DCF Valuation** | Intrinsic value | Long-term fair value |
| **Factor Scoring** | Value/Growth/Quality | Stock selection |
| **Peer Analysis** | Relative valuation | Sector positioning |

---

## REAL-TIME GRADING INTERPRETATION

### During Overnight Training

**Hour 1-2:**
```
Expected: AUC = 0.50-0.51 (learning)
Log shows: "Training XGBoost for SPY..."
Status: NORMAL - Don't panic if metrics low
```

**Hour 3-4:**
```
Expected: AUC = 0.51-0.53 (improving)
Log shows: "SPY | XGBoost | AUC=0.52, Acc=0.53"
Status: GOOD - Models finding patterns
```

**Hour 5-6:**
```
Expected: Final validation
Log shows: "✅ PROMOTED: AUC 0.53 > 0.52"
Status: SUCCESS - Model ready for trading
```

**Morning (9:15 AM):**
```
Check: models/ folder has .pkl files
Run: python src/trading/execution_engine.py
Status: READY FOR MARKET OPEN
```

---

## IMPROVING MODEL PERFORMANCE

### If Models Aren't Passing

1. **More Data**: Longer history improves pattern detection
2. **Better Features**: Add behavioral, sentiment features
3. **Hyperparameter Tuning**: Adjust learning rate, depth
4. **Different Symbols**: Some symbols more predictable

### Hyperparameter Quick Guide

```python
# For HIGHER ACCURACY (slower training):
n_estimators = 800-1000
learning_rate = 0.005-0.01
max_depth = 6-8

# For FASTER TRAINING (lower accuracy):
n_estimators = 300-500
learning_rate = 0.05-0.10
max_depth = 4-5

# For LESS OVERFITTING:
subsample = 0.7
colsample_bytree = 0.7
min_child_weight = 5
```

---

## MONITORING COMMANDS

### Check Training Progress
```powershell
# View live training log
Get-Content logs\overnight_training.log -Tail 50 -Wait

# Check how many models saved
Get-ChildItem models\*.pkl | Measure-Object
```

### Check Model Quality
```powershell
# See all promoted models
Get-ChildItem models\*.pkl | ForEach-Object { $_.Name }
```

---

## CRITICAL UNDERSTANDING

### Why 52% Accuracy is Good

In finance, **52% accuracy with proper risk management = profitable**.

```
Math:
- 100 trades
- 52 wins, 48 losses
- Average win = 1.2% (take profit)
- Average loss = 1.0% (stop loss)

Result: (52 × 1.2%) - (48 × 1.0%) = 62.4% - 48% = +14.4% return
```

### Why AUC 0.55 is Excellent

Most quant funds would be thrilled with AUC 0.55 on out-of-sample data. 
Financial markets are:
- Highly efficient
- Full of noise
- Constantly adapting

Any consistent edge > random is valuable.

---

---

## ACA FEEDBACK LOOP SYSTEM

### Continuous Improvement Architecture

The Agent Creating Agent (ACA) system ensures **training never stops** and performance continuously improves:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACA FEEDBACK LOOP                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ MONITOR  │───▶│ ANALYZE  │───▶│ SPAWN    │───▶│ IMPROVE  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                                                │        │
│       └────────────────────────────────────────────────┘        │
│                    CONTINUOUS CYCLE                             │
└─────────────────────────────────────────────────────────────────┘
```

### ACA Triggers for New Agent Spawning

| Trigger | Action | Expected Outcome |
|---------|--------|------------------|
| **Performance < Phase Target** | Spawn improvement agent | Hyperparameter optimization |
| **Speed Bottleneck** | Spawn parallel agents | 2-10x throughput increase |
| **New Pattern Detected** | Spawn specialist agent | Capture emerging alpha |
| **Model Degradation** | Spawn replacement agent | Seamless transition |
| **Success Pattern** | Clone + Mutate agent | Explore adjacent strategies |

### Logging Confirmation

All operations are logged for audit and improvement:

```python
# Confirmed logging locations:
src/core/agent_base.py     # 17 log points - Agent lifecycle
src/core/learning_engine.py # 41 log points - Training progress
src/core/aca_engine.py     # 12 log points - Agent spawning
src/trading/agent_factory.py # MetaAgent operations
```

**Log outputs:**
- `logs/training_*.log` - All training metrics
- `logs/aca_*.log` - Agent creation/termination
- `logs/performance_*.log` - Live trading metrics
- Azure Blob Storage - Persistent audit trail

---

## ELITE PERFORMANCE TARGETS

### Beat Wall Street Benchmarks

| Metric | Wall Street Average | Alpha Loop Target | Why |
|--------|--------------------|--------------------|-----|
| **Sharpe Ratio** | 0.8 - 1.2 | > 2.0 | Risk-adjusted returns |
| **Hit Rate** | 52-54% | > 58% | Directional accuracy |
| **AUC** | 0.52-0.55 | > 0.58 | Signal quality |
| **Max Drawdown** | 15-20% | < 10% | Capital preservation |
| **Recovery Time** | 3-6 months | < 4 weeks | Resilience |

### Continuous Improvement Mindset

```
❌ WRONG: "We passed the threshold, we're done"
✅ RIGHT: "We passed baseline, now let's dominate"

Wall Street's edge: Resources, infrastructure, talent
Our edge: AI agents that NEVER STOP IMPROVING

Every second an agent isn't improving, we're falling behind.
ACA ensures continuous optimization 24/7/365.
```

---

## SUMMARY

1. **Agents** = Specialized ML models that CONTINUOUSLY IMPROVE
2. **Grading** = Phase 1 baseline (52%), Phase 4 target (62%+)
3. **NO STOPPING** = Training never ends, only intensifies
4. **ACA System** = Automatic spawning of improvement agents
5. **Beat Wall Street** = Sharpe > 2.0, AUC > 0.58, DD < 10%
6. **Logging** = 100+ log points across core systems, audit trail confirmed
7. **Speed** = ACA spawns parallel agents for bottleneck elimination

**Alpha Loop Capital Standard: We don't compete with Wall Street. We outperform them.**

*"By end of 2026, they will know Alpha Loop Capital."*


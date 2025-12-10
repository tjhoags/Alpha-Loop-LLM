# Alpha Loop Capital - Model Grading System Explained

**Document Version:** 1.0
**Last Updated:** December 10, 2025
**Author:** Claude Code for Tom Hogan

---

## Table of Contents

1. [Overview](#overview)
2. [The Two Grading Systems](#the-two-grading-systems)
3. [Core Metrics Explained](#core-metrics-explained)
4. [Grade Tiers & What They Mean](#grade-tiers--what-they-mean)
5. [Production Thresholds](#production-thresholds)
6. [Category Scoring Breakdown](#category-scoring-breakdown)
7. [How to Interpret Your Training Results](#how-to-interpret-your-training-results)
8. [Current Training Status](#current-training-status)
9. [Commands & Quick Reference](#commands--quick-reference)

---

## Overview

Your Alpha Loop Capital grading system is designed to compete with **Citadel, Goldman Sachs, Two Sigma, and Renaissance Technologies**. The philosophy is simple:

> "If you're not getting A grades, you're not ready for live capital."

The system uses **institutional-grade thresholds** - these are HARD requirements. No participation trophies.

---

## The Two Grading Systems

### 1. Institutional Grading (`src/core/grading.py`)

The primary grading system that evaluates agents across 6 categories:

| Category | Weight | What It Measures |
|----------|--------|------------------|
| **Performance** | 25% | Raw predictive ability (AUC, accuracy, Sharpe) |
| **Execution** | 15% | Operational excellence (success rate, trade count) |
| **Learning** | 15% | Continuous improvement (adaptation rate) |
| **Battle** | 15% | Resilience under pressure (crash survival, regime adaptation) |
| **Alpha** | 20% | Unique edge (original insights, contrarian wins) |
| **Competitive** | 10% | Performance vs SPY and peers |

### 2. Elite Grading (`src/core/elite_grading.py`)

A higher-bar system with proprietary metrics for hedge fund-level readiness:

- Alpha half-life (edge must persist 30+ days)
- Regime consistency (profitable in 70%+ of market conditions)
- Black swan survival (80%+ survival in tail events)
- Crowding score (strategy can't be >30% crowded)

---

## Core Metrics Explained

### AUC (Area Under ROC Curve)
**What it is:** Measures how well the model separates "winners" from "losers"

| Value | Interpretation | What It Means For You |
|-------|----------------|----------------------|
| 0.50 | Random | No skill - coin flip |
| 0.52 | Minimal edge | Barely better than random |
| 0.55 | Weak signal | Starting to see patterns |
| 0.58 | Decent signal | Model is learning |
| 0.62 | Good signal | Competitive territory |
| 0.65 | Strong signal | Hedge fund quality |
| 0.70+ | Excellent | Rare - verify not overfitting |

**Your threshold:** Minimum 0.52 for production, Elite is 0.58

### Accuracy
**What it is:** Percentage of correct directional predictions (up vs down)

| Value | Interpretation | What It Means For You |
|-------|----------------|----------------------|
| 50% | Random | Coin flip accuracy |
| 52% | Slight edge | Minimum viable |
| 54% | Weak edge | Building skill |
| 56% | Decent edge | Getting there |
| 58% | Good edge | Production territory |
| 60%+ | Strong edge | Excellent performance |

**Your threshold:** Minimum 53% for institutional, Elite is 57%

### Precision
**What it is:** When model says "BUY", how often is it right?

| Value | Interpretation | What It Means For You |
|-------|----------------|----------------------|
| < 40% | Poor | Too many false signals |
| 40-50% | Weak | Needs improvement |
| 50%+ | Acceptable | Minimum for production |
| 60%+ | Good | Reliable buy signals |
| 70%+ | Excellent | High confidence trades |

**Your threshold:** Minimum 50% required (this is why many models are being rejected)

### Sharpe Ratio
**What it is:** Risk-adjusted return (return per unit of risk)

| Value | Interpretation | What It Means For You |
|-------|----------------|----------------------|
| 0.0 | No edge | Not making money risk-adjusted |
| 0.5 | Below average | Work in progress |
| 1.0 | Acceptable | Industry minimum |
| 1.5 | Good | **Your minimum threshold** |
| 2.0 | Very good | Competitive with top funds |
| 2.5 | Excellent | Elite territory |
| 3.0+ | Outstanding | Top 1% funds |

**Your threshold:** Minimum 1.5 for production, Elite is 2.5

### Max Drawdown
**What it is:** Worst peak-to-trough loss (risk control measure)

| Value | Interpretation | What It Means For You |
|-------|----------------|----------------------|
| < 3% | Excellent | Ultra-conservative risk |
| < 5% | Good | **Your elite threshold** |
| < 8% | Acceptable | **Your minimum threshold** |
| < 10% | Moderate | Getting risky |
| < 15% | High | Critical limit |
| > 15% | Dangerous | **CRITICAL FAILURE** - caps grade at C |

**Your threshold:** Maximum 8% for institutional, Elite is 5%

### Win Rate
**What it is:** Percentage of profitable trades

| Value | Interpretation |
|-------|----------------|
| < 50% | Losing more than winning |
| 52% | **Minimum threshold** |
| 55% | Good |
| 58% | **Elite threshold** |
| 60%+ | Excellent |

### Profit Factor
**What it is:** Gross profits divided by gross losses

| Value | Interpretation |
|-------|----------------|
| < 1.0 | Losing money overall |
| 1.2 | **Minimum threshold** - $1.20 made for every $1 lost |
| 1.5 | Good |
| 1.8 | **Elite threshold** |
| 2.0+ | Excellent |

---

## Grade Tiers & What They Mean

### Letter Grade Mapping

| Grade | Score Range | Meaning | Production Status |
|-------|-------------|---------|-------------------|
| **A+** | 95-100 | Elite - Top 1% hedge fund quality | PRODUCTION READY |
| **A** | 85-94 | Excellent - Institutional quality | PRODUCTION READY |
| **B+** | 80-84 | Good - Needs polish | PAPER TRADING READY |
| **B** | 70-79 | Acceptable - Significant gaps | PAPER TRADING READY |
| **C** | 60-69 | Below standard - Major issues | NEEDS RETRAINING |
| **D** | 50-59 | Failing - Not production ready | START OVER |
| **F** | 0-49 | Failed completely | START OVER |

### Critical Failure Rules

Even with a high numeric score, certain failures **cap your grade at C**:

1. **AUC < 0.51** - No predictive ability
2. **Max Drawdown > 15%** - Unacceptable risk
3. **Success Rate < 80%** (with 50+ executions) - Unreliable
4. **Negative Sharpe Ratio** - Losing money risk-adjusted
5. **No Learning** (< 10 outcomes with 100+ executions) - Not adapting

---

## Production Thresholds

### Minimum Requirements for Live Trading

```
AUC          ≥ 0.52  (must beat random significantly)
Accuracy     ≥ 52%   (must be right more than wrong)
Precision    ≥ 50%   (buy signals must be profitable)
Sharpe       ≥ 1.5   (risk-adjusted returns)
Max Drawdown ≤ 8%    (risk control)
Win Rate     ≥ 52%   (more winners than losers)
Profit Factor ≥ 1.2  (profits exceed losses)
```

### Elite Requirements (Top 5%)

```
AUC          ≥ 0.58
Accuracy     ≥ 57%
Sharpe       ≥ 2.5
Max Drawdown ≤ 5%
Win Rate     ≥ 58%
Profit Factor ≥ 1.8
```

---

## Category Scoring Breakdown

### 1. Performance Score (25% of total)

Calculated from:
- AUC vs thresholds (+15 if elite, -20 if below minimum)
- Accuracy vs thresholds (+15 if elite, -20 if below minimum)
- Sharpe ratio vs thresholds (+15 if elite, -15 if below minimum)
- Drawdown vs thresholds (+10 if elite, -15 if above maximum)

### 2. Execution Score (15% of total)

Calculated from:
- Success rate (90% minimum, 98% elite)
- Execution count (100 minimum, 1000 elite)
- Capability breadth (5 minimum, 15 elite)

### 3. Learning Score (15% of total)

Calculated from:
- Learning events (100 minimum, 1000 elite)
- Recent accuracy (above 70% = +15, below 50% = -15)
- Adaptations made (10+ = +15)

### 4. Battle Score (15% of total)

Calculated from:
- Crashes survived (5 minimum, 50 elite)
- Drawdowns navigated (3 minimum, 20 elite)
- Regime changes adapted (2 minimum, 10 elite)
- Black swans handled (0 minimum, 3 elite)

### 5. Alpha Score (20% of total) - **THIS IS WHAT MAKES YOU DIFFERENT**

Calculated from:
- Unique insights generated (10 minimum, 100 elite)
- Contrarian wins (5 minimum, 50 elite)
- Information edges exploited (3 minimum, 30 elite)

### 6. Competitive Score (10% of total)

Calculated from:
- SPY outperformance (2% minimum, 10% elite)
- Peer percentile (60th minimum, 95th elite)

---

## How to Interpret Your Training Results

### Reading Training Logs

When you see output like:
```
TRAINING: AME
xgboost: AUC=0.767, Acc=0.823, Prec=0.228 | [FAIL] REJECTED: Prec 0.228 < 0.5
lightgbm: AUC=0.782, Acc=0.831, Prec=0.409 | [FAIL] REJECTED: Prec 0.409 < 0.5
catboost: AUC=0.765, Acc=0.812, Prec=0.503 | [PASS] PROMOTED: All criteria met
```

**What this means:**
- **High AUC (0.76-0.78)** = Model is learning patterns well
- **High Accuracy (0.81-0.83)** = Directional predictions are good
- **Low Precision (0.22-0.41)** = When it says "BUY", it's often wrong
- **Pass/Fail** = Based on ALL criteria, not just one

### Why Most Models Are Being Rejected

Looking at your current training, the main issue is **Precision < 0.50**:

| Symbol | AUC | Accuracy | Precision | Status |
|--------|-----|----------|-----------|--------|
| AME | 0.767 | 0.823 | 0.228 | REJECTED |
| AMCX | 0.765 | 0.812 | 0.503 | **PASSED** |
| AMG | 0.765 | 0.842 | 0.271 | REJECTED |

**The pattern:** AUC and Accuracy are strong, but Precision is weak.

**What this means:** The models are good at predicting the general direction but bad at specifically identifying winning buy opportunities.

### How to Improve Precision

1. **Class imbalance** - If there are more "down" days than "up" days, the model learns to predict "down" more often, hurting precision on "up" predictions
2. **Threshold tuning** - Adjust the probability threshold for what counts as a "buy" signal
3. **Feature engineering** - Add features that better identify up-moves specifically
4. **More data** - More training examples help distinguish signal from noise

---

## Current Training Status

### Summary (as of December 10, 2025, ~5:55 AM)

| Metric | Value |
|--------|-------|
| **Total Models Trained** | ~128 |
| **Models Passed** | ~10-15 (estimated) |
| **Pass Rate** | ~10-12% |
| **Primary Failure Reason** | Precision < 50% |
| **Training Status** | **ACTIVE** (still running) |

### What's Working

- AUC scores are generally strong (0.75-0.82)
- Accuracy scores are good (0.80-0.85)
- Models are training with 65 features
- Using 1453 samples per symbol

### What Needs Attention

- Precision is the main bottleneck
- Fundamental data is unavailable (SQL parameter error)
- Many models rejected despite good AUC/Accuracy

### Recommendations

1. **Let training continue** - More models = more chances for passes
2. **Fix SQL query** - The fundamental data query has a parameter issue
3. **Consider precision threshold** - Current 50% may be too strict for initial training
4. **Monitor overnight** - Best models emerge after 6-8+ hours

---

## Commands & Quick Reference

### Check Model Grades
```powershell
# Windows
cd C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1
.\venv\Scripts\activate
python scripts/model_dashboard.py

# Or use the batch file
.\scripts\CHECK_MODEL_GRADES.bat
```

### View Training Progress
```powershell
# Watch live training
Get-Content logs\overnight_training.log -Tail 50 -Wait

# Check how many models passed
Select-String "\[PASS\]" logs\overnight_training.log | Measure-Object
```

### Count Trained Models
```powershell
(Get-ChildItem models\*.pkl).Count
```

### Check Production Readiness
```powershell
python -c "from src.core.grading import is_production_ready; print(is_production_ready({'auc': 0.55, 'sharpe': 1.6, 'max_drawdown': 0.04}))"
```

---

## Grade Report Example

When you run the grading system, you'll get a report like this:

```
Agent: AAPL_momentum
Grade: B+ (82.5/100)

Category Scores:
  Performance:  78.0  (AUC=0.56, Acc=0.54, Sharpe=1.8)
  Execution:    85.0  (95% success, 250 trades)
  Learning:     75.0  (500 learning events)
  Battle:       80.0  (10 crashes survived)
  Alpha:        85.0  (25 unique insights)
  Competitive:  90.0  (5% over SPY)

Strengths:
  - Solid Sharpe ratio (1.8) - good risk-adjusted returns
  - High success rate (95%) - reliable execution

Weaknesses:
  - AUC (0.56) below elite threshold
  - Only 250 executions - needs more testing

Improvement Actions:
  - Add behavioral features (sentiment, fear/greed)
  - Run stress tests via NOBUS agent

Competitive Analysis:
  vs Citadel: APPROACHING - Close but not there yet
  vs Goldman: COMPETITIVE - Institutional quality
  vs Two Sigma: BEHIND - Focus on unique alpha

Production Ready: NO (B+ grade requires more testing)
Paper Trading Ready: YES
```

---

## Summary

| What You Need | Minimum | Elite | Your Current Status |
|---------------|---------|-------|---------------------|
| AUC | 0.52 | 0.58 | Strong (0.75-0.82) |
| Accuracy | 53% | 57% | Strong (80-85%) |
| Precision | 50% | 60% | **WEAK (22-50%)** |
| Sharpe | 1.5 | 2.5 | TBD (need backtesting) |
| Drawdown | < 8% | < 5% | TBD (need backtesting) |

**Bottom Line:** Your models are learning well (high AUC/Accuracy) but need work on precision. Continue training and consider adjusting the precision threshold or adding features specifically for identifying winning buy opportunities.

---

*This document is part of the Alpha Loop Capital trading system. By end of 2026, they will know Alpha Loop Capital.*

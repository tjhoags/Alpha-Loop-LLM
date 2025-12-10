# ALC-Algo Training Workflow - Step-by-Step

**Author:** Tom Hogan | **Organization:** Alpha Loop Capital, LLC  
**Purpose:** Detailed step-by-step training workflow for tonight's session

---

## 🎯 Tonight's Mission: Get Training Environment Ready

Training begins **December 9, 2025**. This document provides the exact steps to execute.

---

## 📋 Training Phases Overview

```
Phase 1: Environment Setup        [TONIGHT - 30 min]
Phase 2: Data Preparation         [TONIGHT - 1 hour]
Phase 3: Agent Initialization     [TONIGHT - 30 min]
Phase 4: Calibration Run          [TONIGHT - 2 hours]
Phase 5: Paper Trading Start      [TONIGHT - Ongoing]
Phase 6: Continuous Learning      [ONGOING - Forever]
```

---

## 🔧 PHASE 1: Environment Setup (30 minutes)

### Step 1.1: Verify Python Environment

```powershell
# Open PowerShell as Administrator
cd C:\Users\tom\ALC-Algo

# Check Python version
python --version
# Required: 3.10 or higher

# Create fresh virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate
```

### Step 1.2: Install Dependencies

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Verify installation
python -c "import pandas, numpy, openai, anthropic; print('Core packages: OK')"
```

### Step 1.3: Configure Secrets

```powershell
# Create secrets file
copy config\secrets.py.example config\secrets.py

# Edit with your master_alc_env path
notepad config\secrets.py
```

**Set in secrets.py:**
```python
ENV_FILE_PATH = "C:/Users/tom/Alphaloopcapital Dropbox/master_alc_env"
```

### Step 1.4: Verify Configuration

```python
# Run verification script
python -c "
from config.settings import settings

checks = [
    ('OpenAI', settings.openai_api_key),
    ('Anthropic', settings.anthropic_api_key),
    ('Google', settings.google_api_key_1),
    ('Alpha Vantage', settings.alpha_vantage_api_key),
    ('IBKR Account', settings.ibkr_account_id),
]

print('Configuration Check:')
for name, value in checks:
    status = '✓' if value else '✗'
    print(f'  {name}: {status}')
"
```

**Expected Output:**
```
Configuration Check:
  OpenAI: ✓
  Anthropic: ✓
  Google: ✓
  Alpha Vantage: ✓
  IBKR Account: ✓
```

---

## 📊 PHASE 2: Data Preparation (1 hour)

### Step 2.1: Create Data Directories

```powershell
# Ensure data directories exist
New-Item -ItemType Directory -Force -Path "data\raw"
New-Item -ItemType Directory -Force -Path "data\processed"
New-Item -ItemType Directory -Force -Path "data\portfolio_history"
New-Item -ItemType Directory -Force -Path "logs"
New-Item -ItemType Directory -Force -Path "models\trained"
```

### Step 2.2: Export Your IBKR Trades

1. **Log into IBKR Client Portal**
2. **Navigate to:** Performance & Reports → Flex Queries
3. **Create/Run Trade Confirmation Flex Query**
4. **Export as CSV** to `data\raw\ibkr_trades.csv`

Required columns:
- Date/Time
- Symbol
- Side (BUY/SELL)
- Quantity
- Price
- Commission

### Step 2.3: Import Historical Trades

```powershell
# Import your trades
python scripts\ingest_portfolio.py "data\raw\ibkr_trades.csv"
```

**What this does:**
- Parses your historical trades
- Calculates performance metrics
- Stores in standardized format
- Prepares for agent learning

### Step 2.4: Fetch Market Data

```powershell
# Fetch historical market data
python scripts\fetch_market_data.py --symbols SPY,QQQ,AAPL,MSFT,GOOGL --years 5
```

This downloads:
- Daily OHLCV data
- Adjusted close prices
- Volume data
- Dividend data

### Step 2.5: Verify Data

```python
python -c "
import pandas as pd
import os

# Check raw data
raw_files = os.listdir('data/raw')
print(f'Raw data files: {len(raw_files)}')

# Check processed data
processed_files = os.listdir('data/processed')
print(f'Processed data files: {len(processed_files)}')

# Check portfolio history
portfolio_files = os.listdir('data/portfolio_history')
print(f'Portfolio history files: {len(portfolio_files)}')
"
```

---

## 🤖 PHASE 3: Agent Initialization (30 minutes)

### Step 3.1: Initialize Core Agents

```python
# init_training.py
from main import initialize_agents
from src.core.logger import alc_logger

# Initialize all 51+ agents
print("Initializing ALC-Algo Agent Ecosystem...")
print("=" * 60)

agents = initialize_agents(user_id="TJH")

# Display summary
print(f"\nTier 1 (Master): GhostAgent")
print(f"Tier 2 (Senior): {len(agents['senior'])} agents")
print(f"Tier 3 (Swarm):  {len(agents['swarm'])} agents")
print(f"\nTotal: {1 + len(agents['senior']) + len(agents['swarm'])} agents")
```

### Step 3.2: Verify Agent Health

```python
# health_check.py
for name, agent in agents['senior'].items():
    stats = agent.get_stats()
    print(f"{stats['name']:20} | Status: {stats['status']} | Toughness: {stats['toughness']}")
```

**Expected Output:**
```
DataAgent            | Status: battle_ready | Toughness: TOM_HOGAN
StrategyAgent        | Status: battle_ready | Toughness: TOM_HOGAN
RiskAgent            | Status: battle_ready | Toughness: TOM_HOGAN
ExecutionAgent       | Status: battle_ready | Toughness: TOM_HOGAN
PortfolioAgent       | Status: battle_ready | Toughness: TOM_HOGAN
ResearchAgent        | Status: battle_ready | Toughness: TOM_HOGAN
ComplianceAgent      | Status: battle_ready | Toughness: TOM_HOGAN
SentimentAgent       | Status: battle_ready | Toughness: TOM_HOGAN
```

### Step 3.3: Initialize Senior Agents (New)

```python
from src.agents import get_skills, get_author, get_bookmaker, get_scout

# Initialize new senior agents
skills = get_skills()
author = get_author()
bookmaker = get_bookmaker()
scout = get_scout()

print("New Senior Agents Initialized:")
print(f"  SKILLS: {skills.name}")
print(f"  THE_AUTHOR: {author.name}")
print(f"  BOOKMAKER: {bookmaker.name}")
print(f"  SCOUT: {scout.name}")
```

---

## ⚡ PHASE 4: Calibration Run (2 hours)

### Step 4.1: Initial Confidence Calibration

```python
# calibrate.py
from src.agents import GhostAgent

ghost = agents['ghost']

# Run calibration for each senior agent
for name, agent in agents['senior'].items():
    print(f"\nCalibrating {name}...")
    
    # Set initial calibration
    agent._confidence_adjustment = 1.0  # Start neutral
    agent._calibration_history = []
    
    # Run 50 calibration predictions
    for i in range(50):
        # Simulate learning outcome
        agent.learn_from_outcome(
            prediction=f"calibration_test_{i}",
            actual="calibration_result",
            confidence=0.5 + (i * 0.01),  # Gradually increase confidence
            context={'mode': 'calibration'}
        )
    
    print(f"  Confidence adjustment: {agent._confidence_adjustment:.3f}")
```

### Step 4.2: Regime Detection Training

```python
# regime_training.py
# Train regime detection with historical scenarios

scenarios = [
    # Normal markets
    {'vix': 15, 'trend': 0.3, 'avg_correlation': 0.5, 'breadth': 0.6, 'credit_spread': 1.0, 'expected': 'normal'},
    # Risk-on
    {'vix': 12, 'trend': 0.8, 'avg_correlation': 0.4, 'breadth': 0.8, 'credit_spread': 0.8, 'expected': 'risk_on'},
    # Risk-off
    {'vix': 22, 'trend': -0.3, 'avg_correlation': 0.6, 'breadth': 0.4, 'credit_spread': 1.5, 'expected': 'risk_off'},
    # Stress
    {'vix': 28, 'trend': -0.5, 'avg_correlation': 0.75, 'breadth': 0.3, 'credit_spread': 2.5, 'expected': 'stress'},
    # Crisis
    {'vix': 40, 'trend': -1.0, 'avg_correlation': 0.9, 'breadth': 0.15, 'credit_spread': 4.0, 'expected': 'crisis'},
]

print("Regime Detection Training:")
for scenario in scenarios:
    expected = scenario.pop('expected')
    detected, conf = agents['senior']['risk'].detect_regime_change(scenario)
    match = "✓" if detected == expected else "✗"
    print(f"  Expected: {expected:10} | Detected: {detected:10} | {match}")
```

### Step 4.3: Learning Method Verification

```python
# verify_learning.py
print("\nLearning Methods Active:")
for agent in agents['senior'].values():
    print(f"\n{agent.name}:")
    for method in agent.learning_methods:
        print(f"  - {method.value}")
```

---

## 📈 PHASE 5: Paper Trading Start (Ongoing)

### Step 5.1: Verify Paper Mode

```python
# CRITICAL: Must be paper trading!
assert agents['senior']['execution'].broker_port == 7497, "ERROR: Not in paper mode!"
print("✓ Paper trading mode verified (port 7497)")
```

### Step 5.2: Start Paper Trading Monitor

```python
# paper_monitor.py
import time
from datetime import datetime

print("=" * 60)
print("ALC-ALGO PAPER TRADING MONITOR")
print("Mode: PAPER (Port 7497)")
print("=" * 60)

# Run continuous monitoring
while True:
    # Get market regime
    regime, conf = agents['senior']['risk'].detect_regime_change({
        'vix': 18,  # Real-time VIX
        'trend': 0.2,
    })
    
    # Get swarm signals
    swarm_signals = agents['swarm_factory'].get_stats()
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
    print(f"  Regime: {regime} (confidence: {conf:.1%})")
    print(f"  Swarm Signals: {swarm_signals['total_signals_generated']}")
    
    time.sleep(60)  # Update every minute
```

### Step 5.3: Execute First Paper Trade

```python
# first_trade.py
result = agents['senior']['execution'].execute({
    'type': 'execute_trade',
    'broker': 'ibkr',
    'ticker': 'AAPL',
    'action': 'BUY',
    'quantity': 1,
    'mode': 'PAPER',
})

print(f"First paper trade: {result}")
```

---

## 🔄 PHASE 6: Continuous Learning (Forever)

### Daily Tasks

| Time | Task | Agent |
|------|------|-------|
| 06:00 | Market data update | DataAgent |
| 06:30 | Regime assessment | RiskAgent |
| 07:00 | Signal generation | Swarm |
| 08:00 | Trading session | ExecutionAgent |
| 16:00 | Daily reconciliation | PortfolioAgent |
| 17:00 | Learning synthesis | GhostAgent |
| 18:00 | Daily report | THE_AUTHOR |

### Weekly Tasks

| Day | Task | Agent |
|-----|------|-------|
| Monday | Skills assessment | SKILLS |
| Wednesday | Model retraining | STRINGS |
| Friday | Weekly report | THE_AUTHOR |
| Sunday | Full backup | System |

---

## ✅ Training Checklist for Tonight

```
□ Phase 1: Environment Setup
  □ Python 3.10+ verified
  □ Virtual environment created
  □ Dependencies installed
  □ Secrets configured
  □ API keys verified

□ Phase 2: Data Preparation
  □ Data directories created
  □ IBKR trades exported
  □ Trades imported
  □ Market data fetched
  □ Data verified

□ Phase 3: Agent Initialization
  □ Core agents initialized
  □ Health check passed
  □ Senior agents ready
  □ Swarm agents ready

□ Phase 4: Calibration
  □ Confidence calibration complete
  □ Regime detection trained
  □ Learning methods verified

□ Phase 5: Paper Trading
  □ Paper mode verified (7497)
  □ Monitor started
  □ First trade executed

□ Phase 6: Continuous Learning
  □ Daily schedule configured
  □ Weekly schedule configured
  □ Monitoring active
```

---

## 🚨 Emergency Procedures

### If Training Fails

```powershell
# Reset and restart
deactivate
rmdir /s /q venv
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### If Agent Errors

```python
# Reset agent state
agent._learning_outcomes.clear()
agent._calibration_history.clear()
agent._confidence_adjustment = 1.0
agent.status = AgentStatus.BATTLE_READY
```

### If IBKR Connection Fails

1. Verify IBKR Gateway/TWS is running
2. Check port (7497 for paper)
3. Enable API in IBKR settings
4. Check firewall

---

## 📞 Quick Reference

| Resource | Location |
|----------|----------|
| Main entry | `main.py` |
| Agent base | `src/core/agent_base.py` |
| Settings | `config/settings.py` |
| Logs | `logs/` |
| Data | `data/` |

---

*Training Workflow v1.0*  
*Tom Hogan | Alpha Loop Capital, LLC*  
*"TONIGHT WE BEGIN. BY 2026 THEY WILL KNOW US."*


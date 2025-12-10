# ALC-Algo Training Guide - Day 0

**Author:** Tom Hogan | **Organization:** Alpha Loop Capital, LLC  
**Date:** December 9, 2025  
**Mission:** Training begins TONIGHT

---

## 🎯 Training Overview

This guide provides comprehensive instructions for training the ALC-Algo 76+ agent ecosystem. Training is not a one-time event—it's a **continuous learning loop** that improves with every trade, every decision, and every market regime change.

---

## 📋 Pre-Training Checklist

### Environment Requirements

| Component | Requirement | Status |
|-----------|-------------|--------|
| Python | 3.10+ | ☐ |
| RAM | 16GB minimum, 32GB+ recommended | ☐ |
| GPU | NVIDIA CUDA capable (optional but recommended) | ☐ |
| Storage | 50GB+ for data and models | ☐ |
| Network | Stable internet for API calls | ☐ |

### API Keys Required

```bash
# ML Protocols (ALL REQUIRED - NO LIMITS)
OPENAI_API_KEY=sk-...           # GPT-4 for complex reasoning
ANTHROPIC_API_KEY=sk-ant-...    # Claude for long-context analysis
GOOGLE_API_KEY_1=...            # Gemini for real-time
GOOGLE_API_KEY_2=...            # Backup
GOOGLE_API_KEY_3=...            # Backup
PERPLEXITY_API_KEY=pplx-...     # Web-connected research

# Data Sources
ALPHA_VANTAGE_API_KEY=...       # Market data
FMP_API_KEY=...                 # Financial Modeling Prep
FRED_API_KEY=...                # Federal Reserve data

# Execution (Paper Mode First!)
IBKR_ACCOUNT_ID=...             # Your paper account ID
IBKR_HOST=127.0.0.1
IBKR_PORT=7497                  # PAPER trading port

# Optional but Recommended
NOTION_API_KEY=...              # Documentation
SLACK_WEBHOOK_URL=...           # Alerts
```

---

## 🚀 Phase 1: Environment Setup (Tonight)

### Step 1: Clone and Configure

```powershell
# Windows PowerShell
cd C:\Users\tom\ALC-Algo

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.agents import TOTAL_AGENTS; print(f'Agents ready: {TOTAL_AGENTS}')"
```

### Step 2: Configure Secrets

```powershell
# Copy secrets template
copy config\secrets.py.example config\secrets.py

# Edit secrets.py - set your master_alc_env path
notepad config\secrets.py
```

Set `ENV_FILE_PATH` to your Dropbox location:
```python
ENV_FILE_PATH = "C:/Users/tom/Alphaloopcapital Dropbox/master_alc_env"
```

### Step 3: Verify API Connections

```python
# test_connections.py
from config.settings import settings

# Test ML APIs
print(f"OpenAI: {'✓' if settings.openai_api_key else '✗'}")
print(f"Anthropic: {'✓' if settings.anthropic_api_key else '✗'}")
print(f"Google: {'✓' if settings.google_api_key_1 else '✗'}")
print(f"Perplexity: {'✓' if settings.perplexity_api_key else '✗'}")

# Test Data APIs
print(f"Alpha Vantage: {'✓' if settings.alpha_vantage_api_key else '✗'}")

# Test Broker
print(f"IBKR Account: {'✓' if settings.ibkr_account_id else '✗'}")
print(f"IBKR Port: {settings.ibkr_port} ({'PAPER' if settings.ibkr_port == 7497 else 'LIVE'})")
```

---

## 🧠 Phase 2: Agent Initialization Training

### Step 1: Initialize All Agents

```python
from main import initialize_agents

# Initialize 51+ agents
agents = initialize_agents(user_id="TJH")

print(f"GhostAgent: {agents['ghost'].name}")
print(f"Senior Agents: {len(agents['senior'])}")
print(f"Swarm Agents: {len(agents['swarm'])}")
```

### Step 2: Run Initial Calibration

Each agent needs initial calibration to establish baseline confidence:

```python
# calibrate_agents.py
from src.agents import GhostAgent, get_skills, get_author

# Initialize master controller
ghost = GhostAgent(user_id="TJH")

# Run calibration workflow
for agent in agents['senior'].values():
    # Process calibration task
    result = agent.execute({
        'type': 'calibration',
        'mode': 'initial',
        'baseline_confidence': 0.5,
    })
    print(f"{agent.name}: Calibrated")
```

---

## 📊 Phase 3: Historical Data Training

### Data Sources to Load

1. **Your Historical Trades** (Priority 1)
   - Import from IBKR Flex Query
   - Learn from your past decisions
   
2. **Market Data** (Priority 2)
   - Historical price data (5+ years)
   - Volume and breadth data
   - Volatility regime data

3. **Alternative Data** (Priority 3)
   - Sentiment indicators
   - Flow data
   - Macro indicators

### Import Your Trades

```powershell
# Import IBKR trades
python scripts\ingest_portfolio.py "C:\path\to\your\trades.csv"

# This will:
# 1. Parse your historical trades
# 2. Calculate performance metrics
# 3. Feed to learning agents
# 4. Establish baseline patterns
```

### Backtest Historical Data

```python
from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from datetime import datetime

config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 1),
    initial_capital=100000,
    mode=BacktestMode.WALK_FORWARD,  # Out-of-sample testing
)

engine = BacktestEngine(config)

# Run backtest with all strategies
# Note: The new API requires a strategy function - see backtesting/backtest_engine.py for examples
def strategy_func(price_data, current_date):
    # Your strategy logic here
    return signals  # Dict[str, float] of target weights

results = engine.run_backtest(strategy_func, price_data)

# Analyze results
print(f"Total Return: {results.total_return:.1%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.1%}")
```

---

## 🔄 Phase 4: Continuous Learning Loop

### Learning Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS LEARNING LOOP                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌─────────┐  │
│   │  TRADE  │ ──▶ │  OUTCOME │ ──▶ │  LEARN  │ ──▶ │  ADAPT  │  │
│   └─────────┘     └──────────┘     └─────────┘     └─────────┘  │
│        ▲                                                   │      │
│        └───────────────────────────────────────────────────┘      │
│                                                                   │
│   Every trade → Outcome captured → Learning updated → Model adapt │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Learning Methods (All Active)

| Method | Purpose | Update Frequency |
|--------|---------|------------------|
| **Reinforcement** | Learn from trade outcomes | Per trade |
| **Bayesian** | Update probability beliefs | Per signal |
| **Adversarial** | Learn from mistakes | Per mistake |
| **Ensemble** | Combine model predictions | Per decision |
| **Meta** | Learn when methods work | Daily |
| **Deep** | Pattern recognition | Weekly |
| **Evolutionary** | Optimize parameters | Weekly |
| **Multi-Agent** | Learn from other agents | Continuous |

### Configure Learning Rates

```python
# learning_config.py
LEARNING_CONFIG = {
    'reinforcement': {
        'learning_rate': 0.01,
        'discount_factor': 0.95,
        'exploration_rate': 0.1,
    },
    'bayesian': {
        'prior_strength': 0.5,
        'update_smoothing': 0.9,
    },
    'adversarial': {
        'mistake_weight': 2.0,  # Learn more from mistakes
        'pattern_threshold': 3,  # Alert after 3 repeated mistakes
    },
    'calibration': {
        'window_size': 500,
        'adjustment_rate': 0.03,
    },
}
```

---

## 📈 Phase 5: Paper Trading Validation

### CRITICAL: Paper Before Live

**RULE: Paper trading (port 7497) for minimum 30 days before live (port 7496)**

```python
# paper_trading_monitor.py
from src.agents import ExecutionAgent

execution = ExecutionAgent(user_id="TJH")

# Verify paper mode
assert execution.broker_port == 7497, "MUST be paper trading!"

# Run paper trades
result = execution.execute({
    'type': 'execute_trade',
    'broker': 'ibkr',
    'ticker': 'AAPL',
    'action': 'BUY',
    'quantity': 10,
    'mode': 'PAPER',
})
```

### Paper Trading Metrics to Track

| Metric | Target | Timeframe |
|--------|--------|-----------|
| Win Rate | > 55% | 30 days |
| Sharpe Ratio | > 1.5 | 30 days |
| Max Drawdown | < 15% | 30 days |
| Signal Accuracy | > 60% | 30 days |
| Confidence Calibration | < 10% error | 30 days |

---

## 🎛️ Phase 6: Regime Training

Markets change. Agents must adapt.

### Regime Detection Training

```python
from src.core.agent_base import BaseAgent

# Train regime detection
market_data = {
    'vix': 25,
    'trend': -0.5,
    'avg_correlation': 0.7,
    'breadth': 0.3,
    'credit_spread': 1.5,
}

for agent in agents['senior'].values():
    regime, confidence = agent.detect_regime_change(market_data)
    print(f"{agent.name}: Detected {regime} regime (conf: {confidence:.1%})")
```

### Regime-Specific Strategies

| Regime | Strategy Emphasis | Risk Level |
|--------|-------------------|------------|
| **risk_on** | Momentum, Growth | Higher |
| **risk_off** | Value, Quality | Lower |
| **crisis** | Defensive, Cash | Minimal |
| **normal** | Balanced | Moderate |
| **correlated** | Diversification | Lower |

---

## 📊 Monitoring & Reporting

### Daily Training Report

```python
from src.agents.senior.author_agent import get_author

author = get_author()

# Generate daily training summary
report = author.summarize_training({
    'date': '2025-12-09',
    'agents_active': 51,
    'trades_simulated': 150,
    'learning_updates': 450,
    'regime': 'normal',
    'win_rate': 0.58,
    'sharpe': 1.8,
})

print(report.content)
```

### Weekly Skills Assessment

```python
from src.agents.senior.skills_agent import get_skills

skills = get_skills()

# Assess all agents
result = skills.execute({
    'action': 'assess_all',
})

# Generate weekly report
report = skills.generate_weekly_report()
print(report.to_email_body())
```

---

## ⚡ Quick Start Commands

```powershell
# Tonight's Training Session

# 1. Activate environment
.\venv\Scripts\Activate

# 2. Initialize agents
python main.py

# 3. Run calibration
python scripts\calibrate_agents.py

# 4. Import historical trades
python scripts\ingest_portfolio.py "path\to\trades.csv"

# 5. Start paper trading monitor
python scripts\paper_trading_monitor.py

# 6. Generate training report
python scripts\generate_training_report.py
```

---

## 🚨 Important Reminders

### DO:
- ✅ Start with paper trading (port 7497)
- ✅ Import your historical trades for personalized learning
- ✅ Monitor confidence calibration daily
- ✅ Review regime detection accuracy
- ✅ Track all trades in audit log
- ✅ Use ALL ML protocols (no limits on compute)

### DON'T:
- ❌ Skip to live trading without 30 days paper
- ❌ Ignore the 30% margin of safety rule
- ❌ Override risk limits without approval
- ❌ Disable learning loops
- ❌ Run single-model analysis (use ensembles)

---

## 📞 Support

- **Documentation:** `docs/` folder
- **Logs:** `logs/` folder
- **Audit Trail:** Automatic in ComplianceAgent
- **Notion:** Skills page for agent tracking

---

*Training Guide for ALC-Algo*  
*Tom Hogan | Alpha Loop Capital, LLC*  
*"By end of 2026, they will know us."*


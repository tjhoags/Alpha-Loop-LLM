# Quick Start: Live Trading by 9:30am

**Target Go-Live: 9:30am EST**

## ⚡ 5-Minute Setup

### Step 1: Pull Latest Code (1 min)
```bash
git fetch origin
git checkout docs/azure-setup
git pull origin docs/azure-setup
```

### Step 2: Install Dependencies (2 min)
```bash
pip install -r requirements-production.txt
```

### Step 3: Configure Environment (1 min)
```bash
# Copy and edit environment file
cp config/env_template.env .env

# Required variables:
export IBKR_USERNAME="your_username"
export IBKR_PASSWORD="your_password"
export IBKR_ACCOUNT="your_account"
export APPLICATIONINSIGHTS_CONNECTION_STRING="your_azure_connection"
```

### Step 4: Verify Setup (1 min)
```bash
python scripts/test_imports.py
python scripts/verify_infrastructure.py
```

### Step 5: Launch! (immediate)
```bash
python run_production.py
```

---

## 🎯 What's Running

### Active Trading Agents
1. **Enhanced Momentum Agent**
   - Multi-timeframe momentum (6M/3M/1M)
   - Regime-aware filtering
   - Target: 20%+ annual, 2.0+ Sharpe

2. **Mean Reversion Agent**
   - RSI + Z-score signals
   - Quick 3-10 day holds
   - Target: 65%+ win rate

### Data Sources
- **Primary**: Yahoo Finance (free, historical)
- **Realtime**: Alpha Vantage (free tier)
- **Backup**: Polygon.io, IEX Cloud (if API keys set)

### Monitoring
- Azure Application Insights (real-time metrics)
- Local logging to `data/logs/`
- Performance dashboard at Azure Portal

---

## 📊 Pre-Market Checklist (Before 9:30am)

- [ ] **System Health**
  ```bash
  python scripts/verify_infrastructure.py
  ```
  - All imports working
  - Database connected
  - Azure monitoring active

- [ ] **Market Data**
  ```python
  from src.ingest.multi_source_aggregator import MultiSourceAggregator
  agg = MultiSourceAggregator()
  data = agg.get_realtime_data(["AAPL", "SPY"])
  print(data)  # Should show latest prices
  ```

- [ ] **Agent Status**
  ```python
  from src.agents.strategies.enhanced_momentum_agent import EnhancedMomentumAgent
  agent = EnhancedMomentumAgent()
  print("Momentum agent ready:", agent)
  ```

- [ ] **Risk Limits**
  - Max 10 positions per agent
  - 10% position sizing
  - 8% stop loss
  - 20% take profit

- [ ] **Capital Allocation**
  - Starting capital: $100,000 (default)
  - Max portfolio leverage: 1.0x (fully invested)
  - Cash reserve: 20% minimum

---

## 🚀 Trading Day Workflow

### Market Open (9:30am)
1. Agents generate signals
2. Orders submitted to IBKR
3. Execution logged to Azure
4. Real-time monitoring begins

### Intraday (9:30am - 4:00pm)
- Continuous price updates (1-minute bars)
- Signal regeneration every 5 minutes
- Stop-loss monitoring every minute
- Position rebalancing as needed

### Market Close (4:00pm)
- Final P&L calculation
- Performance metrics logged
- Daily report generated
- Positions carried overnight or closed

### After Hours
- Performance analysis
- Agent retraining (if needed)
- Next day preparation

---

## 📈 Monitoring Dashboard

### Azure Portal
1. Go to Application Insights resource
2. View:
   - **Live Metrics**: Real-time performance
   - **Failures**: Any errors or exceptions
   - **Performance**: API latency, execution speed
   - **Custom Events**: Trades, signals, alerts

### Key Metrics to Watch
- **Portfolio Value**: Should increase steadily
- **Sharpe Ratio**: Keep > 1.5
- **Max Drawdown**: Keep < -15%
- **Win Rate**: Monitor vs target (60-65%)
- **Trade Count**: Verify reasonable activity

### Alerts (Auto-configured)
- ⚠️ Drawdown > -15%
- ⚠️ API errors > 10 in 5 min
- ⚠️ Position > 25% of portfolio
- ⚠️ Sharpe < 0.5

---

## 🛠️ Troubleshooting

### "No market data"
```bash
# Check data sources
python -c "from src.ingest.multi_source_aggregator import MultiSourceAggregator; agg = MultiSourceAggregator(); print(agg.sources)"

# Test Yahoo Finance
pip install yfinance
python -c "import yfinance as yf; print(yf.Ticker('AAPL').history(period='1d'))"
```

### "IBKR connection failed"
- Verify TWS or IB Gateway running
- Check port 7497 (paper) or 7496 (live)
- Confirm API enabled in TWS settings
- Verify credentials in .env

### "No signals generated"
```python
# Test signal generation
from src.agents.strategies.enhanced_momentum_agent import EnhancedMomentumAgent
import pandas as pd
import numpy as np
from datetime import datetime

agent = EnhancedMomentumAgent()
# Create test data
dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
prices = pd.DataFrame({
    "AAPL": 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
}, index=dates)

signals = agent.generate_signals(prices, {}, datetime.now())
print("Signals:", signals)
```

### "High drawdown"
1. **Immediate**: Reduce position sizes
   ```python
   # In enhanced_momentum_agent.py, change:
   position_size=0.05  # Reduce to 5% from 10%
   ```

2. **Check regime**: May be in unfavorable market
   ```python
   # Manually check regime
   agent._detect_regime(price_series, returns_series)
   ```

3. **Stop trading**: If severe
   ```python
   # Set in run_production.py:
   EMERGENCY_STOP = True  # Closes all positions
   ```

---

## 📞 Emergency Contacts

### Stop All Trading
```python
python run_production.py --stop-all-trading
```

### Close All Positions
```python
python run_production.py --close-all-positions
```

### Emergency Shutdown
```bash
# Ctrl+C in terminal, or:
pkill -f run_production.py
```

---

## 💰 Expected Performance (Live Trading)

### Conservative Scenario (Bear Market)
- Annual Return: 10-15%
- Sharpe Ratio: 1.2-1.5
- Max Drawdown: -10% to -15%
- Win Rate: 55-60%

### Base Scenario (Normal Market)
- Annual Return: 20-25%
- Sharpe Ratio: 1.8-2.2
- Max Drawdown: -8% to -12%
- Win Rate: 60-65%

### Optimistic Scenario (Bull Market)
- Annual Return: 30-40%
- Sharpe Ratio: 2.2-2.8
- Max Drawdown: -5% to -8%
- Win Rate: 65-70%

---

## 🎓 Training & Optimization

### Retrain Agents (Weekly Recommended)
```bash
# Quick retrain (10 minutes)
python src/training/rapid_training_pipeline.py

# Full retrain with hyperparameter tuning (30 minutes)
python src/training/rapid_training_pipeline.py --full --tune-hyperparameters
```

### Backtest New Strategies
```python
from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from datetime import datetime

config = BacktestConfig(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_capital=100000,
    mode=BacktestMode.WALK_FORWARD
)

engine = BacktestEngine(config)
result = engine.run_backtest(your_strategy, price_data)

print(f"Sharpe: {result.sharpe_ratio}")
print(f"Return: {result.total_return}")
```

---

## ✅ Success Criteria

After 1 week of live trading, you should see:
- [ ] Positive P&L
- [ ] Sharpe ratio > 1.0
- [ ] Max drawdown < -10%
- [ ] All trades executed without errors
- [ ] Azure monitoring showing all green

After 1 month:
- [ ] Monthly return > 1.5%
- [ ] Sharpe ratio > 1.5
- [ ] Win rate > 55%
- [ ] Less than 5 error events

After 3 months:
- [ ] Quarterly return > 5%
- [ ] Sharpe ratio > 2.0
- [ ] Max drawdown < -12%
- [ ] Consistent profitable weeks

---

## 📚 Additional Resources

- **Full Documentation**: `docs/setup/TRAINING_GUIDE.md`
- **Agent Details**: `src/agents/strategies/README.md` (if exists)
- **API Reference**: `docs/api/`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
- **Azure Setup**: `docs/azure/COMPLETE_SETUP.md`

---

## 🎯 Ready to Launch!

```bash
# Final pre-flight check
echo "System Check:" && python scripts/test_imports.py && \
echo "Infrastructure Check:" && python scripts/verify_infrastructure.py && \
echo "Ready for launch! Run: python run_production.py"
```

**Good luck! May the Sharpe be with you!** 📈🚀

---

*Last Updated: 2025-12-09*
*Author: Tom Hogan | Alpha Loop Capital, LLC*

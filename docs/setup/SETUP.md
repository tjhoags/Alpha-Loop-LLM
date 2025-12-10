# 📚 ALC-Algo Setup Guide

## Complete Instructions for Alpha Loop Capital Trading Platform

This guide provides step-by-step instructions to set up and run the ALC-Algo trading algorithm platform.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [API Configuration](#api-configuration)
4. [Loading Your Data](#loading-your-data)
5. [Running the Platform](#running-the-platform)
6. [Daily Workflow](#daily-workflow)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Python 3.10+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure "Add Python to PATH" is checked during installation

2. **Git**
   - Download from [git-scm.com](https://git-scm.com/downloads)

3. **Interactive Brokers TWS or Gateway** (for live trading)
   - Download from [interactivebrokers.com](https://www.interactivebrokers.com/en/trading/tws.php)

### Recommended

- Visual Studio Code or Cursor IDE
- PostgreSQL (for production database)
- Redis (for caching)

---

## Installation

### Step 1: Clone the Repository

```powershell
# Navigate to your projects directory
cd C:\Users\YourName\Projects

# Clone the repository
git clone https://github.com/AlphaLoopCapital/ALC-Algo.git
cd ALC-Algo
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Or for Command Prompt
venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**Note:** Some packages like `ta-lib` may require additional setup:

```powershell
# For TA-Lib on Windows, download the pre-built wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.28-cp310-cp310-win_amd64.whl
```

### Step 4: Verify Installation

```python
# Run verification script
python -c "import pandas; import numpy; import yfinance; print('Core packages OK')"
```

---

## API Configuration

### Step 1: Create Environment File

```powershell
# Copy template
copy config\env_template.env config\.env
```

### Step 2: Add Your API Keys

Edit `config/.env` with your actual API keys:

```env
# Market Data
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here

# AI Services
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Trading
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# Notifications
SLACK_BOT_TOKEN=xoxb-your-token
```

### Step 3: Import from master_alc_env (if available)

If you have your `master_alc_env` file in Dropbox:

```python
# In Python
import os
from pathlib import Path

# Read your master env file
dropbox_env = Path.home() / "Dropbox" / "master_alc_env"
if dropbox_env.exists():
    with open(dropbox_env) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                print(f"{key}=...")  # Print keys (not values) to verify
```

### API Key Sources

| Service | Where to Get Key |
|---------|------------------|
| Alpha Vantage | [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key) |
| Polygon.io | [polygon.io/dashboard/api-keys](https://polygon.io/dashboard/api-keys) |
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) |
| IBKR | Enable in TWS: Configure > API > Settings |
| Slack | [api.slack.com/apps](https://api.slack.com/apps) |

---

## Loading Your Data

### Option 1: CSV Import (Most Common)

Export your trades from your broker as CSV, then:

```python
from src.data_ingestion.portfolio_ingestion import PortfolioIngestion

# Initialize
ingestor = PortfolioIngestion()

# Load trades from CSV
trades = ingestor.load_from_csv(
    filepath="data/portfolio_history/my_trades.csv",
    broker="ibkr",  # or "schwab", "generic"
    account="main"
)

# Build portfolio
portfolio = ingestor.build_portfolio_from_trades()

# Analyze performance
analysis = ingestor.analyze_performance()

# View report
print(ingestor.generate_report())
```

### Option 2: IBKR Direct Import

```python
from src.api_clients.ibkr_client import IBKRClient

# Connect to TWS (must be running)
client = IBKRClient()
client.connect()

# Get positions
positions = client.get_positions()
print(f"Loaded {len(positions)} positions")

# Get historical data
df = client.get_historical_data("NVDA", duration="1 Y")

client.disconnect()
```

### Option 3: JSON Import

```python
# Your trades as JSON
trades_data = {
    "trades": [
        {
            "date": "2024-01-15",
            "ticker": "NVDA",
            "trade_type": "buy",
            "quantity": 100,
            "price": 450.00
        },
        # ... more trades
    ]
}

# Save to file
import json
with open("data/portfolio_history/trades.json", "w") as f:
    json.dump(trades_data, f)

# Load
ingestor = PortfolioIngestion()
trades = ingestor.load_from_json("data/portfolio_history/trades.json")
```

### CSV Format Requirements

Your CSV should have these columns (names can vary):

```csv
Date,Symbol,Action,Quantity,Price,Fees
2024-01-15,NVDA,BUY,100,450.00,1.00
2024-03-20,NVDA,SELL,50,550.00,1.00
```

---

## Running the Platform

### Daily Morning Scan

```python
from src.core.sleuth_master import SleuthMaster

# Initialize master
master = SleuthMaster()

# Run morning scan
print(master.get_daily_brief())
```

### Analyze a Specific Thesis

```python
from src.core.chain_mapper import ChainMapper, OrderLevel, ConnectionType

# Create mapper
mapper = ChainMapper()

# Create thesis
chain = mapper.create_chain(
    thesis_name="AI Data Center Boom",
    thesis_description="Massive power demand from AI training",
    thesis_category="energy_transition",
    confidence=0.8
)

# Add beneficiaries
mapper.add_beneficiary(
    chain=chain,
    ticker="NVDA",
    company_name="NVIDIA",
    order_level=OrderLevel.ZERO,
    connection_type=ConnectionType.DIRECT,
    connection_strength=0.95
)

mapper.add_beneficiary(
    chain=chain,
    ticker="VST",
    company_name="Vistra Corp",
    order_level=OrderLevel.TWO,
    connection_type=ConnectionType.INFRASTRUCTURE,
    connection_strength=0.75,
    connection_description="Power provider for data centers"
)

# Get recommendations
recs = mapper.get_recommendations(chain)
for rec in recs:
    print(f"{rec['ticker']}: Tom Score {rec['tom_score']} - {rec['suggested_action']}")
```

### Track a Narrative

```python
from src.core.narrative_tracker import NarrativeTracker

tracker = NarrativeTracker()

# Create narrative
tracker.create_narrative(
    name="Nuclear Renaissance",
    description="Nuclear energy revival for clean baseload power",
    keywords=["uranium", "nuclear", "SMR", "data center power"],
    primary_tickers=["CCJ", "UEC", "DNN"],
    secondary_tickers=["BWXT", "NXE"]
)

# Add mentions as you find them
tracker.add_mention(
    "Nuclear Renaissance",
    source="substack",
    title="Why Uranium is the Trade of 2025",
    url="https://example.com",
    author="UraniumGuy",
    sentiment="bullish"
)

# Get signal
signal = tracker.get_trading_signal("Nuclear Renaissance")
print(f"Signal: {signal['signal']} (Stage: {signal['stage']})")
```

---

## Daily Workflow

### Morning Routine (8:00 AM)

1. **Run Morning Scan**
   ```python
   from src.core.sleuth_master import SleuthMaster
   master = SleuthMaster()
   report = master.run_morning_scan()
   ```

2. **Check Flow Regime**
   ```python
   from src.core.passive_flow import PassiveFlowDetector
   detector = PassiveFlowDetector()
   print(detector.generate_report())
   ```

3. **Review Priority Actions**
   ```python
   for action in report['actions'][:5]:
       print(f"[{action['category']}] {action['ticker']}: {action['action']}")
   ```

### During Market Hours

1. **Monitor Narratives**
   ```python
   signals = tracker.get_all_signals()
   for s in signals:
       if s['signal'] in ['entry', 'exit']:
           print(f"⚠️ {s['narrative_name']}: {s['signal'].upper()}")
   ```

2. **Check Distortions Before Trading**
   ```python
   from src.core.liquidity_distortion import LiquidityDistortionDetector
   detector = LiquidityDistortionDetector()
   analysis = detector.get_ticker_analysis("NVDA")
   print(f"Short timing: {analysis['short_timing']}")
   ```

### End of Day (4:30 PM)

1. **Update Portfolio**
   ```python
   portfolio = ingestor.build_portfolio_from_trades()
   print(ingestor.generate_report())
   ```

2. **Log Trades**
   ```python
   # Export updated portfolio
   ingestor.export_portfolio("data/portfolio_history/portfolio_snapshot.json")
   ```

---

## Troubleshooting

### Common Issues

**"Module not found" errors:**
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall requirements
pip install -r requirements.txt
```

**IBKR Connection Failed:**
- Ensure TWS/Gateway is running
- Check API settings: Configure > API > Settings
- Verify port number (7497 for paper, 7496 for live)
- Enable "Enable ActiveX and Socket Clients"

**Rate Limit Errors (Alpha Vantage):**
```python
# The client auto-handles rate limiting, but you can check:
print(f"Calls made: {client.calls_made}")
# Free tier: 5/min, 500/day
```

**Slow Performance:**
- Enable Redis caching in `.env`
- Use PostgreSQL for large datasets
- Consider Polygon.io for faster data

### Getting Help

1. Check the [docs/](docs/) folder for detailed guides
2. Review error logs in [logs/](logs/)
3. Contact Tom Hogan for Alpha Loop Capital support

---

## Next Steps

1. ✅ Complete this setup guide
2. 📊 Load your historical trades
3. 🧠 Create your first thesis chain
4. 📈 Run your first morning scan
5. 🔔 Set up Slack notifications

---

*Alpha Loop Capital - Trading where algos aren't competing*


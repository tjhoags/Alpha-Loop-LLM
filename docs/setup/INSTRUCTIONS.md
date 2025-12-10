# 📋 ALC-Algo Complete Instructions

## A Natural Language Guide to the Alpha Loop Capital Trading Platform

---

## What is ALC-Algo?

ALC-Algo is your personal trading algorithm co-pilot. Think of it as a "Secondary Tom" - a system that continuously monitors markets, tracks your investment theses, and tells you what to do each morning. It's built on your existing behavioral investment philosophy and integrates with all your data sources.

---

## The Core Philosophy

The platform is built around one key insight:

> **"The best trades are Order 2-4 beneficiaries of obvious theses that algorithms haven't mapped."**

What does this mean?

- **Order 0** = The obvious play (e.g., "AI is big → buy NVDA")
- **Order 1** = Sell-side consensus (covered by analysts, crowded)
- **Order 2-4** = Hidden beneficiaries (your sweet spot)

Example: 
- Order 0: "Nuclear energy revival" → Buy uranium miners (CCJ)
- Order 2: Sprott Inc (SII) manages the uranium physical trust → AUM grows with price
- Order 3: BWX Technologies (BWXT) makes reactor components → Navy + commercial exposure

The platform helps you systematically identify and track these chains.

---

## Your 5 Behavioral Modules

### 1. Chain Mapper
Maps multi-order beneficiary chains from any investment thesis. You input a thesis, it identifies Order 0→4 opportunities and scores them using the "Tom Score."

**How to use:**
1. Think of a thesis (e.g., "AI data centers need massive power")
2. Add it to the chain mapper
3. Add beneficiaries at each order level
4. Get recommendations sorted by opportunity quality

### 2. Passive Flow Detector
Monitors structural flows into passive investments. The market's bias is "stocks only go up" because of 401k contributions and passive investing. But what happens when people lose jobs and start withdrawing?

**What it tracks:**
- Weekly ETF/mutual fund flows
- 401k contribution estimates
- Hardship withdrawal rates
- Consumer stress indicators

**The signal:** When the passive bid weakens, reduce exposure. When it strengthens, lean in.

### 3. Narrative Tracker
Investment narratives drive retail flows. This module tracks stories from emergence (Substack/podcasts) through saturation (CNBC coverage).

**Key insight:** CNBC coverage is an EXIT signal, not an entry.

**Stages:**
1. **Emergence** - Entry signal (buy)
2. **Early Adoption** - Add signal
3. **Acceleration** - Hold
4. **Mainstream** - Reduce
5. **Saturation** - EXIT
6. **Decline** - Potential short

### 4. Liquidity Distortion Detector
Identifies when price is artificially supported or suppressed by:
- Market maker gamma hedging
- Corporate buybacks
- Index rebalancing
- Short squeeze mechanics

**Use case:** Don't short a stock with massive gamma support and an active buyback. Wait for the support to expire.

### 5. Sleuth Master
The orchestrator that runs all modules and generates your daily priority list. This is your "morning briefing" - what to do today, ranked by importance.

---

## Daily Workflow

### Morning (8:00-8:30 AM)

1. **Run the Morning Scan**
   ```
   python main.py scan
   ```
   This gives you:
   - Portfolio status
   - Priority actions
   - Flow regime
   - Active signals

2. **Review Priority Actions**
   The system outputs actions ranked by urgency:
   - CRITICAL: Do immediately
   - HIGH: Do today
   - MEDIUM: Do this week
   - LOW: Monitor

3. **Check Specific Tickers**
   If you're considering a trade:
   ```
   python main.py analyze NVDA
   ```

### During Market Hours

- Monitor narrative signals for changes
- Check liquidity distortions before opening new shorts
- Update positions based on morning recommendations

### End of Day (4:30 PM)

1. Log any trades you made
2. Review portfolio changes
3. Export updated portfolio snapshot

---

## Setting Up Your Data

### Importing Your Trade History

The platform needs your historical trades to analyze your performance and improve recommendations.

**Option A: CSV Import**

Export trades from your broker and save as CSV:
```csv
Date,Symbol,Action,Quantity,Price,Fees,Notes
2024-01-15,NVDA,BUY,100,450.00,1.00,AI thesis
```

Then import:
```python
from src.data_ingestion.portfolio_ingestion import PortfolioIngestion

ingestor = PortfolioIngestion()
trades = ingestor.load_from_csv("data/portfolio_history/my_trades.csv")
portfolio = ingestor.build_portfolio_from_trades()
analysis = ingestor.analyze_performance()
```

**Option B: IBKR Direct**

If you use Interactive Brokers with TWS running:
```python
from src.api_clients.ibkr_client import IBKRClient

client = IBKRClient()
client.connect()
positions = client.get_positions()
```

### Setting Up API Keys

Copy your keys to `config/.env`:

```env
# Required for basic functionality
ALPHA_VANTAGE_API_KEY=your_key

# For AI analysis
OPENAI_API_KEY=your_key

# For notifications
SLACK_BOT_TOKEN=your_token
SLACK_CHANNEL_ALERTS=#trading-alerts

# For live trading
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
```

---

## Building Investment Theses

### Step 1: Identify the Thesis

Start with a macro observation:
- "AI requires massive compute power"
- "Nuclear energy is the only scalable clean baseload"
- "Passive flows are at risk if unemployment rises"

### Step 2: Map the Chain

```python
from src.core.chain_mapper import ChainMapper, OrderLevel, ConnectionType

mapper = ChainMapper()

chain = mapper.create_chain(
    thesis_name="AI Power Demand",
    thesis_description="AI training requires 10x the power of traditional data centers",
    thesis_category="energy_transition",
    confidence=0.8
)

# Order 0 - Obvious
mapper.add_beneficiary(chain, "NVDA", "NVIDIA", 
    OrderLevel.ZERO, ConnectionType.DIRECT, 0.95,
    connection_description="GPU manufacturer")

# Order 2 - Hidden (your zone)
mapper.add_beneficiary(chain, "VST", "Vistra Corp",
    OrderLevel.TWO, ConnectionType.INFRASTRUCTURE, 0.75,
    connection_description="Power provider for Texas data centers")

# Order 3 - Deep edge
mapper.add_beneficiary(chain, "CEG", "Constellation Energy",
    OrderLevel.THREE, ConnectionType.INFRASTRUCTURE, 0.7,
    connection_description="Nuclear utility with PPA potential")
```

### Step 3: Get Recommendations

```python
recs = mapper.get_recommendations(chain, min_tom_score=65)

for rec in recs:
    print(f"{rec['ticker']}: {rec['suggested_action']}")
```

---

## Understanding the Tom Score

The Tom Score (0-100) ranks opportunities based on:

| Component | Points | Logic |
|-----------|--------|-------|
| Order Level | 0-25 | Order 2-3 get max points |
| Competition | 0-15 | Low algo competition = bonus |
| Connection Strength | 0-10 | How certain is the link? |
| Conviction | 0-10 | Your confidence |
| Analyst Coverage | 0-10 | Fewer analysts = better |

**Interpretation:**
- 85+ = Strong Buy (full position)
- 75-84 = Buy (75% position)
- 65-74 = Accumulate (50% position)
- <65 = Watchlist

---

## Training ML Models

### Building Training Data

```python
from src.data_ingestion.dataset_builder import DatasetBuilder

builder = DatasetBuilder()

# Build dataset with features
dataset = builder.build_training_dataset(
    tickers=["NVDA", "CCJ", "VST", "SPY"],
    start_date="2020-01-01",
    include_fundamentals=True,
    include_economic=True
)

# Dataset includes:
# - Price data (OHLCV)
# - Technical indicators (RSI, MACD, Bollinger, etc.)
# - Returns (1d, 5d, 20d)
# - Fundamentals (P/E, market cap, etc.)
# - Economic indicators (from FRED)
# - Labels (forward returns)
```

### Available Features

The dataset builder adds 50+ features:
- Moving averages (SMA/EMA: 5, 10, 20, 50, 200)
- RSI, MACD, Bollinger Bands
- Volatility measures
- Volume ratios
- Price relative to MAs
- ATR (Average True Range)

---

## Notification Setup

### Slack Alerts

When enabled, you'll receive:
- Morning scan summaries
- Trade alerts
- Position warnings
- Daily reports

Setup:
1. Create Slack App at api.slack.com
2. Add permissions: `chat:write`
3. Install to workspace
4. Copy Bot Token to `.env`

### Alert Types

```python
from src.api_clients.slack_client import SlackNotifier

notifier = SlackNotifier()

# Trade alert
notifier.send_trade_alert(
    ticker="CCJ",
    action="BUY",
    reason="Uranium thesis Order 0 - accumulate on weakness",
    price=45.00,
    target=65.00,
    stop=38.00
)

# Daily report
notifier.send_daily_report(
    portfolio_value=250000,
    daily_pnl=1500,
    top_actions=[...]
)
```

---

## Troubleshooting

### "No data returned"
- Check internet connection
- Verify API key is set in `.env`
- Try a different ticker to rule out symbol issues

### "Module not found"
- Activate virtual environment: `.\venv\Scripts\Activate.ps1`
- Reinstall: `pip install -r requirements.txt`

### "IBKR connection failed"
- Ensure TWS/Gateway is running
- Check API is enabled in TWS settings
- Verify port (7497 paper, 7496 live)

### Rate limit errors
- Alpha Vantage: Wait 12 seconds between calls
- The client handles this automatically

---

## Quick Reference

### Commands

```bash
python main.py scan           # Morning scan
python main.py portfolio      # View portfolio
python main.py analyze NVDA   # Analyze ticker
python main.py thesis         # Get recommendations
python main.py dataset        # Build training data
```

### Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point for commands |
| `src/core/sleuth_master.py` | Orchestrator |
| `src/core/chain_mapper.py` | Thesis chain mapping |
| `src/data_ingestion/portfolio_ingestion.py` | Trade import |
| `config/.env` | API keys (never commit!) |

---

## Next Steps

1. ✅ Read this guide
2. 📊 Import your trade history
3. 🧠 Create your first thesis chain
4. 📈 Run your first morning scan
5. 🔔 Set up Slack notifications
6. 🤖 Build a training dataset
7. 📚 Explore the notebooks folder

---

*"Trade where algos aren't competing. The edge is in the connections they haven't mapped."*

— Alpha Loop Capital


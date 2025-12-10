# Alpha Loop LLM - Algorithmic Trading System

## Mission-Critical Production Trading System

**This is the v3.0, LLM-enhanced version** (`tjhoags/alpha-loop-llm`) built from lessons learned in:
- `/alc-algo` (original version)
- `/alc-algo-clean` (cleaned up version)
- Multiple iterations and improvements

This is a **sophisticated, institutional-grade algorithmic trading system** designed to run overnight training and execute trades by market open (9:30 AM ET).

---

## Quick Start

### Prerequisites
- Python 3.10+
- Interactive Brokers TWS/Gateway (for live/paper trading)
- Azure SQL Database access
- API keys: Polygon, Alpha Vantage, Coinbase (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/tjhoags/alpha-loop-llm.git
cd alpha-loop-llm

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
# Windows:
Copy-Item "path/to/.env" -Destination ".env"
# Mac/Linux:
cp ~/path/to/.env .env

# Test database connection
python scripts/test_db_connection.py
```

---

## System Architecture

```
alpha-loop-llm/
├── src/
│   ├── agents/           # 50+ AI trading agents
│   │   ├── specialized/  # Strategy-specific agents
│   │   ├── sectors/      # Sector analysis agents
│   │   └── core/         # Base agent infrastructure
│   ├── config/           # Configuration (Pydantic)
│   ├── core/             # Core utilities and base classes
│   ├── data_ingestion/   # Multi-source data collection
│   ├── database/         # Azure SQL Server integration
│   ├── ml/               # ML models & feature engineering
│   ├── trading/          # Execution engine & order management
│   ├── risk/             # Risk management & position sizing
│   ├── signals/          # Signal generation and aggregation
│   └── integrations/     # External service integrations
├── scripts/              # Utility scripts & helpers
├── models/               # Trained model files (.pkl)
├── data/                 # Market data storage
├── logs/                 # System logs
├── .env                  # API keys (DO NOT COMMIT)
└── requirements.txt      # Python dependencies
```

---

## Core Commands

### Data Collection
```bash
# Start data ingestion from all sources
python src/data_ingestion/collector.py

# Hydrate from Alpha Vantage
python scripts/hydrate_alpha_vantage.py

# Full universe hydration
python scripts/hydrate_full_universe.py
```

### Model Training
```bash
# Train all ML models
python src/ml/train_models.py

# Train specific agent types
python scripts/train_all_agents.sh  # Mac/Linux
scripts\TRAIN_ALL_AGENTS.bat        # Windows

# Massive parallel training
python scripts/hydrate_massive.py
```

### Trading Execution
```bash
# Start trading engine (simulation by default)
python src/trading/execution_engine.py

# Paper trading (port 7497)
python src/trading/execution_engine.py --paper

# Live trading (port 7496) - USE WITH CAUTION
python src/trading/execution_engine.py --live
```

### Monitoring
```bash
# View logs in real-time
# Windows:
Get-Content logs/trading_engine.log -Tail 50 -Wait

# Mac/Linux:
tail -f logs/trading_engine.log

# Check model count
ls models/*.pkl | wc -l  # Mac/Linux
(Get-ChildItem models\*.pkl).Count  # Windows
```

---

## Agent System

The system uses a hierarchical multi-agent architecture:

### Agent Tiers
- **Executive**: Top-level decision makers
- **Senior**: Strategy specialists
- **Junior**: Execution and data processing

### Specialized Agents
| Agent | Focus |
|-------|-------|
| `MomentumAgent` | Price momentum and trend following |
| `ValueAgent` | Fundamental value analysis |
| `ArbitrageAgent` | Cross-market arbitrage |
| `OptionsAgent` | Options strategies |
| `SentimentAgent` | NLP sentiment analysis |
| `RiskAgent` | Portfolio risk management |

---

## ML Pipeline

### Features (100+)
- Price: OHLCV, returns, volatility
- Technical: RSI, MACD, Bollinger Bands
- Volume: OBV, VWAP, volume ratios
- Microstructure: Bid-ask spread, order flow

### Models
- XGBoost (primary)
- LightGBM (fast iteration)
- CatBoost (categorical features)

### Validation
- Time-series cross-validation
- Walk-forward analysis
- Out-of-sample testing

---

## Risk Management

- **Kelly Criterion**: Confidence-weighted position sizing
- **Daily Loss Limit**: 2% max daily loss
- **Drawdown Protection**: 10% max drawdown
- **Position Limits**: Max 20 positions, 5% per position
- **Kill Switches**: Automatic halt on extreme events

---

## Data Sources

| Source | Data Type | Rate Limit |
|--------|-----------|------------|
| Polygon | Equities, 1-min bars | 5 req/min |
| Alpha Vantage | Equities, fundamentals | 5 req/min |
| Coinbase | Crypto OHLCV | 10 req/sec |
| FRED | Macro indicators | No limit |
| Interactive Brokers | Real-time quotes | Varies |

---

## Configuration

### Environment Variables (.env)
```
# Database
AZURE_SQL_SERVER=your-server.database.windows.net
AZURE_SQL_DATABASE=trading_db
AZURE_SQL_USERNAME=your_user
AZURE_SQL_PASSWORD=your_password

# APIs
POLYGON_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
COINBASE_API_KEY=your_key
COINBASE_API_SECRET=your_secret

# IBKR
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # 7497=paper, 7496=live
IBKR_CLIENT_ID=1
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | Activate venv: `source venv/bin/activate` |
| "Execution policy error" | Run: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Database connection fails | Check `.env` credentials |
| "No models found" | Run model training first |
| API rate limits | System handles with exponential backoff |
| Mac goes to sleep | Use `caffeinate -d python ...` |

---

## Best Practices

1. **Never commit `.env`** - Contains API keys
2. **Test in paper trading first** - Port 7497 is paper
3. **Monitor logs** - Check `logs/` folder
4. **Start data collection early** - Needs historical data
5. **Keep machines awake** - Use caffeinate/power settings

---

## Pre-Trading Checklist

- [ ] Virtual environment activated
- [ ] All packages installed
- [ ] `.env` file configured
- [ ] Database connection test passed
- [ ] Data collection completed
- [ ] Models trained (check `models/` folder)
- [ ] IBKR TWS/Gateway running
- [ ] Risk parameters reviewed

---

## Development

### Running Tests
```bash
pytest tests/
```

### Code Review
```bash
python scripts/full_review.py
```

### Training Agents
```bash
python src/training/agent_trainer.py
```

---

## License

Proprietary - Alpha Loop Capital, LLC

**Built for Alpha Loop Capital - Institutional-Grade Trading System**

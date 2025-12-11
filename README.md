# Alpha Loop Capital - Algorithmic Trading System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

## Mission-Critical Production Trading System

**Alpha Loop LLM** is an institutional-grade algorithmic trading system designed for:
- **Overnight training** of ML models on market data
- **Real-time signal generation** at market open
- **Automated execution** via Interactive Brokers
- **Multi-agent orchestration** for comprehensive market analysis

---

## Quick Start

### Prerequisites
- Python 3.10+
- Interactive Brokers TWS/Gateway (for live trading)
- API keys for data providers (Polygon, Alpha Vantage, FRED)

### Installation

```bash
# Clone repository
git clone https://github.com/tjhoags/alpha-loop-llm.git
cd alpha-loop-llm/hfc

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy environment variables to the project:

**Windows:**
```powershell
Copy-Item "C:\Users\tom\Alphaloopcapital Dropbox\ALC Tech Agents\API - Dec 2025.env" -Destination ".env"
```

**Mac:**
```bash
cp ~/Alphaloopcapital\ Dropbox/ALC\ Tech\ Agents/API\ -\ Dec\ 2025.env .env
```

### Test Database Connection

```bash
python scripts/test_db_connection.py
```

---

## Daily Operations

### Terminal 1: Data Collection
```bash
python src/data_ingestion/collector.py
```

### Terminal 2: Model Training
```bash
python src/ml/train_models.py
```

### Terminal 3: Trading Engine (6:30 AM ET)
```bash
python src/trading/execution_engine.py
```

**Note:** Use `caffeinate -d` on Mac to prevent sleep during overnight operations.

---

## Project Structure

```
alpha-loop-llm/
├── src/
│   ├── agents/           # 93 specialized trading agents
│   │   ├── hoags_agent/  # Master investment authority
│   │   ├── ghost_agent/  # Autonomous coordinator
│   │   ├── senior/       # Senior operational agents
│   │   ├── specialized/  # Strategy-specific agents
│   │   └── sectors/      # Sector-focused agents
│   ├── config/           # Configuration (Pydantic settings)
│   ├── core/             # Agent base classes & orchestration
│   ├── data_ingestion/   # Multi-source data collection
│   ├── database/         # Azure SQL Server integration
│   ├── ml/               # ML models & feature engineering
│   ├── trading/          # Execution engine & order management
│   ├── risk/             # Risk management & position sizing
│   └── signals/          # Signal generation & aggregation
├── scripts/              # Utility scripts & batch files
├── models/               # Trained model files (.pkl)
├── data/                 # Market data storage
├── logs/                 # System logs
└── tests/                # Test suite
```

---

## Key Features

### ML Pipeline
- **30+ Features**: Price, volume, volatility, momentum, microstructure
- **Ensemble Models**: XGBoost, LightGBM, CatBoost
- **Time-Series CV**: No data leakage, walk-forward validation
- **Model Versioning**: Timestamped saves with metadata

### Risk Management
- **Kelly Criterion**: Position sizing with confidence weighting
- **Daily Loss Limits**: 2% maximum daily loss
- **Drawdown Protection**: 5% maximum drawdown
- **Position Limits**: Max 10 positions, 10% per position

### Data Infrastructure
- **Multi-Source**: Polygon, Alpha Vantage, Coinbase, FRED
- **Azure SQL**: Centralized cloud database
- **Retry Logic**: Exponential backoff for API failures
- **Rate Limiting**: Automatic handling

### Trading Execution
- **IBKR Integration**: Full Interactive Brokers support
- **Paper Trading**: Safe testing on port 7497
- **Live Trading**: Real execution on port 7496
- **Order Management**: Market orders with fill tracking

---

## Agent Architecture

### Ownership Structure
| Owner | Role | Authority Agent |
|-------|------|-----------------|
| Tom Hogan | Founder & CIO | HOAGS |
| Chris Friedman | COO | FRIEDS |

### Agent Hierarchy
- **Master Agents (3)**: HOAGS, GHOST, FRIEDS
- **Senior Agents (10)**: SCOUT, HUNTER, ORCHESTRATOR, KILLJOY, etc.
- **Operational Agents (8)**: Data, Execution, Compliance, Portfolio, etc.
- **Strategy Agents (34)**: Momentum, Value, Arbitrage, etc.
- **Sector Agents (11)**: Tech, Healthcare, Energy, etc.

**Total: 93 Specialized Agents**

---

## Commands Reference

### Using Make (Recommended)
```bash
make install      # Install dependencies
make collect      # Run data collection
make train        # Run model training
make trade        # Start trading (simulation)
make lint         # Run code linters
make test         # Run test suite
```

### Direct Python Commands
```bash
# Data Collection
python src/data_ingestion/collector.py

# Model Training
python src/ml/train_models.py

# Trading Engine
python src/trading/execution_engine.py

# Model Dashboard
python scripts/model_dashboard.py
```

---

## Documentation

| Document | Description |
|----------|-------------|
| `SETUP_WINDOWS.md` | Windows-specific setup |
| `SETUP_MAC.md` | MacBook-specific setup |
| `DUAL_MACHINE_TRAINING.md` | Running on both machines |
| `TRAINING_GUIDE.md` | ML model training details |
| `COMPREHENSIVE_DATA_GUIDE.md` | Maximum data ingestion |

---

## Troubleshooting

### "Module not found"
→ Activate virtual environment first

### "Execution policy error" (Windows)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Database connection fails
→ Check `.env` file has correct credentials

### "No models found"
→ Ensure model training completed (check `models/` folder)

### Mac goes to sleep
→ Use `caffeinate -d python script.py`

---

## Security

- **Never commit `.env` file** - Contains API keys
- **Test in paper trading first** - Port 7497 is safe
- **Monitor logs** - Check `logs/` folder for errors
- **Rotate API keys** - Every 90 days recommended

---

## Pre-Trading Checklist

- [ ] Virtual environment activated
- [ ] All packages installed
- [ ] `.env` file configured
- [ ] Database connection tested
- [ ] Data collection ran overnight
- [ ] Models trained (check `models/`)
- [ ] Trading engine starts without errors

---

**Built for Alpha Loop Capital | Institutional-Grade Algorithmic Trading**

*Author: Tom Hogan | © 2025 Alpha Loop Capital, LLC*

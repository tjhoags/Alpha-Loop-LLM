# Alpha Loop LLM - Institutional-Grade Algorithmic Trading System

**Version 2.0** | Alpha Loop Capital, LLC

A sophisticated, production-ready algorithmic trading system designed for overnight training and market-open execution. Built with institutional-grade risk management, multi-source data ingestion, and ensemble ML models.

---

## Quick Start

### Prerequisites
- Python 3.10+
- ODBC Driver 17 for SQL Server
- Interactive Brokers TWS/Gateway (for live trading)

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
```

### Configuration

Copy your environment file with API keys:
```bash
# Windows
Copy-Item "path/to/your/.env" -Destination ".env"

# Mac/Linux
cp path/to/your/.env .env
```

### Running the System

```bash
# Start data collection
python src/data_ingestion/collector.py

# Train models
python src/ml/train_models.py

# Start trading engine (paper mode by default)
python src/trading/execution_engine.py
```

---

## Architecture

```
alpha-loop-llm/
├── src/
│   ├── agents/              # 93 specialized trading agents
│   │   ├── master/          # Master-tier agents (HOAGS, GHOST, FRIEDS)
│   │   ├── senior/          # Senior agents (10)
│   │   ├── specialized/     # Strategy agents (34)
│   │   ├── sectors/         # Sector specialists (11)
│   │   └── swarm/           # Agent swarm coordination
│   ├── config/              # Pydantic configuration management
│   ├── core/                # Base classes, learning engine, grading
│   ├── data_ingestion/      # Multi-source data collection
│   ├── database/            # Azure SQL Server integration
│   ├── integrations/        # External service clients
│   ├── ml/                  # Machine learning models & training
│   ├── nlp/                 # NLP sentiment & document analysis
│   ├── risk/                # Risk management & position sizing
│   ├── signals/             # Signal generators (15+ types)
│   ├── trading/             # Execution engine & IBKR integration
│   └── training/            # Agent training orchestration
├── scripts/                 # Utility scripts
├── tests/                   # Test suite
├── data/                    # Market data storage
├── models/                  # Trained model files (.pkl)
└── logs/                    # System logs
```

---

## Key Features

### Multi-Agent Architecture
- **93 specialized agents** organized by tier (Master → Senior → Strategy → Sector)
- **Agent Creating Agents (ACA)** - Dynamic agent generation based on capability gaps
- **Cross-machine learning synchronization** via Azure storage
- **Regime-aware adaptation** - Automatic strategy adjustment based on market conditions

### Machine Learning Pipeline
- **100+ engineered features** (price, volume, volatility, momentum, microstructure)
- **Ensemble models**: XGBoost, LightGBM, CatBoost
- **Time-series cross-validation** with no data leakage
- **Continuous model retraining** every hour

### Data Infrastructure
- **Multi-source ingestion**: Polygon, Alpha Vantage, Coinbase, FRED
- **Azure SQL Server** for centralized storage
- **Parallel API collection** with automatic rate limiting
- **Real-time and historical data support**

### Risk Management
- **Kelly Criterion** position sizing with confidence weighting
- **Daily loss limits**: 2% max daily drawdown
- **Portfolio limits**: Max 10 positions, 10% per position
- **Automatic kill switches** for extreme conditions

### Trading Execution
- **Interactive Brokers** integration (TWS/Gateway)
- **Paper trading mode** (Port 7497) for safe testing
- **Live trading mode** (Port 7496) for production
- **Order tracking** with fill confirmation

---

## Agent Hierarchy

| Tier | Examples | Count | Authority |
|------|----------|-------|-----------|
| **Master** | HOAGS, GHOST, FRIEDS | 3 | Strategic decisions, ACA approval |
| **Senior** | SCOUT, HUNTER, ORCHESTRATOR | 10 | Domain expertise |
| **Operational** | DATA_AGENT, EXECUTION_AGENT | 8 | System operations |
| **Strategy** | MOMENTUM, VALUE, ARBITRAGE | 34 | Trading strategies |
| **Sector** | TECH, HEALTHCARE, ENERGY | 11 | Sector analysis |
| **Support** | SWARM agents | 5+ | Coordination |

---

## Platform Commands

### Windows (PowerShell)

```powershell
# Navigate to project
cd "C:\path\to\alpha-loop-llm"

# Activate environment
.\venv\Scripts\Activate.ps1

# Data collection
python src/data_ingestion/collector.py

# Model training
python src/ml/train_models.py

# Start trading
python src/trading/execution_engine.py

# Check logs
Get-Content logs/data_collection.log -Tail 50 -Wait
```

### Mac/Linux

```bash
# Navigate to project
cd ~/alpha-loop-llm

# Activate environment
source venv/bin/activate

# Data collection (with caffeinate to prevent sleep)
caffeinate -d python src/data_ingestion/collector.py

# Model training
caffeinate -d python src/ml/train_models.py

# Start trading
python src/trading/execution_engine.py

# Check logs
tail -f logs/data_collection.log
```

---

## Configuration Options

Key environment variables (in `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `SQL_SERVER` | Azure SQL server address | Required |
| `SQL_DB` | Database name | `alc_market_data` |
| `DB_USERNAME` | Database username | Required |
| `DB_PASSWORD` | Database password | Required |
| `ALPHAVANTAGE_API_KEY` | Alpha Vantage API key | Required |
| `PolygonIO_API_KEY` | Polygon.io API key | Required |
| `COINBASE_API_KEY` | Coinbase API key | Optional |
| `FRED_DATA_API` | FRED API key | Optional |
| `IBKR_PORT` | IBKR port (7497=paper, 7496=live) | `7497` |

---

## Documentation

| Document | Description |
|----------|-------------|
| `SETUP_WINDOWS.md` | Windows setup guide |
| `SETUP_MAC.md` | Mac setup guide |
| `DUAL_MACHINE_TRAINING.md` | Multi-machine training |
| `AGENT_ARCHITECTURE.md` | Agent system documentation |
| `TRAINING_GUIDE.md` | ML training details |
| `PAT_SETUP_INSTRUCTIONS.md` | GitHub PAT setup |

---

## Troubleshooting

### Module not found
```bash
# Ensure venv is activated
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Mac
```

### Database connection fails
- Check `.env` file has correct Azure SQL credentials
- Ensure ODBC Driver 17 is installed

### No models found
- Run `python src/ml/train_models.py` first
- Check `models/` directory for `.pkl` files

### IBKR connection issues
- Ensure TWS/Gateway is running
- Check correct port (7497 for paper, 7496 for live)
- Verify API connections are enabled in TWS settings

---

## Security Notes

- **NEVER commit `.env` file** - contains API keys
- **Default to paper trading** (port 7497) when testing
- **Review risk limits** before live trading
- **Monitor logs** regularly during operation

---

## Daily Checklist

- [ ] Virtual environment activated
- [ ] All packages installed
- [ ] `.env` file configured
- [ ] Database connection verified
- [ ] Data collection running overnight
- [ ] Models trained (check `models/` folder)
- [ ] IBKR TWS/Gateway running (for trading)

---

**Built for Alpha Loop Capital - Institutional-Grade Trading**

*"By end of 2026, they will know Alpha Loop Capital."*

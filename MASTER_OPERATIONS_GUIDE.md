# MASTER OPERATIONS GUIDE

## Complete Guide to Alpha Loop Capital Algorithmic Trading System

---

## TABLE OF CONTENTS

1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Data Hydration](#data-hydration)
4. [Agent Training](#agent-training)
5. [Elite Grading System](#elite-grading-system)
6. [Running Scripts](#running-scripts)
7. [Monitoring & Dashboards](#monitoring--dashboards)
8. [Troubleshooting](#troubleshooting)

---

## QUICK START

### Windows (PowerShell)
```powershell
# 1. Navigate to project
cd C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1

# 2. Activate environment
.\venv\Scripts\activate

# 3. Start data hydration (run overnight)
python scripts/hydrate_full_universe.py

# 4. Train agents (in another terminal)
python -m src.training.agent_trainer --all
```

### Mac (Terminal)
```bash
# 1. Navigate to project
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1

# 2. Activate environment
source venv/bin/activate

# 3. Start data hydration
python scripts/hydrate_full_universe.py

# 4. Train agents
python -m src.training.agent_trainer --all
```

---

## SYSTEM ARCHITECTURE

```
+------------------+     +------------------+     +------------------+
|   DATA SOURCES   |     |   PROCESSING     |     |   EXECUTION      |
+------------------+     +------------------+     +------------------+
| Polygon.io       |---->| Feature Engine   |---->| Signal Generator |
| Alpha Vantage    |     | ML Training      |     | Risk Manager     |
| Massive S3       |     | Agent Training   |     | Order Executor   |
| FRED             |     | Elite Grading    |     | IBKR Connection  |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+     +------------------+
|   STORAGE        |     |   AGENTS         |     |   OUTPUTS        |
+------------------+     +------------------+     +------------------+
| Azure SQL        |     | 112 Total Agents |     | Trading Signals  |
| CSV Backups      |     | HOAGS (Master)   |     | P&L Reports      |
| FAISS Vectors    |     | Strategy Agents  |     | Risk Alerts      |
+------------------+     +------------------+     +------------------+
```

---

## DATA HYDRATION

### What Gets Pulled

| Source | Data Type | Volume | Time |
|--------|-----------|--------|------|
| Polygon | Stocks, ETFs | 8,000+ symbols | 4-6 hours |
| Polygon | Options | 100,000+ contracts | 2-3 hours |
| Polygon | Crypto | 500+ pairs | 30 min |
| Polygon | Forex | 50+ pairs | 30 min |
| Alpha Vantage | Fundamentals | 100+ stocks | 2-3 hours |
| Alpha Vantage | Daily prices | 100+ stocks | 2-3 hours |
| FRED | Macro data | 50+ series | 15 min |

### Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `hydrate_full_universe.py` | ALL Polygon data | `python scripts/hydrate_full_universe.py` |
| `hydrate_alpha_vantage.py` | Alpha Vantage data | `python scripts/hydrate_alpha_vantage.py` |
| `hydrate_massive.py` | Deep historical data | `python scripts/hydrate_massive.py` |

### Batch Files (Windows - Double-Click)

- `scripts/HYDRATE_QUICK.bat` - Quick test (30 min)
- `scripts/HYDRATE_ALPHA_VANTAGE.bat` - Alpha Vantage with menu
- `scripts/START_DATA_COLLECTION.bat` - Full hydration

---

## AGENT TRAINING

### Agent Hierarchy

```
TIER 1 - MASTERS (Supreme Authority)
  HOAGS     - Master strategist, approves all major decisions
  GHOST     - Autonomous operations, independent judgment

TIER 2 - SENIOR (Core Operations)
  SCOUT     - Market reconnaissance, opportunity identification
  HUNTER    - Trade execution, alpha capture
  ORCHESTRATOR - Multi-agent coordination
  KILLJOY   - Risk management, kill switches
  BOOKMAKER - Options pricing, volatility
  STRINGS   - Position management, allocation
  SKILLS    - Capability development
  AUTHOR    - Documentation, learning

TIER 3 - STRATEGY (Trading Strategies)
  MOMENTUM, MEAN_REVERSION, VALUE, GROWTH, VOLATILITY
  SENTIMENT, LIQUIDITY, MACRO, OPTIONS, CRYPTO, PAIRS, ARBITRAGE

TIER 4 - SECTOR (Industry Focus)
  TECH, HEALTHCARE, ENERGY, FINANCIALS, etc.
```

### Training Commands

```powershell
# Train ALL agents
python -m src.training.agent_trainer --all

# Train specific agent
python -m src.training.agent_trainer --agent HOAGS

# Train specific tier
python -m src.training.agent_trainer --tier SENIOR
python -m src.training.agent_trainer --tier STRATEGY
```

### Training Output

Results saved to: `data/training_results/`

```json
{
  "agent_name": "MOMENTUM",
  "grade": "ELITE",
  "score": 87.5,
  "passed_elite": true,
  "sharpe_ratio": 2.45,
  "win_rate": 0.58,
  "unique_edges": ["Fast adaptation: 4 days", "Survives tail events"]
}
```

---

## ELITE GRADING SYSTEM

### What Makes It Different

Standard quant funds measure basic metrics. We measure UNIQUE features:

| Metric | Standard Fund | Our System |
|--------|---------------|------------|
| Sharpe Ratio | Accept 1.0+ | Require 2.0+ |
| Max Drawdown | Accept 20%+ | Max 10% |
| Alpha Decay | Not measured | Track half-life |
| Regime Consistency | Not measured | Must work in all regimes |
| Conviction Calibration | Not measured | High confidence = high accuracy |
| Crowding Score | Not measured | Penalize popular strategies |
| Black Swan Survival | Not measured | Must survive tail events |

### Grade Levels

| Grade | Score | Meaning |
|-------|-------|---------|
| ELITE | 85+ | Ready for live capital |
| BATTLE_READY | 70-84 | Paper trading approved |
| DEVELOPING | 50-69 | Continue training |
| INADEQUATE | 30-49 | Needs major work |
| FAILED | <30 | Critical flaws, rebuild |

### Unique Features We Measure

1. **Alpha Half-Life** - How long does the edge last? (min 30 days)
2. **Regime Consistency** - Profitable in bull, bear, sideways? (70%+ of regimes)
3. **Conviction-Accuracy Correlation** - When 90% confident, 90% accurate? (0.6+ correlation)
4. **Crowding Score** - Is everyone doing this? (max 30% crowded)
5. **Black Swan Survival** - Performance during crashes? (80%+ survival)
6. **Reflexivity Score** - Does trading affect the market? (max 5% impact)
7. **Capacity Estimate** - How much AUM before alpha decays? (min $10M)

---

## RUNNING SCRIPTS

### Windows Batch Files (Double-Click)

| File | Purpose |
|------|---------|
| `scripts/TRAIN_ALL_AGENTS.bat` | Train all agents with elite grading |
| `scripts/HYDRATE_ALPHA_VANTAGE.bat` | Pull Alpha Vantage data |
| `scripts/INGEST_RESEARCH.bat` | Ingest research documents |
| `scripts/CHECK_MODEL_GRADES.bat` | View model performance |
| `scripts/START_PAPER_TRADING.bat` | Start paper trading |

### Manual Commands

```powershell
# Data Collection
python scripts/hydrate_full_universe.py          # All Polygon data
python scripts/hydrate_alpha_vantage.py --quick  # Quick AV test
python scripts/ingest_research.py                # Research docs

# Training
python -m src.training.agent_trainer --all       # All agents
python -m src.ml.advanced_training               # ML models

# Trading
python scripts/model_dashboard.py                # View grades
```

---

## MONITORING & DASHBOARDS

### Log Files

| Log | Location | Purpose |
|-----|----------|---------|
| Training | `logs/agent_training.log` | Agent training progress |
| Hydration | `logs/hydration.log` | Data collection |
| Alpha Vantage | `logs/alpha_vantage_hydration.log` | AV specific |
| Research | `logs/ingest_research.log` | Document ingestion |

### Check Training Progress

```powershell
# View latest training results
Get-Content data/training_results/*.json | ConvertFrom-Json | Format-Table

# Watch log in real-time
Get-Content logs/agent_training.log -Wait -Tail 50

# Count promoted agents
(Get-Content data/training_results/*.json | ConvertFrom-Json | Where-Object {$_.passed_elite}).Count
```

### Dashboard

```powershell
python scripts/model_dashboard.py
```

---

## TROUBLESHOOTING

### API Key Issues

```
Error: Polygon API Key: MISSING
```
Solution: Check your `.env` file has `PolygonIO_API_KEY=your_key`

### SQL Connection Issues

```
Error: Could not open a connection to SQL Server
```
Solutions:
1. Check SQL server name ends with `.database.windows.net`
2. Check firewall allows your IP
3. Verify credentials in settings

### Import Errors

```
Error: ModuleNotFoundError: No module named 'xxx'
```
Solution:
```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Training Fails with No Data

```
Error: No training data available
```
Solution: Run data hydration first:
```powershell
python scripts/hydrate_full_universe.py --quick
```

---

## OVERNIGHT TRAINING SETUP

### Windows - Keep Running All Night

1. **Prevent Sleep:**
   - Settings > System > Power > Never sleep when plugged in

2. **Run in Background:**
   ```powershell
   # Terminal 1: Data Hydration
   python scripts/hydrate_full_universe.py 2>&1 | Tee-Object logs/hydration.log

   # Terminal 2: Agent Training
   python -m src.training.agent_trainer --all 2>&1 | Tee-Object logs/training.log

   # Terminal 3: ML Model Training
   python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()"
   ```

3. **Auto-Restart Script:**
   ```powershell
   # Save as overnight.ps1
   while ($true) {
       try {
           python -m src.training.agent_trainer --all
       } catch {
           Write-Host "Error, restarting in 60s..."
           Start-Sleep -Seconds 60
       }
   }
   ```

### Mac - Keep Running All Night

1. **Prevent Sleep:**
   ```bash
   caffeinate -i python -m src.training.agent_trainer --all
   ```

2. **Use tmux/screen:**
   ```bash
   tmux new -s training
   python -m src.training.agent_trainer --all
   # Ctrl+B, D to detach
   # tmux attach -t training to reconnect
   ```

---

## FILE REFERENCE

### Key Files

| Path | Purpose |
|------|---------|
| `src/config/settings.py` | All configuration |
| `src/core/elite_grading.py` | Elite grading system |
| `src/training/agent_trainer.py` | Agent training orchestrator |
| `src/ml/advanced_training.py` | ML model training |
| `src/agents/` | All 112 agent files |

### Data Directories

| Path | Contents |
|------|----------|
| `data/csv_backup/` | CSV backups of all data |
| `data/training_results/` | Agent training results |
| `data/models/` | Trained ML models |
| `data/vectorstore/` | Research embeddings |
| `logs/` | All log files |

---

## CONTACT

For issues: Check logs first, then refer to this guide.

System designed for Alpha Loop Capital institutional trading.


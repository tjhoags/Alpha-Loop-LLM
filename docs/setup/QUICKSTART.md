# ALC-Algo Quickstart Guide

**Get running in 10 minutes**

---

## ⚡ Express Setup

### 1. Clone & Setup

```powershell
# Clone (if not already done)
cd C:\Users\tom
git clone https://github.com/AlphaLoopCapital/ALC-Algo.git
cd ALC-Algo

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Secrets

```powershell
# Copy template
copy config\secrets.py.example config\secrets.py

# Edit and set your master_alc_env path
notepad config\secrets.py
```

Set this line in `secrets.py`:
```python
ENV_FILE_PATH = "C:/Users/tom/Alphaloopcapital Dropbox/master_alc_env"
```

### 3. Verify Installation

```python
python -c "from src.agents import TOTAL_AGENTS; print(f'Ready: {TOTAL_AGENTS} agents')"
```

### 4. Run

```powershell
python main.py
```

---

## 🎯 What Happens

When you run `main.py`:

1. **Tier 1** initialized: GhostAgent (Master Controller)
2. **Tier 2** initialized: 8 Senior Agents
3. **Tier 3** initialized: 35+ Swarm Agents
4. Daily workflow executed (paper mode)
5. Statistics displayed

---

## 🔑 Required API Keys

Your `master_alc_env` file should contain:

```bash
# Essential
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
ALPHA_VANTAGE_API_KEY=...

# IBKR (paper trading)
IBKR_ACCOUNT_ID=...
IBKR_PORT=7497
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point |
| `config/settings.py` | Configuration loader |
| `src/agents/` | All 50+ agents |
| `src/core/agent_base.py` | Base agent class |

---

## ⚠️ Important Rules

1. **Paper First**: Always use port 7497 (paper) before 7496 (live)
2. **30% MoS**: All trades require 30% margin of safety
3. **Audit Trail**: All actions are logged
4. **Tom Hogan**: All outputs attributed to founder

---

## 📚 Next Steps

1. Read [Training Guide](TRAINING_GUIDE.md)
2. Import your historical trades
3. Run calibration
4. Start paper trading

---

*Quickstart Guide - ALC-Algo*  
*Tom Hogan | Alpha Loop Capital, LLC*


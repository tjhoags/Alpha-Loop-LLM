# Multi-Machine Setup - Windows + MacBook

## Running Training on Both Machines Overnight

You can run the training system on BOTH your Windows PC and MacBook simultaneously to maximize compute power.

---

## Recommended Setup

### Windows PC:
- **Run:** Data Collection (`src/data_ingestion/collector.py`)
- **Why:** Windows often has better API connectivity
- **Keep running:** Overnight

### MacBook:
- **Run:** Model Training (`src/ml/train_models.py`)
- **Why:** MacBook can use GPU/CPU for training
- **Keep running:** Overnight

**OR** run both processes on both machines for redundancy!

---

## Setup Both Machines

### Windows Setup:
See `SETUP_WINDOWS.md` for complete instructions.

**Quick start:**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\bek"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item "C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env" -Destination ".env"
python scripts/test_db_connection.py
```

### MacBook Setup:
See `SETUP_MAC.md` for complete instructions.

**Quick start:**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/bek
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ~/path/to/.env .env
python scripts/test_db_connection.py
```

---

## Shared Database Configuration

Both machines will write to the same Azure SQL database. Make sure:

1. **Same .env file** on both machines (copy it)
2. **Database accessible** from both networks
3. **No conflicts** - processes coordinate via database

---

## Running Processes

### Windows - Data Collection:
```powershell
python src/data_ingestion/collector.py
```

### MacBook - Model Training:
```bash
python src/ml/train_models.py
```

### Or Run Both on Each Machine:

**Windows Terminal 1:**
```powershell
python src/data_ingestion/collector.py
```

**Windows Terminal 2:**
```powershell
python src/ml/train_models.py
```

**MacBook Terminal 1:**
```bash
python src/data_ingestion/collector.py
```

**MacBook Terminal 2:**
```bash
python src/ml/train_models.py
```

---

## Monitoring Both Machines

### Windows:
```powershell
Get-Content logs\data_collection.log -Tail 50
Get-Content logs\model_training.log -Tail 50
```

### MacBook:
```bash
tail -f logs/data_collection.log
tail -f logs/model_training.log
```

---

## Keep Machines Awake

### Windows:
- Power Settings → Never sleep when plugged in
- Or: `powercfg /change standby-timeout-ac 0`

### MacBook:
```bash
# Prevent sleep
caffeinate -d

# Or run training with caffeinate
caffeinate -d python src/ml/train_models.py
```

---

## Benefits of Multi-Machine Setup

1. **Faster Training:** More compute power
2. **Redundancy:** If one machine fails, other continues
3. **Parallel Processing:** Different symbols/models on each machine
4. **Resource Optimization:** Use each machine's strengths

---

## Troubleshooting Multi-Machine

### Database Conflicts
- System handles concurrent writes automatically
- Each process has unique identifiers

### Network Issues
- Both machines need internet for APIs
- Database must be accessible from both

### Sync Issues
- Models saved independently on each machine
- Copy models from both machines before trading

---

## Next Morning - Collect Results

### From Windows:
```powershell
Get-ChildItem models\*.pkl
```

### From MacBook:
```bash
ls -la models/*.pkl
```

### Copy Best Models:
Copy the best performing models from both machines to your trading machine.

---

**You're ready to run training on both machines simultaneously!**


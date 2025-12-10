#!/bin/bash
# =============================================================================
# MAC FULL TRAINING SCRIPT - RUNS EVERYTHING
# =============================================================================
# This script runs ALL training components in parallel:
# 1. Data Hydration (Azure SQL)
# 2. Core ML Training
# 3. Research Ingestion
#
# HOW TO RUN:
# cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
# chmod +x scripts/mac_full_training.sh
# caffeinate -d ./scripts/mac_full_training.sh
# =============================================================================

PROJECT_DIR="$HOME/Alpha-Loop-LLM/Alpha-Loop-LLM-1"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"
source venv/bin/activate

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  ALPHA LOOP FULL TRAINING (MAC)${NC}"
echo -e "${CYAN}  Started: $(date)${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# -----------------------------------------------------------------------------
# PHASE 1: DATA HYDRATION (Background)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[PHASE 1] Starting Data Hydration...${NC}"
python scripts/hydrate_full_universe.py > "$LOG_DIR/hydration_$TIMESTAMP.log" 2>&1 &
HYDRATION_PID=$!
echo "  PID: $HYDRATION_PID"
echo "  Log: $LOG_DIR/hydration_$TIMESTAMP.log"

# Wait for some data before starting training
echo "  Waiting 60 seconds for initial data..."
sleep 60

# -----------------------------------------------------------------------------
# PHASE 2: ML TRAINING (Background)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[PHASE 2] Starting ML Training...${NC}"
python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()" > "$LOG_DIR/training_$TIMESTAMP.log" 2>&1 &
TRAINING_PID=$!
echo "  PID: $TRAINING_PID"
echo "  Log: $LOG_DIR/training_$TIMESTAMP.log"

# -----------------------------------------------------------------------------
# PHASE 3: RESEARCH INGESTION (Background)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[PHASE 3] Starting Research Ingestion...${NC}"
python scripts/ingest_research.py > "$LOG_DIR/research_$TIMESTAMP.log" 2>&1 &
RESEARCH_PID=$!
echo "  PID: $RESEARCH_PID"
echo "  Log: $LOG_DIR/research_$TIMESTAMP.log"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ALL PROCESSES STARTED${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Running processes:"
echo "  - Hydration: PID $HYDRATION_PID"
echo "  - Training:  PID $TRAINING_PID"
echo "  - Research:  PID $RESEARCH_PID"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/hydration_$TIMESTAMP.log"
echo "  tail -f $LOG_DIR/training_$TIMESTAMP.log"
echo "  tail -f $LOG_DIR/research_$TIMESTAMP.log"
echo ""

# -----------------------------------------------------------------------------
# WAIT FOR ALL PROCESSES
# -----------------------------------------------------------------------------
echo "Waiting for all processes to complete..."
echo "(This will take several hours)"
echo ""

wait $HYDRATION_PID
HYDRATION_EXIT=$?
echo "[$(date)] Hydration finished with exit code: $HYDRATION_EXIT"

wait $TRAINING_PID
TRAINING_EXIT=$?
echo "[$(date)] Training finished with exit code: $TRAINING_EXIT"

wait $RESEARCH_PID
RESEARCH_EXIT=$?
echo "[$(date)] Research finished with exit code: $RESEARCH_EXIT"

# -----------------------------------------------------------------------------
# FINAL REPORT
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  FULL TRAINING COMPLETE${NC}"
echo -e "${GREEN}  Finished: $(date)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Count models
MODEL_COUNT=$(ls "$PROJECT_DIR/models/"*.pkl 2>/dev/null | wc -l | tr -d ' ')
echo "Models trained: $MODEL_COUNT"

# Check Azure SQL row count
echo ""
echo "Checking Azure SQL..."
python -c "
import pyodbc
try:
    conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};Server=alc-sql-server.database.windows.net;Database=alc_market_data;UID=CloudSAb3fcbb35;PWD=ALCadmin27!')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM price_bars')
    print(f'Azure SQL rows: {cursor.fetchone()[0]:,}')
    conn.close()
except Exception as e:
    print(f'Could not check SQL: {e}')
"

echo ""
echo "READY FOR TRADING AT 9:30 AM!"
echo ""



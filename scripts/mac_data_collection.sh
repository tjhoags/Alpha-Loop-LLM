#!/bin/bash
# =============================================================================
# MAC DATA COLLECTION SCRIPT
# =============================================================================
# Pulls FULL UNIVERSE of market data into Azure SQL
#
# HOW TO RUN:
# caffeinate -d ./scripts/mac_data_collection.sh
# =============================================================================

PROJECT_DIR="$HOME/Alpha-Loop-LLM/Alpha-Loop-LLM-1"
LOG_FILE="$PROJECT_DIR/logs/data_collection_mac.log"

cd "$PROJECT_DIR"
mkdir -p logs

echo "========================================"
echo "  ALPHA LOOP DATA COLLECTION (MAC)"
echo "========================================"
echo "Project: $PROJECT_DIR"
echo "Log: $LOG_FILE"
echo ""

source venv/bin/activate

echo "[$(date)] Starting full universe hydration..." | tee -a "$LOG_FILE"

# Run hydration with full output
python scripts/hydrate_full_universe.py 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "[$(date)] Data collection complete!" | tee -a "$LOG_FILE"
echo "========================================"

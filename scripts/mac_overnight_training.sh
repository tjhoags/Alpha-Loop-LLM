#!/bin/bash
# =============================================================================
# MAC OVERNIGHT TRAINING SCRIPT
# =============================================================================
# HOW TO RUN:
# 1. Open Terminal (not in Cursor)
# 2. cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
# 3. chmod +x scripts/mac_overnight_training.sh
# 4. caffeinate -d ./scripts/mac_overnight_training.sh
#
# The 'caffeinate -d' prevents Mac from sleeping!
# =============================================================================

PROJECT_DIR="$HOME/Alpha-Loop-LLM/Alpha-Loop-LLM-1"
LOG_FILE="$PROJECT_DIR/logs/overnight_mac.log"
MAX_RETRIES=5
RETRY_DELAY=60

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  MAC OVERNIGHT TRAINING${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

log "=========================================="
log "OVERNIGHT TRAINING STARTED (MAC)"
log "=========================================="
log "Project: $PROJECT_DIR"

# -----------------------------------------------------------------------------
# PREVENT SLEEP (backup - caffeinate should handle this)
# -----------------------------------------------------------------------------
# Disable sleep via pmset (requires sudo, optional)
# sudo pmset -a disablesleep 1

# -----------------------------------------------------------------------------
# ACTIVATE VENV AND RUN TRAINING
# -----------------------------------------------------------------------------
source "$PROJECT_DIR/venv/bin/activate"

retry_count=0
success=false

while [ "$success" = false ] && [ $retry_count -lt $MAX_RETRIES ]; do
    ((retry_count++))
    log "----------------------------------------"
    log "TRAINING ATTEMPT $retry_count of $MAX_RETRIES"
    log "----------------------------------------"
    
    start_time=$(date +%s)
    
    python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()"
    exit_code=$?
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        log "TRAINING COMPLETED SUCCESSFULLY!"
        log "Duration: $((duration / 3600))h $((duration % 3600 / 60))m $((duration % 60))s"
        success=true
    else
        log "ERROR: Training exited with code $exit_code"
        log "Waiting $RETRY_DELAY seconds before retry..."
        sleep $RETRY_DELAY
    fi
done

# -----------------------------------------------------------------------------
# REPORT
# -----------------------------------------------------------------------------
log "=========================================="
if [ "$success" = true ]; then
    log "OVERNIGHT TRAINING COMPLETE - SUCCESS"
else
    log "OVERNIGHT TRAINING FAILED AFTER $MAX_RETRIES ATTEMPTS"
fi
log "=========================================="

# Show models
log "MODELS CREATED:"
ls -la "$PROJECT_DIR/models/"*.pkl 2>/dev/null | while read line; do
    log "  $line"
done

model_count=$(ls "$PROJECT_DIR/models/"*.pkl 2>/dev/null | wc -l)
log "Total models: $model_count"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}TRAINING COMPLETE!${NC}"
echo -e "${GREEN}Models saved: $model_count${NC}"
echo -e "${GREEN}========================================${NC}"


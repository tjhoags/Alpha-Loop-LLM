#!/bin/bash
# =============================================================================
# MAC TRADING ENGINE
# =============================================================================
# Run at 9:15 AM after training completes
#
# HOW TO RUN:
# ./scripts/mac_trading_engine.sh
# =============================================================================

PROJECT_DIR="$HOME/Alpha-Loop-LLM/Alpha-Loop-LLM-1"
cd "$PROJECT_DIR"

echo "========================================"
echo "  ALPHA LOOP TRADING ENGINE (MAC)"
echo "========================================"
echo ""
echo "WARNING: This will execute trades via IBKR!"
echo "Make sure TWS/Gateway is running."
echo ""
echo "Press Enter to start..."
read

source venv/bin/activate
python src/trading/execution_engine.py


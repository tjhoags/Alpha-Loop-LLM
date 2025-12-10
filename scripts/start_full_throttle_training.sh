#!/bin/bash
# ================================================================================
# FULL THROTTLE TRAINING STARTUP SCRIPT (Mac/Linux)
# ================================================================================
# Starts all data ingestion and training processes simultaneously:
# 1. Massive S3 hydration (5 years backfill)
# 2. Alpha Vantage Premium hydration (continuous)
# 3. Model training (continuous)
# ================================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "🚀 STARTING FULL THROTTLE TRAINING..."
echo "=========================================="
echo ""

# Terminal 1: Massive S3 Hydration
echo "📁 Starting Terminal 1: Massive S3 Hydration (5 years backfill)..."
gnome-terminal -- bash -c "cd '$PROJECT_DIR' && source venv/bin/activate && python scripts/hydrate_massive.py; exec bash" 2>/dev/null || \
osascript -e "tell app \"Terminal\" to do script \"cd '$PROJECT_DIR' && source venv/bin/activate && python scripts/hydrate_massive.py\"" 2>/dev/null || \
xterm -e "cd '$PROJECT_DIR' && source venv/bin/activate && python scripts/hydrate_massive.py" &

sleep 2

# Terminal 2: Alpha Vantage Premium Hydration
echo "📊 Starting Terminal 2: Alpha Vantage Premium Hydration..."
gnome-terminal -- bash -c "cd '$PROJECT_DIR' && source venv/bin/activate && python scripts/hydrate_all_alpha_vantage.py; exec bash" 2>/dev/null || \
osascript -e "tell app \"Terminal\" to do script \"cd '$PROJECT_DIR' && source venv/bin/activate && python scripts/hydrate_all_alpha_vantage.py\"" 2>/dev/null || \
xterm -e "cd '$PROJECT_DIR' && source venv/bin/activate && python scripts/hydrate_all_alpha_vantage.py" &

sleep 2

# Terminal 3: Model Training
echo "🤖 Starting Terminal 3: Model Training..."
gnome-terminal -- bash -c "cd '$PROJECT_DIR' && source venv/bin/activate && python src/ml/train_models.py; exec bash" 2>/dev/null || \
osascript -e "tell app \"Terminal\" to do script \"cd '$PROJECT_DIR' && source venv/bin/activate && python src/ml/train_models.py\"" 2>/dev/null || \
xterm -e "cd '$PROJECT_DIR' && source venv/bin/activate && python src/ml/train_models.py" &

sleep 2

echo ""
echo "✅ All processes started!"
echo ""
echo "📊 Monitor logs:"
echo "  - tail -f logs/massive_ingest.log"
echo "  - tail -f logs/alpha_vantage_hydration.log"
echo "  - tail -f logs/model_training.log"
echo ""
echo "🛑 To stop: Close the terminal windows or Ctrl+C in each"
echo ""
echo "⏰ TOMORROW MORNING (9:15 AM ET):"
echo "  python src/trading/execution_engine.py"


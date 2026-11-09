#!/usr/bin/env bash
# ==============================================================================
# CRAfter AI — Visual Reinforcement Learning Lab Control Script
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
VENV="$DIR/.venv"

# Ensure venv exists
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install -r "$DIR/requirements.txt"
    pip install -e "$DIR/crafter-game"
    pip install -e "$DIR/crafter-ai-engine"
else
    source "$VENV/bin/activate"
fi

MODE="${1:-all}"

case "$MODE" in
    backend)
        echo "Starting FastAPI backend server on http://localhost:8000..."
        cd "$DIR/crafter-ai-backend"
        exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
        ;;

    frontend|dashboard)
        echo "Starting Next.js research dashboard on http://localhost:3000..."
        cd "$DIR/crafter-ai-dashboard"
        exec npm run dev
        ;;

    train)
        echo "Launching Crafter CNN + LSTM PPO training..."
        exec python "$DIR/crafter-ai-engine/scripts/train.py" --config "$DIR/crafter-ai-engine/configs/default.yaml" "${@:2}"
        ;;

    figures|plot)
        echo "Generating all 10 post-training scientific figures into figures/..."
        exec python "$DIR/crafter-ai-engine/scripts/plot_figures.py" --output-dir "$DIR/figures"
        ;;

    test)
        echo "Running complete Crafter AI test suite..."
        echo "--- 1. Testing crafter-ai-engine ---"
        (cd "$DIR/crafter-ai-engine" && pytest tests/ -v)
        echo "--- 2. Testing crafter-ai-backend ---"
        (cd "$DIR/crafter-ai-backend" && pytest tests/ -v)
        echo "All tests passed successfully!"
        ;;

    all|start)
        echo "=================================================================="
        echo "  CRAfter AI — Visual Reinforcement Learning Observatory Lab"
        echo "=================================================================="
        echo "  Backend API:  http://localhost:8000"
        echo "  API Docs:     http://localhost:8000/docs"
        echo "  Dashboard:    http://localhost:3000"
        echo "  Figures:      $DIR/figures"
        echo "=================================================================="
        echo "Starting services (Press Ctrl+C to stop both)..."

        # Trap Ctrl+C to cleanly kill background processes
        cleanup() {
            echo ""
            echo "Stopping CRAfter AI Lab services..."
            kill 0
            exit 0
        }
        trap cleanup SIGINT SIGTERM EXIT

        # Start backend in background
        (cd "$DIR/crafter-ai-backend" && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload) &
        BACKEND_PID=$!

        # Start dashboard in background
        (cd "$DIR/crafter-ai-dashboard" && npm run dev) &
        DASHBOARD_PID=$!

        wait
        ;;

    *)
        echo "Usage: ./start_lab.sh [all|backend|frontend|train|figures|test]"
        exit 1
        ;;
esac

#!/bin/bash
# Run calibration gate: Stage 2
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source ~/venv/bin/activate 2>/dev/null || source ~/venv_debate_study/bin/activate 2>/dev/null || true

echo "=== Stage 2: Calibration Gate ==="
cd "$PROJECT_ROOT"
python3 -m src.pipeline_orchestrator --stage calibration
echo "=== Stage 2 Complete ==="

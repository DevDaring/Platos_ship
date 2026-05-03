#!/bin/bash
# Run main experiment: Stage 1 (C1-C4 + cross-model)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source ~/venv/bin/activate 2>/dev/null || source ~/venv_debate_study/bin/activate 2>/dev/null || true

echo "=== Stage 1: Main Experiment (C1-C4) ==="
cd "$PROJECT_ROOT"
python3 -m src.pipeline_orchestrator --stage main
echo "=== Stage 1 Complete ==="

#!/bin/bash
# Run mitigation experiment: Stage 3 (C5, only if gate passed)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source ~/venv_debate_study/bin/activate 2>/dev/null || true

echo "=== Stage 3: Mitigation Experiment (C5) ==="
cd "$PROJECT_ROOT"
python3 -m src.pipeline_orchestrator --stage mitigation
echo "=== Stage 3 Complete ==="

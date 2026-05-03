#!/bin/bash
# Run dry-run validation
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source ~/venv_debate_study/bin/activate 2>/dev/null || true

echo "=== Running Dry Run ==="
cd "$PROJECT_ROOT"
python3 -m src.pipeline_orchestrator --dry-run --stage dry_run
echo ""
echo "=== Running Dry Run Assertions ==="
python3 -m tests.dry_run_assertions
echo "=== Dry Run Complete ==="

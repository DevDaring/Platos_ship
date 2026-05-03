#!/bin/bash
# Run environment checks
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source ~/venv/bin/activate 2>/dev/null || source ~/venv_debate_study/bin/activate 2>/dev/null || true

echo "=== Running Environment Checks ==="
python3 -m src.environment_check --project-root "$PROJECT_ROOT"
echo "=== Environment Checks Complete ==="

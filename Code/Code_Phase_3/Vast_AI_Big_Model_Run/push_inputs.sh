#!/usr/bin/env bash
# push_inputs.sh — send the probe's inputs to a Vast.ai instance.
#
# Only two small files are needed: the 300-question pool and the wrong-anchored
# persona pool (~530 KB together). The 132 GB model checkout is pulled on the
# instance itself by vast_bootstrap.sh, not pushed from here.
#
#   bash push_inputs.sh <ssh-host> <ssh-port> [remote-dir]

set -euo pipefail

HOST="${1:?usage: push_inputs.sh <ssh-host> <ssh-port> [remote-dir]}"
PORT="${2:?usage: push_inputs.sh <ssh-host> <ssh-port> [remote-dir]}"
REMOTE="${3:-~/platos/Code/Code_Phase_3/Vast_AI_Big_Model_Run}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PHASE3="$(dirname "$HERE")"
LOCAL_INPUTS="$PHASE3/results/processed"

for f in question_pool.parquet dumb_personas.parquet; do
  [ -f "$LOCAL_INPUTS/$f" ] || {
    echo "Missing $LOCAL_INPUTS/$f — run: python3 tools/fetch_artefacts.py" >&2
    exit 1
  }
done

echo "==> Creating $REMOTE/inputs on $HOST"
ssh -p "$PORT" "$HOST" "mkdir -p $REMOTE/inputs"

echo "==> Copying the question and persona pools"
scp -P "$PORT" \
  "$LOCAL_INPUTS/question_pool.parquet" \
  "$LOCAL_INPUTS/dumb_personas.parquet" \
  "$HOST:$REMOTE/inputs/"

echo "==> Copying the probe code"
ssh -p "$PORT" "$HOST" "mkdir -p $REMOTE/../GPU_Only/src $REMOTE/../src"
scp -P "$PORT" "$HERE/run_big_probe.py" "$HERE/requirements_big.txt" \
  "$HERE/vast_bootstrap.sh" "$HOST:$REMOTE/"
scp -P "$PORT" "$PHASE3/GPU_Only/src/corrected_probe.py" \
  "$HOST:$REMOTE/../GPU_Only/src/"
scp -P "$PORT" "$PHASE3/src/contexts.py" "$PHASE3/src/extraction.py" \
  "$PHASE3/src/seeding.py" "$PHASE3/src/__init__.py" \
  "$HOST:$REMOTE/../src/"
ssh -p "$PORT" "$HOST" "mkdir -p $REMOTE/../src/agent_wrappers"
scp -P "$PORT" "$PHASE3/src/agent_wrappers/judge_agent.py" \
  "$PHASE3/src/agent_wrappers/base_agent.py" \
  "$PHASE3/src/agent_wrappers/__init__.py" \
  "$HOST:$REMOTE/../src/agent_wrappers/"
ssh -p "$PORT" "$HOST" "touch $REMOTE/../GPU_Only/__init__.py $REMOTE/../GPU_Only/src/__init__.py"

echo
echo "Done. On the instance:"
echo "  cd $REMOTE && export HUGGINGFACE_TOKEN=... && bash vast_bootstrap.sh"
echo "  python3 run_big_probe.py --tensor-parallel-size 2"

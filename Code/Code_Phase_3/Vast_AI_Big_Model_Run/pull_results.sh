#!/usr/bin/env bash
# pull_results.sh — retrieve the 70B probe outputs, then remind you to destroy
# the instance. Vast.ai bills until the instance is destroyed, not stopped.
#
#   bash pull_results.sh <ssh-host> <ssh-port> [remote-dir]

set -euo pipefail

HOST="${1:?usage: pull_results.sh <ssh-host> <ssh-port> [remote-dir]}"
PORT="${2:?usage: pull_results.sh <ssh-host> <ssh-port> [remote-dir]}"
REMOTE="${3:-~/platos/Code/Code_Phase_3/Vast_AI_Big_Model_Run}"

HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/results"
mkdir -p "$DEST"

echo "==> Pulling results from $HOST"
scp -P "$PORT" -r "$HOST:$REMOTE/results/*" "$DEST/" || {
  echo "Nothing to pull — did the run finish? Check $REMOTE/big_probe.log" >&2
  exit 1
}
scp -P "$PORT" "$HOST:$REMOTE/big_probe.log" "$DEST/" 2>/dev/null || true

echo
echo "==> Retrieved:"
ls -la "$DEST"

python3 - <<'PY'
import json, pathlib
meta = pathlib.Path(__file__).resolve().parent / "results" / "big_probe_meta.json"
if meta.exists():
    d = json.loads(meta.read_text())
    c = d.get("contrast", {})
    print(f"\n  model      : {d.get('model')} ({d.get('dtype')}, tp={d.get('tensor_parallel_size')})")
    print(f"  GPUs       : {d.get('n_gpus')} x {d.get('gpu')}")
    print(f"  wall clock : {d.get('wall_clock_minutes')} min")
    print(f"  WR - E     : {c.get('estimate')}  CI [{c.get('ci_low')}, {c.get('ci_high')}]  n={c.get('n_pairs')}")
    print(f"  same-answer: {c.get('estimate_stated_answer_unchanged')} "
          f"(n={c.get('n_pairs_stated_answer_unchanged')})")
PY

cat <<'MSG'

==> NEXT
  1. Fold into the analysis:   cd .. && python3 -m analysis.run_analysis
  2. DESTROY the Vast.ai instance now. Stopping it does not stop billing.

MSG

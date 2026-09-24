#!/usr/bin/env bash
# start_run.sh — start the full run, detached from the SSH session. Runs ON the VM.
#
#   bash GCP_API_Run/start_run.sh              # real run
#   bash GCP_API_Run/start_run.sh --dry-run    # one unit per cell (smoke test)
#
# Two detached processes:
#   autopush  every 30 minutes, shard outputs and logs to GitHub
#   run       tools/run_parallel.sh (prepare -> 8 shards -> merge -> analyse),
#             then one final push with the merged outputs and analysis.
# logs/launcher/RUN_EXIT holds the launcher's exit code when it finishes.

set -euo pipefail
ROOT="$HOME/platos/Code/Code_Phase_3"
PY="$HOME/platos/.venv/bin/python"
cd "$ROOT"
mkdir -p logs/launcher

if pgrep -f "tools/run_parallel.sh" >/dev/null; then
  echo "a run is already active; refusing to start a second one" >&2
  exit 1
fi
rm -f logs/launcher/RUN_EXIT

EXTRA="${*:-}"
# A smoke or dry run must never reach GitHub: its 4-question outputs would sit
# next to the real results. Only a full run starts the pushes.
case " $EXTRA " in
  *" --max-questions "*|*" --dry-run "*) PUSH=0 ;;
  *) PUSH=1 ;;
esac

if [ "$PUSH" = 1 ] && ! pgrep -f "GCP_API_Run/autopush.sh$" >/dev/null; then
  nohup setsid bash GCP_API_Run/autopush.sh > logs/autopush.log 2>&1 < /dev/null &
fi

nohup setsid bash -c "
  PYTHON='$PY' bash tools/run_parallel.sh $EXTRA
  echo \$? > logs/launcher/RUN_EXIT
  [ '$PUSH' = 1 ] && bash GCP_API_Run/autopush.sh --final >> logs/autopush.log 2>&1
" > logs/launcher/run.out 2>&1 < /dev/null &

sleep 2
echo "started: $(pgrep -fa 'run_parallel.sh|autopush.sh' | wc -l) processes"
echo "watch:   tail -f $ROOT/logs/launcher/exits.txt"

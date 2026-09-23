#!/usr/bin/env bash
# run_full.sh — the full X8 probe, with the 30-minute autopush alongside it.
# Runs ON the VM.
#
#   GITHUB_TOKEN=... bash run_full.sh
#
# Both processes are detached with nohup, so closing the SSH session does not
# kill the run. Progress is visible from GitHub alone via STATUS.md.

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
QUESTIONS="${QUESTIONS:-300}"
REPLICATES="${REPLICATES:-3}"
HOME_DIR="${HOME_DIR:-$HOME/platos}"
SEED="${SEED:-20260502}"
# --include-numeric is a store_true flag that defaults to OFF, so leaving
# it out silently drops GSM8K and yields a multiple-choice-only run. The
# first run was launched by hand with the flag and the script was never
# updated, so re-running this file would NOT have reproduced it.
INCLUDE_NUMERIC="${INCLUDE_NUMERIC:-1}"
NUMERIC_ARGS=()
# An `&&` here would return non-zero when numeric is off, and `set -e`
# would abort the whole run on it.
if [ "$INCLUDE_NUMERIC" = "1" ]; then NUMERIC_ARGS+=(--include-numeric); fi

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

[ -n "${GITHUB_TOKEN:-}" ] || { echo "set GITHUB_TOKEN for the autopush" >&2; exit 1; }

cd "$HOME_DIR"
mkdir -p logs results
# shellcheck disable=SC1090
[ -f "$HOME/platos_env.sh" ] && source "$HOME/platos_env.sh"

say "Starting the 30-minute autopush"
pkill -f "autopush.sh" 2>/dev/null || true
# The one-shot runs FIRST and completes the clone. Starting the loop first
# and firing a --once three seconds later put two processes into the same
# empty working directory: both began cloning, the second died on
# "destination path already exists", and the loop went down with it, which
# left the run with no way to get results off the box.
GITHUB_TOKEN="$GITHUB_TOKEN" bash autopush.sh --once || true
nohup env GITHUB_TOKEN="$GITHUB_TOKEN" bash autopush.sh \
  > logs/autopush.log 2>&1 &
echo "    autopush pid $!"

say "Starting the probe: $QUESTIONS questions x $REPLICATES replicates, seed $SEED, numeric=$INCLUDE_NUMERIC"
nohup python3 run_probe.py \
  --model "$MODEL" \
  --tensor-parallel-size 1 \
  --questions "$QUESTIONS" \
  --replicates "$REPLICATES" \
  --seed "$SEED" \
  "${NUMERIC_ARGS[@]}" \
  --out "$HOME_DIR/results" \
  > logs/probe.log 2>&1 &
PROBE_PID=$!
echo "    probe pid $PROBE_PID"

cat <<MSG

Both are running detached; the SSH session can be closed.

  follow    : tail -f $HOME_DIR/logs/probe.log
  progress  : github.com/DevDaring/Platos_ship  branch l4-probe-results
              -> Code/Code_Phase_3/GCP_L4_Run/results/STATUS.md
  finished? : ls $HOME_DIR/results/big_probe_contrast.parquet

When it finishes: one final push, then DELETE the VM.
  GITHUB_TOKEN=... bash autopush.sh --once
  gcloud compute instances delete <name> --zone <zone> --quiet
MSG

wait "$PROBE_PID" && say "Probe finished — pushing the final results" \
  && GITHUB_TOKEN="$GITHUB_TOKEN" bash autopush.sh --once \
  && say "Done. DELETE THE VM: a stopped instance still bills for its disk."

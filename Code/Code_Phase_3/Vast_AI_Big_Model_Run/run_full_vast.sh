#!/usr/bin/env bash
# run_full_vast.sh — the full 70B X8 probe, detached. Runs ON the instance.
#
#   bash run_full_vast.sh
#
# Detached with nohup, so closing the SSH session does not kill the run. The
# probe is checkpointed per condition, so an interrupted run resumes without
# re-paying for GPU time already spent.

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
QUESTIONS="${QUESTIONS:-300}"
REPLICATES="${REPLICATES:-3}"
HOME_DIR="${HOME_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SEED="${SEED:-20260502}"
TP="${TP:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"

# --include-numeric is a store_true flag that defaults to OFF, so leaving it
# out silently drops GSM8K and yields a multiple-choice-only run. The L4 run
# was launched by hand with the flag and its script was never updated, so
# re-running that file would NOT have reproduced it. Same seed as both L4
# runs, so the three are directly comparable.
INCLUDE_NUMERIC="${INCLUDE_NUMERIC:-1}"
NUMERIC_ARGS=()
if [ "$INCLUDE_NUMERIC" = "1" ]; then NUMERIC_ARGS+=(--include-numeric); fi

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

[ -n "${GITHUB_TOKEN:-}" ] || { echo "set GITHUB_TOKEN for the autopush" >&2; exit 1; }

cd "$HOME_DIR"
mkdir -p logs results
# shellcheck disable=SC1090
[ -f "$HOME/platos_env.sh" ] && source "$HOME/platos_env.sh"

say "Starting the 30-minute autopush to GitHub"
# A rented instance can be reclaimed and its disk goes with it, so results
# must not live only here. The one-shot runs FIRST and completes the clone;
# starting the loop first and firing a --once three seconds later put two
# processes into the same empty directory on the L4 run, the second died on
# "destination path already exists", and the loop went down with it.
GITHUB_TOKEN="$GITHUB_TOKEN" bash autopush_vast.sh --once || true
setsid nohup env GITHUB_TOKEN="$GITHUB_TOKEN" bash autopush_vast.sh \
  > logs/autopush.log 2>&1 < /dev/null &
echo "    autopush pid $!"

say "Starting the probe: $QUESTIONS questions x $REPLICATES replicates, tp=$TP, seed $SEED, numeric=$INCLUDE_NUMERIC"
nohup python3 run_big_probe.py \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --questions "$QUESTIONS" \
  --replicates "$REPLICATES" \
  --seed "$SEED" \
  "${NUMERIC_ARGS[@]}" \
  --out "$HOME_DIR/results" \
  > logs/probe.log 2>&1 &
PROBE_PID=$!
echo "    probe pid $PROBE_PID"

cat <<MSG

Running detached; the SSH session can be closed.

  follow    : tail -f $HOME_DIR/logs/probe.log
  progress  : python3 -c "import pandas as pd; d=pd.read_parquet('$HOME_DIR/results/big_probe_trials.parquet'); print(len(d), d.groupby('condition').size().to_dict())"
  finished? : ls $HOME_DIR/results/big_probe_contrast.parquet

When it finishes: pull the results, then DESTROY the instance. A stopped
instance still bills for its disk, and 132 GB of checkpoint is not cheap.
MSG

if wait "$PROBE_PID"; then
  say "Probe finished — final push"
else
  say "Probe exited non-zero — pushing what exists, then check logs/probe.log"
fi
GITHUB_TOKEN="$GITHUB_TOKEN" bash autopush_vast.sh --once || true

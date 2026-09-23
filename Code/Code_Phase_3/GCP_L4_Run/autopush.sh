#!/usr/bin/env bash
# autopush.sh — push results and logs to GitHub every 30 minutes. Runs ON the VM.
#
# The probe is checkpointed, but a checkpoint on a VM disk is lost when the VM
# goes away. This pushes the outputs off the box on a fixed interval so a
# pre-emption, a crash or an accidental delete costs at most one interval.
#
#   GITHUB_TOKEN=... bash autopush.sh            # loop until stopped
#   GITHUB_TOKEN=... bash autopush.sh --once     # single push, then exit
#
# TOKEN HANDLING. A classic PAT has broad scope. It is written to a file that
# only this user can read and handed to git through a credential file, never
# passed on a command line (where it would appear in `ps`) and never echoed.
# Revoke it when the run is done: https://github.com/settings/tokens

set -euo pipefail

INTERVAL="${INTERVAL:-1800}"                 # 30 minutes
REPO="${REPO:-https://github.com/DevDaring/Platos_ship.git}"
BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-$HOME/platos_push}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/platos/results}"
LOG_DIR="${LOG_DIR:-$HOME/platos/logs}"
REMOTE_SUBDIR="${REMOTE_SUBDIR:-Code/Code_Phase_3/GCP_L4_Run/results}"

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

[ -n "${GITHUB_TOKEN:-}" ] || { echo "set GITHUB_TOKEN" >&2; exit 1; }

# ── one-time git setup ────────────────────────────────────────────────────
if [ ! -d "$WORKDIR/.git" ]; then
  say "Cloning $BRANCH (shallow)"
  mkdir -p "$WORKDIR"

  # Credentials in a 0600 file, not in the remote URL and not in argv.
  CRED="$HOME/.git-credentials"
  HOST="${REPO#https://}"; HOST="${HOST%%/*}"
  PATHPART="${REPO#https://$HOST/}"
  umask 077
  printf 'https://x-access-token:%s@%s\n' "$GITHUB_TOKEN" "$HOST" > "$CRED"
  chmod 600 "$CRED"
  git config --global credential.helper "store --file=$CRED"
  git config --global user.email "l4-probe@noreply.invalid"
  git config --global user.name  "Platos Ship L4 probe"

  if ! git clone --depth 1 --branch "$BRANCH" "$REPO" "$WORKDIR" 2>/dev/null; then
    say "Branch $BRANCH does not exist yet — creating it from the default branch"
    git clone --depth 1 "$REPO" "$WORKDIR"
    git -C "$WORKDIR" checkout -b "$BRANCH"
  fi
fi

push_once() {
  mkdir -p "$WORKDIR/$REMOTE_SUBDIR" "$WORKDIR/$REMOTE_SUBDIR/logs"

  # Results and logs only — never the checkpoint, the model, or the .env.
  if [ -d "$RESULTS_DIR" ]; then
    rsync -a --delete \
      --include='*/' --include='*.parquet' --include='*.json' --include='*.csv' \
      --exclude='*' "$RESULTS_DIR/" "$WORKDIR/$REMOTE_SUBDIR/" 2>/dev/null || true
  fi
  if [ -d "$LOG_DIR" ]; then
    rsync -a --include='*/' --include='*.log' --exclude='*' \
      "$LOG_DIR/" "$WORKDIR/$REMOTE_SUBDIR/logs/" 2>/dev/null || true
  fi
  for extra in big_probe.log smoke.log install.log; do
    cp -f "$HOME/platos/$extra" "$WORKDIR/$REMOTE_SUBDIR/logs/" 2>/dev/null || true
  done

  # A short status file makes the run legible from GitHub alone.
  {
    echo "# L4 probe status"
    echo
    echo "- updated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "- host: $(hostname)"
    echo "- gpu: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
              --format=csv,noheader 2>/dev/null || echo 'n/a')"
    if [ -f "$RESULTS_DIR/big_probe_trials.parquet" ]; then
      # The VM's system python3 is 3.14 with no pandas; use the venv.
      "$HOME/platos/.venv/bin/python" - <<'PY' 2>/dev/null || true
import os, pandas as pd
p = os.path.expanduser("~/platos/results/big_probe_trials.parquet")
d = pd.read_parquet(p)
print(f"- trials written: {len(d)}")
print(f"- by condition: {d.groupby('condition').size().to_dict()}")
PY
    else
      echo "- trials written: 0 (not started or still loading the model)"
    fi
  } > "$WORKDIR/$REMOTE_SUBDIR/STATUS.md"

  cd "$WORKDIR"
  git add -A "$REMOTE_SUBDIR" >/dev/null 2>&1 || true
  if git diff --cached --quiet; then
    say "no change"
    return 0
  fi
  git commit -q -m "L4 probe results $(date -u '+%Y-%m-%d %H:%M UTC')"
  if git push -q origin "$BRANCH" 2>/dev/null; then
    say "pushed to $BRANCH"
  else
    say "push failed (will retry next interval)"
  fi
}

if [ "${1:-}" = "--once" ]; then
  push_once
  exit 0
fi

say "Pushing every $((INTERVAL / 60)) minutes to $BRANCH. Ctrl-C to stop."
while true; do
  push_once || say "push cycle errored; continuing"
  sleep "$INTERVAL"
done

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
# TWO INSTANCES MUST NOT RACE.
# run_full.sh starts the loop and then a --once three seconds later. Both
# found no $WORKDIR/.git, both began cloning the same path, and the second
# died on "destination path already exists and is not an empty directory",
# leaving a half-populated tree and killing the loop under `set -e`. The run
# then had no way of getting results off the box. So: the clone lands in a
# temporary directory and is moved into place atomically, and every push is
# serialised behind a lock.
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
LOCK="${LOCK:-$HOME/.platos_autopush.lock}"

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

[ -n "${GITHUB_TOKEN:-}" ] || { echo "set GITHUB_TOKEN" >&2; exit 1; }

# ── one-time git setup, safe to run concurrently ──────────────────────────
setup_repo() {
  CRED="$HOME/.git-credentials"
  HOST="${REPO#https://}"; HOST="${HOST%%/*}"
  umask 077
  printf 'https://x-access-token:%s@%s\n' "$GITHUB_TOKEN" "$HOST" > "$CRED"
  chmod 600 "$CRED"
  git config --global credential.helper "store --file=$CRED"
  git config --global user.email "l4-probe@noreply.invalid"
  git config --global user.name  "Platos Ship L4 probe"

  [ -d "$WORKDIR/.git" ] && return 0

  # Clone somewhere private, then move into place in one step. A second
  # instance either sees no .git and clones its own copy (discarded below),
  # or sees a complete one. It never sees a half-written tree.
  local staging
  staging="$(mktemp -d "${WORKDIR}.tmp.XXXXXX")"
  rm -rf "$staging"
  if ! git clone --quiet --depth 1 --branch "$BRANCH" "$REPO" "$staging"; then
    say "branch $BRANCH not found — cloning the default branch"
    git clone --quiet --depth 1 "$REPO" "$staging"
    git -C "$staging" checkout -q -b "$BRANCH"
  fi
  if [ -d "$WORKDIR/.git" ]; then
    rm -rf "$staging"            # someone else won; theirs is complete
  else
    rm -rf "$WORKDIR"
    mv "$staging" "$WORKDIR"
    say "cloned $BRANCH into $WORKDIR"
  fi
}

push_once() {
  setup_repo
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
  for extra in big_probe.log smoke.log install.log run.out; do
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
      "$HOME/platos/.venv/bin/python" - <<'PY' 2>/dev/null || true
import os, pandas as pd
p = os.path.expanduser("~/platos/results/big_probe_trials.parquet")
d = pd.read_parquet(p)
print(f"- trials written: {len(d)}")
print(f"- by condition: {d.groupby('condition').size().to_dict()}")
if "round1_text" in d.columns:
    kept = d["round1_text"].astype(str).str.strip().ne("").sum()
    print(f"- raw generations kept: {kept}/{len(d)}")
else:
    print("- raw generations kept: NO — round1_text column absent")
PY
    else
      echo "- trials written: 0 (not started or still loading the model)"
    fi
  } > "$WORKDIR/$REMOTE_SUBDIR/STATUS.md"

  cd "$WORKDIR"
  # The remote may have moved on since the shallow clone; rebase onto it
  # rather than failing the push and dropping an interval of results.
  git fetch --quiet --depth 1 origin "$BRANCH" 2>/dev/null || true
  git add -A "$REMOTE_SUBDIR" >/dev/null 2>&1 || true
  if git diff --cached --quiet; then
    say "no change"
    return 0
  fi
  git commit -q -m "L4 probe results $(date -u '+%Y-%m-%d %H:%M UTC')"
  if git push -q origin "$BRANCH" 2>/dev/null; then
    say "pushed to $BRANCH"
  else
    say "push rejected — re-cloning and retrying once"
    rm -rf "$WORKDIR"
    setup_repo
    return 1
  fi
}

# Serialise every push behind one lock so two instances cannot interleave.
guarded_push() {
  if command -v flock >/dev/null 2>&1; then
    flock -w 600 "$LOCK" bash -c "$(declare -f say setup_repo push_once); \
      INTERVAL='$INTERVAL' REPO='$REPO' BRANCH='$BRANCH' WORKDIR='$WORKDIR' \
      RESULTS_DIR='$RESULTS_DIR' LOG_DIR='$LOG_DIR' \
      REMOTE_SUBDIR='$REMOTE_SUBDIR' GITHUB_TOKEN='$GITHUB_TOKEN' push_once"
  else
    push_once
  fi
}

if [ "${1:-}" = "--once" ]; then
  guarded_push || guarded_push || say "push failed twice; next interval will retry"
  exit 0
fi

say "Pushing every $((INTERVAL / 60)) minutes to $BRANCH. Ctrl-C to stop."
while true; do
  guarded_push || guarded_push || say "push cycle errored; continuing"
  sleep "$INTERVAL"
done

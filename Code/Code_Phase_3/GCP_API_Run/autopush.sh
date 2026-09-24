#!/usr/bin/env bash
# autopush.sh — push the run's results and logs to GitHub. Runs ON the VM.
#
#   bash autopush.sh            # every 30 minutes until stopped
#   bash autopush.sh --once     # one push of shard outputs and logs
#   bash autopush.sh --final    # one push that also includes the merged
#                               # outputs, analysis, figures, tables and the
#                               # Phase 3 generated inputs (results/processed)
#
# Nothing may live only on the VM's disk. Each shard writes immutable
# part-files, so a push adds only new blobs and the repository does not grow
# by a full copy each interval.
#
# SAFETY
#   * Only one push at a time (flock).
#   * `rsync --delete` is used for ONE folder, the shard tree, and only when
#     source and destination are both named `shards`. A mismatched pair is
#     how an earlier sweep erased results on main; it is refused here.
#   * Everything else is copied without --delete, so nothing already on
#     GitHub can be removed by this script.
#   * The token is read from ~/platos/Code/.env (Github_Classic_Token), put
#     in a 600 credential file, and never echoed or placed on a command line.

set -uo pipefail

INTERVAL="${INTERVAL:-1800}"
REPO="${REPO:-https://github.com/DevDaring/Platos_ship.git}"
BRANCH="${BRANCH:-main}"
PHASE3="$HOME/platos/Code/Code_Phase_3"
WORKDIR="$HOME/platos_push"
REMOTE_PHASE3="Code/Code_Phase_3"
LOCK="$HOME/.platos_autopush.lock"

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

token() {
  grep -E '^Github_Classic_Token=' "$HOME/platos/Code/.env" | head -1 \
    | cut -d= -f2- | tr -d '"'"'"' \r'
}

setup_repo() {
  local cred="$HOME/.git-credentials" tok
  tok="$(token)"
  [ -n "$tok" ] || { say "Github_Classic_Token missing from .env"; return 1; }
  ( umask 077; printf 'https://x-access-token:%s@github.com\n' "$tok" > "$cred" )
  git config --global credential.helper "store --file=$cred"
  git config --global user.email "platos-api-run@noreply.invalid"
  git config --global user.name  "Platos Ship Phase 3 API run"
  [ -d "$WORKDIR/.git" ] && return 0
  local staging
  staging="$(mktemp -d "$WORKDIR.tmp.XXXXXX")"
  rm -rf "$staging"
  git clone --quiet --depth 1 --branch "$BRANCH" "$REPO" "$staging" || return 1
  rm -rf "$WORKDIR"
  mv "$staging" "$WORKDIR"
}

sync_tree() {
  local final=$1 src dst
  # 1. Shard tree, mirrored. Guard: both ends must be the `shards` folder.
  src="$PHASE3/results/outputs/shards"
  dst="$WORKDIR/$REMOTE_PHASE3/results/outputs/shards"
  if [ "$(basename "$src")" != "$(basename "$dst")" ]; then
    say "REFUSING: $src and $dst are different folders"; return 1
  fi
  if [ -d "$src" ]; then
    mkdir -p "$dst"
    rsync -a --delete --include='*/' --include='*.parquet' --include='*.json' \
      --include='*.csv' --exclude='*' "$src/" "$dst/"
  fi
  # 2. Logs, never deleted.
  mkdir -p "$WORKDIR/$REMOTE_PHASE3/logs/gcp_api_run"
  rsync -a --include='*/' --include='*.log' --include='*.txt' --exclude='*' \
    "$PHASE3/logs/" "$WORKDIR/$REMOTE_PHASE3/logs/gcp_api_run/"
  # 3. Final: merged outputs, analysis, figures, tables, generated inputs.
  if [ "$final" = "1" ]; then
    for sub in results/outputs results/processed results/figures results/tables; do
      [ -d "$PHASE3/$sub" ] || continue
      mkdir -p "$WORKDIR/$REMOTE_PHASE3/$sub"
      rsync -a --exclude='shards/' --exclude='_shards/' --exclude='*.lock' \
        --include='*/' --include='*.parquet' --include='*.json' --include='*.csv' \
        --include='*.png' --include='*.pdf' --include='*.tex' --include='*.md' \
        --exclude='*' "$PHASE3/$sub/" "$WORKDIR/$REMOTE_PHASE3/$sub/"
    done
  fi
}

status_file() {
  {
    echo "# Phase 3 API run status"
    echo
    echo "- updated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "- host: $(hostname)"
    echo
    echo '```'
    cat "$PHASE3/logs/launcher/exits.txt" 2>/dev/null || echo "launcher not started"
    echo '```'
    echo
    echo "Units written per shard (revision rows, part files):"
    echo
    for dir in "$PHASE3"/results/outputs/shards/*/; do
      [ -d "$dir" ] || continue
      n=$(find "$dir" -path '*_shards/revision_log/part-*.parquet' | wc -l)
      echo "- $(basename "$dir"): $n revision part-files"
    done
  } > "$WORKDIR/$REMOTE_PHASE3/logs/gcp_api_run/STATUS.md"
}

push_once() {
  local final=$1
  setup_repo || return 1
  cd "$WORKDIR" || return 1
  git pull --quiet --rebase --depth 1 origin "$BRANCH" 2>/dev/null || true
  sync_tree "$final" || return 1
  status_file
  git add -A "$REMOTE_PHASE3/results" "$REMOTE_PHASE3/logs/gcp_api_run" 2>/dev/null
  if git diff --cached --quiet; then say "no change"; return 0; fi
  git commit -q -m "Phase 3 API run: $( [ "$final" = 1 ] && echo final || echo progress ) $(date -u '+%Y-%m-%d %H:%M UTC')"
  if git push -q origin "$BRANCH"; then
    say "pushed ($( [ "$final" = 1 ] && echo final || echo progress ))"
  else
    say "push rejected; fresh clone next attempt"
    cd "$HOME" && rm -rf "$WORKDIR"
    return 1
  fi
}

locked_push() {
  (
    flock -w 900 9 || { say "lock busy"; exit 1; }
    push_once "$1"
  ) 9>"$LOCK"
}

case "${1:-}" in
  --once)  locked_push 0 || locked_push 0 ;;
  --final) locked_push 1 || locked_push 1 ;;
  *)
    say "pushing every $((INTERVAL / 60)) minutes to $BRANCH"
    while true; do
      locked_push 0 || locked_push 0 || say "push cycle failed; next interval retries"
      sleep "$INTERVAL"
    done ;;
esac

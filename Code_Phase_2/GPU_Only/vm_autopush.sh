#!/usr/bin/env bash
# vm_autopush.sh — push GPU-probe results from the rented box to GitHub every 15 min.
#
# Runs on the VM only. Commits GPU_Only outputs into Code_Phase_2/results/gpu_probe/
# and pushes to main, rebasing over whatever the local machine has pushed in the
# meantime. Secrets are never staged: the .env on the box is gitignored, and the
# add list below is explicit.
set -uo pipefail

REPO=/workspace/Platos_ship
GPU_OUT="$REPO/Code_Phase_2/GPU_Only/data/outputs"
PUB="$REPO/Code_Phase_2/results/gpu_probe"
INTERVAL="${INTERVAL:-900}"

log() { printf '%s [vm-autopush] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

cd "$REPO" || exit 1
git config user.email "gpu-runner@localhost"
git config user.name "platos-ship-gpu-runner"

# Credentials and repo location both come from the environment. This file ships
# in the artefact released for double-blind review, so it must contain no
# github.com/<user>/... string.
ENVFILE="$REPO/Code_Phase_2/.env"
TOKEN=$(grep -m1 -E '^Github_Classic_Token=' "$ENVFILE" 2>/dev/null | cut -d= -f2- | tr -d '"'"'"' \r')
SLUG=$(grep -m1 -E '^GIT_REPO_SLUG=' "$ENVFILE" 2>/dev/null | cut -d= -f2- | tr -d '"'"'"' \r')
if [ -z "${SLUG:-}" ]; then
    echo "FATAL: set GIT_REPO_SLUG=owner/repo in $ENVFILE" >&2
    exit 1
fi
REMOTE="https://${TOKEN}@github.com/${SLUG}.git"

while true; do
    mkdir -p "$PUB"
    # Stage-cache pickles are resume scaffolding, not results — never published.
    for f in "$GPU_OUT"/*.parquet; do [ -e "$f" ] && cp -u "$f" "$PUB/"; done
    cp -u "$REPO/Code_Phase_2/GPU_Only/logs/gpu_probe.log" "$PUB/gpu_probe.log" 2>/dev/null || true

    git add -- Code_Phase_2/results/gpu_probe 2>/dev/null
    if git diff --cached --quiet; then
        log "nothing new"
    elif git diff --cached --name-only | grep -qE '(^|/)\.env($|\.)'; then
        log "ABORT: .env staged"; git reset >/dev/null
    else
        n=$(ls -1 "$PUB" 2>/dev/null | wc -l)
        git commit -q -m "GPU probe: publish outputs ($n files) @ $(date -u +%Y-%m-%dT%H:%MZ)"
        git fetch -q "$REMOTE" main && git rebase -q FETCH_HEAD || git rebase --abort
        if git push -q "$REMOTE" HEAD:main 2>&1 | grep -v '^remote:'; then :; fi
        log "pushed"
    fi
    sleep "$INTERVAL"
done

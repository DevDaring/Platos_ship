#!/usr/bin/env bash
# autopush.sh — publish experiment progress to GitHub on a timer.
#
#   bash scripts/autopush.sh            # light push (safe every 15 min)
#   bash scripts/autopush.sh --heavy    # also publishes trial_log.parquet (~40 MB)
#   bash scripts/autopush.sh --loop     # run forever, light every 15 min, heavy hourly
#
# Light pushes copy only the small artefacts (checkpoint, metrics, gate reports,
# status snapshot) into Code_Phase_2/results/. trial_log.parquet is 40 MB and is
# rewritten in full on every run, so committing it every 15 minutes would add
# gigabytes of git objects per day — it goes out on --heavy only.
#
# Secrets: .env is gitignored AND a staged-content scan runs before every
# commit; the push aborts if anything key-shaped appears in the diff.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE2="$REPO_ROOT/Code_Phase_2"
LIVE="$PHASE2/CPU_Only/data"
PUB="$PHASE2/results"
PY="${PY:-$HOME/venv/bin/python}"
[ -x "$PY" ] || PY=python3

HEAVY=0
LOOP=0
for a in "$@"; do
    case "$a" in
        --heavy) HEAVY=1 ;;
        --loop)  LOOP=1 ;;
        *) echo "unknown flag: $a" >&2; exit 2 ;;
    esac
done

log() { printf '%s [autopush] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# ── credentials ────────────────────────────────────────────────────────────
load_token() {
    for cand in "$REPO_ROOT/.env" "$PHASE2/.env" "$REPO_ROOT/../Code_Phase_2/.env"; do
        [ -f "$cand" ] || continue
        local t
        t=$(grep -m1 -E '^Github_Classic_Token=' "$cand" | cut -d= -f2- | tr -d '"'"'"' \r')
        if [ -n "${t:-}" ]; then GH_TOKEN="$t"; return 0; fi
    done
    return 1
}

# ── sync live run artefacts into the published results/ tree ───────────────
sync_results() {
    mkdir -p "$PUB/processed" "$PUB/outputs" "$PUB/progress"
    if [ -d "$LIVE/processed" ]; then
        cp -u "$LIVE/processed"/*.parquet "$PUB/processed/" 2>/dev/null || true
    fi
    if [ -d "$LIVE/outputs" ]; then
        for f in "$LIVE/outputs"/*; do
            base=$(basename "$f")
            # the big one only on a heavy push
            if [ "$base" = "trial_log.parquet" ] && [ "$HEAVY" -eq 0 ]; then continue; fi
            cp -u "$f" "$PUB/outputs/" 2>/dev/null || true
        done
    fi
    # GPU probe outputs, if the probe has produced any
    if [ -d "$PHASE2/GPU_Only/data/outputs" ]; then
        mkdir -p "$PUB/gpu_probe"
        cp -u "$PHASE2/GPU_Only/data/outputs"/* "$PUB/gpu_probe/" 2>/dev/null || true
    fi
    "$PY" "$REPO_ROOT/scripts/run_status.py" --phase2-root "$PHASE2/CPU_Only" >/dev/null 2>&1 || true
}

# ── refuse to commit anything key-shaped ───────────────────────────────────
scan_staged_for_secrets() {
    # Never allow an env file through, whatever it is called.
    if git -C "$REPO_ROOT" diff --cached --name-only | grep -qE '(^|/)\.env($|\.)'; then
        log "ABORT: a .env file is staged"
        return 1
    fi
    # Scan added text for provider key formats. Binary (parquet) diffs are skipped
    # by git as "Binary files differ", so this only ever sees text.
    local hits
    hits=$(git -C "$REPO_ROOT" diff --cached -U0 | grep -E '^\+' | grep -oE \
        'sk-or-v1-[A-Za-z0-9]{16,}|sk-[A-Za-z0-9]{32,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AIza[A-Za-z0-9_-]{30,}' \
        | head -5)
    if [ -n "$hits" ]; then
        log "ABORT: staged diff contains key-shaped strings"
        return 1
    fi
    return 0
}

push_once() {
    cd "$REPO_ROOT" || return 1
    sync_results

    # Explicit paths only — never `git add -A`, so an untracked stray secret
    # cannot be swept in.
    git add -- Code_Phase_2/results scripts src config \
        Code_Phase_2/CPU_Only/src Code_Phase_2/CPU_Only/config \
        Code_Phase_2/CPU_Only/run_all.py Code_Phase_2/CPU_Only/README.md \
        Code_Phase_2/GPU_Only Code_Phase_2/README.md \
        README.md CLAUDE.md Recharge.md ACTION_REQUIRED.md requirements.txt \
        .gitattributes .gitignore 2>/dev/null

    if git diff --cached --quiet; then
        log "nothing to publish"
        return 0
    fi
    scan_staged_for_secrets || { git reset >/dev/null; return 1; }

    local n kind
    n=$("$PY" - <<'EOF' 2>/dev/null || echo "?"
import json, pathlib
p = pathlib.Path("Code_Phase_2/results/progress/run_status.json")
print(json.loads(p.read_text())["total_completed_trials"] if p.exists() else "?")
EOF
)
    kind=$([ "$HEAVY" -eq 1 ] && echo "full" || echo "progress")
    git commit -q -m "Auto-publish ($kind): $n completed trials @ $(date -u +%Y-%m-%dT%H:%MZ)" || return 1

    if load_token; then
        local url
        url=$(git remote get-url origin | sed -E 's#https://([^@]*@)?#https://#')
        git push -q "https://${GH_TOKEN}@${url#https://}" HEAD:main 2>&1 | grep -v '^remote:' || true
    else
        git push -q origin HEAD:main || { log "push failed (no token, no cached creds)"; return 1; }
    fi
    log "pushed ($kind, $n trials)"
}

if [ "$LOOP" -eq 1 ]; then
    log "loop mode: light push every 15 min, heavy push every 4th cycle (hourly)"
    i=0
    while true; do
        i=$((i + 1))
        HEAVY=$([ $((i % 4)) -eq 0 ] && echo 1 || echo 0)
        push_once
        sleep 900
    done
else
    push_once
fi

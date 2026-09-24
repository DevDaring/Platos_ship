#!/usr/bin/env bash
# deploy_vast.sh — stage and upload the probe to a Vast.ai instance, then
# bootstrap it. Runs on YOUR machine.
#
#   bash deploy_vast.sh <ssh-host> <ssh-port>
#   SKIP_SMOKE=1 bash deploy_vast.sh <ssh-host> <ssh-port>
#
# Replaces the per-file scp sequence in push_inputs.sh, which omitted two
# modules the probe imports:
#
#   analysis/stats.py                     imported by corrected_probe at
#                                         CONTRAST time, i.e. after every
#                                         minute of GPU work is already paid
#   src/agent_wrappers/openai_compatible_agent.py
#                                         imported at START-UP through
#                                         src/extraction.py -> judge_agent
#
# The first cost the L4 run a full re-run. Staging one tree and copying it
# once makes the payload auditable before anything is billed.

set -euo pipefail

HOST="${1:?usage: deploy_vast.sh <ssh-host> <ssh-port>}"
PORT="${2:?usage: deploy_vast.sh <ssh-host> <ssh-port>}"
REMOTE="${REMOTE:-/workspace/platos}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PHASE3="$(dirname "$HERE")"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

[ -n "${HUGGINGFACE_TOKEN:-}" ] || fail \
  "export HUGGINGFACE_TOKEN first — Llama-3.1-70B is gated and the 132 GB
   download would fail only after the GPUs are already billing."

# Vast authenticates with whichever key is registered on the account; it is
# not necessarily ~/.ssh/id_rsa. Pass SSH_KEY to pick it explicitly.
SSH_KEY="${SSH_KEY:-}"
KEY_OPTS=()
[ -n "$SSH_KEY" ] && KEY_OPTS=(-i "$SSH_KEY")
SSH_OPTS=("${KEY_OPTS[@]}" -p "$PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=120)
SCP_OPTS=("${KEY_OPTS[@]}" -P "$PORT" -o StrictHostKeyChecking=accept-new)

# ── stage the payload locally ─────────────────────────────────────────────
say "Staging the upload"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/GPU_Only/src" "$STAGE/src/agent_wrappers" "$STAGE/analysis" \
         "$STAGE/inputs"

# autopush_vast.sh belongs in this list too. A re-deploy does `rm -rf $REMOTE`,
# so anything not staged is DELETED from the box. Omitting it once meant the
# full run started with no 30-minute push and nothing protecting the results
# on an instance that can be reclaimed at any moment.
cp "$HERE/run_big_probe.py" "$HERE/requirements_big.txt" \
   "$HERE/vast_bootstrap.sh" "$HERE/smoke_test_vast.sh" \
   "$HERE/run_full_vast.sh" "$HERE/autopush_vast.sh" \
   "$HERE/run_sweep_vast.sh" "$STAGE/"
cp "$PHASE3/GPU_Only/src/corrected_probe.py" "$STAGE/GPU_Only/src/"
cp "$PHASE3/GPU_Only/vram_planner.py"        "$STAGE/GPU_Only/"
cp "$PHASE3/src/contexts.py" "$PHASE3/src/extraction.py" \
   "$PHASE3/src/seeding.py"  "$PHASE3/src/__init__.py" "$STAGE/src/"
cp "$PHASE3/src/agent_wrappers/judge_agent.py" \
   "$PHASE3/src/agent_wrappers/base_agent.py" \
   "$PHASE3/src/agent_wrappers/openai_compatible_agent.py" \
   "$PHASE3/src/agent_wrappers/__init__.py" "$STAGE/src/agent_wrappers/"
cp "$PHASE3/analysis/stats.py" "$STAGE/analysis/"
: > "$STAGE/analysis/__init__.py"
: > "$STAGE/GPU_Only/__init__.py"
: > "$STAGE/GPU_Only/src/__init__.py"

for f in question_pool.parquet dumb_personas.parquet; do
  [ -f "$PHASE3/results/processed/$f" ] || fail \
    "missing $f — run: python3 tools/fetch_artefacts.py"
  cp "$PHASE3/results/processed/$f" "$STAGE/inputs/"
done

# Scripts must be LF. A CRLF autopush.sh on the L4 run died with
# "set: pipefail: invalid option name".
# Git Bash on Windows ships `python`, not `python3`, so resolve it rather
# than assuming the Linux name on the machine running this script.
PYBIN="$(command -v python3 || command -v python)"
[ -n "$PYBIN" ] || fail "no python on PATH to normalise line endings"
"$PYBIN" - "$STAGE" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
for f in root.rglob("*.sh"):
    b = f.read_bytes()
    if b"\r\n" in b:
        f.write_bytes(b.replace(b"\r\n", b"\n"))
        print(f"    LF-normalised {f.name}")
PY

say "Payload: $(du -sh "$STAGE" | cut -f1) — no keys, no model, no results"
find "$STAGE" -type f | sed "s|$STAGE|  .|" | sort

# ── upload ────────────────────────────────────────────────────────────────
say "Uploading to $HOST:$REMOTE"
ssh "${SSH_OPTS[@]}" "$HOST" "rm -rf '$REMOTE' && mkdir -p '$REMOTE'"
scp "${SCP_OPTS[@]}" -r "$STAGE"/* "$HOST:$REMOTE/"
ssh "${SSH_OPTS[@]}" "$HOST" "chmod +x '$REMOTE'/*.sh"

# ── bootstrap ─────────────────────────────────────────────────────────────
say "Bootstrapping (vLLM + FlashAttention, then the 132 GB checkpoint)"
ssh "${SSH_OPTS[@]}" "$HOST" \
  "cd '$REMOTE' && HUGGINGFACE_TOKEN='$HUGGINGFACE_TOKEN' MODEL='${MODEL:-meta-llama/Llama-3.1-70B-Instruct}' bash vast_bootstrap.sh"

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  say "Smoke test — 2 questions, all three conditions, output checked"
  ssh "${SSH_OPTS[@]}" "$HOST" "cd '$REMOTE' && MODEL='${MODEL:-meta-llama/Llama-3.1-70B-Instruct}' bash smoke_test_vast.sh"
fi

cat <<MSG

Deployed and verified.

Full run (detached):
  ssh -p $PORT $HOST "cd $REMOTE && nohup bash run_full_vast.sh > run.out 2>&1 &"

Destroy the instance when the results are pulled — that is the only thing
that stops the cost.
MSG

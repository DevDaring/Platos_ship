#!/usr/bin/env bash
# deploy.sh — upload the GPU-only code to the L4 VM and bootstrap it.
# Runs on YOUR machine.
#
#   bash deploy.sh                    # upload + bootstrap + smoke test
#   SKIP_SMOKE=1 bash deploy.sh       # upload + bootstrap only
#
# Only what the probe needs is uploaded: the GPU probe, the few src modules it
# imports, and the two small input parquets (~530 KB). No .env, no API keys,
# no results, no model.

set -euo pipefail

PROJECT="${PROJECT:-silicon-guru-472717-q9}"
ZONE="${ZONE:-us-central1-c}"
NAME="${NAME:-platos-ship}"
REMOTE="${REMOTE:-platos}"

# `gcloud compute ssh` resolves the Linux username and key for you, which
# matters here: the key comment is `DESKTOP-9952NT0\Debz@...`, so the username
# GCP derived is not guessable from the key alone. If you would rather use the
# key directly, set SSH_USER and SSH_HOST and the script will use plain ssh/scp.
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa_gcp}"
SSH_USER="${SSH_USER:-}"
SSH_HOST="${SSH_HOST:-}"

if [ -n "$SSH_USER" ] && [ -n "$SSH_HOST" ]; then
  USE_GCLOUD=0
  SSH_TARGET="$SSH_USER@$SSH_HOST"
else
  USE_GCLOUD=1
fi

remote_exec() {
  if [ "$USE_GCLOUD" = "1" ]; then
    gcloud compute ssh "$NAME" --zone "$ZONE" --project "$PROJECT" --quiet \
      --command "$1"
  else
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$SSH_TARGET" "$1"
  fi
}

remote_copy() {   # remote_copy <local-glob-dir> <remote-dir>
  if [ "$USE_GCLOUD" = "1" ]; then
    gcloud compute scp --recurse --zone "$ZONE" --project "$PROJECT" --quiet \
      "$1"/* "$NAME:$2"
  else
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -r \
      "$1"/* "$SSH_TARGET:$2"
  fi
}

HERE="$(cd "$(dirname "$0")" && pwd)"
PHASE3="$(dirname "$HERE")"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

[ -n "${HUGGINGFACE_TOKEN:-}" ] || fail \
  "export HUGGINGFACE_TOKEN first — Llama-3.1-8B is gated and the download
   would fail only after the GPU is already billing."

# ── stage the payload locally ─────────────────────────────────────────────
say "Staging the upload"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/GPU_Only/src" "$STAGE/src/agent_wrappers" "$STAGE/inputs"

# The probe itself, renamed to a neutral entry point. It already accepts
# --model and --tensor-parallel-size, so one L4 is just tp=1 with the 8B.
cp "$PHASE3/Vast_AI_Big_Model_Run/run_big_probe.py" "$STAGE/run_probe.py"
cp "$PHASE3/GPU_Only/src/corrected_probe.py" "$STAGE/GPU_Only/src/"
cp "$PHASE3/GPU_Only/vram_planner.py"          "$STAGE/GPU_Only/"

# The handful of Phase-3 modules the probe imports.
cp "$PHASE3/src/contexts.py" "$PHASE3/src/extraction.py" \
   "$PHASE3/src/seeding.py"  "$PHASE3/src/__init__.py" "$STAGE/src/"
cp "$PHASE3/src/agent_wrappers/judge_agent.py" \
   "$PHASE3/src/agent_wrappers/base_agent.py" \
   "$PHASE3/src/agent_wrappers/openai_compatible_agent.py" \
   "$PHASE3/src/agent_wrappers/__init__.py" "$STAGE/src/agent_wrappers/"
: > "$STAGE/GPU_Only/__init__.py"
: > "$STAGE/GPU_Only/src/__init__.py"

# Inputs.
for f in question_pool.parquet dumb_personas.parquet; do
  [ -f "$PHASE3/results/processed/$f" ] || fail \
    "missing $f — run: python3 tools/fetch_artefacts.py"
  cp "$PHASE3/results/processed/$f" "$STAGE/inputs/"
done

cp "$HERE/vm_bootstrap.sh" "$HERE/smoke_test.sh" "$HERE/run_full.sh" \
   "$HERE/autopush.sh" "$HERE/requirements_l4.txt" "$STAGE/"

say "Payload: $(du -sh "$STAGE" | cut -f1) — no keys, no model, no results"
find "$STAGE" -type f | sed "s|$STAGE|  .|" | sort

# ── upload ────────────────────────────────────────────────────────────────
say "Uploading to $NAME:~/$REMOTE"
remote_exec "rm -rf ~/$REMOTE && mkdir -p ~/$REMOTE"
remote_copy "$STAGE" "~/$REMOTE/"

# ── bootstrap ─────────────────────────────────────────────────────────────
say "Bootstrapping (installs vLLM + FlashAttention, downloads the checkpoint)"
remote_exec "cd ~/$REMOTE && HUGGINGFACE_TOKEN='$HUGGINGFACE_TOKEN' bash vm_bootstrap.sh"

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  say "Smoke test — 2 questions, all three conditions, output checked"
  remote_exec "cd ~/$REMOTE && bash smoke_test.sh"
fi

cat <<MSG

Deployed and verified.

Full run (detached, pushes every 30 min):

  gcloud compute ssh $NAME --zone $ZONE --command \\
    "cd ~/$REMOTE && GITHUB_TOKEN='<token>' nohup bash run_full.sh > run.out 2>&1 &"

Then watch it from GitHub:
  github.com/DevDaring/Platos_ship  branch l4-probe-results
  -> Code/Code_Phase_3/GCP_L4_Run/results/STATUS.md

Delete the VM when it is done — a stopped instance still bills for its disk:
  gcloud compute instances delete $NAME --zone $ZONE --quiet
MSG

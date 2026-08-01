#!/usr/bin/env bash
# vm_bootstrap.sh — one-shot setup of a rented GPU box for the E9 logprob probe.
#
# Usage on the VM (after scp-ing gpu.env next to it):
#   bash vm_bootstrap.sh          # install + fetch pools + dry run
#   bash vm_bootstrap.sh --full   # ... then launch the full probe under nohup
#
# gpu.env must define exactly two secrets:
#   HUGGINGFACE_TOKEN=...   (Llama-3.1-8B-Instruct is a gated repo)
#   Github_Classic_Token=... (to push results back)
# Nothing else from the local .env is ever copied to a rented machine.
set -euo pipefail

# Repo location comes from the environment, never hardcoded: this file is part
# of the artefact released for double-blind review, and a hardcoded
# github.com/<user>/... URL would de-anonymise it. Set GIT_REPO_SLUG in gpu.env
# (e.g. "owner/repo"); the clone URL is assembled from it at runtime.
WORK=/workspace
REPO="$WORK/Platos_ship"
GPU_DIR="$REPO/Code_Phase_2/GPU_Only"
CPU_DIR="$REPO/Code_Phase_2/CPU_Only"

log() { printf '\n=== %s ===\n' "$*"; }

log "system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl rsync >/dev/null

log "python deps (vllm brings its own torch build)"
pip install -q --upgrade pip
# `openai` is needed because CPU_Only's judge_agent (whose regex extractor the
# probe reuses) imports the OpenAI-compatible client at module load.
pip install -q vllm pandas pyarrow python-dotenv pyyaml transformers openai scipy

log "secrets (loaded before the clone — the repo is private)"
if [ -f "$WORK/gpu.env" ]; then
    set -a; . "$WORK/gpu.env"; set +a
    export HF_TOKEN="${HUGGINGFACE_TOKEN:-}"
else
    echo "FATAL: $WORK/gpu.env not found (needs HUGGINGFACE_TOKEN + Github_Classic_Token)" >&2
    exit 1
fi

log "clone repo"
if [ -z "${GIT_REPO_SLUG:-}" ]; then
    echo "FATAL: set GIT_REPO_SLUG=owner/repo in gpu.env" >&2
    exit 1
fi
mkdir -p "$WORK" && cd "$WORK"
REPO_URL="https://github.com/${GIT_REPO_SLUG}.git"
AUTH_URL="https://${Github_Classic_Token}@github.com/${GIT_REPO_SLUG}.git"
if [ ! -d "$REPO/.git" ]; then
    git clone --depth 1 "$AUTH_URL" "$REPO" 2>&1 | sed "s/${Github_Classic_Token}/***/g"
fi
cd "$REPO"
# Keep the token out of .git/config; fetch with it explicitly instead.
git remote set-url origin "$REPO_URL"
git fetch -q "$AUTH_URL" main 2>&1 | sed "s/${Github_Classic_Token}/***/g" || true
git reset -q --hard FETCH_HEAD || true

log "stage the CPU_Only pools the probe reuses"
mkdir -p "$CPU_DIR/data/processed" "$GPU_DIR/data/outputs" "$GPU_DIR/logs"
cp -n "$REPO/Code_Phase_2/results/processed/question_pool.parquet"  "$CPU_DIR/data/processed/"
cp -n "$REPO/Code_Phase_2/results/processed/dumb_personas.parquet"  "$CPU_DIR/data/processed/"
ls -la "$CPU_DIR/data/processed/"

log "place secrets inside the repo for the autopush loop (gitignored)"
cp "$WORK/gpu.env" "$REPO/Code_Phase_2/.env"
chmod 600 "$REPO/Code_Phase_2/.env"

log "GPU visible?"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

log "DRY RUN (2 questions x 1 replication, all three conditions, full code path)"
cd "$GPU_DIR"
python run_all.py --dry-run 2>&1 | tail -40

if [ "${1:-}" = "--full" ]; then
    log "FULL PROBE (background)"
    cd "$GPU_DIR"
    setsid nohup python run_all.py > logs/full_probe.out 2>&1 < /dev/null &
    echo "probe pid $!"
    log "auto-push loop (results -> GitHub every 15 min)"
    setsid nohup bash "$GPU_DIR/vm_autopush.sh" > logs/vm_autopush.log 2>&1 < /dev/null &
    echo "autopush pid $!"
fi

log "bootstrap done"

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

REPO_URL="https://github.com/DevDaring/Platos_ship.git"
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
pip install -q vllm pandas pyarrow python-dotenv pyyaml transformers

log "clone repo"
mkdir -p "$WORK" && cd "$WORK"
[ -d "$REPO/.git" ] || git clone --depth 1 "$REPO_URL" "$REPO"
cd "$REPO" && git pull --ff-only 2>/dev/null || true

log "stage the CPU_Only pools the probe reuses"
mkdir -p "$CPU_DIR/data/processed" "$GPU_DIR/data/outputs" "$GPU_DIR/logs"
cp -n "$REPO/Code_Phase_2/results/processed/question_pool.parquet"  "$CPU_DIR/data/processed/"
cp -n "$REPO/Code_Phase_2/results/processed/dumb_personas.parquet"  "$CPU_DIR/data/processed/"
ls -la "$CPU_DIR/data/processed/"

log "secrets"
if [ -f "$WORK/gpu.env" ]; then
    cp "$WORK/gpu.env" "$REPO/Code_Phase_2/.env"
    chmod 600 "$REPO/Code_Phase_2/.env"
    set -a; . "$REPO/Code_Phase_2/.env"; set +a
    export HF_TOKEN="${HUGGINGFACE_TOKEN:-}"
else
    echo "WARNING: $WORK/gpu.env not found — a gated model download will fail." >&2
fi

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

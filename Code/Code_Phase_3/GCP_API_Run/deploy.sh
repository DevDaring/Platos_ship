#!/usr/bin/env bash
# deploy.sh — upload the code and the .env to the VM, then bootstrap it.
# Runs on YOUR machine, after create_vm.sh.
#
# Uploaded: Code_Phase_3 without its run outputs, logs, GPU-only code and the
# earlier VM folders; plus Code/.env (API keys and the GitHub token), which
# lands at ~/platos/Code/.env with mode 600. The VM has no service account,
# and it is deleted after the run, which removes the keys with it.

set -euo pipefail

PROJECT="${PROJECT:-silicon-guru-472717-q9}"
ZONE="${ZONE:-us-central1-a}"
NAME="${NAME:-platos-api-run}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PHASE3="$(dirname "$HERE")"
CODE="$(dirname "$PHASE3")"

say()  { printf '\n==> %s\n' "$*"; }
fail() { printf '\nFAIL: %s\n' "$*" >&2; exit 1; }
remote() { gcloud compute ssh "$NAME" --zone "$ZONE" --project "$PROJECT" --quiet --command "$1"; }

[ -f "$CODE/.env" ] || fail "no $CODE/.env"

say "Staging"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/Code"
( cd "$CODE" && tar \
    --exclude='./Code_Phase_3/results/outputs' \
    --exclude='./Code_Phase_3/logs' \
    --exclude='./Code_Phase_3/GPU_Only' \
    --exclude='./Code_Phase_3/Vast_AI_Big_Model_Run' \
    --exclude='./Code_Phase_3/GCP_L4_Run' \
    --exclude='__pycache__' --exclude='.pytest_cache' \
    -czf "$STAGE/payload.tgz" ./Code_Phase_3 ./.env )
say "Payload $(du -h "$STAGE/payload.tgz" | cut -f1)"

say "Uploading"
REMOTE_HOME="$(remote 'printf %s "$HOME"' | tr -d '\r\n')"
[ -n "$REMOTE_HOME" ] || fail "could not resolve the remote HOME"
remote "rm -rf $REMOTE_HOME/platos/Code && mkdir -p $REMOTE_HOME/platos/Code"
gcloud compute scp --zone "$ZONE" --project "$PROJECT" --quiet \
  "$STAGE/payload.tgz" "$NAME:$REMOTE_HOME/platos/payload.tgz"
remote "cd ~/platos/Code && tar -xzf ../payload.tgz && rm ../payload.tgz && chmod 600 .env"

say "Bootstrapping"
remote "bash ~/platos/Code/Code_Phase_3/GCP_API_Run/vm_bootstrap.sh"

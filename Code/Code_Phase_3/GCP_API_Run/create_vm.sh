#!/usr/bin/env bash
# create_vm.sh — one small CPU VM for the Phase 3 API run. Runs on YOUR machine.
#
# The run makes HTTP calls and waits on them; it needs almost no CPU. One
# e2-standard-2 (2 vCPU, 8 GB) holds eight shard processes comfortably.
#
# Nothing recurring, as required for project silicon-guru-472717-q9:
#   * no snapshot schedule, no backup policy, no machine image, no reserved IP
#     (the external IP is ephemeral and is released with the VM);
#   * no Shielded VM features, no service account, no API scopes (the VM
#     calls model providers and GitHub, never a Google Cloud API);
#   * the boot disk is deleted with the VM (--boot-disk-auto-delete).
# Deleting the VM (delete_vm.sh) therefore ends every charge.
#
# Standard provisioning, not Spot: a pre-emption mid-run would need a manual
# restart, and the whole run costs about a dollar of compute either way.

set -euo pipefail

PROJECT="${PROJECT:-silicon-guru-472717-q9}"
ZONE="${ZONE:-us-central1-a}"
NAME="${NAME:-platos-api-run}"
MACHINE="${MACHINE:-e2-standard-2}"
DISK_GB="${DISK_GB:-20}"

say()  { printf '\n==> %s\n' "$*"; }
fail() { printf '\nFAIL: %s\n' "$*" >&2; exit 1; }

if gcloud compute instances describe "$NAME" --zone "$ZONE" --project "$PROJECT" \
     >/dev/null 2>&1; then
  say "Instance '$NAME' already exists; reusing it."
  exit 0
fi

say "Creating $NAME ($MACHINE, ${DISK_GB} GB pd-standard) in $PROJECT/$ZONE"
gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size="${DISK_GB}GB" \
  --boot-disk-type=pd-standard \
  --boot-disk-auto-delete \
  --no-service-account --no-scopes \
  --no-shielded-secure-boot \
  --no-shielded-vtpm \
  --no-shielded-integrity-monitoring \
  --provisioning-model=STANDARD \
  --maintenance-policy=MIGRATE \
  --labels=purpose=platos-phase3-api,delete-after-run=true

say "Waiting for SSH"
for attempt in $(seq 1 30); do
  if gcloud compute ssh "$NAME" --zone "$ZONE" --project "$PROJECT" --quiet \
       --command "echo ready" >/dev/null 2>&1; then
    say "SSH is up."
    exit 0
  fi
  sleep 10
done
fail "SSH did not come up after ~5 minutes"

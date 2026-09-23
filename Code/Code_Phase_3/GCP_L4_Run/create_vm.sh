#!/usr/bin/env bash
# create_vm.sh — one L4 VM for the X8 output-distribution probe.
#
# Deliberately minimal, as requested: no snapshot schedule, no backup policy,
# no Shielded VM, no OS Login enforcement. The instance exists to run one
# experiment and then be deleted.
#
#   bash create_vm.sh                 # standard instance
#   SPOT=1 bash create_vm.sh          # ~60% cheaper, can be pre-empted
#
# Cost note: g2-standard-8 (1x L4) is roughly $0.85/hour on-demand in
# us-central1, about $0.30/hour as Spot. The probe itself is well under an
# hour; the model download is usually the longer part. The run is checkpointed
# and pushes every 30 minutes, so a Spot pre-emption loses at most one
# interval — but it does need a manual restart.

set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-us-central1-a}"
NAME="${NAME:-platos-l4-probe}"
MACHINE="${MACHINE:-g2-standard-8}"     # 1x NVIDIA L4 (24 GB), 8 vCPU, 32 GB RAM
DISK_GB="${DISK_GB:-200}"               # 8B checkpoint ~16 GB + CUDA image headroom

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

[ -n "$PROJECT" ] || fail "no project set: gcloud config set project <ID>"

say "Project $PROJECT | zone $ZONE | $MACHINE (1x L4) | ${DISK_GB}GB"

if gcloud compute instances describe "$NAME" --zone "$ZONE" >/dev/null 2>&1; then
  say "Instance '$NAME' already exists — reusing it."
  gcloud compute instances describe "$NAME" --zone "$ZONE" \
    --format="table(name,status,machineType.basename(),guestAccelerators[0].acceleratorType.basename())"
  exit 0
fi

SPOT_ARGS=()
if [ "${SPOT:-0}" = "1" ]; then
  # Spot: cheapest, and safe here because the probe is checkpointed per stage
  # and results are pushed every 30 minutes.
  SPOT_ARGS=(--provisioning-model=SPOT --instance-termination-action=DELETE)
  say "Spot provisioning requested (pre-emptible)."
fi

say "Creating $NAME"
# The Deep Learning VM image ships CUDA and the NVIDIA driver, which removes
# the slowest and most failure-prone part of the setup.
gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --accelerator="type=nvidia-l4,count=1" \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu124-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size="${DISK_GB}GB" \
  --boot-disk-type=pd-balanced \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform \
  --no-shielded-secure-boot \
  --no-shielded-vtpm \
  --no-shielded-integrity-monitoring \
  "${SPOT_ARGS[@]}"

say "Waiting for SSH"
for attempt in $(seq 1 30); do
  if gcloud compute ssh "$NAME" --zone "$ZONE" --command "echo ready" \
       --quiet >/dev/null 2>&1; then
    say "SSH is up."
    break
  fi
  [ "$attempt" -eq 30 ] && fail "SSH did not come up after ~5 minutes."
  sleep 10
done

say "GPU check"
gcloud compute ssh "$NAME" --zone "$ZONE" --quiet \
  --command "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"

cat <<MSG

Instance ready: $NAME ($ZONE)

  Next:   bash deploy.sh
  Delete: gcloud compute instances delete $NAME --zone $ZONE --quiet

A stopped instance still bills for its disk. Delete it when the results are
pulled; that is the only thing that stops the cost.
MSG

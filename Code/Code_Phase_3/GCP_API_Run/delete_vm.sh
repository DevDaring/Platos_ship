#!/usr/bin/env bash
# delete_vm.sh — delete the VM, then prove nothing billable is left.
# Runs on YOUR machine. Only after the results are verified and on GitHub.

set -uo pipefail

PROJECT="${PROJECT:-silicon-guru-472717-q9}"
ZONE="${ZONE:-us-central1-a}"
NAME="${NAME:-platos-api-run}"

gcloud compute instances delete "$NAME" --zone "$ZONE" --project "$PROJECT" \
  --delete-disks=all --quiet

echo
echo "Remaining billable compute resources in $PROJECT (all must be empty):"
count() {  # count <label> <gcloud compute args...>
  printf '  %-18s %s\n' "$1:" \
    "$(gcloud compute "${@:2}" --project "$PROJECT" --format='value(name)' \
         2>/dev/null | grep -c .)"
}
count instances         instances list
count disks             disks list
count snapshots         snapshots list
count images            images list --no-standard-images
count machine-images    machine-images list
count addresses         addresses list
count resource-policies resource-policies list

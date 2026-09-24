# Phase 3 API run on a GCP CPU VM

All API-bound experiments (X1–X7) run on one small VM in project
`silicon-guru-472717-q9`. X5 is offline and runs in the analysis step.

## Order

```bash
# on your machine (Git Bash), logged in with gcloud
bash GCP_API_Run/create_vm.sh        # e2-standard-2, no backups/snapshots/Shielded VM
bash GCP_API_Run/deploy.sh           # upload code + Code/.env, install, run the test suite

# smoke test on the VM (4 questions, every experiment, real providers)
gcloud compute ssh platos-api-run --zone us-central1-a --command \
  "cd ~/platos/Code/Code_Phase_3 && bash GCP_API_Run/start_run.sh --max-questions 4"
# ...check, then wipe the smoke outputs before the real run:
#   rm -rf results/outputs logs results/processed/{honest_peer_bank,hedged_personas,gsm_symbolic_pool,gsm_symbolic_personas}.parquet

# full run, detached
gcloud compute ssh platos-api-run --zone us-central1-a --command \
  "cd ~/platos/Code/Code_Phase_3 && bash GCP_API_Run/start_run.sh"

# after the results are verified and on GitHub
bash GCP_API_Run/delete_vm.sh        # deletes the VM and lists what remains (all zero)
```

## What runs

`tools/run_parallel.sh`:

1. `run_all.py --prepare` once. It builds the honest bank, hedged pool,
   GSM-Symbolic pool and its personas. No shard starts before this exits 0.
2. One process per focal model, all eight at once, each writing only to
   `results/outputs/shards/<model>/`. Inside a process, units run on 4–8
   threads (`src/concurrency.py`); `tests/test_concurrency.py` shows the output
   is identical to one thread.
3. Exit code 3 means some API calls failed and were not recorded
   (`src/call_guard.py`). The shard is re-run and pays only for what is
   missing. Any other non-zero exit stops that shard, and the merge does not
   happen.
4. `tools/merge_shards.py` refuses unless every check passes. Then
   `run_all.py --analyse` runs.

`GCP_API_Run/autopush.sh` pushes shard outputs and logs to `main` every
30 minutes, and everything else once at the end. Watch
`Code/Code_Phase_3/logs/gcp_api_run/STATUS.md` on GitHub.

## Cost and teardown

About $0.07/hour for the VM plus an ephemeral IP, for roughly 10 hours. No
snapshot schedule, machine image, reserved IP or service account is created.
The boot disk is deleted with the VM. `delete_vm.sh` ends every charge, and
it prints the remaining instances, disks, snapshots, images, machine images,
addresses and resource policies, which should all be 0.

The VM holds `Code/.env` (API keys, GitHub token) at mode 600 until it is
deleted. Revoke the GitHub token after the run.

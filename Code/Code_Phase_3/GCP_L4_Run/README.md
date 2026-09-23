# GCP L4 run — the X8 output-distribution probe

Runs the corrected output-distribution probe on **Llama-3.1-8B-Instruct in
bf16** on a single **NVIDIA L4 (24 GB)**, pushing results and logs to GitHub
every 30 minutes so nothing is lost if the VM goes away.

This is the GPU half of Phase 3. X1–X7 are API-bound and need no GPU.

---

## One-time setup on your machine

```bash
gcloud auth login                      # the session token expires; this is interactive
gcloud config set project <PROJECT_ID>
export HUGGINGFACE_TOKEN=hf_...        # Llama-3.1-8B is a gated repo
```

## Run it

```bash
cd Code/Code_Phase_3/GCP_L4_Run

bash create_vm.sh                      # 1x L4, no snapshot/backup/shielding
bash deploy.sh                         # upload, install vLLM + FlashAttention,
                                       # download the checkpoint, SMOKE TEST

# full run, detached, with the 30-minute autopush
gcloud compute ssh platos-l4-probe --zone us-central1-a --command \
  "cd ~/platos && GITHUB_TOKEN='<token>' nohup bash run_full.sh > run.out 2>&1 &"

# when it finishes — this is the only thing that stops the cost
gcloud compute instances delete platos-l4-probe --zone us-central1-a --quiet
```

`deploy.sh` runs the smoke test automatically and **fails loudly if the output
is wrong**, so the full run only starts from a verified state.

## Why an L4 is the right card here

| | |
|---|---|
| Llama-3.1-8B weights, bf16 | **15.0 GB** |
| KV cache | 128 KB/token → 512 MB per 4096-token sequence |
| L4 usable at `gpu_memory_utilization=0.90` | ~20.7 GB |
| Concurrent sequences after weights | ~13 |

Comfortable. `python3 GPU_Only/vram_planner.py` recomputes this for any model
or context length.

**Do not quantise.** The probe measures probability mass over answer options;
int4/fp8 change the output distribution, which is the quantity being measured.
`vm_bootstrap.sh` refuses to start if the card cannot hold bf16 rather than
silently falling back.

## FlashAttention

vLLM ships FlashAttention kernels (`vllm-flash-attn`) and an L4 is sm89 (Ada),
which FlashAttention-2 supports. `vm_bootstrap.sh` sets

```
VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

so the choice is explicit and logged rather than left to a heuristic, and
verifies the kernels import. A standalone `flash-attn` wheel is also installed
**only if a prebuilt one matches** this torch/CUDA pair — a source build takes
about 30 minutes on a GPU that is already billing, which is a poor trade for a
path vLLM already covers.

## The smoke test is a real test

`smoke_test.sh` runs 2 questions through all three conditions and then checks
the output rather than trusting the exit code:

1. every condition (R, E, WR) produced rows;
2. the target is **fixed per (question, replicate) across conditions** —
   otherwise the difference-in-differences compares unlike things;
3. probability mass is finite and within [0, 1] — catches a broken chat
   template or a tokenizer mismatch;
4. off-candidate mass is present, i.e. not normalised away;
5. Round-1 answers parsed;
6. all four output files exist.

One caveat by construction: `--dry-run` uses a single replicate, so the two
self-sample peers in condition E are the same text. That is degenerate for E
but harmless — the smoke test checks plumbing, not effects.

## The 30-minute push

`autopush.sh` rsyncs `results/*.parquet|json|csv` and `logs/*.log` into the
branch **`l4-probe-results`** under
`Code/Code_Phase_3/GCP_L4_Run/results/`, and writes a `STATUS.md` with the
trial count, per-condition breakdown and live GPU utilisation — so the run is
legible from GitHub without an SSH session.

It pushes to a **branch, not `main`**, so a half-finished experiment never
lands on the default branch.

### About the token

A classic PAT has broad scope. `autopush.sh` writes it to a `0600` credential
file and hands it to git through a credential helper — never on a command line
(where `ps` would expose it) and never echoed. It is not in the uploaded
payload; you pass it at run time.

**Revoke it when the run is done:** <https://github.com/settings/tokens>.
A VM with no backups and no shielding is also a VM whose disk is readable by
anyone who can reach it — which is the trade you asked for, and it is fine for
a throwaway box, but the token outlives the VM unless you revoke it.

## Cost

| Item | Rate | Typical |
|---|---|---|
| `g2-standard-8` (1× L4) on demand | ~$0.85/h | |
| same, Spot (`SPOT=1 bash create_vm.sh`) | ~$0.30/h | |
| 200 GB pd-balanced | ~$0.03/h | |
| Model download + bootstrap | | ~20–30 min |
| Probe, 300 questions × 3 replicates | | well under 1 h |

Budget **~$2–3 on demand**, about **$1 on Spot**. The run is checkpointed per
stage and pushes every 30 minutes, so a Spot pre-emption loses at most one
interval — but it needs a manual restart.

## Files

| File | Runs on | Purpose |
|---|---|---|
| `create_vm.sh` | your machine | creates the L4, waits for SSH, checks the GPU |
| `deploy.sh` | your machine | stages and uploads the payload, bootstraps, smoke-tests |
| `vm_bootstrap.sh` | VM | GPU/disk/driver checks, vLLM + FlashAttention, gated-repo check, checkpoint download |
| `smoke_test.sh` | VM | 2 questions, output verified |
| `run_full.sh` | VM | detached full run + autopush |
| `autopush.sh` | VM | 30-minute push to `l4-probe-results` |
| `requirements_l4.txt` | VM | pinned dependencies |

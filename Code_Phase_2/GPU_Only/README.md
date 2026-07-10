# GPU_Only — Experiment E9: mechanistic logprob probe

**This is the only part of Phase 2 that needs a GPU.** Everything else (E1–E8)
is API-only in `../CPU_Only`. E9 is **optional** — no reviewer strictly requires
it; it buys one mechanistic figure that can lift the Excitement score.

## What it measures

An open-weight focal model (default `Llama-3.1-8B-Instruct`) is served locally
with vLLM. For C1/C2/C4 it records, at each debate round, the focal's
probability distribution over the candidate answers. The headline number:

```
delta_wrong_mass = P_round1(peer's wrong answer) - P_round0(peer's wrong answer)
```

Averaged over trials, `delta_wrong_mass` in **C4** (two wrong peers) minus the
same in **C2** (homogeneous, no wrong peers) is the probability mass the focal
moves toward the adversarial answer — a representation-level sycophancy signal
*beneath* the discrete flip the behavioural experiments measure.

## Why it stays symmetric with the main experiment

It **reuses `../CPU_Only`'s** question pool, personas, prompt builders, seed, and
answer-extraction — imported directly, not re-implemented. The C4 weak peers are
the same pre-generated persona texts, so peers are identical to the main runs.
Run the CPU pipeline first (it builds the pool + personas this probe loads).

## Hardware

| Focal | VRAM | Card | Time (300q × 3 reps × 3 conds) | Rent cost |
|---|---|---|---|---|
| **Llama-3.1-8B, bf16 (recommended)** | ~18–20 GB | 1× RTX 4090 24 GB | 10–15 h | ~$5–8 |
| Qwen3-32B / Gemma-3-27B, 4-bit AWQ | ~20 GB | 1× RTX 4090 24 GB | 15–25 h | ~$8–13 |

## Run

```bash
python3 -m venv ~/venv_gpu && source ~/venv_gpu/bin/activate
pip install -r requirements_gpu.txt
# HUGGINGFACE_TOKEN is read from ../.env for gated repos

python3 run_all.py --max-questions 30      # ~1 h pilot; sanity-check the metric first
python3 run_all.py                         # full probe
python3 run_all.py --model Qwen/Qwen3-32B --conditions C1_smart_solo,C4_one_smart_two_dumb
```

Outputs → `GPU_Only/data/outputs/`:
- `logprob_probe_trials.parquet` — per-trial answer distributions and mass shift.
- `logprob_probe_summary.parquet` — mean R0→R1 mass shift toward the peer-wrong answer, by condition.

Pin the exact HF checkpoint hash you served into your notes — reviewers reward a
pinned open-weights snapshot.

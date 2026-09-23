# Vast.ai big-model run — Llama-3.1-70B in bf16

The output-distribution probe in the reviewed paper reads one open-weight
model: **Llama-3.1-8B-Instruct**, the most flip-prone model in the sweep. The
Limitations section says what is missing:

> "we cannot say whether the strong models that resist wrong peers do so by
> feeling less pull or by overriding more of it. Separating those two accounts
> would need the same read-out on a strong open-weight model."

This directory runs exactly that read-out on **Llama-3.1-70B-Instruct**, which
converts a stated limitation into a result: if the strong model shows small
distributional movement, it feels less pull; if it shows movement comparable to
the 8B but does not flip, it overrides more.

Everything else (X1–X7) runs on the GCP L4 and needs no GPU. Only this run
needs rented hardware.

---

## Why 2×80 GB and why bf16

| Precision | Weights | Fits on | Distribution fidelity |
|---|---|---|---|
| **bf16** | **131.5 GB** | **2×80 GB** | exact — use this |
| fp8 | 65.8 GB | 1×80 GB | altered |
| int4 (AWQ/GPTQ) | 32.9 GB | 1×48 GB | altered |

The probe measures probability mass over answer options. Quantisation changes
the output distribution, so a quantised run would confound the very thing being
measured. A 4-bit 70B would fit one card and would not be measuring the model
the paper names.

KV cache is 320 KB per token, so a 4096-token sequence costs 1.25 GB. On
2×80 GB at `gpu_memory_utilization=0.90`, that leaves room for roughly 9–10
concurrent sequences after weights — ample, because the probe issues one batch
per stage rather than serving traffic.

Verify for any other model or context length:

```bash
python3 ../GPU_Only/vram_planner.py --model llama-3.1-70b --max-len 4096
```

## Choosing an instance on Vast.ai

Filter for **2× A100 80 GB SXM** or **2× H100 80 GB**, and require:

- `Disk space ≥ 180 GB` — the checkpoint alone is ~132 GB
- `Inet Down ≥ 500 Mbps` — otherwise the download dominates the bill
- CUDA ≥ 12.1
- Direct SSH port, not proxy, if you want `scp` to be quick

Typical cost at the time of writing: **$2–4/hour** for 2×A100-80. The run
itself is well under an hour; the checkpoint download is usually the longer
part. Budget **2–3 hours** end to end, i.e. roughly **$6–12**.

A single 80 GB card is *not* enough for bf16 — `vram_planner.py` reports "no"
for every 80 GB row, which is the honest answer.

## Running it

```bash
# 1. On the Vast.ai instance
git clone <your release URL> platos && cd platos/Code/Code_Phase_3/Vast_AI_Big_Model_Run
bash vast_bootstrap.sh                 # installs vLLM, verifies 2 GPUs, gates HF access

# 2. Upload the inputs the probe needs (from your laptop)
bash push_inputs.sh <ssh-host> <ssh-port>

# 3. Run
python3 run_big_probe.py --model meta-llama/Llama-3.1-70B-Instruct \
                         --tensor-parallel-size 2 \
                         --questions 300 --replicates 3

# 4. Pull results back, then DESTROY the instance
bash pull_results.sh <ssh-host> <ssh-port>
```

`run_big_probe.py` is checkpointed per stage, so an interrupted run resumes
without re-paying for GPU time already spent.

## What it produces

| File | Contents |
|---|---|
| `results/big_probe_trials.parquet` | one row per (question, replicate, condition) with probability mass on the fixed target and on the correct answer, both rounds |
| `results/big_probe_contrast.parquet` | the paired WR-minus-E difference-in-differences, with CI and permutation p |
| `results/big_probe_candidates.parquet` | per-question tokenisation audit: which candidates are single-token, which collide |
| `results/big_probe_meta.json` | model revision hash, GPU type, vLLM version, seeds, wall-clock |

Feed them to the main analysis with:

```bash
cd .. && python3 -m analysis.run_analysis
```

## Measurement correctness

This run uses `GPU_Only/src/corrected_probe.py`, not the Phase-2 probe. Three
differences matter:

1. **Exact candidate scoring.** The released probe matched candidates by their
   first character, so numeric answers sharing a leading digit collided. Here
   each candidate is tokenised and single-token candidates are read from the
   first-position logprobs; collisions are reported rather than merged.
2. **Off-candidate mass is kept.** The released probe renormalised over the
   candidates it found, discarding mass the model placed outside the answer
   set. That residual is recorded here, so "moved toward the peer's answer" can
   be distinguished from "moved out of the option set entirely".
3. **One target per (question, replicate), fixed across conditions.** The
   difference-in-differences is only interpretable if WR and E track the same
   target.

Because of (1), the default run restricts to the **multiple-choice** items,
where single-token letters A–J are exact. Pass `--include-numeric` to score
GSM8K items as well; they are scored by full-sequence teacher forcing and are
slower.

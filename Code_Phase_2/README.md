# Plato's Ship — Phase 2 (reviewer-driven follow-up experiments)

This directory implements every experiment the ACL rejection asked for (see
`../Code/Submission/Review_Fix.md`). It is a **superset of Phase 1**: it reuses
the same 300-question pool, the same personas, the same seed (`20260502`), the
same answer-extraction cascade, the same trial/final-answer schemas, and the
same McNemar/Bonferroni statistics — so the new numbers **merge into the same
paper** without a methods mismatch.

```
Code_Phase_2/
├── .env                      # your keys (already here)
├── API_AND_RECHARGE.md       # cheapest provider per model + how much to keep where
├── README.md                 # this file
├── CPU_Only/                 # E1–E8: API-only, no GPU. The main pipeline.
│   ├── run_all.py            # single entry point
│   ├── config/               # models.yaml (multi-provider), experiment.yaml, paths.yaml
│   └── src/                  # Phase-1 core + Phase-2 extensions
└── GPU_Only/                 # E9 only: local vLLM logprob probe (needs 1 GPU)
    └── run_all.py
```

## What maps to what

| Exp | Reviewer ask | New condition(s) | Where | GPU? |
|---|---|---|---|---|
| **E1** | solo re-answer control (sTG4) | `C1R_solo_reanswer` | CPU | no |
| **E2** | honest (non-anchored) peers (sTG4, Tf6M) | `C3H`, `C4H` | CPU | no |
| **E3** | confidence filter with confidence actually elicited (Tf6M) | `C5R`, `C5H` + corrected gate | CPU | no |
| **E4** | more focal models + capability metric (Tf6M, gL73) | 6-model sweep, C1/C2/C4 | CPU | no |
| **E5** | larger sample for GPT-4o-mini (gL73) | C1–C4 on full 300 | CPU | no |
| **E6** | rule out contamination | perturbed GSM8K, C1/C4 | CPU | no |
| **E7** | split-peer (unanimity) | `C4split` (off by default) | CPU | no |
| **E8** | heterogeneous-smart control | `C2het` (off by default) | CPU | no |
| **E9** | mechanistic probe (optional) | logprob answer-mass shift | GPU | **yes** |

## Quick start

```bash
# 1) Recharge: $50 OpenRouter + $15 DeepSeek covers everything (see API_AND_RECHARGE.md)

# 2) CPU pipeline
cd CPU_Only
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install -r requirements_cpu.txt
python3 run_all.py --list          # show the plan
python3 run_all.py --dry-run       # 1 trial/condition smoke test (all wiring)
python3 run_all.py --p0            # run the must-do experiments E1–E4
python3 run_all.py --p1            # P0 + P1 (adds E5, E6)
python3 run_all.py --experiments E1,E3   # a specific subset
python3 run_all.py --analyse-only  # recompute metrics/gates/stats from saved parquets

# 3) (optional) GPU probe, on a 24 GB card
cd ../GPU_Only
pip install -r requirements_gpu.txt
python3 run_all.py --max-questions 30   # pilot first
python3 run_all.py                      # full probe
```

Everything is **resume-capable**: re-running after a crash skips completed
`(question, condition, trial, focal)` tuples. Runs are safe to leave unattended.

## Symmetry guarantees (so it goes in the same paper)

- Same seed, same 300-question pool, same difficulty stratification.
- Same wrong-anchored persona pool for C4/sweep/split — **the main pool has no
  confidence line, exactly like Phase 1.** Only E3 uses a *separate*
  confidence-bearing pool (`persona_pool: confidence`), so the sweep's C4 peers
  are byte-for-byte comparable to the published C3/C4.
- Same `trial_log` / `final_answers` schemas (full-form column names).
- Same paired McNemar + Bonferroni machinery
  (`statistical_analyzer.mcnemar_paired_test`) for the new contrasts.
- Provenance written to `data/outputs/experiment_metadata.json`.

See `API_AND_RECHARGE.md` for provider routing and the DeepSeek alias deadline
(2026-07-24), and `../Code/Submission/Review_Fix.md` for the full rationale.

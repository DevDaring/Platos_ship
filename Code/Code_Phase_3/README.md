# Phase 3 — Protocol B: matched revision under capability asymmetry

Phase 3 answers the ARR August 2026 reviews of *Correction and Corruption
Signals* by repairing the experimental design rather than adding more models to
the old one. Everything runs from a single entry point and regenerates every
number the paper reports.

```bash
python3 -m venv ~/venv && ~/venv/bin/pip install -r requirements.txt
cp .env.example .env                  # fill in your keys
python3 tools/fetch_artefacts.py      # question pool + persona pools
python3 run_all.py --list             # the plan and its call counts
python3 run_all.py --dry-run          # 1 unit per cell; exercises every path
python3 run_all.py --all              # prepare, run, analyse
python3 run_all.py --analyse          # offline: tables and figures only
python3 -m pytest tests/ -q           # offline tests, no API key needed
python3 tools/routing_report.py       # which account pays for which call
python3 tools/routing_report.py --probe   # confirm every route's served model
```

**Regenerate every submitted table, Figure 1 and every number in the paper,
offline, from the released logs** (no API key, a few minutes on a laptop):

```bash
python3 run_all.py --analyse && python3 -m analysis.phase4 && \
python3 tools/paper_tables.py --out <paper dir>
```

`run_all.py --analyse` rebuilds `results/outputs/paper_numbers.json` from the
Protocol-B logs; `analysis.phase4` rebuilds `results/phase4/phase4_numbers.json`
from the Phase 4 logs; `tools/paper_tables.py` writes every LaTeX table,
Figure 1 and `paper_facts.json`. Phase 4 (pre-registered in
`PREREG_PHASE4.md`) is re-run with `tools/run_phase4_all.sh`.

---

## Why Phase 3 exists

Two ARR cycles scored the paper at mean 2.5 despite every requested control
being added. The reviews point at two things, and both are design problems:

**1. The controls were not matched.** The focal model's debate Round-1 prompt
omitted its own Round-0 answer, while the re-answer control C1R included it,
and the API wrapper carries no conversation state. So `C4 − C1R` changed three
things at once — peers present, own answer removed, instruction reworded — and
a "harmful flip" in the debate conditions was partly ordinary temperature-0.7
resampling of a fresh answer. This is the substance behind oEjr's W4 ("the
controls do not isolate the mechanism") and YVDD's W1.

**2. The headline was the fragile result.** "Wrong peers help strong models" is
concentrated on GSM8K, vanishes under re-answering for GPT-4o-mini, and does
not survive the question-level test after correction. The robust result — the
cost of wrong peers rises as focal ability falls — was contribution 2.

Protocol B fixes the first. The reframing and the new experiments address the
second.

## Protocol B

One Round-0 answer is generated and **cached** per (focal model, question,
replicate). Every revision condition clones that exact record and issues one
call with an identical template:

```
Question: {question}{options}

Your previous response:
{cached Round-0 text}

{CONTEXT_BLOCK}          <- the only thing that differs

Review your previous response[, and the other responses above]. Reconsider
the problem carefully and state your final answer with reasoning. You may
keep or revise your answer. End with:
Final answer: <your answer>
Confidence: <integer from 0 to 100>
```

Same temperature, same 600-token cap, same instruction everywhere. The cache
also makes the study affordable: 7,200 initial calls are reused by every
condition rather than re-paid per condition.

| Condition | CONTEXT_BLOCK | Legacy | Isolates |
|---|---|---|---|
| `R` | (empty) | C1R | second attempt alone |
| `G` | content-free challenge | — | being challenged, without content |
| `W` | bare "Final answer: X" lines | — | the alternative answer, no rationale |
| `WR` | two wrong-anchored rationales | C4 | the main treatment |
| `WRh` | the same rationales, hedged wording | — | expressed confidence |
| `SF` | the same content, unattributed | — | source framing |
| `H` | two natural weak-peer messages | C4H | natural low-capability peers |
| `E` | two further samples of the focal itself | C2 | homogeneous / self-consistency |
| `CR` | one wrong + one correct rationale | C4split | correctness of the rationale |
| `WR1` / `WR4` | one / four wrong rationales | C3 / — | dose |
| `WRagree` / `WRdiff` | same / different wrong targets | — | agreement, holding all peers wrong |
| `WRfilt` / `Hfilt` | confidence-filtered peers | C5R / C5H | the filter |

## Experiments

| ID | What it does | Answers | Priority |
|---|---|---|---|
| **X1** | the common matrix (`R, E, WR, H, CR`) on **all eight** focal models, 3 replicates, identical items | YVDD W1, oEjr W1/W2 | P0 |
| **X2** | message decomposition (`G, W, WRh, SF`) on one strong / mid / weak model | oEjr W4 | P0 |
| **X3** | scaling: peer count {1, 2, 4}, agreement, rounds {1, 2, 3} | oEjr's scaling request | P1 |
| **X4** | contamination probe on **GSM-Symbolic** with template-computed answers | replaces the invalid probe | P1 |
| **X5** | plurality-vote baseline from the cached replicates (free) | Choi 2025, Zhang 2025 | P0 |
| **X6** | verification safeguard: adopt a change only if an independent model backs it | oEjr W3 | P2 |
| **X7** | the confidence filter under the corrected retention gap | oEjr W3 | P1 |
| **X8** | output-distribution probe with exact candidate scoring (GPU) | mechanism evidence | P2 |

`python3 run_all.py --list` prints the call count for each.

## Repairs carried out here

Each is pinned by a test (77 offline tests in total, no API key needed).

| # | Defect | Repair |
|---|---|---|
| 1 | `trial_runner.py` seeded with `hash(question_id)`, which is salted per process, so peer ordering and persona choice were not reproducible | `src/seeding.py` uses `zlib.crc32`; a test runs two interpreters under different `PYTHONHASHSEED` values and requires identical seeds |
| 2 | `phase2_analyzer._flip_rate()` returned `P(A0=1 ∧ Y=0)` while the paper defined `P(Y=0 \| A0=1)` — 8.67% vs 13.43% on real data | `analysis/metrics.py` returns every rate with its denominator attached; both quantities are reported |
| 3 | `corrected_gate.py` computed `P(loud\|wrong) − P(loud\|correct)` and passed when positive — the sign of the *harmful* case for a retain-high filter — and substituted 0 for an absent class, producing a spurious "passed" at 0.99 | `analysis/gate.py` computes `Δ_ret = P(retained\|correct) − P(retained\|wrong)`; a single-class substrate returns `undefined_single_class` |
| 4 | revision prompts were not matched (above) | `src/contexts.py`: one template, own answer always shown |
| 5 | perturbed-GSM8K gold answers invalid — every integer scaled by 4, including percentages and fractions; the pool's own probe agreed on 11 of 97 items | `tools/quarantine_perturbed.py` isolates them with a README; X4 replaces the probe |
| 6 | 75 "GPT-4o-mini" focal responses served by `gpt-4.1-mini` via a fallback chain | a focal chain is allowed but built with `src/pinned_fallback.py`, which checks every link's served model against `expected_served_prefix` and skips a link answering with a different model; `src/snapshots.py` records and enforces the pin per call |
| 7 | every anchored peer message was written by Llama-3.1-8B but logged under whichever slot it filled | `message_generator_model`, `nominal_peer_slot_model` and `served_model` are separate columns |
| 8 | `capability_sweep_analysis.json` reports ρ = −0.22 and solo accuracy 0.2165 because perturbed trials leaked into the sweep grouping | `analysis/registry.py` keys every row by dataset scope; the stale file is quarantined |
| 9 | GPU probe scored candidates by first character, so numeric answers sharing a leading digit collided, and renormalised away off-candidate mass | `GPU_Only/src/corrected_probe.py`: exact tokenised scoring, collisions reported, off-candidate mass kept |
| 10 | a resumed multi-round run skipped a completed round without restoring its output, so the next round received the Round-0 text instead of the previous round's | `run_condition` restores the prior round from the log, and re-runs the unit rather than guessing when it cannot; pinned by `tests/test_multiround_resume.py` |
| 11 | `run_big_probe.py` used `zip(strict=)` (Python 3.10+) while the bootstrap accepted 3.9, so a 3.9 box would have crashed *after* the 70B was loaded and billing | removed; generation is also batched rather than issued one prompt at a time |

## Layout

```
Code_Phase_3/
├── run_all.py                  single entry point
├── ANALYSIS_PLAN.md            frozen before the confirmatory run
├── config/
│   ├── experiment.yaml         conditions, experiments, contrast families
│   ├── models.yaml             pinned snapshots, no focal fallback
│   └── paths.yaml
├── src/
│   ├── seeding.py              process-stable seeds            [repair 1]
│   ├── snapshots.py            served-model assertion          [repair 6]
│   ├── contexts.py             Protocol B prompt construction  [repair 4]
│   ├── r0_cache.py             Stage 0: the cached initial answer
│   ├── peer_pools.py           peer assembly + provenance      [repair 7]
│   ├── revision_runner.py      the execution engine
│   ├── honest_bank.py          natural weak-peer messages
│   ├── hedged_personas.py      the WRh pool + validation
│   ├── gsm_symbolic.py         X4 pool                         [repair 5]
│   ├── verifier.py             X6 safeguard
│   ├── extraction.py           parsing and grading
│   ├── store.py                checkpoints + incremental parquet
│   └── agent_wrappers/         copied verbatim from Phase 2
├── analysis/
│   ├── registry.py             one table, scoped               [repair 8]
│   ├── metrics.py              rates with denominators         [repair 2]
│   ├── stats.py                bootstrap, permutation, Holm, GEE
│   ├── contrasts.py            the pre-registered families
│   ├── gate.py                 corrected retention gap         [repair 3]
│   ├── voting.py               X5
│   ├── safeguard.py            X6 scoring
│   ├── make_tables.py          LaTeX the paper \input{}s
│   ├── make_figures.py         figures, names matched to the text
│   └── run_analysis.py         offline: logs in, numbers out
├── tools/
│   ├── fetch_artefacts.py      inputs + SHA-256 manifest
│   ├── routing_report.py       which account pays for which call
│   ├── quarantine_perturbed.py [repair 5]
│   └── audit_release.py        anonymity, secrets, manifest
├── GPU_Only/
│   ├── src/corrected_probe.py  exact candidate scoring       [repair 9]
│   └── vram_planner.py         what GPU the probe actually needs
├── Vast_AI_Big_Model_Run/      Llama-3.1-70B bf16 on rented 2x80 GB
│   ├── vast_bootstrap.sh       fails fast BEFORE the 132 GB download bills
│   ├── run_big_probe.py        checkpointed per stage
│   └── push_inputs.sh / pull_results.sh
└── tests/                      77 offline tests, no API key
    ├── test_repairs.py         the four original defects
    ├── test_pinned_fallback.py a fallback changes the route, not the model
    ├── test_multiround_resume.py  a resumed round sees the right context
    ├── test_pipeline_integration.py  the whole pipeline, stubbed
    └── test_big_probe_offline.py  the 70B probe, before renting a GPU
```

## Reproducibility

- **Resumable.** Every unit is checkpointed by `(protocol, focal, condition,
  question, replicate, round)`. A killed run restarts where it stopped and
  never pays twice for the same call.
- **Deterministic.** One master seed (20260502) plus `crc32` derivation; no
  process-dependent randomness anywhere.
- **Offline-verifiable.** `python3 run_all.py --analyse` regenerates every
  table, figure and number from the parquet logs with no API key.
- **Auditable.** `tools/audit_release.py` writes `MANIFEST.sha256.json` and
  checks for author names, absolute paths and key-shaped literals before the
  artefact ships.

## Cost and routing

`python3 tools/routing_report.py` prints the current estimate and which account
pays for what; `--probe` confirms each route's served model for a fraction of a
cent. At the routing in `config/models.yaml` the whole programme is roughly
**$6**, dominated by OpenRouter and nano-gpt. Phase-2 throughput was about 950
focal trials per hour end to end.

A provider is used only when it serves the SAME model under the same
identifier. Llama-3.1-70B and both Gemma models stay on OpenRouter because
nano-gpt carries only a fine-tune (`hermes-3-llama-3.1-70b`) or a third-party
re-upload (`unsloth/gemma-3-*`).

The optional 70B probe (`Vast_AI_Big_Model_Run/`) needs 2x80 GB for bf16 and
costs roughly $6-12 for 2-3 hours on Vast.ai. Everything else is API-bound and
runs on a GCP L4 or no GPU at all.

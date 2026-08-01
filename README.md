# Plato's Ship — Capability-Asymmetric Multi-Agent LLM Debate

A fully reproducible research pipeline that measures what happens to a strong
language model — the **focal model** — when it debates weaker peers that argue
confidently for wrong answers, and whether a confidence-weighted peer filter
can remove those bad peers.

Named after Plato's ship-of-state metaphor: do unskilled crew drag down a
skilled navigator? This repository contains the complete code, prompts,
question pool, and per-trial logs behind the accompanying paper, so that every
number and figure can be regenerated from scratch.

> **Anonymity note.** This repository is prepared for double-blind review. It
> contains no author names, institutions, or personal identifiers. The `.env`
> holding API keys is never committed; copy `.env.example` and fill your own.

## Headline findings

Across ~37,000 trials on MMLU-Pro + GSM8K, with **eight focal models** spanning
a wide capability range:

- **Correction, not corruption, for strong models.** No focal model loses
  accuracy under two confidently wrong peers, and the stronger ones gain (up to
  +7.9 points). The wrong answers prompt the focal model to reconsider and
  recompute rather than to copy.
- **The gain is causal, and specific.** A *re-answering* control (revise with no
  peers) reproduces none of the gain; two wrong peers beat it by a significant
  margin. An *honest-peer* control (the same weak models answering naturally)
  leaves accuracy at solo level. The effect is specific to confidently wrong
  peers.
- **The cost falls on the weak.** The rate at which a focal model drops a
  correct answer under wrong peers rises steeply as its own ability falls
  (Spearman ρ ≈ −0.95 across the eight models).
- **Confidence cannot filter peers.** Even with confidence properly elicited,
  weak peers sound equally confident when right and when wrong (AUROC ≈ 0.58),
  so no threshold separates them. A corrected pre-flight test predicts this
  before any filtering is run.
- **Not memorisation.** The math gain survives when GSM8K numbers are perturbed
  and answers recomputed, supporting the recomputation mechanism.
- **The pull is real, below the answer.** Reading an open-weight focal model's
  output distribution, two confidently wrong peers move probability mass *onto*
  their answer (+0.064) where homogeneous peers move it *away* (−0.018);
  paired, the shift is +0.082 (Wilcoxon p ≈ 1e-42). Flip rates cannot see this,
  because it happens on trials where the stated answer never changes.
- **Adversarial framing is conservative, not alarmist.** Weak peers that are
  wrong *naturally* flip the focal model about twice as often as peers
  *instructed* to be wrong (29.7% vs 14.4%, paired by question). The reported
  harmful-flip rates are therefore a floor on real mixed-ability ensembles.

## Repository layout

```
.
├── src/                    # Phase-1 pipeline (main experiment C1–C4 + gate + C5)
│   ├── agent_wrappers/     # API agent abstraction, round-robin keys, judge cascade
│   ├── dataset_builder.py  # MMLU-Pro + GSM8K sampling, difficulty stratification
│   ├── persona_generator.py / persona_validator.py   # wrong-anchored weak peers
│   ├── debate_protocol.py / confidence_weighted_protocol.py
│   ├── trial_runner.py     # resumable, checkpointed trial execution
│   ├── calibration_gate.py # pre-flight test for the confidence filter
│   └── metrics_calculator.py / statistical_analyzer.py
├── config/                 # experiment.yaml, models.yaml, paths.yaml (no hardcoding)
├── results/                # released per-trial logs + question/persona pools
├── Code_Phase_2/           # follow-up experiments (self-contained)
│   ├── CPU_Only/           # API-only: causal controls, honest peers, 8-model
│   │                       #   sweep, corrected filter, contamination probe
│   ├── GPU_Only/           # optional local logprob probe (vLLM)
│   └── results/            # Phase-2 per-trial logs + analysis outputs
├── requirements.txt
└── .env.example            # copy to .env and fill API keys
```

`src/` + `config/` run the main experiment; `Code_Phase_2/CPU_Only/run_all.py`
is a single entry point for the follow-up experiments. Both share the same
question pool, seed, prompts, and schemas, so results are directly comparable.

## Experimental conditions

| Code | Focal | Weak peers | Purpose |
|------|-------|-----------|---------|
| C1   | 1 | 0                | solo baseline, no debate |
| C1R  | 1 | 0 (own answer)   | control: is a gain just a second attempt? |
| C2   | 3 | 0                | homogeneous debate (equal peers) |
| C2het | 3 | 0               | control: three *architecturally distinct* strong models |
| C3   | 2 | 1 wrong-anchored | one confidently wrong peer |
| C4   | 1 | 2 wrong-anchored | two confidently wrong peers |
| C4split | 1 | 1 wrong + 1 correct | control: unanimity vs. a wrong answer being present |
| C4H  | 1 | 2 honest         | control: wrong-anchored vs. natural peers |
| C5   | 1 | 2 wrong + filter | confidence-weighted peer filtering |

Each trial has two rounds: every agent answers independently in Round 0, then
sees the others' Round 0 messages and gives a final answer in Round 1 (the solo
condition has no Round 1). Weak peers are two small open models (Llama-3.1-8B,
Gemma-3-4B); in the wrong-anchored conditions a persona prompt makes them argue
one assigned wrong answer, and in the honest control they answer naturally.

## Reproducing the results

```bash
# 1. Environment
python3 -m venv ~/venv && ~/venv/bin/pip install -r requirements.txt

# 2. Secrets — copy the template and fill your own API keys
cp .env.example .env      # DeepSeek + OpenRouter keys are the minimum

# 3. Main experiment (C1–C4, calibration gate, then C5 if the gate passes)
python3 -m src.pipeline_orchestrator --stage all

# 4. Recompute metrics/statistics from saved logs without re-running models
python3 -m src.metrics_calculator
python3 -m src.statistical_analyzer

# 5. Follow-up experiments (causal controls, 8-model sweep, corrected filter, …)
cd Code_Phase_2/CPU_Only
pip install -r requirements_cpu.txt
python3 run_all.py --list      # show the experiment plan
python3 run_all.py --p1        # run the recommended set
```

All model access is via commercial APIs; **no GPU is required** for the main
results (the optional mechanistic probe in `Code_Phase_2/GPU_Only` is the only
GPU component). The runner is checkpointed and idempotent: re-running any stage
after a crash skips completed `(question, condition, trial, focal)` tuples.

## Released data

`results/` and `Code_Phase_2/results/` contain the per-trial logs and analysis
outputs behind the paper:

| File | Grain | Contents |
|------|-------|----------|
| `trial_log.parquet` | one agent-round | every agent's raw + parsed response, per round |
| `final_answers.parquet` | one focal trial | Round-0/Round-1 answers, correctness, flip flags |
| `metrics_summary.parquet` | one (condition, focal) | accuracy and flip rates |
| `statistical_tests.parquet` | one contrast | paired McNemar, corrected p-values |
| `capability_sweep_*` | one focal model | solo accuracy vs. flip outcome (8 models) |
| `corrected_calibration_gate_report.parquet` | one substrate | discriminative gap + AUROC |
| `gpu_probe/logprob_probe_trials.parquet` | one probe trial | per-round probability mass on the correct and peer-asserted wrong answers |
| `gpu_probe/logprob_probe_contrast.parquet` | one contrast | paired C4−C2 mass shift with bootstrap CI |

Analysis scripts live in `Submission/Analyse/`: `verify_paper_numbers.py`
recomputes every headline number from the logs, `make_tables.py` emits the
paper's data tables as LaTeX, and `make_figures.py` regenerates the figures.
The paper `\input`s the generated tables, so the manuscript cannot drift from
the released data.

Column names are full-form and self-describing (e.g.
`extracted_self_reported_confidence_integer`, `condition_identifier`). A single
random seed propagates to every randomised step, so runs are deterministic.

## Cost

The entire study — all conditions and eight focal models — runs for well under
US$100 of API inference, on a single CPU machine. No model training is
performed.

## Licence

Code released under the MIT Licence; benchmark data (MMLU-Pro, GSM8K) retains
its original licence. See the paper for full dataset and model citations.

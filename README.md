# Plato's Ship - Capability-Asymmetric LLM Debate Study

A fully reproducible research pipeline that measures how a strong LLM
focal agent's reasoning changes when it is forced to debate with
capability-weaker ("dumb") peer agents, and whether a confidence-weighted
peer-filtering rule is available as a mitigation.

Named after Plato's ship-of-state metaphor: do unskilled crew members
drag down a skilled navigator? The empirical answer reported in the
accompanying paper is **focal-model-dependent** — DeepSeek-v4-flash
gains +7.9 pp Round-1 accuracy under two weak peers while GPT-4o-mini
loses 4.0 pp under one. The gain for DeepSeek is also not uniform: 33
of 300 questions regress at the question level even as the aggregate
rises. Confidence-weighted filtering is unavailable as a mitigation
because weak agents emit no usable confidence signal.

The paper accompanying this repository (`Submission/CSR_Paper.tex` +
`Submission/CSR_Supplement.tex`) targets ACL ARR; the prior CSR
formatting is retained.

## Headline findings

- **Trial total:** 7,300 across 5 conditions and 2 focal models.
  Dataset: 200 MMLU-Pro + 100 GSM8K = 300 questions.
- **DeepSeek-v4-flash, C4 (1 smart + 2 weak peers):** Round-1 accuracy
  rises from 76.2% solo to 84.1% (+7.9 pp). The +7.9 pp average hides
  a per-question regression: paired McNemar shows 33 questions
  regress vs 12 improve (Holm-corrected p = 0.017).
- **GPT-4o-mini, C3 (1 weak peer):** accuracy drops 4.0 pp. C→I flip
  rate reaches 17.2% in C4, ~3× DeepSeek's.
- **Self-consistency-of-3 baseline (free from C2's R0 samples):** debate
  adds +1.9 pp over sample aggregation; the three smart agents in C2
  are highly correlated (Cohen's κ = 0.72), so SC-3 is roughly 1.5
  effective independent draws.
- **C5 confidence-weighted filter strips 100% of peers** in every
  trial. Two failure modes: (a) dumb agents emit no confidence
  integer under the persona-anchored Round-0 prompt (0 of 5,850 R0
  parses); (b) when they do emit one in Round-1, the conditional rates
  P(conf≥60|wrong) = 0.995 and P(conf≥60|correct) = 0.998 differ by
  -0.003, so confidence carries no discriminative signal. The
  originally specified calibration gate metric is shown to be
  misspecified.
- **GSM8K drives the headline gain.** DeepSeek C4 on GSM8K: +16.0 pp.
  On MMLU-Pro: +3.9 pp.

The full attack-and-fix review for ACL ARR readiness is in
`Submission/final_review.md`. Every numerical claim in the paper
traces to one of the parquet artefacts in `results/data/outputs/` via
`Submission/Analyse/free_analyses.py` or
`Submission/Analyse/inter_agent_dependence.py`.

---

## Table of Contents

1. [Research Design](#1-research-design)
2. [Architecture Overview](#2-architecture-overview)
3. [Repository Layout](#3-repository-layout)
4. [Prerequisites](#4-prerequisites)
5. [First-Time Setup on a Fresh VM](#5-first-time-setup-on-a-fresh-vm)
6. [Secrets - .env File](#6-secrets---env-file)
7. [Configuration Files](#7-configuration-files)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [SMS Notifications](#9-sms-notifications)
10. [Output Files](#10-output-files)
11. [Reproducing Results](#11-reproducing-results)
12. [Cost Estimate](#12-cost-estimate)

---

## 1. Research Design

### Experimental Conditions

| ID | Label | Smart | Dumb | Aggregation | Purpose |
|----|-------|-------|------|-------------|---------|
| C1 | `C1_smart_solo` | 1 | 0 | none | Baseline - solo, no debate |
| C2 | `C2_three_smart` | 3 | 0 | standard debate | Homogeneous peer control |
| C3 | `C3_two_smart_one_dumb` | 2 | 1 | standard debate | Plato condition |
| C4 | `C4_one_smart_two_dumb` | 1 | 2 | standard debate | Capability-collapse |
| C5 | `C5_one_smart_two_dumb_confidence_weighted` | 1 | 2 | confidence-weighted | Mitigation (gate-gated) |

### Dataset

- **MMLU-Pro** - 200 questions across 10 subjects (20/subject), stratified by difficulty
- **GSM8K** - 100 grade-school math word problems, stratified by difficulty
- Difficulty probe: Llama 3.1 8B zero-shot accuracy splits each subject 50/50

### Trial Protocol

```
Round 0 (independent):
  Each agent answers alone.
  Required format:
    Final answer: <X>
    Confidence: <0-100>

Round 1 (debate):
  Each agent sees peers' Round-0 responses (random order), then re-answers.

C5 only: dumb peers with confidence < 60 are filtered from the focal
         agent's context before Round 1.
```

### Calibration Gate (Stage 2)

```
Gate as implemented:
  P(confidence >= 60 | wrong, dumb agent, condition in {C3,C4}) >= 0.40
    -> PASS: run C5
    -> FAIL: skip C5

In this experiment the gate PASSED (metric = 0.995) but C5
nonetheless stripped 100% of peer messages. The gate metric as
specified is misspecified: it should test the discriminative gap
  P(loud | wrong) - P(loud | correct)
against a small positive threshold, not the absolute high-confidence
rate. The observed gap is -0.003. A correctly-specified gate would
have FAILED and Stage 3 would not have been run. See Section 3.4 of
the paper and Submission/Analyse/free_analyses_output.json
("calibration_by_dumb_model").
```

### Key Metrics

- Round-0 and Round-1 accuracy per condition, per focal agent
- Flip rates C→I (revision pressure) and I→C (correction signal)
- Self-consistency-of-3 baseline derived from C2's Round-0 samples
  (free, no extra API calls)
- Inter-agent Cohen's κ on Round-0 correctness (how independent are
  three "independent" smart-agent samples in C2?)
- AUROC of self-reported confidence against correctness (peer-quality
  signal usability)
- Cohen's h effect size and post-hoc power for every paired
  proportion comparison
- Trial-level McNemar with Holm-Bonferroni correction over six
  pairwise comparisons per focal agent
- GEE logistic regression of R1 correctness on weak-peer count,
  clustered by question identifier (accounts for the 5-replications-
  per-question nesting)

---

## 2. Architecture Overview

```
Smart agents (API)           Dumb agents (API)          Judge cascade (API)
-------------------          -----------------          -------------------
DeepSeek  <-- focal          Llama 3.1 8B  }            Tier 1: Gemini 2.5 Flash
GPT-4o-mini <-- cross-val    Gemma 3 4B    } OpenRouter  (4 keys, round-robin)
                                                        Tier 2: Mistral Small
                                                        Tier 3: DeepSeek (fallback)

- Judge fires ONLY on regex parse failure
- All calls: exponential backoff (2-4-8-16-32s + jitter, max 5 retries)
- Key rotation: OpenRouter 2 keys, DeepSeek 2 keys, Gemini 4 keys
- NO GPU. NO local model loading. Pure API pipeline.
```

---

## 3. Repository Layout

```
Platos_ship/
+-- .env                          # Secrets - NEVER commit
+-- .env.example                  # Template with blank values
+-- .gitignore
+-- README.md
+-- CLAUDE.md                     # Orientation for future Claude Code sessions
+-- requirements.txt
+-- TextBelt.py                   # SMS notifier
+-- coding_agent.md               # Full implementation spec (v2; historical)
+-- config/
|   +-- experiment.yaml           # Conditions, sizes, seeds
|   +-- models.yaml               # Model registry + API params
|   +-- paths.yaml                # All filesystem paths
+-- src/
|   +-- pipeline_orchestrator.py  # MAIN ENTRY POINT
|   +-- trial_runner.py           # Core execution loop
|   +-- debate_protocol.py        # C1-C4 Round-0 + Round-1
|   +-- confidence_weighted_protocol.py  # C5 Round-1
|   +-- dataset_builder.py        # Downloads + stratifies datasets
|   +-- persona_generator.py      # 1500 dumb-persona texts via Llama API
|   +-- persona_validator.py      # Validates + regenerates personas
|   +-- calibration_gate.py       # Stage-2 gate logic
|   +-- metrics_calculator.py     # Accuracy, flip rates, bootstrap CIs
|   +-- statistical_analyzer.py   # McNemar, Bonferroni, dose-response
|   +-- environment_check.py      # Pre-flight checks
|   +-- agent_wrappers/
|       +-- base_agent.py         # Abstract base + RoundRobinKeyManager
|       +-- deepseek_agent.py
|       +-- openrouter_agent.py   # GPT-4o-mini + Llama + Gemma
|       +-- judge_agent.py        # 3-tier cascade
|       +-- local_huggingface_agent.py  # Reference only; unused on CPU
+-- dry_run/                      # Per-module smoke tests
|   +-- run_all_dry_tests.py
|   +-- test_api_connectivity.py  # Real ping to every provider
|   +-- test_dataset_pipeline.py
|   +-- test_persona_pipeline.py
|   +-- test_debate_protocol.py
|   +-- test_answer_extraction.py
|   +-- test_judge_cascade.py
|   +-- test_round_robin.py
|   +-- test_metrics_and_stats.py
|   +-- test_local_models.py
+-- scripts/
|   +-- run_environment_check.sh
|   +-- run_dry_run.sh            # Mandatory before full run
|   +-- run_main_experiment.sh    # Stage 1
|   +-- run_calibration_gate.sh   # Stage 2
|   +-- run_mitigation_experiment.sh  # Stage 3
+-- tests/
|   +-- dry_run_assertions.py
+-- Submission/                   # ACL ARR submission bundle (LaTeX + analyses)
|   +-- CSR_Paper.tex             # Main manuscript
|   +-- CSR_Supplement.tex        # Supplement S1-S8
|   +-- cover_letter.tex          # CSR cover letter (legacy)
|   +-- highlights.tex            # 5 Elsevier highlights
|   +-- references.bib            # 22 entries (some flagged in
|   |                             #   bibliography_verification_log.md)
|   +-- final_review.md           # Adversarial ACL ARR review (30 attacks)
|   +-- generate_figures.py       # Builds the three PNGs in images/
|   +-- images/                   # figure1-3 PNG renderings
|   +-- Analyse/
|       +-- analysis.md           # Narrative results doc
|       +-- generate_analysis.py  # Loads parquets, prints every number
|       +-- free_analyses.py      # SC-of-3, GEE, effect sizes,
|       |                         #   difficulty/subject/dataset splits
|       +-- free_analyses_output.json
|       +-- inter_agent_dependence.py  # Cohen's kappa + AUROC
|       +-- inter_agent_dependence_output.json
+-- results/                      # Experiment artefacts (post-run)
|   +-- data/
|   |   +-- processed/            # question_pool, dumb_personas
|   |   +-- outputs/              # trial_log, final_answers, metrics,
|   |                             #   statistical_tests, calibration_gate,
|   |                             #   mitigation_summary, experiment_metadata
|   +-- logs/                     # pipeline.log, api_failures.log
+-- data/                         # Runtime only - not in git
+-- logs/                         # Runtime only - not in git
```

---

## 4. Prerequisites

| Requirement | Value |
|-------------|-------|
| OS | Ubuntu 22.04 or 24.04 LTS (x86-64) |
| VM | >= 8 vCPUs, >= 32 GB RAM (NO GPU needed) |
| Python | 3.10, 3.11, or 3.12 |
| Disk | >= 20 GB free |
| Network | Outbound HTTPS to deepseek, openrouter, google, mistral, huggingface |

### API Keys Required

| Provider | Purpose | Keys |
|----------|---------|------|
| DeepSeek | Smart focal agent + tertiary judge | 2 (round-robin) |
| OpenRouter | GPT-4o-mini + Llama 3.1 8B + Gemma 3 4B | 2 (round-robin) |
| Gemini | Primary judge tier | 4 (round-robin) |
| Mistral | Secondary judge tier | 1 |
| HuggingFace | Dataset download | 1 |
| Github_Classic_Token | Clone this private repo | 1 |
| TextBelt | SMS notifications (optional) | 1 |

---

## 5. First-Time Setup on a Fresh VM

### 5.1 Clone the Repository

This is a **private repo**. Use a GitHub Classic Personal Access Token
with `repo` scope. It is stored as `Github_Classic_Token` in `.env`.

```bash
# Read your token (from your local .env or paste directly)
GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"

# Clone
git clone https://${GITHUB_TOKEN}@github.com/DevDaring/Platos_ship.git
cd Platos_ship

# Embed token in remote URL so git pull works without prompts
git remote set-url origin https://${GITHUB_TOKEN}@github.com/DevDaring/Platos_ship.git
```

### 5.2 Upload Your .env File

The `.env` is never committed. Upload it manually after each VM provision.

**Windows PowerShell (from local machine):**
```powershell
scp -i C:\Users\YourName\.ssh\id_rsa_gcp `
    D:\path\to\Platos_ship\Code\.env `
    debz@<VM_IP>:/home/debz/Platos_ship/.env
```

**Linux / macOS:**
```bash
scp -i ~/.ssh/id_rsa_gcp .env debz@<VM_IP>:/home/debz/Platos_ship/.env
```

### 5.3 Install Python Dependencies (CPU VM)

Ubuntu 24.04 enforces PEP 668 and blocks system-wide pip installs.
**Use a virtual environment:**

```bash
# Create venv (one-time)
python3 -m venv ~/venv

# Install all dependencies
~/venv/bin/pip install -r /home/debz/Platos_ship/requirements.txt
```

The project scripts are pre-configured to call `~/venv/bin/python3` directly,
so you do not need to `source activate` the venv to run them.

> torch, bitsandbytes, and flash_attn are NOT installed.
> All inference is via API only.

### 5.4 Verify Imports

```bash
~/venv/bin/python3 -c "import openai, datasets, pandas, scipy, statsmodels; print('imports ok')"
```

### 5.5 Run Pre-Flight Checks

```bash
bash scripts/run_environment_check.sh
```

| Check | What it verifies |
|-------|-----------------|
| Python version | >= 3.10 |
| DeepSeek | Model listing; both model names present |
| OpenRouter | Model listing; GPT-4o-mini + Llama + Gemma all available |
| Gemini | Test call on each of the 4 keys |
| Mistral | Test call |
| HuggingFace | whoami() succeeds |
| Paths | All required directories created |
| Disk | >= 10 GB free |

**Fix every failure before continuing.**

### 5.6 Run the Dry Run (Mandatory)

```bash
bash scripts/run_dry_run.sh
```

What the dry run does:
- 2 questions (1 MMLU-Pro + 1 GSM8K)
- All 5 conditions (C5 force-enabled)
- 1 trial per question
- Every model invoked at least once
- Judge fallback path triggered deliberately
- C5 empty-peer-messages branch tested

Expected time: **under 5 minutes**.

Acceptance criteria (all must pass before full run):
- [ ] All API keys return valid responses
- [ ] trial_log.parquet has >= 1 row per model
- [ ] All output column names match schema exactly
- [ ] Metrics + stats run without error
- [ ] mitigation_summary.parquet has >= 1 C5 row

---

## 6. Secrets - .env File

```dotenv
# DeepSeek - 2 keys round-robin
DEEPSEEK_API_KEY_1=sk-...
DEEPSEEK_API_KEY_2=sk-...
DEEPSEEK_API_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_PRIMARY_MODEL_NAME=deepseek-chat
DEEPSEEK_JUDGE_MODEL_NAME=deepseek-chat

# OpenRouter - 2 keys round-robin (GPT-4o-mini + Llama + Gemma)
OPENROUTER_API_KEY_1=sk-or-v1-...
OPENROUTER_API_KEY_2=sk-or-v1-...
OPENROUTER_API_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_PRIMARY_MODEL_NAME=openai/gpt-4o-mini
OPENROUTER_LLAMA_MODEL_NAME=meta-llama/llama-3.1-8b-instruct
OPENROUTER_GEMMA_MODEL_NAME=google/gemma-3-4b-it

# Gemini - 4 keys round-robin (primary judge)
GEMINI_API_KEY_1=AIzaSy...
GEMINI_API_KEY_2=AIzaSy...
GEMINI_API_KEY_3=AIzaSy...
GEMINI_API_KEY_4=AIzaSy...
GEMINI_MODEL_NAME=gemini-2.5-flash-lite

# Mistral - secondary judge
MISTRAL_API_KEY=...
MISTRAL_MODEL_NAME=mistral-small-latest

# HuggingFace
HUGGINGFACE_TOKEN=hf_...

# GitHub Classic Token (repo scope) - clone private repo
Github_Classic_Token=ghp_...

# Reproducibility
RANDOM_SEED=20260502

# SMS (optional)
PHONE_NO=+91XXXXXXXXXX
TextBelt_API_KEY=...
```

---

## 7. Configuration Files

### config/experiment.yaml

| Parameter | Default | Meaning |
|-----------|---------|---------|
| random_seed | 20260502 | Master seed |
| total_questions_in_pool | 300 | 200 MMLU-Pro + 100 GSM8K |
| trials_per_question_main_conditions | 5 | C1-C4 |
| trials_per_question_mitigation_condition | 3 | C5 |
| calibration_gate.precondition_metric_threshold | 0.40 | Gate threshold |
| calibration_gate.high_confidence_threshold | 60 | Confidence cutoff |

### config/models.yaml

Model registry: provider, env-var names, temperature, timeouts, retry backoff.
No model names hardcoded in source.

### config/paths.yaml

Every filesystem path. Resolved and asserted at startup.

---

## 8. Running the Pipeline

```bash
cd /home/debz/Platos_ship

# Pre-flight
bash scripts/run_environment_check.sh

# Mandatory dry run
bash scripts/run_dry_run.sh

# Stage 1: main experiment (C1-C4, 7000 trials)
bash scripts/run_main_experiment.sh

# Stage 2: calibration gate (< 5 min)
bash scripts/run_calibration_gate.sh

# Stage 3: C5 mitigation (only runs if gate passed)
bash scripts/run_mitigation_experiment.sh
```

Or all stages at once:
```bash
python3 -m src.pipeline_orchestrator --stage all
```

### Stage Timing (CPU VM)

| Stage | Trials | Est. time |
|-------|--------|-----------|
| Dry run | ~10 | < 5 min |
| Stage 1 (C1-C4, DeepSeek + GPT-4o-mini) | 7 000 | 18-28 h |
| Stage 2 (gate) | 0 | < 5 min |
| Stage 3 (C5, DeepSeek only) | 300 | 2-4 h |
| **Total** | **7 300** | **20-32 h** |

### Crash Recovery

The runner is **idempotent**. Re-run any script after a crash - completed
trials in `data/outputs/completed_trials.parquet` are automatically skipped.

---

## 9. SMS Notifications

`TextBelt.py` sends SMS to `PHONE_NO` on:

| Event | Message |
|-------|---------|
| Completed | Plato's Ship pipeline completed |
| Exception | Plato's Ship FAILED: <error summary> |
| SIGTERM/Ctrl-C | Plato's Ship: interrupted |
| Unexpected exit | Plato's Ship: unexpected exit |

Loaded from `.env` automatically. Silent if keys missing.

> **Note:** TextBelt's default plan covers US/Canada numbers only.
> For Indian (+91) or other international numbers, purchase an international
> plan at https://textbelt.com or use a different SMS provider.

---

## 10. Output Files

Runtime pipeline output lives in `data/outputs/` (gitignored); the
committed snapshot of the May 2026 experiment lives in
`results/data/outputs/`.

| File | Row = | Purpose |
|------|-------|---------|
| trial_log.parquet | Agent response | Raw record of every call (35,050 rows) |
| final_answers.parquet | (question, condition, trial, focal) | Analysis view (7,300 rows) |
| completed_trials.parquet | Trial tuple | Resume checkpoint |
| metrics_summary.parquet | (condition, focal) | Accuracy, flip rates, CIs |
| statistical_tests.parquet | Condition pair | McNemar + Bonferroni |
| calibration_gate_report.parquet | Gate run | P(loud-and-wrong) + decision |
| mitigation_summary.parquet | C4 vs C5 | Filter outcome (100% peer removal) |
| experiment_metadata.json | Run | Full provenance |

Paper-side analysis artefacts (in `Submission/Analyse/`):

| File | Purpose |
|------|---------|
| analysis.md | Narrative walkthrough of every number reported in the paper |
| generate_analysis.py | Original printer that produced analysis.md |
| free_analyses.py | SC-of-3 baseline, GEE clustered regression, Cohen's h + power, difficulty/subject/dataset stratifications, trial-level McNemar with Holm-Bonferroni, net-flip arithmetic audit, calibration-by-model |
| free_analyses_output.json | Output JSON cited throughout the paper |
| inter_agent_dependence.py | Pairwise Cohen's κ on R0 correctness, AUROC of confidence-to-correctness |
| inter_agent_dependence_output.json | Output JSON for §4.5 of the paper |

---

## 11. Reproducing Results

Recompute paper numbers from the saved parquet artefacts without
rerunning any model:

```bash
# Original pipeline-side metrics (uses the runtime outputs)
python3 -m src.metrics_calculator
python3 -m src.statistical_analyzer
python3 -m src.calibration_gate

# Paper-side analyses (loads results/data/outputs/, no API calls)
python3 Submission/Analyse/free_analyses.py
python3 Submission/Analyse/inter_agent_dependence.py
python3 Submission/Analyse/generate_analysis.py
```

Every numerical claim in `Submission/CSR_Paper.tex` and
`Submission/CSR_Supplement.tex` can be re-derived from
`results/data/outputs/*.parquet` by running the two analysis scripts
above. The outputs are committed for inspection.

---

## 12. Cost Estimate

| Provider | ~Calls | ~Cost |
|----------|--------|-------|
| DeepSeek (focal) | 14 000 | $4-8 |
| OpenRouter GPT-4o-mini | 2 000 | $1-2 |
| OpenRouter Llama 3.1 8B | 4 500 | $0.50-1 |
| OpenRouter Gemma 3 4B | 4 500 | $0.50-1 |
| Gemini (judge) | 500 | ~$0.10 |
| Mistral (judge) | 100 | ~$0.05 |
| **Total API** | | **~$6-13** |

GCP VM (n2-highmem-8): ~$0.38/h x 28h = ~$11.

---

## Licence

Research use only.

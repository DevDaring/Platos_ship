# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reproducible research pipeline for the "Plato's Ship" study: does a smart LLM's accuracy degrade when forced to debate capability-weaker peers, and can a confidence-weighted aggregation rule recover the loss? Five conditions (C1–C5), MMLU-Pro + GSM8K datasets, statistical analysis with McNemar + Bonferroni.

**Important deviation from older docs:** `coding_agent.md` (the v2 implementation spec) describes a GPU/A100 setup with locally loaded HuggingFace models (Llama, Gemma) under bfloat16 + flash-attention. The current code is **API-only**:
- `requirements.txt` has no `torch`/`bitsandbytes`/`flash_attn`.
- `config/models.yaml` routes every dumb agent through OpenRouter (`provider: openrouter`).
- `bit_precision_policy.policy_description` explicitly says "All models … are accessed via remote API. No local models are loaded."
- `src/agent_wrappers/local_huggingface_agent.py` is reference-only and unused on CPU.

Treat `coding_agent.md` as historical design intent; `config/` and `src/agent_wrappers/` are the source of truth.

## Commands

All scripts assume a venv at `~/venv` (preferred) or `~/venv_debate_study` and activate it themselves.

```bash
# Setup (one-time)
python3 -m venv ~/venv && ~/venv/bin/pip install -r requirements.txt

# Pre-flight: validates API keys, paths, disk
bash scripts/run_environment_check.sh

# Mandatory before any real run — exercises every model + judge fallback + C5 filter branch
bash scripts/run_dry_run.sh

# Three sequenced stages
bash scripts/run_main_experiment.sh      # Stage 1: C1–C4 (~7000 trials)
bash scripts/run_calibration_gate.sh     # Stage 2: decides whether C5 runs
bash scripts/run_mitigation_experiment.sh # Stage 3: C5 (only if gate passed)

# All stages in one process
python3 -m src.pipeline_orchestrator --stage all

# Recompute metrics from saved trial_log.parquet without rerunning models
python3 -m src.metrics_calculator
python3 -m src.statistical_analyzer
python3 -m src.calibration_gate
```

`scripts/run_dry_run.sh` also runs `python3 -m tests.dry_run_assertions`. Individual smoke tests live in `dry_run/` and can be run with `python3 -m dry_run.test_<name>`; the umbrella driver is `python3 -m dry_run.run_all_dry_tests`.

## Architecture

### Three-stage pipeline (orchestrated by `src/pipeline_orchestrator.py`)

1. **Stage 1 — Main experiment.** `TrialRunner` executes conditions C1–C4 across the full 300-question pool, then a cross-model validation subset (50 questions) with GPT-4o-mini as focal.
2. **Stage 2 — Calibration gate** (`src/calibration_gate.py`). Computes `P(confidence >= 60 | wrong answer, dumb agent, C3 or C4)`. The gate **passes only if this is >= 0.40**. A failed gate is a publishable negative result — do not lower the threshold or run C5 anyway. The pipeline must exit cleanly with status 0 on gate failure.
3. **Stage 3 — Mitigation experiment.** C5 (confidence-weighted aggregation, 100-question subset, 3 trials/question) runs only when the gate passed.

### Agent abstraction

`src/agent_wrappers/base_agent.py` defines:
- `BaseAgent.generate_response(system_prompt, user_prompt, temperature, maximum_output_tokens, request_metadata) -> AgentResponse` — every wrapper conforms.
- `AgentResponse` — uniform dataclass (raw text, latency, tokens, error_status, retry_attempts_used, optional judge_tier_used).
- `RoundRobinKeyManager` — thread-safe key rotation. DeepSeek uses 2 keys, OpenRouter uses 2 keys (shared across smart + both dumb models), Gemini uses 4 keys.

All API agents (`deepseek_agent.py`, `openrouter_agent.py`) implement exponential backoff: 2-4-8-16-32s + jitter, max 5 retries, on 429/5xx. Failures log to `logs/api_failures.log`.

### Judge cascade (`src/agent_wrappers/judge_agent.py`)

Three tiers with **no retry inside a tier — only fallback**:
- Tier 1: Gemini 2.x flash (round-robin across 4 keys)
- Tier 2: Mistral Small
- Tier 3: DeepSeek (reuses smart-agent keys)

The judge fires **only** when regex extraction of `Final answer: X` fails. Regex patterns (in `judge_agent.py`) are intentionally permissive to minimize judge usage. Confidence has its own simpler regex.

### Trial protocol

`src/debate_protocol.py` handles Round 0 (independent) and Round 1 standard debate for C1–C4. `src/confidence_weighted_protocol.py` handles C5's Round 1, which filters out dumb peers whose self-reported confidence is below 60 before showing peer messages to the focal agent. Round 0 is identical across all conditions.

**Universal output contract** (asked of every agent in every condition):
```
Final answer: <X>
Confidence: <integer 0 to 100>
```
This must be in the Round 0 prompt for all conditions — the calibration gate analysis depends on dumb agents emitting confidence from the start, not retroactively.

### Resumability

`TrialRunner` is idempotent. Every completed `(question_id, condition_id, trial_index, focal_agent)` tuple is appended to `data/outputs/completed_trials.parquet` and skipped on resume. The runner flushes every 50 trials and on SIGINT. Re-running any stage script after a crash is safe.

### Config-driven, no hardcoding

- All filesystem paths come from `config/paths.yaml`. Resolved relative to project root at startup.
- All model names and API parameters come from `config/models.yaml`. Model name strings live in `.env` (`*_MODEL_NAME` vars).
- All experiment knobs (conditions, sample sizes, seeds, gate threshold) come from `config/experiment.yaml`.

The single `RANDOM_SEED` from `.env` propagates to every randomized step (deterministic per-step derivations like `seed + 1` for mitigation subset selection).

## Schemas — do not abbreviate column names

`trial_log.parquet`, `final_answers.parquet`, `metrics_summary.parquet`, `mitigation_summary.parquet`, `calibration_gate_report.parquet`, and `statistical_tests.parquet` use **full-form** column names exactly as specified in `coding_agent.md` §10 and §11–§13 (e.g. `extracted_self_reported_confidence_integer`, not `confidence`; `condition_identifier`, not `cond_id`). Dry-run assertions in `tests/dry_run_assertions.py` enforce this. Any new column should follow the same convention.

## Secrets

`.env` (gitignored) holds: `DEEPSEEK_API_KEY_{1,2}`, `OPENROUTER_API_KEY_{1,2}`, `GEMINI_API_KEY_{1..4}`, `MISTRAL_API_KEY`, `HUGGINGFACE_TOKEN` (dataset download), `Github_Classic_Token` (private-repo clone), `RANDOM_SEED`, and optional `PHONE_NO` / `TextBelt_API_KEY` for SMS notifications via `TextBelt.py`. Copy `.env.example` and fill values. Never echo key values to stdout/logs — `RoundRobinKeyManager` masks them when logging.

## Runtime artifacts (gitignored)

`data/raw/`, `data/processed/`, `data/outputs/`, and `logs/` are runtime only. The pipeline writes `experiment_metadata.json` at run start and updates it at end with served model names, gate decision, trial counts, and parse-failure rates — this is the provenance record for reproducibility.

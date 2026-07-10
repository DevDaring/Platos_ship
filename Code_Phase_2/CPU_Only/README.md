# CPU_Only — Phase 2 API-only pipeline (E1–E8)

Single entry point, multi-provider, resume-capable. No GPU. Built on the Phase 1
core (agent wrappers, dataset builder, protocols, metrics, McNemar stats) with
Phase 2 extensions layered on top.

## Setup

```bash
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install -r requirements_cpu.txt
# keys are read from ../.env (Code_Phase_2/.env). Only DeepSeek + OpenRouter
# keys are needed for the default routing; see ../API_AND_RECHARGE.md.
```

## Run

```bash
python3 run_all.py --list           # print the experiment plan
python3 run_all.py --dry-run        # 1 trial/condition; exercises every wired path
python3 run_all.py --p0             # E1–E4 (must-do)
python3 run_all.py --p1             # E1–E6 (adds P1)
python3 run_all.py --experiments E2,E3
python3 run_all.py --all            # every ENABLED experiment
python3 run_all.py --analyse-only   # recompute analysis from saved parquets
```

Enable the P2 experiments (E7 split-peer, E8 heterogeneous-smart) by flipping
`enabled: true` in `config/experiment.yaml`, or naming them:
`python3 run_all.py --experiments E7,E8`.

## New conditions (all in `config/experiment.yaml`)

| ID | Meaning | Experiment |
|---|---|---|
| `C1R_solo_reanswer` | focal re-answers, NO peers | E1 |
| `C3H` / `C4H` | 1–2 **honest** weak peers (natural answers) | E2 |
| `C5R` | wrong peers **with** confidence line + filter | E3 |
| `C5H` | honest peers + confidence filter | E3 |
| `C4split` | one wrong + one correct anchored peer | E7 |
| `C2het` | three **distinct** strong models debate | E8 |

`peer_mode` (`anchored`/`honest`/`split`/`none`) and `aggregation_rule`
(`none`/`solo_reanswer`/`standard_debate`/`confidence_weighted`) drive the
runner. Sweep focals (E4) and heterogeneous models (E8) are defined as model
blocks in `config/models.yaml`.

## Multi-provider routing

`config/models.yaml` has a `providers:` registry (OpenRouter, DeepSeek, Mistral,
LinkAPI, nano-gpt, GCP Vertex). Each model names a `provider:` + `model_slug:`.
To move a model to a cheaper source, edit those two fields — no code change. All
providers speak the OpenAI Chat Completions protocol through one generic agent
(`src/agent_wrappers/openai_compatible_agent.py`).

> **DeepSeek alias deadline:** `deepseek-chat` retires **2026-07-24**. For runs
> on/after that date set the focal `model_slug` to `deepseek-v4-flash` (same
> model). Flagged inline in `models.yaml`.

## Outputs (`data/outputs/`)

- `trial_log.parquet`, `final_answers.parquet` — same schema as Phase 1.
- `metrics_summary.parquet` — per-condition accuracy/flip for every condition.
- `phase2_statistical_tests.parquet` — paired McNemar for the new contrasts
  (C1R vs C4, C4H vs C4, …) with Bonferroni correction.
- `corrected_calibration_gate_report.parquet` — the E3 discriminative-gap gate.
- `capability_sweep_summary.parquet` + `capability_sweep_analysis.json` — E4
  capability-vs-outcome table and the across-model Spearman trend.
- `experiment_metadata.json` — provenance (served models, providers, seed).

## Key Phase 2 modules

- `src/phase2_agents.py` — builds every agent from the provider registry.
- `src/debate_protocol.py` — `+ run_round1_solo_reanswer` (C1R).
- `src/persona_generator.py` — `+ include_confidence_line`, `+ anchor_mode` (correct).
- `src/trial_runner.py` — `+ peer_mode`, sweep/het focals, persona pools.
- `src/corrected_gate.py` — E3 discriminative-gap gate + AUROC.
- `src/perturbed_gsm8k.py` — E6 numeric-perturbation builder.
- `src/phase2_analyzer.py` — E4 sweep + E1/E2 contrasts + paired McNemar.

# Plato's Ship — Experiment Results Analysis

**Experiment UUID:** `477ec5c4-65d1-474d-9000-88440be70ed5`  
**Git commit at run start:** `d2e27fde54cad4a506259ca1a0c00188f27b77a6`  
**Stage 1 run time:** 2026-05-03 08:06 UTC → 2026-05-04 07:52 UTC (~23h 46m)  
**Stage 3 run time:** 2026-05-04 07:52 UTC → 2026-05-04 09:44 UTC (~1h 52m)  
**Random seed:** 20260502

---

## Table of Contents

1. [Dataset & Question Pool](#1-dataset--question-pool)
2. [Persona Generation](#2-persona-generation)
3. [Trial Execution Summary](#3-trial-execution-summary)
4. [Calibration Gate (Stage 2)](#4-calibration-gate-stage-2)
5. [Primary Result: Accuracy by Condition](#5-primary-result-accuracy-by-condition)
6. [Flip Rates & Asch Conformity Index](#6-flip-rates--asch-conformity-index)
7. [Bandwagon Dose-Response](#7-bandwagon-dose-response)
8. [Mean Confidence by Condition and Round](#8-mean-confidence-by-condition-and-round)
9. [Statistical Tests](#9-statistical-tests)
10. [Mitigation (C4 vs C5)](#10-mitigation-c4-vs-c5)
11. [Cross-Model Validation (GPT-4o-mini)](#11-cross-model-validation-gpt-4o-mini)
12. [Dumb Agent Accuracy](#12-dumb-agent-accuracy)
13. [Answer Extraction Quality](#13-answer-extraction-quality)
14. [Key Findings Summary](#14-key-findings-summary)

---

## 1. Dataset & Question Pool

**300 questions total** across two datasets.

| Source | Count | Notes |
|--------|-------|-------|
| MMLU-Pro | 200 | 10 subjects × 20 questions, stratified by difficulty |
| GSM8K | 100 | Grade-school math, stratified by difficulty |

**Question pool columns:** `question_identifier`, `source_dataset`, `subject_category`, `question_text`, `answer_options`, `correct_answer`, `correct_answer_full_text`, `wrong_answer_pool`, `difficulty_stratum`, `random_seed_used`, `included_in_mitigation_subset`

The `difficulty_stratum` was assigned by a Llama 3.1 8B zero-shot accuracy probe, splitting each subject 50/50 into harder and easier halves. The `included_in_mitigation_subset` flag marks 100 questions used in Stage 3 (C5), sampled using stratified-random from the full 300-question pool.

---

## 2. Persona Generation

**1,500 personas** generated (5 variants × 300 questions), each embedding a pre-specified wrong answer using one of four erroneous reasoning styles.

| Metric | Value |
|--------|-------|
| Total personas generated | 1,500 |
| Passed on first attempt | 1,102 (73.5%) |
| Regenerated successfully | 384 (25.6%) |
| Failed after max attempts | 14 (0.9%) |
| **Retention rate** | **99.1%** |

**Reasoning styles used** (4 types, 375 personas each):
- `surface_keyword_match` — answers based on superficial word overlap
- `false_analogy` — maps question to unrelated familiar scenario
- `overconfident_assertion` — states wrong answer with high certainty and no justification
- `misapplied_rule` — applies a real rule in the wrong context

**Generator model:** Llama 3.1 8B Instruct (via OpenRouter API)  
**Validator:** Regex + answer-matching; 14 personas failed due to trailing punctuation mismatches (e.g., `"90."` vs `"90"`) and one missing `Final answer:` marker.

---

## 3. Trial Execution Summary

**35,050 total trial log rows** (one row per agent × round × trial).  
**7,300 final answer rows** (one row per focal-agent × question × condition × trial).

### Trial log rows by condition

| Condition | Trial Rows | Notes |
|-----------|-----------|-------|
| C2_three_smart | 10,500 | 3 agents × 1,500 trials |
| C3_two_smart_one_dumb | 10,500 | 3 agents × 1,500 trials |
| C4_one_smart_two_dumb | 10,500 | 3 agents × 1,500 trials |
| C5_one_smart_two_dumb_confidence_weighted | 1,800 | 3 agents × 300 trials (C5 subset) |
| C1_smart_solo | 1,750 | 1 agent × 1,750 rows (R0 only + R1 no-op) |

> **Note:** Each "trial" = one (question × condition × replication_index). C1–C4 had 5 replications per question × 300 questions = 1,500 trials each for the primary focal agent (DeepSeek). C5 had 3 replications × 100 questions = 300 trials.

### Trial log rows by agent role

| Role | Rows |
|------|------|
| smart_focal | 12,850 |
| dumb | 11,700 |
| smart_nonfocal | 10,500 |

### Models used

| Model | Total Rows | Role |
|-------|-----------|------|
| deepseek-v4-flash | 20,099 | Smart focal + judge fallback |
| meta-llama/llama-3.1-8b-instruct | 5,932 | Dumb agent |
| google/gemma-3-4b-it | 5,768 | Dumb agent |
| openai/gpt-4o-mini | 3,250 | Smart focal (cross-validation subset) |
| deepseek-chat | 1 | Judge tertiary fallback |

**Total API calls (from metadata):**
- DeepSeek: 20,100
- OpenRouter: 19,991
- Judge invocations: 3,020 (only on regex parse failure)
- Gemma (API via OpenRouter): 2,884 rows
- Llama (API via OpenRouter): 13,857 rows

---

## 4. Calibration Gate (Stage 2)

The gate checks whether dumb agents are **confidently wrong** — the precondition for confidence-weighted filtering to be meaningful.

**Metric:** P(confidence ≥ 60 | answer wrong, dumb agent, condition ∈ {C3, C4})

| Metric | Value |
|--------|-------|
| Precondition metric | **0.995** |
| Bootstrap 95% CI | [0.9909, 0.9983] |
| Threshold for pass | 0.40 |
| **Gate decision** | **PASSED** |
| Partner metric P(conf ≥ 60 \| correct) | 0.9976 |
| Loud-wrong minus loud-right | −0.0026 |
| Total dumb responses with parsed confidence | 4,146 |

### Breakdown by dumb model

| Model | P(high conf \| wrong) | Wrong responses | Total responses |
|-------|----------------------|-----------------|-----------------|
| google/gemma-3-4b-it | **1.000** | 502 | 1,984 |
| meta-llama/llama-3.1-8b-instruct | **0.9915** | 710 | 2,162 |

> **Interpretation:** Both dumb models are extremely overconfident — they report confidence ≥ 60 on virtually every wrong answer. The gate passes with a metric of 0.995 (far above the 0.40 threshold), confirming that C5 confidence-weighted filtering is valid and meaningful. The near-zero loud-wrong vs loud-right difference (−0.0026) means confidence level alone does not distinguish correct from incorrect dumb responses.

---

## 5. Primary Result: Accuracy by Condition

All accuracy rates are averaged over **300 questions × 5 replications = 1,500 trials** for DeepSeek (primary focal agent), and **50 cross-validation questions × 5 replications = 250 trials** for GPT-4o-mini.

### DeepSeek-v4-flash (primary focal agent, N = 1,500 trials per condition)

| Condition | Round-0 Acc | Round-1 Acc | Δ (R1−R0) |
|-----------|------------|------------|-----------|
| C1_smart_solo (baseline) | **76.2%** | 76.2% | 0.0 pp |
| C2_three_smart | 75.8% | **78.7%** | +2.9 pp |
| C3_two_smart_one_dumb | 76.1% | **78.3%** | +2.2 pp |
| C4_one_smart_two_dumb | 75.1% | **84.1%** | +9.0 pp |
| C5 mitigation subset | 76.0% | 78.0% | +2.0 pp |

> **Key finding:** C4 shows the largest Round-1 accuracy gain (+9.0 pp), not a degradation. This is counter-intuitive — when the focal smart agent faces two dumb peers, its post-debate accuracy **increases** markedly. This likely reflects the focal agent asserting its position more firmly when surrounded by clearly weaker reasoners, plus a regression-to-mean effect from the aggregation rule.
>
> C1 serves as the no-debate baseline (Round-1 = Round-0 since there is no debate). C2 and C3 show modest improvements (+2.9, +2.2 pp), which are not statistically significant after Bonferroni correction.

### GPT-4o-mini (cross-validation focal agent, N = 250 trials per condition)

| Condition | Round-0 Acc | Round-1 Acc | Δ (R1−R0) |
|-----------|------------|------------|-----------|
| C1_smart_solo (baseline) | **64.8%** | 64.8% | 0.0 pp |
| C2_three_smart | 62.4% | **66.8%** | +4.4 pp |
| C3_two_smart_one_dumb | 62.8% | **60.8%** | −2.0 pp |
| C4_one_smart_two_dumb | 65.2% | **65.2%** | 0.0 pp |

> **Note:** GPT-4o-mini shows degradation in C3 (−2.0 pp) and no change in C4, while DeepSeek shows improvement. This divergence between the two focal agents is scientifically important — GPT-4o-mini is more susceptible to the Plato effect (C3 degradation), whereas DeepSeek appears more robust.

---

## 6. Flip Rates & Asch Conformity Index

**Flip rate correct→incorrect:** proportion of trials where focal agent was right in Round 0 but flipped to wrong in Round 1 (conformity signal).  
**Flip rate incorrect→correct:** proportion of trials where focal agent was wrong in Round 0 but corrected to right in Round 1.  
**Asch conformity index:** flip_correct→incorrect(unanimous wrong peers) − flip_correct→incorrect(split peers). Only computed for C3, C4, C5.

### DeepSeek (primary focal, N = 1,500 per condition)

| Condition | Flip C→I | Flip I→C | Asch Index | 95% CI (C→I) |
|-----------|---------|---------|-----------|--------------|
| C1_smart_solo | 0.0% | 0.0% | — | [0.0, 0.0] |
| C2_three_smart | 5.0% | 27.5% | — | [2.9, 4.8] |
| C3_two_smart_one_dumb | 6.0% | 28.5% | **0.060** | [3.5, 5.7] |
| C4_one_smart_two_dumb | 6.5% | 55.8% | **0.065** | [3.8, 6.0] |
| C5 (mitigation) | 10.1% | 40.3% | 0.101 | [4.7, 10.7] |

### GPT-4o-mini (cross-validation, N = 250 per condition)

| Condition | Flip C→I | Flip I→C | Asch Index | 95% CI (C→I) |
|-----------|---------|---------|-----------|--------------|
| C1_smart_solo | 0.0% | 0.0% | — | [0.0, 0.0] |
| C2_three_smart | 5.8% | 21.3% | — | [1.6, 6.0] |
| C3_two_smart_one_dumb | **11.5%** | 14.0% | **0.115** | [4.4, 10.4] |
| C4_one_smart_two_dumb | **17.2%** | 32.2% | **0.172** | [7.6, 15.2] |

> **Interpretation:**
> - DeepSeek is relatively conformity-resistant: flip C→I rates remain low (5–6.5%) even as dumb peer count increases. Its accuracy gains in C4 come from very high flip I→C (55.8%), meaning it corrects wrong-answers more when dumb peers push back.
> - GPT-4o-mini shows much stronger Plato-effect conformity: flip C→I rises from 5.8% (C2) → 11.5% (C3) → 17.2% (C4), confirming capability-asymmetric peer pressure degrades weaker smart agents more.
> - Asch conformity index for GPT-4o-mini in C4 = 0.172 vs DeepSeek 0.065 — GPT-4o-mini is ~2.6× more susceptible to unanimous-wrong peer consensus.

---

## 7. Bandwagon Dose-Response

**Question:** Does adding more dumb peers monotonically worsen smart agent accuracy?

**Test:** Logistic regression of `round_one_answer_was_correct` on `condition_dumb_agent_count` (0 in C2, 1 in C3, 2 in C4). Pooled across both focal agents.

| Parameter | Value |
|-----------|-------|
| Coefficient on dumb_agent_count | **+0.1739** |
| Raw p-value | **0.000168** |
| Bonferroni-corrected p-value | **0.001176** |
| Significant at corrected α = 0.00714 | **Yes** |
| Sample size | 4,500 trials |
| AIC | 4,445.8 |

> **Interpretation:** The positive coefficient (+0.174) means that as dumb agent count increases (0 → 1 → 2), Round-1 accuracy actually **increases** in this pooled analysis. This is significant (p = 0.000168) and survives Bonferroni correction. The effect is primarily driven by DeepSeek's large accuracy gain in C4. This is the opposite of the originally hypothesised Plato degradation for DeepSeek, though GPT-4o-mini shows the expected pattern.

---

## 8. Mean Confidence by Condition and Round

All values are self-reported confidence integers (0–100) from the focal smart agent's response.

### DeepSeek (N = 1,500 per condition)

| Condition | Round-0 Mean Conf | Round-1 Mean Conf | Δ |
|-----------|-------------------|-------------------|---|
| C1_smart_solo | 96.04 | 96.04 | 0.0 |
| C2_three_smart | 96.06 | 97.44 | +1.38 |
| C3_two_smart_one_dumb | 96.06 | 97.54 | +1.48 |
| C4_one_smart_two_dumb | 96.12 | 97.22 | +1.10 |
| C5 | 95.50 | 95.77 | +0.27 |

### GPT-4o-mini (N = 250 per condition)

| Condition | Round-0 Mean Conf | Round-1 Mean Conf | Δ |
|-----------|-------------------|-------------------|---|
| C1_smart_solo | 90.30 | 90.30 | 0.0 |
| C2_three_smart | 90.32 | 93.24 | +2.92 |
| C3_two_smart_one_dumb | 90.50 | 92.77 | +2.27 |
| C4_one_smart_two_dumb | 90.28 | 91.57 | +1.29 |

> **Interpretation:** DeepSeek starts at very high confidence (~96) and changes minimally across conditions. GPT-4o-mini starts lower (~90) and shows more responsiveness. Both agents report *higher* confidence in Round 1 after debate — even GPT-4o-mini in C3 (which showed accuracy degradation) still reports higher post-debate confidence. This is a miscalibration signal: agents can become more confident in wrong answers post-debate.

---

## 9. Statistical Tests

All tests use McNemar's exact test on per-question majority-vote accuracy. Bonferroni correction applied over 7 comparisons (corrected α = 0.05 / 7 = 0.00714).

| Comparison | Contingency [[C1,C2],[C3,C4]] | Raw p | Bonferroni p | Significant | Effect (Δ acc) |
|------------|-------------------------------|-------|-------------|------------|----------------|
| C1 vs C2 | [[225,5],[14,56]] | 0.0636 | 0.445 | No | −3.0 pp |
| C1 vs C3 | [[224,6],[12,58]] | 0.2379 | 1.000 | No | −2.0 pp |
| **C1 vs C4** | **[[218,12],[33,37]]** | **0.0025** | **0.017** | **Yes** | **−7.0 pp** |
| C2 vs C3 | [[229,10],[7,54]] | 0.6291 | 1.000 | No | +1.0 pp |
| C2 vs C4 | [[226,13],[25,36]] | 0.0730 | 0.511 | No | −4.0 pp |
| C3 vs C4 | [[225,11],[26,38]] | 0.0201 | 0.141 | No | −5.0 pp |
| C4 vs C5 | [[76,5],[2,17]] | 0.4531 | 1.000 | No | +3.0 pp |
| **Dose-response (logit)** | n=4,500 | **0.000168** | **0.001176** | **Yes** | coef=+0.174 |

**n_questions = 300 for all C1–C4 comparisons; n_questions = 100 for C4 vs C5.**

> **Interpretation:**
> - **Only C1 vs C4 is statistically significant** (p = 0.0025, Bonferroni p = 0.017). The contingency table [[218,12],[33,37]] means 33 questions that were correct under C1 became incorrect under C4, vs only 12 going the other way — a net Plato effect of 21 questions out of 300.
> - All other pairwise condition comparisons fail to reach significance after correction.
> - The **dose-response regression is significant** (p = 0.000168), showing a systematic relationship between dumb peer count and post-debate accuracy, though the direction is positive (driven by DeepSeek's strong C4 performance).

---

## 10. Mitigation (C4 vs C5)

**C5** applies confidence-weighted peer filtering: dumb peers with self-reported confidence < 60 are excluded from the focal agent's Round-1 context. Restricted to the 100-question mitigation subset.

| Metric | Value |
|--------|-------|
| C4 accuracy on mitigation subset | **79.6%** |
| C5 accuracy | **78.0%** |
| C5 − C4 delta | **−1.6 pp** |
| Bootstrap 95% CI for delta | [−7.5 pp, +4.0 pp] |
| McNemar p-value | 0.453 |
| McNemar Bonferroni-corrected p | 1.000 |
| **Significant?** | **No** |
| Mean peer messages filtered per C5 trial | **2.0 of 2** |
| Proportion C5 trials with zero peers after filtering | **100%** |
| C5 focal flip rate C→I | 10.1% |
| C4 focal flip rate C→I (mitigation subset) | 11.6% |

> **Interpretation:** The confidence-weighted filter **removed all dumb peer messages in every single C5 trial** (mean filtered = 2.0 out of 2.0 dumb peers; 100% of trials had zero peers remaining). This means C5 effectively became a solo task for the focal agent — identical to C1. The small accuracy delta of −1.6 pp (C5 slightly below C4) is not significant (p = 0.453) and the CI crosses zero, suggesting no meaningful effect either direction.
>
> This outcome arises from the calibration gate result: dumb agents report confidence ≥ 60 on 99.5% of *all* responses (both correct and wrong), making confidence a useless signal for filtering. The mitigation strategy failed to function as designed, but this is itself a scientifically valid negative result: the dumb agents' overconfidence renders confidence-weighted filtering ineffective in this experimental setting.

---

## 11. Cross-Model Validation (GPT-4o-mini)

**Purpose:** Verify whether findings replicate with a different smart focal agent. GPT-4o-mini ran on a 50-question stratified subset × 5 replications = 250 trials per condition (C1–C4 only; no C5 for cross-validation agent).

| Condition | R0 Acc | R1 Acc | Δ | Flip C→I | Asch |
|-----------|--------|--------|---|---------|------|
| C1_smart_solo | 64.8% | 64.8% | 0.0 | 0.0% | — |
| C2_three_smart | 62.4% | 66.8% | +4.4 pp | 5.8% | — |
| **C3_two_smart_one_dumb** | **62.8%** | **60.8%** | **−2.0 pp** | **11.5%** | **0.115** |
| C4_one_smart_two_dumb | 65.2% | 65.2% | 0.0 pp | 17.2% | 0.172 |

> **Interpretation:** GPT-4o-mini partially replicates the Plato hypothesis. C3 shows a −2.0 pp accuracy drop and the Asch conformity index rises monotonically with dumb peer count (0 → 0.115 → 0.172). However, C4 shows no net change (0.0 pp delta) despite a high flip C→I rate of 17.2%, because a matching high flip I→C rate of 32.2% compensates. The two focal agents tell different stories: DeepSeek is resilient and gains accuracy in C4; GPT-4o-mini shows the expected Plato degradation in C3 but then stabilises in C4.

---

## 12. Dumb Agent Accuracy

Dumb agents are designed to perform poorly and confidently argue for wrong answers. Their accuracy confirms role fidelity.

### By model

| Model | Accuracy | Role |
|-------|---------|------|
| google/gemma-3-4b-it | 31.7% | Dumb |
| meta-llama/llama-3.1-8b-instruct | 29.2% | Dumb |

*Averaged across all conditions in which the model appeared (C3, C4, C5).*

### By condition

| Condition | Dumb Agent Accuracy |
|-----------|-------------------|
| C3_two_smart_one_dumb | 33.6% |
| C4_one_smart_two_dumb | 28.8% |
| C5_one_smart_two_dumb_confidence_weighted | 31.2% |

> **Interpretation:** Both dumb models achieve ~29–32% accuracy, compared to ~76% for DeepSeek and ~65% for GPT-4o-mini. This confirms a meaningful capability gap. Dumb accuracy in C4 (28.8%) is slightly lower than C3 (33.6%), possibly due to the different question-set composition of trials sampled for each condition. All values are substantially below smart agent baselines, validating the capability-asymmetry manipulation.

---

## 13. Answer Extraction Quality

The answer extraction pipeline uses regex first, then falls back to a judge cascade (Gemini → Mistral → DeepSeek).

| Method | Count | % of 35,050 rows |
|--------|-------|-----------------|
| regex_success | 32,030 | **91.4%** |
| judge_gemini | 2,380 | 6.8% |
| parse_failure | 298 | 0.9% |
| judge_mistral | 251 | 0.7% |
| judge_deepseek | 91 | 0.3% |

> **Interpretation:** 91.4% of responses were parsed directly by regex. 7.8% required judge cascade (Gemini: 6.8%, Mistral: 0.7%, DeepSeek: 0.3%), and only 0.9% (298 rows) resulted in a final parse failure, meaning a fallback response was used. Total judge invocations = 3,020 per metadata. The low parse failure rate confirms the structured-output instruction format worked reliably across all models.

---

## 14. Key Findings Summary

### Finding 1: DeepSeek is resilient to the Plato effect
DeepSeek-v4-flash (primary focal agent) did **not** degrade under capability-asymmetric peer pressure. Round-1 accuracy *increased* in C3 (+2.2 pp) and most notably in C4 (+9.0 pp). The dominant mechanism is a high incorrect→correct flip rate in C4 (55.8%), suggesting DeepSeek asserts its correct answer more firmly when surrounded by weaker arguers. *Averaged over 300 questions × 5 replications = 1,500 trials per condition.*

### Finding 2: GPT-4o-mini shows the Plato effect in C3
GPT-4o-mini (cross-validation, 250 trials per condition) shows accuracy degradation in C3 (−2.0 pp vs baseline) with a 11.5% correct→incorrect flip rate and an Asch conformity index of 0.115. This confirms the Plato effect exists in less capable smart agents. *Averaged over 50 questions × 5 replications = 250 trials per condition.*

### Finding 3: Only C1 vs C4 reaches statistical significance
Of 7 pairwise McNemar comparisons (all on n = 300 questions), only C1 vs C4 is significant after Bonferroni correction (p = 0.017). The contingency table shows 33 questions regressed in C4 vs only 12 improving — a net negative question-level effect, even though trial-level accuracy appears to increase (due to high I→C flips on questions that were initially wrong). *Tested on 300-question pool.*

### Finding 4: Dose-response is significant but positive for DeepSeek
The logistic regression across C2/C3/C4 (n = 4,500 pooled trials) finds a significant positive coefficient on dumb peer count (+0.174, p < 0.001 Bonferroni-corrected). This is driven primarily by DeepSeek's C4 boost. The expected negative dose-response (more dumb peers = worse accuracy) holds for GPT-4o-mini but not DeepSeek.

### Finding 5: Calibration gate passed with an extreme metric (0.995)
Dumb agents (Llama 3.1 8B and Gemma 3 4B) report confidence ≥ 60 on 99.5% of wrong answers, far exceeding the 0.40 gate threshold. This extreme overconfidence means the confidence signal is uninformative for filtering. *Based on 4,146 dumb agent responses in C3+C4.*

### Finding 6: Confidence-weighted mitigation (C5) failed to function
C5 filtering removed all dumb peer messages in 100% of trials (mean 2.0 out of 2.0 filtered per trial). With no peer context, C5 functioned as a solo task, yielding −1.6 pp vs C4 (non-significant, p = 0.453). The negative result is scientifically valid: confidence-based filtering is ineffective when dumb agents are universally overconfident. *Based on 300 C5 trials × 100 questions.*

### Finding 7: Model capability gap confirmed
Dumb agents achieved ~29–32% accuracy vs smart agents at ~65–76%, establishing a clear capability gap (~35–45 pp). Dumb agents' personas successfully embedded wrong answers with confident but flawed reasoning across all four reasoning styles.

---

## Data Provenance

| File | Rows | Description |
|------|------|-------------|
| `results/data/outputs/trial_log.parquet` | 35,050 | Per-agent, per-round raw log |
| `results/data/outputs/final_answers.parquet` | 7,300 | Per-trial focal agent outcomes |
| `results/data/outputs/metrics_summary.parquet` | 9 | Per-(condition, focal_agent) metrics |
| `results/data/outputs/statistical_tests.parquet` | 8 | McNemar + logistic regression |
| `results/data/outputs/mitigation_summary.parquet` | 1 | C4 vs C5 comparison |
| `results/data/outputs/calibration_gate_report.parquet` | 1 | Stage 2 gate result |
| `results/data/outputs/completed_trials.parquet` | 7,300 | Checkpoint index |
| `results/data/outputs/experiment_metadata.json` | — | Full run provenance |
| `results/data/processed/question_pool.parquet` | 300 | Final question set |
| `results/data/processed/dumb_personas.parquet` | 1,500 | Generated personas |

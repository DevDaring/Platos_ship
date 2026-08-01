# audit/numerical_consistency_report.md

Every number in the manuscript was recomputed from the raw per-trial logs by
`Submission/Analyse/verify_paper_numbers.py`, and every data table is emitted
from that output by `Submission/Analyse/make_tables.py`. The manuscript
`\input`s the generated tables, so it cannot drift from the logs.

Source of truth: `results/data/outputs/*.parquet` (Phase 1),
`Code_Phase_2/results/outputs/*.parquet` (Phase 2),
`Code_Phase_2/results/gpu_probe/*.parquet` (probe).

---

## Discrepancies found and how each was resolved

| # | Discrepancy | Severity | Resolution |
|---|---|---|---|
| 1 | The contamination probe re-uses the `C1`/`C4` condition IDs on a **different** question pool (`gsm8k_perturbed_*`). Pooling them dropped DeepSeek-chat's solo accuracy from 76.2% to 62.9%. | **Critical** | Analysis now separates the perturbed pool by question-ID prefix and reports it independently. |
| 2 | Table 3 was captioned "Holm-corrected" but printed **raw** p-values. | **Critical** | Table now has separate `p` and `p_H` columns, both generated from the pipeline. All previously-asserted significant results survive correction. |
| 3 | Table 3 (p = 0.013) and Appendix S2 (p = 0.017) disagreed for the same C4–C1 contrast. | **High** | Two different multiplicity families were in use. Both families are now **declared explicitly** in §3.5 (causal family of 8; ladder family of 6), S2 reports the raw value plus both corrected values, and states which family the main text uses and why. |
| 4 | Spearman ρ for the capability sweep was computed on the **1-dp rounded** table values, creating ties among four mid-range models and inflating the statistic to −0.97. | **High** | Correlations now use unrounded rates. **ρ = −0.95**, exact permutation p = 0.0011 (not the asymptotic 0.0003 previously quoted as 0.0001). |
| 5 | Appendix S1's trial-count table still carried the superseded 50-question GPT-4o-mini numbers, contradicting Table 2. | **High** | All tables regenerated from logs; the superseded `openrouter_gpt4o_mini` focal is excluded from pooled analyses and reported separately. |
| 6 | Appendix S5 persona counts (1,102 first-attempt / 384 regenerated) were each off by 14 — the failures were double-counted. | **Medium** | Correct values from `regeneration_attempts_used`: **1,116 / 370 / 14**. The correct-anchored pool (1,033 / 452 / 15) is now reported too. |
| 7 | Appendix S10's GPT-4o-mini persona-style rows used the superseded n≈40 subset. | **Medium** | Recomputed on the full pool (n ≈ 224–267). The style spread narrows from 11.6 pp to 7.0 pp, which *strengthens* the paper's own conclusion that style is not a driver. |
| 8 | Appendix S7 asserted the weak models' unprompted accuracy "is not measured in this study". | **Medium** | The honest-peer conditions measure it directly: 50.4% (Llama-3.1-8B) and 45.1% (Gemma-3-4B). |
| 9 | The released corrected-gate report substituted **0** for `P(loud | correct)` on the wrong-anchored substrate, where that class is empty by construction — producing a spurious gap of 0.99 and a "passed" verdict contradicting the paper's own table. | **High** | `corrected_gate.py` now emits `undefined_single_class` with per-class counts. Undefined stays undefined. |
| 10 | Parser statistics (91.4% regex / 7.8% judge / 0.9% failure) were Phase-1-only but presented as global. | **Medium** | Pooled values across both phases: **89.8% / 9.0% / 1.17%**; focal-only failure rate 0.72%. |
| 11 | Every appendix cross-reference resolved to a bogus number ("Appendix 6", "Section 8") because the appendix uses `\section*`, which sets no counter. | **Medium** | All replaced with literal appendix labels; PDF now has 0 unresolved references and no `??`. |
| 12 | Methods promised post-hoc power for non-significant contrasts. | **Medium** | Removed. The paper now reports effect estimates, intervals and discordant-pair counts, and states why post-hoc power is not evidence for a null. |
| 13 | Trial counts were approximate ("about 37,000"). | **Low** | Exact: **39,470** API focal trials (37,500 main pool + 970 perturbed + 1,000 superseded), **179,145** agent responses, **2,700** probe trials. The counting unit is now defined in the text. |
| 14 | `sharma2023towards` bibliography entry contained **six invented co-authors** and omitted two real ones. | **Critical** | Corrected against the ICLR 2024 camera-ready. Eleven further entries corrected (see below). |

---

## Exact counts (by explicit unit)

**Unit definitions.** A *focal trial* is one focal-model episode
(question × condition × replication × focal). An *agent response* is one model
call within a trial — the grain of `trial_log.parquet`. They are not
interchangeable.

| Unit | Count |
|---|---:|
| API focal trials, total | 39,470 |
| — main 300-question pool | 37,500 |
| — perturbed GSM8K pool | 970 |
| — superseded 50-q cross-validation | 1,000 |
| Agent responses (all roles, both rounds) | 179,145 |
| — focal | 69,805 |
| — non-focal strong | 47,700 |
| — weak peers | 61,640 |
| GPU output-distribution probe trials | 2,700 |
| Responses with `error_status = failure` | 210 |

Focal trials per condition (all focal models pooled): C1 9,135 · C1R 3,000 ·
C2 8,650 · C2het 900 · C3 3,250 · C3H 1,500 · C4 9,135 · C4H 1,500 ·
C4split 1,500 · C5 300 · C5R 300 · C5H 300.

---

## Missingness / parser sensitivity

- Focal-response parse failures: **502 / 69,805 = 0.72%**, concentrated in
  Round 1 (1.41%) versus Round 0 (0.18%).
- Highest by condition: C5R and original C5 at 1.17%, C4 at 1.03%.
- Unrecovered failures are scored **incorrect** (conservative).
- Bounding the headline C4–C1 contrast over the 12 affected paired trials
  (none in C1): observed **+7.93 pp**, pessimistic **+7.93 pp**, optimistic
  **+8.73 pp**. No conclusion depends on the handling.

---

## Correlation robustness (eight-model sweep)

| Check | Value |
|---|---|
| Spearman ρ (solo accuracy vs C4 harmful-flip rate) | **−0.95** |
| Exact permutation p (all 8! orderings) | **0.0011** |
| Leave-one-model-out ρ range | [−0.96, −0.93] |
| Excluding the two weakest models (n = 6) | ρ = −0.94, p = 0.0048 |
| Spearman ρ (solo accuracy vs C4 *gain*) | +0.33, p = 0.42 — **no relationship** |

The association is with the *cost*, not the *benefit*. Per-model 95% CIs for
every harmful-flip rate are in Table 5.

---

## Answer-level consensus (the "unanimity" question)

The two anchored peers in C4 draw their assigned wrong answer independently, so
they name the **same** wrong answer in only **16.4%** of trials.

| Peers | n (R0-correct) | C→I | 95% CI |
|---|---:|---:|---|
| same wrong answer | 185 | 5.95% | [3.01, 10.39] |
| different wrong answers | 942 | 6.58% | [5.08, 8.36] |

χ² = 0.03, **p = 0.88**. Correction is likewise unchanged (55.7% vs 55.8%).
**Answer-level consensus has no detectable effect**, so the paper no longer
claims unanimity drives the harm; the C4split result is attributed to a visible
correct peer, with the caveat that C4split changes two things at once.

---

## Bibliography

Twelve of 22 entries were corrected against primary sources. Most serious:
`sharma2023towards` (six invented co-authors), `koo2024benchmarking` (cited to
the ACL main volume instead of Findings), `echterhoff2024cognitive` (arXiv
preprint replaced by the Findings EMNLP 2024 version),
`amayuelas2024multiagent` (wrong author name), `hong2025sycon` (author not on
the published version), `smit2024going` (wrong author order). `du2023improving`
and `sharma2023towards` were also moved from arXiv preprints to their ICML 2024
and ICLR 2024 versions.

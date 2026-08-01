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
| 6 | ~~Appendix S5 persona counts were each off by 14.~~ **This entry was itself wrong and is retracted.** The original 1,102 / 384 / 14 was correct; "correcting" it to 1,116 / 370 counted the 14 failed personas as first-attempt passes. | **Medium** | Reverted to **1,102 / 384 / 14**, recomputed as `regeneration_attempts_used` crossed with `validation_pass_status`. The correct-anchored pool is 1,033 / 452 / 15. |
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

---

## Second round: discrepancies found by adversarial re-review

Three independent red-team reviews were run against the artefact after the
first round of fixes. They found six further defects, all now resolved.

| # | Discrepancy | Severity | Resolution |
|---|---|---|---|
| 15 | §4.5 claimed the math gain "persists at a similar size" under perturbation, under a section titled "The math gain is not memorisation". The comparison set a **GSM8K-only perturbed gain against a full-pool original gain**, concealing a two-thirds collapse. | **Critical** | On the matched comparator — the same 97 items in their original form — the gain falls from **+16.1 pp to +5.4 pp**. §4.5 is retitled to a question, reports the matched comparison, and states that the drop is consistent with partial contamination *and* with the items simply being harder, which the design cannot separate. |
| 16 | Holm families covered the accuracy contrasts only, while harmful-revision contrasts were asserted uncorrected in the prose — letting the family boundary decide the outcome. | **Critical** | A third **revision-rate family** is declared, containing all seven revision contrasts stated anywhere in the paper. **Three do not survive correction** (C4split–C4, C2het–C2, C4–C2) and are reported as descriptive. The C4split claim now rests on its accuracy contrast, which does survive. |
| 17 | The confidence-filter retention gap was reported over **parsed** confidences only. The deployed rule also drops peers whose confidence cannot be parsed, and all 22 such peers in C5H were wrong. | **High** | The deployed gap is **+0.072** over 600 messages, not +0.007 over 578, and is now the headline figure. Still far below the 0.10 threshold, and it arises from a parsing failure rather than from confidence — conclusion unchanged. |
| 18 | The capability gradient was reported only against the wrong-peer condition, but weak models revise more under **any** debate. | **High** | The C2 baseline correlation is ρ = −0.45 and is now a column in Table 4. The wrong-peer-specific excess is **ρ = −0.83** (exact p = 0.015, leave-one-out [−0.96, −0.75]) — ordering preserved, estimate less stable, both stated. |
| 19 | The peer-consensus comparison used a χ² test on trials that are not independent (five replications per question), and concluded that agreement "makes no difference". | **Medium** | Replaced with a question-clustered bootstrap: −0.65 pp, 95% CI [−4.2, +3.4]. The text now says the data cannot resolve a difference of this size, not that none exists. |
| 20 | `make_figures.py` crashed on `rho['p']` and resolved `REPO` one level above the repository, so it regenerated **no** figures and silently skipped main-text Figure 1. Table 7 was hand-written, with a substrate label ("Phase-1") that did not match its own counts. | **High** | Both path bugs and the key error fixed; the script writes to `Submission/images/` and regenerates all three figures. Table 7 is now generated by `make_tables.py` from the pooled logs, with substrate labels that match the data. |

**Artefact fixes in the same round.** Added the missing `LICENSE` (the README
promised MIT and none shipped). `crosscheck_paper_numbers.py` now exits 0 with
an explanation instead of `FileNotFoundError` when `Submission/` is absent, and
a claim whose pattern no longer matches the manuscript is now a **failure**
rather than a silently skipped check — rewrapping a paragraph had been quietly
disabling checks. `make_tables.py` writes to `Submission/tables/` rather than a
directory nothing read. README corrected to point at `analysis/` and to
document the Python 3.12 requirement. Three team-authored bibliography entries
changed to corporate authors, so `acl_natbib.bst` no longer renders "and 1
others". The uncited `deepseekai2024v3` entry was removed rather than cited,
because citing DeepSeek-V3 would contradict the paper's own statement that the
served snapshot is not tied to any published report.

**One reviewer claim checked and rejected.** A reviewer reported that Appendix
S5's correct-anchored counts should be 1,048 / 437 / 15. The crosstab of
`regeneration_attempts_used` against `validation_pass_status` shows all 15
failures sit at zero regeneration attempts, so 1,048 counts them as
first-attempt *passes*. The paper's 1,033 / 452 / 15 is correct. Checking it,
however, exposed that entry 6 above had made the identical error in the
wrong-anchored pool — see the retraction there.

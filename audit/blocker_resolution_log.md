# audit/blocker_resolution_log.md

Every blocker raised by the three adversarial reviews, with the verification
that closed it. "Verified" means recomputed from the released per-trial logs,
not taken from the reviewer's report.

## Reviewer B — rejection blockers

### M1 — C1R contrast direction (raised, already fixed)
Reviewer B read a superseded build. The C1R reversal had been corrected in
round one. No further action; confirmed against `tab_stats.tex`.

### M2 — "persists at a similar size" — CONFIRMED, FIXED
**Claim:** the perturbed-GSM8K gain was compared against a full-pool original
gain, concealing a collapse.

**Verification:** the perturbed pool is 97 items × 5 replications × 2
conditions, each `gsm8k_perturbed_<n>` derived from a `gsm8k_<n>` we also ran,
so the comparison is matched at the item level.

| Pool | Solo | C4 | Gain |
|---|---:|---:|---:|
| Original, same 97 items | 76.9% | 93.0% | **+16.1 pp** |
| Perturbed | 21.6% | 27.0% | **+5.4 pp** |

The gain retains 33.5% of its size. The paper had compared +5.4 against the
full-pool +6.2/+7.9, which made it look unchanged.

**Fix:** §4.5 retitled from "The math gain is not memorisation" to "How much of
the math gain survives perturbation?". It now reports the matched comparison,
states the two-thirds reduction, and gives both surviving readings —
contamination of the originals, and the items simply being too hard for the
correction mechanism to operate — noting the design cannot separate them.
Guarded by three new registry checks.

### M4 — multiplicity family excluded the failing tests — CONFIRMED, FIXED
**Verification:** collecting all seven paired harmful-revision contrasts stated
anywhere in the paper and applying Holm within that family:

| Contrast | raw p | Holm p | Survives |
|---|---:|---:|:--:|
| GPT-4o-mini C4−C1R | 3.0e-10 | <0.001 | yes |
| GPT-4o-mini C4−C2 | 1.7e-09 | <0.001 | yes |
| DeepSeek C4H−C4 | 1.2e-04 | 0.0006 | yes |
| DeepSeek C4−C1R | 1.8e-04 | 0.0007 | yes |
| DeepSeek C2het−C2 | 0.021 | 0.062 | **no** |
| DeepSeek C4split−C4 | 0.033 | 0.066 | **no** |
| DeepSeek C4−C2 | 0.059 | 0.066 | **no** |

**Fix:** the family is declared in §3.5 and computed in
`verify_paper_numbers.py` (`revision_rate_family`), so it cannot be redrawn
silently. §4.6 now states the C4split revision contrast does not survive and
rests that claim on the accuracy contrast; §4.1 marks the C2het revision rise
descriptive. Registry checks assert the paper says "seven" and "three".

### M5 — capability gradient not baselined — CONFIRMED, FIXED
**Verification:** C2 harmful-flip rates track solo ability on their own
(ρ = −0.45). The wrong-peer-specific excess gives ρ = −0.83, exact p = 0.015,
leave-one-out range [−0.96, −0.75] — negative throughout but less stable than
the raw ρ = −0.95 (LOO [−0.96, −0.93]).

**Fix:** Table 4 carries a C2 baseline column; §4.3 reports the baseline
correlation and the excess, and labels the excess suggestive rather than robust.

### M6 — deployed vs reported filter gap — CONFIRMED, FIXED
**Verification:** of 600 C5H Round-0 peer messages, 22 have unparseable
confidence and **all 22 are wrong**. The deployed rule drops them.

| Accounting | n | Δ_ret |
|---|---:|---:|
| Parsed only (as reported) | 578 | +0.007 |
| Deployed rule | 600 | **+0.072** |

**Fix:** the deployed figure is now the headline one in §4.4 and Appendix S3.
Table 7 is generated from the pooled logs by `make_tables.py` with substrate
labels that match their own counts — the hand-written version said "Phase-1"
over counts covering both phases. Registry check asserts +0.072.

## Reviewer B — improvable items, all applied

- Consensus χ² on non-independent trials → question-clustered bootstrap
  (−0.65 pp, 95% CI [−4.2, +3.4]); "makes no difference" removed.
- Probe statistic named as a difference in differences; no-flip subset reported
  (+0.074, n = 638).
- Contribution 3 scoped: one model, and the most flip-prone in the sweep.
- C2het revision contrast reported with raw and Holm p.
- Review artefacts removed ("the exact point raised in review", "An earlier
  draft of this manuscript…", "has since been run") — these also carried a mild
  anonymity risk.

## Reviewer C — artefact defects, all applied

| Item | Status |
|---|---|
| No `LICENSE` despite README promising MIT | Added |
| `make_figures.py` crashed (`rho['p']`) and resolved `REPO` outside the repo | Fixed; regenerates all three figures |
| `crosscheck_paper_numbers.py` died in a clean clone | Exits 0 with an explanation |
| README pointed at unreleased `Submission/Analyse/` | Points at `analysis/` |
| Python version undocumented while code hard-fails off 3.12 | Documented |
| Table 1 overflowed the margin by 36.7 pt | `adjustbox` + shorter cells; gone |
| `deepseekai2024v3` uncited | Removed (citing it would contradict §3.2) |
| "and 1 others" rendering for Llama 3 / Gemma 3 | Corporate authors |
| S5 correct-anchored counts wrong | **Rejected** — reviewer was mistaken; see `final_red_team_review.md` |

## Regressions introduced and caught in this round

- Adding the new material pushed content onto page 9. Reclaimed by writing
  compression and by removing genuine duplication (§3.4 restated the round
  structure already defined in §3.1), not by formatting tricks. §1–§5 including
  Discussion and Conclusion again end on page 8.
- Table 3 overflowed once its Holm column was added; wrapped in `adjustbox`.
- Rewrapping prose silently disabled 9 of 29 registry checks, which were
  reported as benign "NOT FOUND". Patterns are now matched against
  whitespace-normalised text, and a non-matching pattern is a **failure**.

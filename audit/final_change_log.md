# audit/final_change_log.md

Chronological record of every substantive change made during the final audit
pass, on branch `final-fix-20260801` (baseline preserved in
`audit/before_final_fix/`).

## P0 — confidence-filter semantics (scientific error)

| # | Change | Files |
|---|---|---|
| 1 | Traced the implementation: the filter **retains** peers with confidence ≥ 60 (behaviour A), matching the prose. | `confidence_weighted_protocol.py` |
| 2 | Found the pre-flight gate's sign reversed for a retain-high filter. Replaced `P(loud\|wrong) − P(loud\|correct)` with `Δ_ret = P(retained\|correct) − P(retained\|wrong)`. | `corrected_gate.py`, §3.4, §4.4, App. S3, Table 7 |
| 3 | Declared the AUROC positive class (**correct**). | `corrected_gate.py`, §3.4, Table 7 |
| 4 | Added 4 unit tests: boundary values 0/59/60/61/100/missing; useful direction; harmful direction; single-class stays undefined. All pass. | `tests/test_filter_semantics.py` |
| 5 | Reported measured removal rates: legacy C5 **100%**, C5R **1.0%**, C5H **4.0%**. | §4.4, App. S3 |

## P0 — page limit

| # | Change |
|---|---|
| 6 | Merged Discussion and Conclusion (they restated each other) and compressed. |
| 7 | Compressed §2.2, §3.1, §3.5, §4.2, §4.4, §4.6, §4.7, §4.8; removed the sentence summarising Appendix S11. |
| 8 | **Result: Sections 1–5 including the complete Discussion and Conclusion end on page 8.** No font, margin, spacing or caption-size changes. |

## P0 — condition definitions

| # | Change |
|---|---|
| 9 | Table 1 now lists all twelve conditions including C3H, C5R, C5H. |
| 10 | Legacy C5 explicitly labelled a broken diagnostic (its peers were never asked for a confidence line); excluded from the filter conclusion. |
| 11 | §3.1 control count corrected to seven; §3.3 counts separated into final-analysis vs archived. |
| 12 | Canonical registry created. | `audit/condition_registry.csv` |

## P0 — claim scoping

| # | Change |
|---|---|
| 13 | Natural-vs-instructed comparison rewritten as a **matched observational association**; removed "conservative", "floor", "understates the risk", "more dangerous", "wrong by nature", "real mixed-ability ensemble" from the paper, contributions, Limitations and revision notes. |
| 14 | C4split retitled *"Replacing a wrong peer with a correct one"*; described as a combined replacement estimating neither component separately, and not a test of dissent. |
| 15 | C2het: standard wording — heterogeneous but capability-asymmetric; does not estimate diversity alone. Applied in Results, Limitations and notes. |
| 16 | "mechanistic probe" → "output-distribution probe" everywhere; contribution 3 claims peer-induced movement, not mechanism. |
| 17 | "recompute"/"re-derive" hedged to "recheck"/"consistent with re-derivation"; "correction and conformity are not alternatives" replaced. |
| 18 | Perturbed GSM8K: no longer said to remove contamination; direction-only claim. |
| 19 | "None loses accuracy" → "all eight point estimates non-negative", with the number of intervals covering zero stated. |

## P0 — numerical

| # | Change |
|---|---|
| 20 | Per-model **paired bootstrap CIs** for Δ_solo added to Table 4 (resampling unit = question). Four of eight include zero; the prose now says so. |
| 21 | Revision notes p-values corrected (0.004→0.006, 0.011→0.013), table numbers (5→4), figure references, contribution count (three→four), and the "identical question pool" claim softened to the accurate description. |
| 22 | Build-failing cross-check script added; **20/20 checks pass, exit 0**. | `analysis/crosscheck_paper_numbers.py`, `audit/numerical_crosscheck.csv` |

## P1

| # | Change |
|---|---|
| 23 | Removed the unverifiable claim that correction families were "fixed before the corrected values were inspected"; renamed *causal family* → **primary contrast family**. |
| 24 | Appendix S4: removed the stale Phase-1-only parser statistics; a single final account now lives in S12. |
| 25 | Appendix S8: removed subjects not in our pool ("engineering", "health") and the "sycophancy mechanism" attribution; now a descriptive association. |
| 26 | Appendix S9: "wrong-peer consensus" → "resists two wrong peers"; "illustrates the C→I mechanism" → "illustrates a harmful revision event". |
| 27 | Appendix S10: removed the inference of no style effect from overlapping CIs; states only that no clear pattern is detectable at this precision, with no omnibus test run. |
| 28 | Appendix S11: heading "Conformity signature" → "Harmful-revision rate by condition"; the future heterogeneous control now points at C2het. |
| 29 | Appendix S12: "cross-validation focal agent" → second focal agent evaluated on the full pool. |
| 30 | Appendix S1: removed "confirms that no condition is under-powered"; states realized sample sizes. |
| 31 | Tables 6 and 7 wrapped in `adjustbox` to stop column overlap; verified by rendering pages 12–13. |

## Earlier in the session (pre-audit), recorded for completeness

Perturbed-pool contamination of main-pool means; raw p-values captioned as
Holm-corrected; Spearman ρ computed on rounded values (−0.97 → **−0.95**, exact
permutation p 0.0011); stale Appendix S1 and S10 numbers; persona counts off by
14; gate substituting 0 for an undefined class; parser statistics; broken
appendix cross-references; post-hoc power removed; 12 of 22 bibliography
entries corrected (one had six fabricated co-authors); `.gitignore` excluding
new results; analysis scripts moved out of the gitignored `Submission/` so the
release claim is true; a relocated script silently reading stale data.

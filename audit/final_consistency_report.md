# audit/final_consistency_report.md

State of the submission after two audit rounds and three adversarial reviews.
Every figure below was produced by re-running the released scripts against the
released logs, not copied from an earlier report.

## Verification suite — all green

```
analysis/verify_paper_numbers.py       exit 0   regenerates paper_numbers.json
analysis/make_tables.py                exit 0   5 tables written to Submission/tables/
analysis/make_figures.py               exit 0   3 figures written to Submission/images/
pytest test_filter_semantics.py        4 passed
pdflatex + bibtex                      0 errors, 0 undefined control sequences
analysis/crosscheck_paper_numbers.py   29/29 checks pass, 0 failures, exit 0
```

## Compiled PDF

| Check | Result |
|---|---|
| Content (§1–§5 incl. Discussion and Conclusion) ends by page 8 | **Yes** — Limitations begins page 9 |
| Total pages | 20 |
| Page size | A4 (595.276 × 841.89 pts) |
| Fonts not embedded | 0 |
| Unresolved references / `??` | 0 / 0 |
| PDF metadata (Author, Title, Subject, Keywords) | all empty |
| Identity strings in extracted text | 0 |
| Overfull boxes > 10 pt | 0 (Table 1's 36.7 pt and Table 3's 37.9 pt both fixed) |

## What changed scientifically in this round

Four claims changed, one of them a reversal:

1. **The contamination section no longer claims the math gain is not
   memorisation.** On the matched comparator the gain falls from +16.1 pp to
   +5.4 pp. The section reports both surviving readings and says the design
   cannot separate them.
2. **Three of seven revision-rate contrasts do not survive Holm correction**
   and are now reported as descriptive, including the C4split revision result.
   The C4split claim rests on its accuracy contrast, which survives.
3. **The capability gradient is partly baseline churn.** ρ = −0.45 against the
   homogeneous baseline; the wrong-peer-specific excess is ρ = −0.83, and less
   stable under leave-one-out than the raw −0.95.
4. **The deployed filter gap is +0.072, not +0.007.** All 22 unparseable-
   confidence peers in C5H were wrong, and the deployed rule drops them.

None of these overturns a research question. RQ1 (correction under wrong
peers), RQ2 (capability gradient in harmful flips) and RQ3 (confidence
filtering fails) all stand, with narrower scope and honest intervals.

## Guardrails added, so this cannot silently regress

- The revision-rate family, the matched contamination comparison, the C2
  baseline correlation and the deployed retention gap are all **computed** in
  `verify_paper_numbers.py` and asserted by the cross-check registry. They are
  no longer hand-typed numbers.
- Table 7 is generated rather than hand-written; its old substrate label
  ("Phase-1") did not match its own counts.
- Registry patterns match against whitespace-normalised text, and a pattern
  that no longer matches the manuscript **fails the build**. Previously it was
  reported as a benign "NOT FOUND" — nine of 29 checks had been silently
  disabled by ordinary re-wrapping of prose.

## Status

**READY WITH HUMAN VERIFICATION.**

Ready: the artefact reproduces byte-identically from a clean clone, the
verification suite is green, the PDF clears every ARR desk-rejection category,
and the four blockers are closed against the data.

Human verification still required on two points, neither of which I can settle:

1. **The authors must personally check the changed scientific claims**, in
   particular the contamination reading in §4.5 — the paper now states a result
   that is partly unfavourable to it, and that framing is a judgement call the
   authors own.
2. **The AI-assistance disclosure (Responsible NLP checklist item E1)** is
   filed on OpenReview, not in the PDF, and only the authors can submit it. See
   `ACTION_REQUIRED.md`. I did not write a narrower disclosure than the facts
   support; the factual record is in `audit/ai_assistance_log.md`.

The manuscript source (`Submission/`) is gitignored and not part of the
released artefact, so `crosscheck_paper_numbers.py` is an authors-only tool and
now says so rather than crashing.

# audit/page_limit_report.md

**Requirement.** Sections 1–5, including the complete Discussion and
Conclusion, must end on or before the bottom of printed page 8. Limitations,
Ethical Considerations, references and appendices are excluded.

**Result: PASS.**

| Element | Page |
|---|---|
| Last sentence of §5 Discussion and Conclusion ("…insufficiently discriminative here.") | **8** |
| Limitations begins | 9 |
| Ethical Considerations | 9 |
| References | 10 |

Verified by extracting page 8 and page 9 separately and searching for the final
Discussion sentence, and by rendering both pages to PNG.

## How the space was found — writing compression only

No font size, margin, line spacing or caption size was changed, and no negative
vertical spacing was used.

| Change | Approx. lines |
|---|---:|
| Merged Discussion and Conclusion (they restated each other) and compressed | ~20 |
| Compressed §2.2 error-detection paragraph (dropped restatement of our own results) | ~8 |
| §4.7 probe: removed numbers already shown in Figure 1 | ~5 |
| §4.4 filter results tightened while keeping the corrected substance | ~6 |
| §4.2 natural-error paragraph rewritten shorter as an observational association | ~7 |
| §4.6 rewritten as intervention-specific and shorter | ~7 |
| §3.1 compressed now that Table 1 lists every condition | ~7 |
| §3.5 statistics prose tightened | ~10 |
| §4.8 tightened | ~3 |
| Removed the sentence summarising Appendix S11 | ~4 |

## Material moved to the appendix (secondary diagnostics only)

- Related-work comparison table (Table 5, Appendix S0)
- Capability–corruption scatter (Appendix Figure 2) — the numbers it plots are
  in main-text Table 4
- Corrected pre-flight gate table (Table 7, Appendix S3) — its values are
  quoted in §4.4
- GPT-4o-mini per-condition rows (Appendix Table 6) — the one cross-model
  comparison the text makes is quoted inline

No evidence required to assess a headline claim was moved: the causal
contrasts (Table 3), the per-condition results for the primary focal model
(Table 2) and the capability sweep with intervals (Table 4) all remain in the
main text.

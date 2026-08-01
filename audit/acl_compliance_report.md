# audit/acl_compliance_report.md

Checked against the current ARR/ACL rules (CFP, author guidelines, ACLPUB).

| Requirement | Status | Evidence |
|---|---|---|
| Long-paper content ≤ 8 pages | **PASS** | Discussion+Conclusion ends on p8; Limitations begins p9 |
| References outside the limit | PASS | References start p9 |
| Limitations present, mandatory, after conclusion | PASS | `\section*{Limitations}`, p9, before references |
| Ethical Considerations placed at end | PASS | follows Limitations |
| No Acknowledgements in review version | PASS | block commented out |
| Appendices after references, double-column | PASS | `\appendix` after `\bibliography`; S4 uses `\onecolumn` for verbatim prompts then returns to `\twocolumn` |
| A4 paper size | PASS | 595.276 × 841.89 pt |
| Review mode / line numbers | PASS | `\usepackage[review]{acl}` |
| Anonymous author block | PASS | "Anonymous ACL submission" |
| All fonts embedded | PASS | `pdffonts`: 0 non-embedded |
| Searchable PDF | PASS | `pdftotext` extracts full text |
| PDF metadata anonymous | PASS | Author/Title/Subject/Keywords all empty; Creator "LaTeX with hyperref" |
| No identity strings in PDF | PASS | 0 hits for author names, affiliation, username, `/home/`, `C:\` |
| No non-anonymous URLs | PASS | only `anonymous.4open.science`, anthology/PMLR/NeurIPS/OpenReview/DOI |
| All citations resolve | PASS | 0 undefined citations, 0 undefined references, 0 `??` in PDF |
| No reviewer-directed meta-text | PASS | no TODO/FIXME/"reviewer" asides in body |
| Overfull boxes | **MINOR** | 6 remain (max 38 pt), all inside tables; pages 3–6 rendered and visually inspected — no clipping or text in the margin |

## Commands run

```
latexmk-equivalent: pdflatex ×3 + bibtex   → 0 errors
pdfinfo ACL_Paper.pdf                      → A4, 18 pages, empty metadata
pdffonts ACL_Paper.pdf                     → all embedded
pdftotext ACL_Paper.pdf audit/final_extracted_text.txt
pdftoppm -r 100 -png (pages 3,4,5)         → visual inspection, clean
```

## Not verified here (requires the authors / OpenReview)

- Responsible NLP checklist answers.
- Prior-submission declaration (#1656) on the submission form.
- Reviewer-registration compliance for all authors (deadline 5 Aug 2026).

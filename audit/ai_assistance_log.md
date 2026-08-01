# audit/ai_assistance_log.md

Record of Claude's role, so the authors can answer the ARR Responsible NLP
checklist (item E1) and the ACL AI-assistance policy accurately. The authors
must decide the final disclosure wording; this is a factual record, not a
proposed disclosure.

ACL policy distinguishes assistance that needs **no** disclosure (language
polishing, short-form input assistance) from assistance that **does**
(literature search, low-novelty text generation, new ideas). Several items
below fall in the second category.

## 1. Language polishing — no disclosure required under ACL policy

- Rewording for concision and readability throughout the manuscript.
- Trimming to fit the eight-page content limit.

## 2. Code assistance — disclosure recommended (README-level)

- Rewrote `Code_Phase_2/GPU_Only/src/logprob_probe.py` to batch through vLLM
  per stage rather than one prompt at a time (2,700 trials in 8.5 min instead
  of an estimated ~15 h), and fixed a bug where a fixed sampling seed made all
  replications byte-identical.
- Redesigned the probe's headline metric to use a common reference wrong answer
  across conditions, so C2 forms a usable baseline.
- Wrote `Submission/Analyse/verify_paper_numbers.py`,
  `make_tables.py`, `make_figures.py`; the paper's data tables are generated
  from these.
- Fixed `Code_Phase_2/CPU_Only/src/corrected_gate.py` to leave an undefined
  single-class gap undefined instead of substituting zero.
- Fixed `Code_Phase_2/.gitignore`, which was silently excluding new result
  files from the released artefact.
- Operational scripts: `scripts/autopush.sh`, `scripts/run_status.py`,
  `scripts/check_balances.sh`, VM bootstrap scripts.

## 3. Experiment execution — disclosure recommended

Claude ran, monitored and collected:

- C4split (split-peer), 1,500 trials.
- C2het (heterogeneous debate), 900 trials.
- The GPU output-distribution probe, 2,700 trials, on a rented Vast.ai RTX 4090
  that was destroyed after results were downloaded.

Experimental *design* for these three came from the authors'
`Improvement_Plan.md`; Claude implemented, executed and audited them.

## 4. Statistical auditing — **must be disclosed**

Claude recomputed every reported number from the raw logs and found fourteen
discrepancies, including four rated critical or high (perturbed-pool
contamination of main-pool means; raw p-values captioned as Holm-corrected;
a Spearman ρ inflated by rounding; a spurious "passed" gate verdict from
substituting zero for an undefined quantity). All are listed in
`audit/numerical_consistency_report.md`.

Claude also ran analyses that were not in the original plan: the answer-level
consensus test within C4, leave-one-model-out and exact-permutation sensitivity
for the capability correlation, and parse-failure bounds.

## 5. Proposed scientific framing — **must be disclosed**

Claude proposed, and drafted text for, interpretations that are substantive
rather than editorial:

- That a **visible correct peer**, rather than broken unanimity, explains the
  split-peer result — and correspondingly that the earlier "unanimity drives
  the harm" framing (also Claude-drafted) was unsupported and had to be
  withdrawn.
- That naturally wrong peers are associated with more harmful revision than
  instructed ones, together with the stratified check and the selection caveat.
- That the heterogeneous control varies diversity and peer strength together
  and therefore cannot isolate diversity.
- Reframing the probe as an output-distribution probe rather than a mechanism.

## 6. Substantive generated text — **must be disclosed**

Claude drafted most of the current wording of: the abstract, contributions,
§2.2 error-detection related-work paragraph, §4.2 natural-vs-instructed
paragraph, §4.6 (answer consensus and split peers), §4.7 (probe), the merged
Discussion and Conclusion, several Limitations paragraphs, and
`Revision_Notes.tex`.

Bibliographic verification was performed against primary sources (ACL
Anthology, PMLR, NeurIPS proceedings, OpenReview, Crossref); twelve entries
were corrected, one of which had six fabricated co-authors in the prior
version.

## 7. What Claude did **not** do

- Did not design the original study, conditions, or Phase-1/Phase-2 experiments.
- Did not fabricate any result, citation, count, or interval; every number in
  the manuscript regenerates from released logs.
- Did not submit anything to OpenReview.
- Did not train or fine-tune any model.

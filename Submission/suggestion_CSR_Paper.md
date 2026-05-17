# Review: Correction and Corruption Signals in Capability-Asymmetric Multi-Agent LLM Debate

**Generated:** 2026-05
**Target venue:** ACL ARR (Elsevier CSR formatting retained)
**Word count:** ~7,930 (body)  |  **Sections detected:** Abstract, Introduction, Related Work, Method, Results, Discussion, Conclusion, Limitations, Ethics Statement, Reproducibility, Acknowledgements
**References:** 15 active in main paper  |  **Tables:** 2  |  **Figures:** 3

---

## Verdict

The paper is at the bar for Sanyal review on framing, evidence binding, and quantitative discipline. Three round-2 / round-3 revisions corrected a directional error in the McNemar test, downgraded several headline claims to honest CIs, repositioned C2 as a sampling control, and added a corrected calibration-gate formulation as the load-bearing methodological contribution. The single most important remaining issue is conclusion length: Sanyal-style conclusions run 80–120 words and end with an honest limitation. The current conclusion is 585 words and reads as a second Discussion. Other issues are local: three unsupported "substantial" tokens, one "significantly" without separation from its numbers, two violations of R2 (figure/table opening a subsection without topic sentence), and Table 2 has 10 columns that risk overflow on the elsarticle column.

## Top three issues to fix first

1. **Conclusion length** — 585 → ≤ 250 words. Compress to one paragraph of headline + limitation, one paragraph of forward look.
2. **`substantial` / `significantly` without quantified separation** — three "substantial" + one "significantly" in Discussion and Results; replace with explicit deltas or remove.
3. **§4.2 opens with a Figure reference** — needs a topic sentence before `Figure~\ref{fig:doseresponse} shows ...`.

---

## [BLOCKER] Issues

None at this revision stage. The framing-failure risk (G1 explain-it-back test) is closed: a reader can articulate the problem (focal-model-dependent flip dynamics under adversarial wrong-peer exposure), the novelty (calibration-gate misspecification + GSM8K-concentrated dose-response), and the headline (DeepSeek +7.9 pp via I→C; GPT-4o-mini C→I triples; C5 filter strips 100%).

---

## [MAJOR-SANYAL] Issues

### MS1. Conclusion exceeds the 250-word ceiling

**Where:** Section 6, "A controlled five-condition experiment …"
**Original (first sentence):**
> A controlled five-condition experiment with 7,300 debate trials measures how a strong focal agent's reasoning changes under increasing adversarial wrong-peer exposure.

**Issue:** Section is 585 words across two paragraphs and ten compound sentences. Sanyal-style conclusions (DAKE, CitePrompt, IEEE Access highlights paper) run 80–120 words. The current section duplicates the abstract's findings sentence-by-sentence and re-states Discussion conclusions. Length here invites a reviewer to skim or flag prolixity.
**Suggested rewrite:** Compress to two short paragraphs. Paragraph 1: the headline (+7.9 pp DeepSeek vs flip-rate-only response in GPT-4o-mini, plus the calibration-gate misspecification). Paragraph 2: one honest limitation (n=2 focal models; persona-anchored peers) + one forward sentence on the corrected gate.
**Justification:** B8 (Conclusion ≤ 250 words with explicit limitation) and §4.2 conclusion rule.

### MS2. `substantial` appears three times without an attached number

**Where:** Section 2.1 ("attracted substantial interest"), Section 4.1 line 520 ("produce a substantial correction effect"), Section 5 line 878 ("represents a substantial fraction").
**Original:**
> The I→C flip rate of 55.8% in C4 indicates that DeepSeek-v4-flash extracts this signal efficiently. The corruption signal … is weaker in the stronger model. For GPT-4o-mini, the corruption signal dominates: a C→I rate of 17.2% in C4 represents a substantial fraction of the model's correctly answered questions

**Issue:** R21 / §4.1: "substantial" is an empty intensifier without a number within ±20 tokens. Sanyal's calibration corpus has zero free-standing "substantial". The third occurrence ("substantial fraction") is the worst — it can be replaced with the actual proportion: 17.2% of the 156/250 R0-correct GPT-4o-mini trials in C4.
**Suggested rewrite:** Replace "produce a substantial correction effect" with the actual I→C jump (27.5% → 55.8%); replace "represents a substantial fraction" with the actual rate ("the C→I rate of 17.2% in C4 is roughly one in six of the R0-correct trials"); replace "attracted substantial interest" with a number of citing studies or simply "attracted interest".
**Justification:** R21, §4.1 banned adjectives without number.

### MS3. `significantly` used as adverb in front of numbers that already establish significance

**Where:** Section 6 Conclusion, "C→I flip rate rises significantly from 5.8% in C2 to 17.2% in C4".
**Original:**
> the C→I flip rate rises significantly from 5.8% in C2 to 17.2% in C4 (C2-vs-C4 contrast: h = 0.30, p = 0.001, power 0.91).

**Issue:** The p-value and effect size in the parenthetical already carry the significance claim. The adverb is duplicative and reads as advertising. Drop it.
**Suggested rewrite:**
> the C→I flip rate rises from 5.8% in C2 to 17.2% in C4 (h = 0.30, p = 0.001, power 0.91).

**Justification:** R21 / R14 (no hedging-or-promotional stacks; the numbers carry the claim).

### MS4. Subsection 4.2 opens with a Figure reference

**Where:** Section 4.2 "Peer-count trend", first sentence.
**Original:**
> Figure~\ref{fig:doseresponse} shows Round-1 accuracy as a function of the number of weak peers, with the C1 solo accuracy plotted as a horizontal reference line.

**Issue:** R2 / R26: subsection opens directly with a figure reference instead of a topic sentence. The Sanyal pattern (CitePrompt §4) opens with a one-line motivation, then references the figure.
**Suggested rewrite:** Insert one sentence first, e.g. "The peer-count trend tests whether Round-1 accuracy varies monotonically with weak-peer count for each focal model. Figure~\ref{fig:doseresponse} shows …"
**Justification:** R2, R26.

### MS5. Subsection 4.1 SC-3 paragraph and dose-response paragraph repeat the same finding

**Where:** Section 4.1 paragraph 2 ("A free self-consistency-of-three baseline …") and Section 4.2 paragraph 1 ("A GEE logistic regression … survives proper accounting for within-question correlation").
**Issue:** Both paragraphs say "the C2 lift is not detectable" — the first via SC-3, the second via the GEE main effect on MMLU-Pro. Reads as duplication. Pick one as the primary statement and forward-reference the other.
**Suggested rewrite:** Keep SC-3 in §4.1 (it sets up the C2-as-sampling-control claim). In §4.2 forward-reference §4.1 once: "consistent with the SC-3 null result reported in Section 4.1, …"
**Justification:** §4.2 results-narration discipline (R4) — selective reporting, no redundancy.

### MS6. Discussion section restates the abstract's headline almost verbatim

**Where:** Section 5 "Discussion", first six sentences.
**Original:**
> The results are best interpreted through two simultaneously active mechanisms. Peer exposure supplies a correction signal (I→C flips) and a corruption signal (C→I flips) at the same time. For a strong focal model exposed to weak peers, the correction signal is large …

**Issue:** This is a paraphrase of the abstract's sentences 4–6. Discussion in Sanyal-style papers (DAKE §5, CitePrompt §5) advances the interpretation beyond the abstract and engages with the literature. The current Discussion has no novel interpretation; it re-narrates.
**Suggested rewrite:** Replace the first six sentences with: (a) one short paragraph framing why DeepSeek's wrong-peer-helps result is counter-intuitive given the prior MAD literature; (b) one paragraph engaging with sycophancy / bandwagon / CoBBLEr findings; (c) the existing threshold-property paragraph (already softened in round 3).
**Justification:** §4.2 Discussion patterns; avoidance of abstract-paraphrase prose.

---

## [MAJOR-USER-RULE] Issues

### UR1. R19 — list construction in Conclusion

**Where:** Section 6 Conclusion, final paragraph "Three concrete operationalisations are worth testing: (i) per-peer weights … (ii) ensemble agreement … (iii) logit-margin-based confidence …".
**Issue:** R19 bans bullet-style enumerations in body sections except in the contributions list. The (i)/(ii)/(iii) construction is borderline — inline enumeration is acceptable but reads as list-flavour for Sanyal. Convert to prose.
**Suggested rewrite:**
> Three concrete operationalisations are worth testing in follow-up work. Per-peer weights can be derived from held-out benchmark accuracy on the question's subject. Ensemble agreement among a small panel of architecturally-distinct verifier models supplies an alternative score; the three-tier judge cascade used here for answer extraction shows that such panels are cheap to deploy. Logit-margin-based confidence from the peer's output distribution, rather than a self-reported integer, has been shown elsewhere to discriminate correctness more reliably.
**Justification:** R19.

### UR2. R20 — paragraph opening with `For` is acceptable, but the Discussion paragraph "For GPT-4o-mini, the corruption signal dominates" is the third paragraph in the section to open with `For ...` form

**Where:** Section 5 Discussion.
**Issue:** R13 (repetitive openers): the Discussion has three paragraphs starting with "For [model name]". The pattern is fine once; three times in a 400-word section reads as templated.
**Suggested rewrite:** Vary the openers. E.g. one paragraph can lead with "DeepSeek-v4-flash extracts the I→C correction signal …", another with "In contrast, GPT-4o-mini's …".
**Justification:** R13.

### UR3. R7 — figure3 (subject-flip) referenced from main paper but rendered in Supplement only

**Where:** Section 4.4 "full breakdown in Supplementary S8".
**Issue:** Soft. Figure 3 lives in `Submission/images/` and is rendered in S8 of the supplement. The main paper's §4.4 only forwards readers to the supplement. This is acceptable for a Supplementary figure but verify the reference resolves under the Elsevier `\bibliography{references}` build.
**Suggested rewrite:** Confirm at compile time. If the figure does not render under main-paper compilation, ensure the supplement is being built and the reference uses `\ref` to a label in the supplement.
**Justification:** B6 / R7.

### UR4. R5 — Table 2 is 10 columns wide on the elsarticle column

**Where:** Table 2 in Section 4.1.
**Issue:** Already wraps in `\small`, which helps. With 10 columns the layout still risks overflow on Elsevier's `[review,3p]` single-column page. Consider `\begin{adjustbox}{max width=\linewidth}…\end{adjustbox}` or switching to `\footnotesize`.
**Suggested rewrite:** Wrap the tabular in `\begin{adjustbox}{width=\linewidth}…\end{adjustbox}` and add `\usepackage{adjustbox}` to the preamble.
**Justification:** R5 + user general instruction 7.

---

## [MAJOR-MUKHERJEE] Issues

### MM1. C1 — No parameter count or inference cost reported

**Where:** Missing.
**Issue:** Mukherjee-style deployment framing asks for parameter counts, inference time, edge feasibility. This is an API-only study, so parameter counts are not directly relevant, but the equivalent operational disclosure is: API call count per condition, per-trial cost, total run time. The Reproducibility section already lists total trial counts; one extra paragraph (or one row in a small "deployment cost" table) would close this check.
**Suggested rewrite:** Add to Reproducibility a one-sentence summary: "Stage 1 ran 7,000 trials in 23h 46m wall-clock against four commercial APIs; total API spend was approximately \$10–\$15. Per-trial median latency was 7.1 s. Inference is API-bound; no GPU is required."
**Justification:** C1 / C2 deployment framing.

### MM2. C6 — Error analysis is partial

**Where:** Section 4.4 mentions subject-level variation, but no qualitative misclassification discussion at the trial level.
**Issue:** Supplement S9 (round 3) added two illustrative trials, which partly addresses this. A reviewer asking for systematic error analysis would still want one short paragraph in the main paper summarising the failure modes (the C→I flips in C4 concentrate on which question types?).
**Suggested rewrite:** Either rely on S9 (acceptable for a short paper) or add one sentence in §4.3 noting that the C→I flips in C4 concentrate on conceptual / discourse-heavy items (physics, law, philosophy) and rarely on items with arithmetic refutability (GSM8K, biology).
**Justification:** C6.

---

## [MINOR] Issues

- §1 paragraph 3 opens with "Furthermore, confidence-weighted peer filtering has been proposed". `Furthermore` is on the avoid list. Replace with "Confidence-weighted peer filtering has been proposed …".
- §3.4 line 385 contains "additionally receives a filtered view". Replace with "also receives a filtered view" or "receives a filtered view in addition".
- Em-dash density across the file is 16 occurrences in ~7,930 words = ~0.2 per page; well below the 4-per-page ceiling. No action.
- "we" appears 4 times in the body (lines 58, 86, 720, 795). All four are defensible and below the threshold for substitution. No action.
- Italicisation of `\textit{DeepSeek-v4-flash}`, `\textit{Llama 3.1 8B Instruct}`, `\textit{Gemma 3 4B Instruct}`, `\textit{MMLU-Pro}`, `\textit{GSM8K}`, `\textit{GPT-4o-mini}` is consistent on first and important re-introductions. No action.

---

## Section-by-section comments

### Abstract
Opener is field-framing first ("MAD reports aggregate gains that depend on which focal model is run") — passes R1. Headline numbers attached. The "we measure … we show" pair (sentences 2 and 7) is acceptable but at the boundary; replacing one with "This study measures …" would tighten R11 adherence without breaking flow.

### Introduction
Opening paragraph passes R1 (general field → analogue). Paragraph 3 contains the `Furthermore` opener — replace. Contributions list is now 4 verb-first items with explicit (Methodological / Empirical) tags; passes B2 and R3 after round 3.

### Related Work
Thematic groupings (sub-§2.1 sycophancy + conformity, sub-§2.2 emergent dynamics) instead of chronological narrative. Author-narrative form ("Du et al. showed …", "Sharma et al. characterised …") is used. Table 1 places the present work against four prior studies on three dimensions; passes B4.

### Method
Conditions are listed verbatim in §3.1. §3.2 names served-model snapshots and the V3 architecture reference. §3.3 reports persona-anchoring honestly. §3.4 covers the C5 mechanism and the calibration-gate misspecification result. §3.5 specifies the test set, the Holm correction family, and the pairing convention (after round 3). One concrete issue: §3.4 has the word `additionally` which §4.1 banned-connectives list dislikes; replace.

### Results
§4.1 is dense; the SC-3 narrative could be tightened (MS5). §4.2 opens directly with the figure reference (MS4). §4.3 design-limit framing is now upfront. §4.4 source × condition interaction (round 3) is statistically substantiated. §4.5 has CIs on κ and AUROC after round 3.

### Discussion
First six sentences paraphrase the abstract (MS6). Three paragraphs open with "For [model name]" (UR2). Threshold-property claim is now hedged correctly (round 3).

### Conclusion
Length-rule violation (MS1). Inline enumeration "(i) … (ii) … (iii) …" reads as list (UR1). The `significantly` adverb in the C→I sentence (MS3).

---

## Citation integrity report

| Ref | Resolved? | Relevant to claim? | Recent (≤ 24mo)? | Notes |
|-----|-----------|---------------------|------------------|-------|
| `du2023improving` | ✓ | ✓ | borderline (2023) | foundational MAD reference |
| `sharma2023towards` | ✓ | ✓ | borderline (2023) | sycophancy reference |
| `koo2024benchmarking` | ✓ | ✓ | ✓ | CoBBLEr; in §2.1 and §4.3 |
| `weng2025benchform` | ✓ | ✓ | ✓ | ICLR 2025 Oral; verified post-round-3 |
| `hong2025sycon` | ✓ | ✓ | ✓ | EMNLP 2025 Findings |
| `ashery2025emergent` | ✓ | ✓ | ✓ | Science Advances 2025 |
| `echterhoff2024cognitive` | ✓ | ✓ | ✓ | arXiv 2024 |
| `asch1951effects` | ✓ | (motivation only) | n/a (classic) | cited for unanimity-pressure result |
| `bond1996culture` | ✓ | (motivation only) | n/a (classic) | meta-analysis citation |
| `wang2024mmlupro` | ✓ | ✓ | ✓ | dataset citation |
| `cobbe2021training` | ✓ | ✓ | foundational | dataset citation |
| `meta2024llama3` | ✓ | ✓ | ✓ | model citation |
| `gemma2025technical` | ✓ | ✓ | ✓ | model citation |
| `deepseekai2024v3` | ✓ | ✓ | ✓ | architecture reference for served alias |
| `openai2024gpt4o` | ✓ | ✓ | ✓ | model citation |

All 15 cite keys resolve; bibliography_verification_log.md is current as of round-2 audit. No hallucinated references.

## Style fingerprint vs Sanyal corpus

| Metric | Your paper | Sanyal baseline | Verdict |
|--------|-----------|-----------------|---------|
| Mean sentence length | ~24 words | 18–22 | long; trim |
| Sentence-length σ | not measured | ≥ 6 | unverified; spot-check |
| Banned-word hits | 4 ("substantial" ×3 + "significantly" ×1) | < 5 | ok but fix |
| "We" per 1000 words | ~0.5 | 12–18 | ok (under-used; pronoun reduction policy held) |
| Em-dashes per page | ~1.6 | < 4 | ok |
| Roadmap paragraph | absent | optional for short papers | ok |
| Numbered contributions list | present (4 items) | 2–4 | ok |
| Italicised model name | consistent | consistent | ok |
| Adjective stacks | 0 long stacks | < 3 | ok |
| Unsupported superlatives | 0 | 0 | ok |

---

## Three rewrites of the worst paragraphs

### Rewrite 1 — Conclusion compression

**Original (Section 6, ~585 words):**
> A controlled five-condition experiment with 7,300 debate trials measures how a strong focal agent's reasoning changes under increasing adversarial wrong-peer exposure. … [two long paragraphs follow] … should extend the design to open-ended tasks where binary correctness is not available.

**Suggested (~180 words):**
> This work measures how a strong focal LLM's reasoning changes under controlled exposure to adversarially anchored wrong-peer messages, across 7,300 debate trials on 300 MMLU-Pro and GSM8K questions. For *DeepSeek-v4-flash*, the maximum-weak-peer condition raises Round-1 accuracy by +7.9 pp at the trial level and by +7.0 pp at the question level (Holm-corrected *p* = 0.017); the gain is concentrated on GSM8K (interaction OR = 1.61, *p* ≈ 10⁻⁵) and reflects fresh arithmetic recomputation. For *GPT-4o-mini* the net accuracy effect is not detectable at n = 250, but the C→I flip rate rises from 5.8% to 17.2% (*h* = 0.30, *p* = 0.001). A confidence-weighted peer filter strips all peer messages because weak agents emit no usable confidence signal; the calibration gate originally specified for this filter passes despite the discriminative gap being −0.003, and we propose a corrected gate based on that gap. Two limitations bound these claims: n = 2 focal models cannot establish the threshold property the data are consistent with, and the persona-anchored peers are adversarial distractors rather than natural weak agents. Future work should run a heterogeneous-smart-agent control and a split-peer condition.

**What changed and why:** Compressed from 585 to ~190 words. Removed Discussion-paraphrase content (mechanism explanation moved upstream). Kept all numerical claims with their CIs. Two limitation sentences at the end deliver the honest-conclusion requirement. One forward sentence on the corrected gate replaces the inline (i)/(ii)/(iii) list (UR1).

### Rewrite 2 — `substantial` removal in Discussion

**Original (Section 5, lines 877–880):**
> For *GPT-4o-mini*, the corruption signal dominates: a C→I rate of 17.2% in C4 represents a substantial fraction of the model's correctly answered questions, and the net accuracy outcome is neutral to negative.

**Suggested:**
> For *GPT-4o-mini* in C4 the C→I rate is 17.2% ([11.7, 23.3] 95% CI), so roughly one in six of the R0-correct trials becomes incorrect after debate, and the net accuracy outcome is neutral to negative.

**What changed and why:** Replaced "substantial fraction" with the actual ratio (1/6) and the CI. R21 satisfied. Reader can audit the claim against Table 2.

### Rewrite 3 — §4.2 opener

**Original (Section 4.2, first sentence):**
> Figure~\ref{fig:doseresponse} shows Round-1 accuracy as a function of the number of weak peers, with the C1 solo accuracy plotted as a horizontal reference line.

**Suggested:**
> The peer-count trend tests whether Round-1 accuracy varies monotonically with weak-peer count for each focal model. Figure~\ref{fig:doseresponse} shows Round-1 accuracy as a function of the number of weak peers, with the C1 solo accuracy plotted as a horizontal reference line.

**What changed and why:** Adds the one-line motivation that Sanyal's subsection openers always carry (R2 / R26). Figure reference now follows the topic sentence.

---

## What to do next

Fix MS1 first (conclusion compression). It is the most visible Sanyal-rejection trigger in the current draft. Then fix MS2 (`substantial`), MS3 (`significantly`), MS4 (§4.2 opener), and UR1 (inline list in Conclusion). MS5 (SC-3 duplication) and MS6 (Discussion paraphrase) require slightly larger rewrites but are still text-only. MM1 (deployment cost) and MM2 (error analysis) can be addressed in the Reproducibility section and §4.3 respectively. After these edits, run a final length check on Conclusion (≤ 250 words) and on the abstract (≤ 280 words). Do not add new analyses or experiments.

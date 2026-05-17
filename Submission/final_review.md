# Final Review — Elevating the CSR Paper to ACL ARR Definite-Accept Territory

> Adversarial review of `CSR_Paper.tex` against expected ACL ARR reviewer attacks, with mapped fixes. Constraint: no new experiments unless absolutely crucial.

---

## Section 1 — The attack surface

Grouped by what kind of damage each attack does. ACL reviewers are NLP-trained, statistically literate, and care about novelty, reproducibility, and mechanism.

### 1.1 Framing attacks (these alone can sink it)

**A1. The headline contradicts itself.** Abstract says "correction signal dominates for DeepSeek, corruption signal dominates for GPT-4o-mini." A reviewer will write: *"Your title implies a general account of asymmetric debate, but your two models show opposite-sign effects. With n=2 focal models you have an anecdote, not a theory."* This is the single most dangerous attack — it threatens the entire thesis.

**A2. Trial-level vs question-level metric inversion.** §4.1 reports +7.9 pp trial-level gain *and* a significant McNemar regression (p=0.0025, 33 regressed vs 12 improved) on the same condition. The paper picks trial-level as "primary." A reviewer will write: *"The authors choose the aggregation that gives them a positive headline, then bury the contradicting result. This is selective reporting."*

**A3. C5 is post-hoc rationalized.** §3.4 reframes C5 as "confirmation of confidence miscalibration, not a mitigation evaluation." The calibration gate predicted failure and you ran it anyway. Reviewer: *"Either C5 was a real mitigation test (in which case it failed and the design is flawed) or it was a calibration confirmation (in which case you needed only the 1,500 C3/C4 dumb-agent confidences, not a 300-trial C5 condition). Pick one and own the consequences."*

**A4. Unanimity confound, explicitly admitted but unaddressed.** §4.3 lines 491–497 concede that the bandwagon claim cannot be isolated from generic distractor pressure. Reviewer: *"The authors title their bias-theoretic account around bandwagon/Asch but then admit the design cannot test it. The Asch citations are decorative."*

**A5. Smart-agent count is also varying.** C2=3S, C3=2S, C4=1S. Reviewer: *"C2-to-C4 conflates 'adding dumb peers' with 'removing smart peers.' Without a 1-smart-only baseline (≠ C1, which has no debate), you cannot attribute the effect to weak-peer presence."*

### 1.2 Novelty attacks

**A6. What's new vs. CoBBLEr, BenchForm, SYCON-Bench?** Table 1 claims "het. capability + bias mech. + cross-model" as the unique combination, but a reviewer will press: *"BenchForm varies group size; SYCON-Bench is multi-turn; CoBBLEr is multi-model. Your delta is 'I varied dumb-peer count from 1 to 2.' That's two data points."*

**A7. Confidence-weighted filtering failing is not new.** Sycophancy + miscalibration literature (Sharma 2023, Tian 2023, Xiong 2024 on self-reported confidence) already shows this. Reviewer: *"The 99.5% finding restates known miscalibration. What's the new measurement?"*

### 1.3 Statistical attacks

**A8. Per-question observations are nested.** Each question gets 5 replications; trials within a question are not independent. The logistic regression in §4.2 (n=4,500) treats them as independent. Reviewer: *"You need question-level random effects. Your standard errors are underestimated, your p-values are anti-conservative, and your 'p=1.68e-4' may be 10× too small."*

**A9. Underpowered claims dressed as findings.** §4.1 line 374: "two-proportion test on C2 vs C4 C→I rates gives p≈0.10, … reported as a trend." With n=1,500 per arm and rates 5.0% vs 6.5%, you have ~25% power for that test. Reviewer: *"You report a 'monotone rise' as a substantive finding but your own test is non-significant. This is a misuse of trend language."*

**A10. Bonferroni count is unclear.** §3.5 says "over five comparisons (α_adj = 0.01)." But the comparisons aren't listed. Six paired comparisons are mentioned elsewhere in repo docs. Reviewer: *"Which five? Why not all C×C pairs? Why not Holm-Bonferroni, which is uniformly more powerful?"*

**A11. No effect sizes.** No Cohen's h for proportion differences, no odds ratios for logistic regression. Reviewer: *"+0.174 is meaningless without the odds-ratio interpretation. Is this a 1.19× OR? That's clinically tiny."*

**A12. No I→C flip in Table 2.** §4.1 mentions 27.5% → 55.8% I→C jump for DeepSeek but only shows C→I in the table. Reviewer: *"You omit the flip direction that drives your headline. Why is the column missing from the table?"*

**A13. Sample-size imbalance unaddressed.** DeepSeek n=1,500/condition; GPT-4o-mini n=250/condition. The "Asch index 11.5/17.2%" claims for GPT-4o-mini have CIs ~±5 pp. Reviewer: *"You compare two models with 6× different statistical power and report tight numerical contrasts. The CIs overlap heavily."*

### 1.4 Methodology attacks

**A14. No self-consistency baseline.** §5 line 553 lists this as Limitation #1, calling it "the most important follow-up." Reviewer: *"If you know this is the most important comparison, why isn't it the headline contribution? Without it, C2's +2.5 pp is potentially just sample aggregation."* **This one stings because the paper itself admits it.**

**A15. Smart non-focal agents in C2/C3 are also DeepSeek.** Limitation #2 admits this. Reviewer: *"You sample the same model three times and call it 'debate.' How is this different from temperature-0.7 self-consistency with explicit critique prompts?"*

**A16. Data contamination.** §5 Limitation #3 handwaves "constant offset." Reviewer: *"Constant offsets don't apply if contamination differs by question. Your 76.2% C1 baseline could be inflated 10 pp on some subjects. You need a held-out contamination probe — at minimum, a quick analysis of MMLU-Pro overlap with DeepSeek-V3 training data, which is unknown."*

**A17. Persona description ↔ code mismatch.** §3.3 names personas *analytical/sceptical/diplomatic/assertive*. The repo's `experiment.yaml` lists *surface_keyword_match/false_analogy/overconfident_assertion/misapplied_rule*. A reviewer who checks the artefact will catch this. Reviewer: *"Either the paper or the code is wrong. The personas in your code are wrong-reasoning styles; the personas in your paper are rhetorical tones. Which were actually used?"*

**A18. Dataset count mismatch.** §3.3 says "250 MMLU-Pro + 50 GSM8K." Repo says "200 + 100." Same problem as A17 — reviewer will check. **Both A17 and A18 will be caught if the reviewer opens the GitHub repo.**

**A19. `deepseek-chat` is an alias.** The model string in §3.2 is `deepseek-chat`, which routes to the current default — not a fixed snapshot. Reviewer: *"Your experiment is not reproducible against a fixed model. Specify the snapshot date and the actual served model ID logged in `experiment_metadata.json`."*

**A20. Judge cascade reliability not characterized for the main result.** Supplement S4 says judges agree 98.3% of the time, but the regex-vs-judge split for the C→I flip determinations is not given. Reviewer: *"If 5% of C→I flips are judge-extracted and the judge is biased toward extracting a 'committed answer' from ambiguous text, your flip rates inherit that bias."*

### 1.5 Reproducibility attacks (ACL takes this very seriously)

**A21. Prompts are TODOs in Supplement S4.** The verbatim Round-0/Round-1/C5 prompts are placeholders. *Hard fail* against ACL reproducibility checklist.

**A22. Data availability is "upon acceptance."** ACL ARR's reproducibility checklist increasingly penalizes this. Reviewer: *"Anonymized release at submission time is the community norm; 'upon acceptance' suggests the authors don't trust their own results."*

**A23. Random seed propagation not documented.** Paper mentions a master seed; the per-trial derivation rule isn't stated. Reviewer: *"How are persona variants and peer-ordering seeds derived? A reproducibility checklist needs explicit statements."*

### 1.6 Scope/generalization attacks

**A24. Only multiple-choice + grade-school math.** Reviewer: *"All findings are gated by the binary-correctness operationalization. Sycophancy in open-ended generation is the more common deployment setting."*

**A25. Two dumb models, same scale (4B/8B).** Reviewer: *"You measure 'capability asymmetry' with one open-weights size class. Where's the gradient — what happens with Phi-3.5, Qwen-7B, or a 70B model fine-tuned for evaluation?"*

**A26. English only.** Standard ACL attack. *"Multilingual MAD has different sycophancy patterns (Hong 2025)."*

### 1.7 Mechanism attacks

**A27. No mechanistic verification.** Flip rates are behavioral. Reviewer: *"You claim 'sycophancy' and 'bandwagon' but provide no representational evidence — attention to peer text, embedding drift between R0 and R1, no probing classifier. The mechanistic claims are speculative."*

**A28. Persona effect not isolated from peer-answer effect.** When a dumb agent says "the answer is X, because [false analogy]," the focal agent's R1 could change due to the *answer* X or the *reasoning*. Reviewer: *"You don't ablate reasoning vs answer. A condition with bare wrong answers (no reasoning) would isolate this — but it's missing."*

### 1.8 Practical/positioning attacks

**A29. What should an NLP practitioner do differently?** The conclusion says "weight by verifiable accuracy, not self-reported confidence" — but this is the prescription the gate already showed. Reviewer: *"Your actionable recommendation is the trivial corollary of your null result. Where is the constructive contribution?"*

**A30. Why ACL and not Cognitive Systems Research?** ACL reviewers will smell a re-target. The cog-sci framing (Plato, Asch, Bond) is heavy. Reviewer: *"The methodological core is fine but the framing is non-NLP. Cut the 'ship of state' framing and lead with the deployment risk for agentic LLM systems."*

---

## Section 2 — Fixes that require zero new model calls

These are the high-leverage moves. Most use data you already have in `trial_log.parquet` and `final_answers.parquet`.

### 2.1 Statistical re-analysis (kills A8, A10, A11, A12, A13)

1. **Mixed-effects logistic regression** of `round_one_answer_was_correct` on `condition_dumb_agent_count` with `(1 | question_identifier)` random intercept. Use `statsmodels.MixedLM` or `pymer4`. Re-run the existing dose-response regression — your effective n is closer to 300 questions, not 4,500 trials. Report OR + 95% CI, not raw coefficient.
2. **Replace question-level majority-vote McNemar** with a **trial-level GEE / clustered McNemar** (e.g., Durkalski–Palesch extension for clustered binary outcomes). This eliminates the trial-vs-question inversion in A2 cleanly.
3. **Add I→C flip rate to Table 2** (existing data in `final_answers.parquet`). Add a **net flip score** column `(I→C × R0_incorrect_rate) − (C→I × R0_correct_rate)` and verify it matches the observed accuracy delta. This makes the "correction dominates" claim numerically auditable.
4. **Effect sizes everywhere**: Cohen's h for all proportion comparisons; odds ratios for regressions.
5. **Holm-Bonferroni** instead of plain Bonferroni. State the comparison set explicitly: list the 5 (or 6) tests by name.
6. **Power analysis post-hoc for the 'monotone rise' claim**: with 1,500 trials and rates 5.0%/6.5%, you have X% power for a two-proportion z-test at α=0.05. Report it. Then either reframe ("the trend is in the predicted direction but underpowered for detection") or pool C3+C4 vs C2 to gain power.

### 2.2 Free baselines you can derive from existing data (kills A14 partially, A15)

7. **Self-consistency baseline from C2's R0 answers.** You have three independent DeepSeek R0 answers per question in C2 (before any debate). Compute the **majority-vote-of-3** accuracy. This is a fair self-consistency control — *no new experiments*. Compare it to C2's R1 accuracy: if SC≈R1, then "debate" adds nothing beyond sampling. If R1>SC, you have isolated genuine revision benefit. **This single analysis converts your most-admitted limitation into a positive contribution.**
8. **Solo-3-trial control from C1.** You ran C1 with 5 replications per question. Use 3 of them for a `C1-majority-vote-of-3` baseline. Compare to C2's SC-of-3 above — if they match, then C2's R0 itself is just sampling.

### 2.3 Subgroup analyses (kills A16 partially; converts A24 into a positive)

9. **Difficulty stratification.** `question_pool.parquet` has `difficulty_stratum`. Recompute every metric separately for `probe_correct` vs `probe_incorrect` items. Hypothesis: corruption signal dominates on `probe_incorrect` (hard) items, correction signal dominates on `probe_correct` (easy) items. **If this holds, it's a paper-worthy finding on its own.**
10. **Subject-domain breakdown** (currently buried in Supplement S8). If law/health show higher flip than math/physics, that's a structural finding about which task types tolerate MAD. Promote to main paper. (Be careful — multiple-testing correction over 10 subjects.)
11. **MMLU-Pro vs GSM8K split.** Two datasets, very different formats (10-option MCQ vs free-form numeric). Split every result. If the bandwagon effect is MCQ-specific, that's an important scope claim.

### 2.4 C5 reframing without losing the C5 condition (kills A3)

12. **Drop "mitigation evaluation" entirely.** Reframe C5 as: *"a stress-test confirming that the calibration-gate prediction held empirically — i.e., the gate is a valid pre-flight check for whether confidence-based filtering would be informative."* This is honest, methodologically interesting (the gate is the new tool), and turns C5's null result into a positive contribution about pre-flight validity.
13. **Compute the "effective filter rate"** in C5 (peer-messages-filtered-out / peer-messages-shown). Show it's ~0%. This nails the gate-prediction confirmation.

### 2.5 Robustness checks against your own results

14. **Regex-vs-judge attribution sensitivity.** From `trial_log.parquet`, split the C→I flip rate by `answer_extraction_method` (regex_success vs judge_fallback). If judge-extracted answers contribute disproportionately to flips, the result depends on judge reliability — flag it. If they don't, you've defended against A20.
15. **Confidence calibration plot.** Plot calibration curves for both focal models and both dumb models (reliability diagram). The 99.5% miscalibration becomes visually undeniable, and you can cite Tian 2023 / Xiong 2024 / Detommaso 2024 properly.

### 2.6 Reframe the headline (kills A1, A2, A29)

The current framing claims a general theory ("correction vs corruption signals") but supports it with two data points. The honest, defensible reframing is:

> **Old:** "Correction dominates for DeepSeek; corruption dominates for GPT-4o-mini."
>
> **New:** "Under capability-asymmetric MAD, two failure modes co-occur in every trial: an information-injection effect (some answers improve) and a conformity-revision effect (some R0-correct answers flip). The dominant net direction depends on focal-model strength in a way that cannot be predicted from baseline accuracy alone — DeepSeek (76.2% solo) gains 7.9 pp while GPT-4o-mini (64.8% solo) loses 4.0 pp under the same protocol. **A deployment-time risk: the average-accuracy metric reported in current MAD evaluations hides a systematic subset-level regression that affects 11% of the questions DeepSeek answers correctly in C1.**"

This is publishable, defensible, and ACL-relevant. The "deployment risk" framing addresses A29 and A30 simultaneously.

### 2.7 Documentation fixes (kills A17, A18, A19, A21, A22, A23)

16. Reconcile dataset counts. Audit `question_pool.parquet`: count actual MMLU-Pro vs GSM8K rows. Fix paper or code to match.
17. Reconcile persona descriptions. The repo's `surface_keyword_match` etc. are *wrong-reasoning styles* used to seed dumb-agent responses; the paper's *analytical/sceptical/diplomatic/assertive* is a different naming scheme. Pick one, use it consistently. If the rhetorical-style description is post-hoc, drop it and use the actual code's labels.
18. **Paste verbatim prompts in Supplement S4.** Hard requirement.
19. **Pin model versions.** Open `experiment_metadata.json`, take the served model strings (DeepSeek's API returns the actual model in its response), and put them in §3.2.
20. **Release code + data at submission**, anonymized. ACL allows anonymous GitHub repos for review. Drop "upon acceptance."
21. **Reproducibility appendix**: per-trial seed derivation (`seed = master_seed + question_index × 1000 + trial_index` or similar), API timeout/backoff settings, judge cascade thresholds.

### 2.8 ACL ARR-specific structural additions

22. **Limitations section** as a separate numbered section after Conclusion (mandatory for ACL ARR). The current §5 Discussion has limitations folded in — split them out.
23. **Ethics statement.** Trivial here: no human subjects, public datasets, but explicitly state it.
24. **Reproducibility checklist** in appendix.
25. **Trim to long-paper length** (8 pages content). Currently ~12 pages. The cog-sci framing (Plato, Asch detail) can be cut by half.

### 2.9 Citation refresh (kills A6, A7, A26 partially)

26. Add 2025–2026 ACL/EMNLP/ICLR/NeurIPS papers on MAD, sycophancy, miscalibration. The `bibliography_verification_log.md` flags 7 unverified ARR-era refs — verify or replace.
27. Specifically cite: Tian/Lin/Goyal on confidence elicitation, Xiong on LLM calibration, Detommaso on calibration repair, and the LLM-as-judge bias literature (Zheng 2023 MT-Bench, Wang 2024 LLM-judge biases).

---

## Section 3 — New experiments that *might* be crucial

Only run these if Section 2 fixes don't carry the paper. In priority order:

### 3.1 (Probably necessary) Smart-non-focal heterogeneity check

**Cost:** ~300 questions × 1 condition × 3 trials = 900 trials, ~$2 in API spend.
**What:** Add C2', three smart agents from *different* providers (DeepSeek + GPT-4o-mini + Gemini 2.x). Run on a 100-question subset.
**Kills:** A15. Lets you separate "homogeneous self-debate" from "true heterogeneous debate."
**Why it might be skippable:** Limitation #2 admits this; if you also add the self-consistency baseline from §2.2 above, you can pin C2's lift to debate vs sampling. C2' would be confirmatory.

### 3.2 (Possibly necessary) Split-peer condition

**Cost:** 300 × 1 × 3 = 900 trials.
**What:** C3-split — 1 dumb agent gets a wrong-anchored persona, 1 dumb agent gets a *correct*-anchored persona. Or 1 wrong dumb + 1 correct smart.
**Kills:** A4 (the unanimity confound). Without this, the paper must drop bandwagon framing entirely.
**Recommendation:** If you can spare the budget, run it. Otherwise, drop "bandwagon" from title, abstract, and all section headings and rebrand as "wrong-peer-induced revision pressure." The cog-sci framing thins out, but the empirics survive.

### 3.3 (Skip) Reasoning-vs-answer ablation

The A28 attack (persona-reasoning vs bare-answer) is real but expensive (~3,000 trials) and ACL reviewers will accept "scope" as a deferral.

### 3.4 (Skip) Open-ended task generalization

A24 is real but huge in scope. Just admit it in Limitations.

### 3.5 (Skip) Larger dumb model

A25 is real but expensive. Cite scaling-law arguments from prior work.

---

## Section 4 — Revision strategy in execution order

1. **Day 1.** Re-run statistical analysis (Section 2.1 + 2.2). The mixed-effects regression, the self-consistency baseline from C2's R0, and the question-level/trial-level reconciliation are the three highest-leverage moves. If C2's SC-of-3 baseline shows ~76.2% (same as C1), your "C2 debate gain" of +2.5 pp is real revision benefit; if SC-of-3 shows ~78%, then C2's gain is sample aggregation. **You need to know this before deciding on framing.**
2. **Day 2.** Difficulty + subject + dataset stratifications (Section 2.3). Surface anything that's interpretable as a structural finding.
3. **Day 3.** Reframe abstract, title, contributions, conclusion (Section 2.6). The new framing must lead with the deployment-risk finding, not the cog-sci theory.
4. **Day 4.** Reconcile code↔paper (Section 2.7). Audit dataset counts, fix persona naming, paste prompts in supplement, release anonymized repo.
5. **Day 5.** Decide on Section 3 experiments. If §2.2's self-consistency baseline shows C2's lift is real, you can probably skip C2'. If you can run anything, prioritize the split-peer C3-split to defend (or honestly drop) the bandwagon claim.
6. **Day 6.** Trim to 8 pages, add Limitations / Ethics / Reproducibility appendix.
7. **Day 7.** Internal review against Section 1 attack list — every attack should map to either a fix in the paper or an honest concession in Limitations.

---

## The one move that matters most

If you do nothing else: **derive the self-consistency baseline from C2's existing R0 data and add it as a column to Table 2.** It is free, it pre-empts the single most damaging methodological attack (A14), and depending on what it shows, it either:
- Validates the paper's debate-as-revision framing (best case), or
- Forces an honest rewrite where debate ≈ sampling, and the paper becomes about *capability-asymmetric sampling biases* rather than debate — still publishable, more honest.

Without that baseline analysis the paper is at desk-reject risk regardless of how well-written it is, because the limitation is admitted *and* free to fix.

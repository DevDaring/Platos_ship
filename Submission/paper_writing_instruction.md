# Paper Writing Instruction — Cognitive Systems Research (Elsevier)

> **Audience.** A coding assistant (GitHub Copilot or equivalent) that has read access to the full experiment codebase and the result files produced by the pipeline.
>
> **Goal.** Produce a complete, submission-ready LaTeX manuscript for *Cognitive Systems Research* (Elsevier), built from the actual numbers in the result files. The user will compile in Overleaf (paid plan).
>
> **Style rule.** The user's advisors are Dr. Debarshi Kumar Sanyal and Dr. Imon Mukherjee. Their writing is plain, evidence-bound, and short on adjectives. The companion document `PAPER_REVIEW_INSTRUCTIONS.md` lists the 26 hard rules that govern every line. Read that document first. Do not write anything that violates it.
>
> **Tone discipline.** A human reviewer will read this. Sentences should be short. Long sentences are allowed only when content forces them. No promotional vocabulary. No filler.

---

## 1. What This Paper Is About (one paragraph for the writer's grounding)

The study tests whether a small number of weaker LLM agents in a debate group can degrade or collapse the accuracy of a stronger LLM agent. Smart agents are DeepSeek (latest, via DeepSeek API) and GPT-4o-mini (via OpenRouter). Dumb agents are Llama 3.1 8B Instruct and Gemma 3 4B Instruct (or Gemma 2 2B Instruct, if the fallback path was taken — check `experiment_metadata.json`). The dataset is 200 MMLU-Pro questions and 100 GSM8K questions. Four conditions (one solo smart agent, three smart agents, two smart plus one dumb, one smart plus two dumb) are run with five trials each. The framing draws on Plato's *Ship of State* allegory and on two well-documented LLM cognitive biases: sycophancy and the bandwagon effect. The Asch conformity index is reported as a direct cross-species replication signal.

---

## 2. Pre-Flight: Read Before Writing

Before writing any prose, the assistant must load and inspect these files in this exact order. Stop and ask if any are missing or empty.

1. `data/outputs/experiment_metadata.json` — hardware, model names actually served, gemma fallback flag, completion timestamps.
2. `data/outputs/final_answers.parquet` — one row per (question, condition, trial, focal smart agent).
3. `data/outputs/metrics_summary.parquet` — accuracy, flip rates, Asch index, bootstrap confidence intervals.
4. `data/outputs/statistical_tests.parquet` — McNemar test results for the six paired comparisons.
5. `data/processed/question_pool.parquet` — sample sizes per subject, difficulty stratification.
6. `data/processed/dumb_personas.parquet` — persona retention rate, reasoning style mix.
7. `logs/pipeline.log` — final summary line, parse failure rate.

Every numeric claim in the paper must trace to a specific cell in one of these files. No placeholders. No round numbers invented for fluency. If a number cannot be located, write `[NUMBER UNAVAILABLE — VERIFY]` in the draft and add it to a `pending_numbers.md` checklist at the project root.

---

## 3. Target Venue and Format

**Journal.** *Cognitive Systems Research*, Elsevier. ISSN 1389-0417.

**Article type.** Research article (not opinion essay, not short report). Recommended length 3,500–4,500 words including abstract, excluding references. Up to 6 keywords.

**Manuscript format.** LaTeX using the `elsarticle` document class. Single-column review version for first submission. Reference style: author-year (Harvard) — Elsevier APA-like. Use `\documentclass[review,3p,authoryear]{elsarticle}` and `\bibliographystyle{elsarticle-harv}`.

**Word budget for this paper.**

| Section | Target words |
|---|---|
| Abstract | 180–200 |
| Introduction | 700–850 |
| Related Work | 600–750 |
| Method | 700–850 |
| Results | 700–850 |
| Discussion | 350–450 |
| Conclusion | 180–230 |
| **Body total** | **~3,800** |

These are targets, not ceilings. Cut, do not pad, if any section runs long.

---

## 4. Files To Produce

Create exactly these files in a directory called `manuscript/` at the repository root. The user opens this directory in Overleaf as a project.

```
manuscript/
├── main.tex                       # The manuscript
├── references.bib                 # All cited works
├── highlights.tex                 # Required by Elsevier — 3 to 5 bullets, ≤ 85 chars each
├── cover_letter.tex               # Editor-facing cover letter
├── declaration_of_interests.tex   # COI statement
├── credit_authorship.tex          # CRediT taxonomy contributions
├── data_availability.tex          # Data and code availability statement
├── figures/
│   ├── figure_1_bandwagon_dose_response.pdf
│   ├── figure_2_asch_conformity_index.pdf
│   └── figure_3_flip_rate_by_subject.pdf
└── tables/
    └── (tables are inline in main.tex via \begin{table})
```

Do not create a separate title page file. Use `\title`, `\author`, `\affiliation` inside `main.tex`. Author names and affiliations are placeholders the user will fill — leave them as `\author{[Author One]}` and `\affiliation{[Affiliation One]}` until the user provides the real values.

### 4.1 `main.tex` Skeleton

The skeleton must contain these blocks, in order. Do not deviate.

```latex
\documentclass[review,3p,authoryear]{elsarticle}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{multirow}
\usepackage{url}
\usepackage{hyperref}
\usepackage{lineno}
\linenumbers

\journal{Cognitive Systems Research}

\begin{document}

\begin{frontmatter}
\title{[Working Title — see §6 of writing instruction]}
\author[aff1]{[Author One]}
\author[aff1]{[Author Two]}
\affiliation[aff1]{organization={[Affiliation One]}}

\begin{abstract}
[Abstract — see §7]
\end{abstract}

\begin{keyword}
[6 keywords — see §7]
\end{keyword}
\end{frontmatter}

\section{Introduction}
\section{Related Work}
\section{Method}
\section{Results}
\section{Discussion}
\section{Conclusion}

\bibliographystyle{elsarticle-harv}
\bibliography{references}

\end{document}
```

Acknowledgements, if any, go in a `\section*{Acknowledgements}` block before the bibliography. The user will add this.

---

## 5. Style Rules That Govern Every Sentence

Read the companion file `PAPER_REVIEW_INSTRUCTIONS.md` in full. The rules below are the irreducible minimum that must be enforced while writing. These are the ones the user has had flagged most often.

**Vocabulary blocked entirely.** *leverage, harness, employ, utilize (use "use"), showcase, pave, pioneer, delve, foster, navigate (when metaphorical), underscore, unlock, supercharge, revolutionize, paradigm shift, holistic, robust (more than two times per section), cutting-edge, breakthrough, transformative, novel (more than once outside the contributions list), state-of-the-art (more than twice in the paper), seamless, intricate, pivotal, paramount, realm, landscape, multifaceted, unprecedented*.

**Connectives blocked.** *Furthermore, Moreover (more than once per section), Additionally, In essence, Notably, Importantly, It is important to note, It is worth noting, To this end (more than once per paper), It should be emphasised, It must be mentioned*.

**Connectives allowed, sparingly.** *Therefore, However, In particular, Note that, More precisely, We observe that, In contrast, On the other hand, Specifically*. Use no more than one of these per five sentences.

**Adjectives without numbers.** *Significant, substantial, considerable, remarkable, impressive, compelling* without an attached number or delta is forbidden. If "significant" appears, a `p`-value or a delta must be within twenty tokens of it.

**Sentence length.** Mean 18 to 22 words per sentence. Within any paragraph, length variance must be visible: short sentences of 8 to 12 words alongside longer ones of 25 to 30. A paragraph where every sentence is over 25 words reads as flat and gets flagged. The user has flagged this in prior drafts.

**Em-dashes.** No more than four per page. AI prose over-uses them.

**Hedging.** One hedge word per claim. *"may potentially possibly indicate"* is forbidden. *"suggests"* alone or *"may"* alone is fine.

**Voice.** Active by default. Passive only when the actor is irrelevant or unknown.

**Pronoun choice.** Mix *we*, *this study*, *this work*, *this paper*. Do not replace every *we* — Sanyal's papers use *we* successfully. The signature phrase *we observe that* is preserved as is. If a paragraph has more than four instances of *we* or *our*, recast some.

**No paragraph may open with *While* or *However*.** State the paragraph's claim directly.

**No personification.** A model does not *strive*, *aim*, *aspire*, or *seek*. The authors design or use the model. The model achieves a metric.

**Spelling.** British / Indian English consistently: *modelling*, *analyse*, *behaviour*, *labelling*, *characterised*, *recognised*, *organised*, *neighbouring*, *summarising*. No mixing with US spellings.

**Italics.** Model and dataset names italicised on first use and on important re-introduction. Specifically italicise *DeepSeek*, *GPT-4o-mini*, *Llama 3.1 8B Instruct*, *Gemma 3 4B Instruct*, *MMLU-Pro*, *GSM8K* on first occurrence in each major section.

**Hyphenation.** *Multi-agent*, *capability-asymmetric*, *pre-trained*, *fine-tune*, *flash-attention*, *state-of-the-art*, *peer-exposed*, *cross-model*, *open-weight*, *closed-weight*. Apply consistently.

**Acronyms.** Define on first use. *Large Language Model (LLM)*, *Multi-Agent Debate (MAD)*, *Massive Multitask Language Understanding (MMLU)*, *Grade School Math 8K (GSM8K)*. Use the acronym thereafter.

**No bullet points or numbered lists in body sections.** The single exception is the numbered contributions list at the end of the Introduction. Tables and figure captions are exempt.

**No section opens with a method-detail sentence.** Every section and subsection opens with a general framing sentence that motivates what follows.

**Robotic openers blocked.** Do not start any section with: *In this section we...*, *This section presents...*, *We now describe...*, *This subsection contains...*, *The following describes...*, *In what follows...*. The good pattern is the DAKE form: name the components or state the motivation, then introduce the technical content.

---

## 6. Title and Keywords

**Working title (use this; the user can change later).**

> Capability-Asymmetric Multi-Agent Debate Degrades Strong LLM Reasoners: A Bias-Theoretic Account

If the user prefers the Plato framing, an acceptable alternative is:

> Plato's Sailors: Sycophancy and Bandwagon Effects in Capability-Asymmetric LLM Debate

Pick the first. Reviewers at CSR are technically conservative; the Plato framing belongs in the introduction, not the title.

**Keywords (six, comma-separated).** *Multi-agent debate; Large language models; Cognitive bias; Sycophancy; Bandwagon effect; Conformity*.

---

## 7. Abstract

Length 180 to 200 words. Six elements, in this order, no headings:

1. **General field framing (1–2 sentences).** Open with the broader context: LLMs are increasingly deployed in multi-agent systems where they exchange messages and arrive at joint answers. Do not open with *"This paper proposes..."*. Do not open with *"We present..."*.
2. **Narrowing to the problem (1 sentence).** State that capability-asymmetric groups (a strong reasoner mixed with weaker peers) are common in practice but their failure modes are not well characterised.
3. **What the paper does (1 sentence).** State that this work measures accuracy degradation under capability mixing and connects the failure to two documented LLM cognitive biases.
4. **Method one-liner (1 sentence).** Mention four conditions, two smart and two open-weight dumb agents, and the dataset (300 questions from MMLU-Pro and GSM8K).
5. **Headline numerical results (2 sentences).** Read these from `metrics_summary.parquet` for `deepseek_primary` as the focal agent. Report (a) accuracy of the focal smart agent in C1 versus C4 with absolute and percentage-point delta; (b) flip-rate from correct to incorrect in C4 versus C2; (c) Asch conformity index for C4. Use real numbers from the file.
6. **One-sentence implication.** State that the failure pattern matches sycophancy and the bandwagon effect documented in single-agent settings, and that mitigation should target peer-source weighting.

Do not include any number that does not appear in the result files.

---

## 8. Introduction (700–850 words)

Structure follows the order in the 26 rules: motivation, background, gap, what we do, contributions list. No roadmap paragraph (this is a short paper).

**Paragraph 1 — Motivation (~120 words).** Open with the spread of LLM-based multi-agent systems in deployed settings: research assistants, coding agents, automated review pipelines, and code review. Cite two or three deployment-oriented papers. Then in two sentences raise the basic worry: when agents of different capability work together, who pulls whom? End with a single sentence that names Plato's *Ship of State* as the historical formulation of the same worry. The Plato reference is one sentence — do not lean on it.

**Paragraph 2 — Background (~150 words).** Three threads:

- LLM cognitive biases: cite Echterhoff et al. (2024) survey, Sharma et al. (2023, 2025) on sycophancy, Koo et al. (2024) on bandwagon. State briefly that LLMs inherit human-like biases through preference-based fine-tuning.
- Multi-agent debate: cite Du et al. (2023). Report the original finding that homogeneous debate improves accuracy.
- Conformity in groups of LLM agents: cite Weng et al. (2025) BENCHFORM, Choi et al. (2025) identity bias, Hong et al. (2025) SYCON-Bench.

Each cited claim must come with a `\citep{}`. No uncited factual claims about prior work.

**Paragraph 3 — Gap (~120 words).** State the gap precisely. Wu et al. (2025) documented that adding a weak agent to a strong-agent debate reduces accuracy. Weng et al. (2025) measured conformity but kept capability homogeneous. Choi et al. (2025) varied identity but not capability. No prior work measures the dose-response curve as the count of weak peers increases, and no prior work explicitly maps the failure to specific bias mechanisms (sycophancy, bandwagon) in a heterogeneous-capability group. State this in two clean sentences. Do not over-claim.

**Paragraph 4 — What this paper does (~150 words).** State the four conditions with the smart-to-dumb counts. State the two smart models (DeepSeek, GPT-4o-mini) and the two dumb models (Llama 3.1 8B, Gemma — name the actual model loaded per `experiment_metadata.json`). Name the datasets and the trial count. Give one sentence on metrics: independent (Round 0) versus peer-exposed (Round 1) accuracy, the flip rate from correct to incorrect, the Asch conformity index, and a logistic dose-response.

**Paragraph 5 — Contributions list (numbered, ~120 words).** Three items only. Each starts with a verb. No boilerplate.

```latex
The contributions of this work are as follows:
\begin{enumerate}
    \item We measure the dose-response curve of strong-agent accuracy as the number of weak peers in a debate group increases from zero to two, on 300 questions across two datasets.
    \item We map the observed degradation onto two documented cognitive biases: sycophancy, measured via the correct-to-incorrect flip rate, and the bandwagon effect, measured via the Asch conformity index over peer unanimity.
    \item We replicate the effect across two frontier-class smart agents (\textit{DeepSeek} and \textit{GPT-4o-mini}) and two open-weight dumb agents (\textit{Llama 3.1 8B Instruct} and \textit{Gemma}), establishing cross-model generality.
\end{enumerate}
```

The numbered enumerate is the **only** numbered list allowed in the Introduction. Do not add a fourth item unless the experiment results genuinely support it.

---

## 9. Related Work (600–750 words)

Three thematic subsections. No chronological storytelling. No phrases like *"a key breakthrough came in..."*, *"his team"*, *"fast forward to..."*. Author-narrative form: *"Wu et al. (2025) show that..."*, *"Weng et al. (2025) introduce..."*.

**§2.1 Multi-agent debate and its failure modes (~250 words).** Cover Du et al. (2023) as the positive baseline (homogeneous debate helps). Then cover the failure-mode literature: Wu et al. (2025) "Talk Isn't Always Cheap" on weak-agent disruption, Pitre et al. (2025) CONSENSAGENT on sycophancy in MAD, Choi et al. (2025) on identity-driven sycophancy and obstinacy. End with one sentence stating that this work is closest to Wu et al. (2025) but differs in that it (a) varies the number of weak peers as a dose, and (b) connects the effect to bias taxonomy.

**§2.2 Conformity and bandwagon biases in LLMs (~250 words).** Cover Asch (1951) in one sentence as the human anchor. Then Weng et al. (2025) BENCHFORM, Hong et al. (2025) SYCON-Bench, Sharma et al. (2023, 2025), Koo et al. (2024) CoBBLEr. Bond and Smith (1996) meta-analysis on group-size effects in human conformity is included as the cross-species hook. State that this work tests whether the saturation pattern Bond and Smith (1996) reported in human groups also appears in LLM debate groups.

**§2.3 Emergent dynamics in LLM populations (~200 words).** Brief subsection. Cite Ashery et al. (2025) on emergent conventions and collective bias in *Science Advances*. State that prior work studies large populations and emergent conventions; this work studies small groups (three agents) and capability heterogeneity, which is closer to most deployed multi-agent systems.

End the section with one **summary table** comparing this work to the closest prior studies. The table has columns: *Study*, *Setting*, *Capability heterogeneity?*, *Bias mechanism analysed?*, *Cross-model test?*. Five to seven rows. Wrap in `\small`. Use `\multirow` if any column overflows.

---

## 10. Method (700–850 words)

Five subsections. Each opens with a general framing sentence, not a method-detail sentence.

**§3.1 Experimental conditions (~150 words).** Open by stating that the design varies the count of weak peers from zero to two while holding total group size at three (with C1 as a solo baseline). Describe the four conditions in prose, then a small table. Do not use bullet points.

**§3.2 Models (~150 words).** Open with one sentence on the rationale for the two-by-two grid (two smart, two dumb). Then list the four models. State the precision (`bfloat16`) for the local models and that smart models run at provider-server precision. State the actual model name strings as recorded in `experiment_metadata.json`. Italicise model names on first use.

**§3.3 Datasets and dumb-persona generation (~200 words).** Open with the rationale: MMLU-Pro provides factual breadth across subjects, GSM8K provides arithmetic reasoning. State the sampling plan: 200 MMLU-Pro questions stratified across ten subjects with twenty per subject, 100 GSM8K questions. Use the actual retained counts from `question_pool.parquet`. Then describe the dumb-persona generation procedure: a template that pairs a wrong answer with one of four reasoning style labels (*surface_keyword_match*, *false_analogy*, *overconfident_assertion*, *misapplied_rule*), generated by Llama 3.1 8B at temperature 0.9, validated by an automatic filter, with retention rate read from `logs/persona_validation_report.txt`.

**§3.4 Debate protocol (~150 words).** Two rounds. Round 0: each agent answers independently. Round 1: each agent sees the other agents' Round 0 responses, with peer-message ordering randomised per trial, and produces a final answer. Final answers parsed by regex; the judge model (older DeepSeek variant) handles parse failures.

**§3.5 Metrics and statistical tests (~200 words).** Define every metric explicitly. Each definition is one sentence followed by the symbol. The four metrics: Round 0 accuracy, Round 1 accuracy, flip rate from correct to incorrect, Asch conformity index. State that paired McNemar tests with Bonferroni correction for six comparisons (alpha = 0.0083) are used, and that 95% bootstrap confidence intervals are reported with 10,000 resamples. State that the focal smart agent for the main analysis is *DeepSeek*, and a 50-question replication subset uses *GPT-4o-mini* as focal.

If any equation is included, it must be followed by a *where ... is ...* clause defining every symbol.

---

## 11. Results (700–850 words)

Open the section with a general framing sentence: that the analysis tests three predictions — accuracy degradation, dose-response, and bias-mechanism alignment.

**§4.1 Accuracy degradation across conditions (~250 words).** Anchor on Table 2 (main results table). Report Round 0 and Round 1 accuracy and the deltas for each of the four conditions, for the focal *DeepSeek* agent. Then report McNemar p-values for the six paired comparisons. Use the discipline of Mob-Res result narration: do not narrate every cell. Highlight only the comparisons that lead to a conclusion. A paragraph might read like:

> Table 2 reports Round 0 and Round 1 accuracy for *DeepSeek* across the four conditions. Round 0 accuracy is stable near [X]% across conditions, as expected when agents do not see peers. Round 1 accuracy in the homogeneous control (C2) is [Y]%, a [Δ]-point change from C1. Adding one weaker peer (C3) drops Round 1 accuracy to [Z]%, a [Δ]-point loss relative to C2 (McNemar p = [p], Bonferroni-corrected). Adding a second weaker peer (C4) drops Round 1 accuracy further to [W]%.

Replace bracketed values with the actual numbers from `metrics_summary.parquet` and `statistical_tests.parquet`. Do not invent.

**§4.2 Dose-response curve (~200 words).** Anchor on Figure 1 (bandwagon dose-response). Report the slope coefficient from the logistic regression of Round 1 correctness on the count of weak peers. State whether the curve saturates between one and two peers (compare deltas C2→C3 and C3→C4). Tie back to Bond and Smith (1996) only if the saturation pattern matches; do not force the comparison.

**§4.3 Asch conformity and bias signatures (~150 words).** Anchor on Figure 2. Report the Asch conformity index for C3 and C4: flip rate when the dumb peers were unanimously wrong minus flip rate when they were split. If unanimity-driven flips exceed split-driven flips, state the delta with confidence interval. State the flip-rate breakdown by reasoning style if the variation is meaningful — do not over-interpret narrow style differences.

**§4.4 Cross-model replication (~150 words).** Report the replication on 50 questions with *GPT-4o-mini* as focal. State whether the C1 → C4 accuracy drop direction reproduces. Do not claim full quantitative agreement; this is a smaller sample. Two to three sentences.

End the section with a one-sentence transition to the discussion.

---

## 12. Discussion (350–450 words)

Three short paragraphs.

**Paragraph 1 — Bias-mechanism interpretation (~180 words).** The flip-rate evidence aligns with sycophancy as documented by Sharma et al. (2023, 2025): the strong agent abandons a correct answer under peer disagreement. The Asch conformity index aligns with the bandwagon effect: unanimity amplifies the flip. Both biases were originally documented in single-agent, user-pressure settings; this work shows they also surface under peer-pressure in multi-agent settings.

**Paragraph 2 — Boundaries of the claim (~120 words).** The study uses two smart models, two dumb models, and 300 questions. The dose-response is measured at three points (zero, one, two weak peers). Effects at larger group sizes, longer debate horizons, and other model families are not tested here. The dumb-persona reasoning styles are templated; spontaneous wrong reasoning by an unprompted weak model may differ.

**Paragraph 3 — Implications for system design (~120 words).** State two concrete implications. First, in deployed multi-agent systems where capability is asymmetric, debate aggregation by majority vote is unsafe; weighting by self-reported uncertainty or by an external trust signal is a candidate mitigation. Second, sycophancy mitigation strategies developed in single-agent settings (Pitre et al., 2025; Hong et al., 2025) should be evaluated in capability-asymmetric multi-agent settings before deployment. Do not claim a mitigation method — that is a future paper.

---

## 13. Conclusion (180–230 words)

Sanyal-style conclusions are short and honest about limits. Two paragraphs.

**Paragraph 1 (~110 words).** State what was done. State the headline result with one number. State the bias mapping in one sentence.

**Paragraph 2 (~100 words).** State two concrete future directions. State one explicit limitation that the discussion already acknowledged. Do not introduce new claims. Do not pad.

Forbidden phrases in the conclusion: *novel*, *state-of-the-art*, *significant improvement* (without a number), *extensive experiments*, *demonstrate the superior*, *paving the way*, *broad applicability*. The DAKE conclusion ends with "the predicted research highlights are not yet perfect in terms of syntax and semantics" — that level of honesty is the model.

---

## 14. Tables and Figures

Three figures and two tables maximum. Each must be referenced in prose with `\ref{}`. Each must be interpreted by at least one prose sentence.

**Table 1.** Comparison of this work to closest prior studies. Five to seven rows. Goes in §2 (Related Work). Wrap in `\small`. Use `\multirow` for any cell that spans rows.

**Table 2.** Main results. Rows = conditions (C1, C2, C3, C4). Columns = Round 0 accuracy, Round 1 accuracy, Δ (percentage points), Flip rate (correct → incorrect), Bootstrap 95% CI. Goes in §4.1.

**Figure 1.** Bandwagon dose-response. X-axis: count of weak peers (0, 0, 1, 2). Y-axis: Round 1 accuracy of *DeepSeek*. Error bars: 95% bootstrap CI. PDF format. Single panel.

**Figure 2.** Asch conformity index. Bar plot: flip rate under unanimous-wrong peers vs split peers, for C3 and C4. Error bars: 95% bootstrap CI.

**Figure 3.** Subject-wise flip rate (optional, include only if subject variation is meaningful). Horizontal bar plot, ten subjects from MMLU-Pro plus GSM8K.

Generate the figures with matplotlib at `dpi=300`, save as PDF, place in `manuscript/figures/`. Use a minimal style: no gridlines, axis labels in plain English, font matching Elsevier defaults (sans-serif, 10pt). Do not use chart junk.

Every table and figure has a caption that ends with the source of the numbers (e.g., *"Source: data/outputs/metrics_summary.parquet"*) — this is not standard for journal tables but it is added during draft so the user can verify each number, then removed before submission. Mark these source lines with `% PROVENANCE — REMOVE BEFORE SUBMIT`.

---

## 15. Bibliography (`references.bib`)

Build a single `references.bib` file with all citations. Roughly 25 to 30 entries. Use the BibTeX entry format compatible with `elsarticle-harv`.

**Required citations (must appear in the paper).** Each is followed by a recommended BibTeX key.

| Work | BibTeX key |
|---|---|
| Asch (1951) — line judgment conformity | `asch1951effects` |
| Bond and Smith (1996) — meta-analysis on group conformity | `bond1996culture` |
| Plato's *Republic* — book VI, ship of state allegory | `plato1992republic` |
| Du et al. (2023) — multi-agent debate improves reasoning | `du2023improving` |
| Sharma et al. (2023, 2025) — sycophancy in language models | `sharma2023towards` |
| Koo et al. (2024) — CoBBLEr cognitive bias benchmark | `koo2024benchmarking` |
| Echterhoff et al. (2024) — cognitive biases in LLMs survey | `echterhoff2024cognitive` |
| Weng et al. (2025) — BENCHFORM | `weng2025benchform` |
| Wu et al. (2025) — Talk Isn't Always Cheap, ICML | `wu2025talk` |
| Choi, Zhu, Li (2025) — When Identity Skews Debate | `choi2025identity` |
| Pitre et al. (2025) — CONSENSAGENT, ACL | `pitre2025consensagent` |
| Hong et al. (2025) — SYCON-Bench | `hong2025sycon` |
| Ashery et al. (2025) — Emergent conventions, *Sci. Advances* | `ashery2025emergent` |
| TIGER-Lab (2024) — MMLU-Pro | `wang2024mmlupro` |
| Cobbe et al. (2021) — GSM8K | `cobbe2021training` |
| Touvron et al. — Llama 3.1 model card | `meta2024llama31` |
| Google DeepMind — Gemma technical report | `gemma2024technical` |
| DeepSeek-AI — DeepSeek V3 technical report | `deepseekai2024v3` |
| OpenAI — GPT-4o system card | `openai2024gpt4o` |
| Dao (2023) — FlashAttention | `dao2023flashattention` |

Verify every entry against DBLP, Semantic Scholar, or Google Scholar. **Do not invent BibTeX entries.** If a paper cannot be verified, leave it out and remove the citation. Write a short `bibliography_verification_log.md` at the repository root listing each entry, the source used to verify, and the verification status.

For preprints, include the arXiv identifier and the word "preprint" in the entry, per Elsevier policy.

---

## 16. Highlights and Cover Letter

**`highlights.tex`.** Three to five bullets, each at most 85 characters including spaces. Each is a single declarative claim with a number where possible. Example shape:

```latex
\begin{itemize}
    \item Two weak LLM peers cut DeepSeek accuracy by [X] points on 300 questions.
    \item Flip rate from correct to incorrect rises monotonically with weak-peer count.
    \item Asch conformity index is [Y] under unanimous wrong peers, [Z] under split peers.
    \item Effect replicates with GPT-4o-mini as focal smart agent.
    \item Failure aligns with sycophancy and bandwagon biases documented in single-agent work.
\end{itemize}
```

Replace bracketed values with real numbers. Count characters carefully — Elsevier truncates at 85.

**`cover_letter.tex`.** Six to eight short paragraphs. Address the editor by title (*"Dear Editor-in-Chief"*). State (1) submission to *Cognitive Systems Research* as a research article, (2) the headline finding in two sentences, (3) why the paper fits CSR scope (cognitive systems framing, bias-mechanism account, cross-species replication of Asch and Bond–Smith effects), (4) confirmation that the work is original and not under review elsewhere, (5) conflict-of-interest statement summary (point to declaration), (6) suggested reviewers if the user has any (placeholder list of three names with affiliations and emails — leave blank for the user to fill).

Do not pad the cover letter. Editors skim.

---

## 17. Required Elsevier Artefacts

**`declaration_of_interests.tex`.** Standard Elsevier template. The authors declare no competing financial or personal interests, or list specific declarations the user provides.

**`credit_authorship.tex`.** CRediT taxonomy. Use the exact role names: *Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Resources, Data curation, Writing – original draft, Writing – review and editing, Visualization, Supervision, Project administration, Funding acquisition*. Leave author initials as placeholders for the user to fill.

**`data_availability.tex`.** State that the question pool, persona pool, full trial logs, and analysis code will be released in a public repository upon acceptance. State the planned repository URL as a placeholder. State that MMLU-Pro and GSM8K are publicly available under their original licences.

---

## 18. Numbers Discipline (the rule that fails most papers)

Every numerical claim in the manuscript must be traceable to a specific cell in a specific result file. Build an internal mapping while writing. At the end, produce `numbers_provenance.md` at the repository root with this format:

```
| Claim location in main.tex | Number | Source file | Source row/column | Verified |
|---|---|---|---|---|
| Abstract sentence 5 | 78.2% | metrics_summary.parquet | condition=C1, focal=deepseek_primary, round_one_accuracy_rate | ✓ |
| §4.1 paragraph 2 | -8.3 pp | metrics_summary.parquet | C2 round_one minus C4 round_one | ✓ |
| ... | | | | |
```

The user will spot-check this file before sending the manuscript to the advisor. A single fabricated number invalidates the draft.

---

## 19. Final Quality Gate (run before declaring done)

The following checks must all pass. Treat any failure as a hard stop.

**Format checks.**

1. `main.tex` compiles in Overleaf with no errors. Test by compiling the local copy if `latexmk` is available; otherwise list the imports and structure for the user to verify.
2. Word count of body (Abstract through Conclusion, excluding references and bibliography) is between 3,500 and 4,500.
3. Abstract is between 180 and 200 words.
4. Conclusion is at most 250 words.
5. No section opens with a robotic phrase (the six listed in §5).
6. No paragraph opens with *While* or *However*.
7. Every `\label{tab:X}` and `\label{fig:X}` has at least one `\ref{}` in prose.
8. Every figure and table caption has at least one prose sentence in the body that interprets it.

**Vocabulary checks.**

9. Run a search for every blocked word in §5. Each hit must be justified or replaced.
10. Count em-dashes per page. No page exceeds four.
11. Count *we* and *our* per paragraph. No paragraph has more than four.
12. Count *significant*, *substantial*, *considerable*. Each occurrence has a number within twenty tokens or is removed.
13. Spelling is consistently British / Indian English. No US spellings mixed in.

**Structure checks.**

14. The Introduction has the contributions list as a numbered enumerate at the end.
15. The contributions list has between two and four items. Each starts with a verb. None is boilerplate.
16. The Related Work section has at least one citation per claim about prior work and no chronological storytelling phrases.
17. Every equation is followed by a *where ... is ...* clause defining each symbol.
18. The Conclusion contains an explicit acknowledgement of at least one limitation.

**Numbers checks.**

19. Every numerical claim in the manuscript appears in `numbers_provenance.md` with a source file and row.
20. No invented numbers. No round-number-for-fluency replacements.

**Bibliography checks.**

21. Every `\citep{}` and `\citet{}` resolves to a real BibTeX entry in `references.bib`.
22. `bibliography_verification_log.md` exists and shows verification status for every entry.

**Reviewer-impression checks.**

23. The "explain it back" test: read the abstract and intro, then write three sentences answering — what problem, what novelty, what headline number. If those three sentences are vague, the framing has failed and the draft is not ready for the advisor.
24. Sentence length variance: pick three random paragraphs, compute σ of word-count per sentence. Each must be ≥ 6.

If any check fails, fix it. If a fix would require a number that is not in the result files, mark it `[NUMBER UNAVAILABLE — VERIFY]` and append to `pending_numbers.md`.

---

## 20. Hand-Off Checklist

When done, the assistant produces these artefacts at the repository root:

1. The `manuscript/` directory with all files listed in §4.
2. `numbers_provenance.md` — every number traced to a source.
3. `bibliography_verification_log.md` — every reference verified.
4. `pending_numbers.md` — any unresolved numbers (should be empty if the result files are complete).
5. A short `manuscript_status.md` listing word counts per section, the result of each quality-gate check from §19, and any remaining issues for the user to resolve.

The user will open `manuscript/` in Overleaf, verify provenance, and compile the PDF for review by the advisor.

---

## 21. What Not to Do

- Do not invent results, citations, or numbers.
- Do not write more than the word budget. The draft will be reviewed by hand; padding wastes the reviewer's time and triggers rejection.
- Do not include this instruction file or any of its content in the manuscript.
- Do not insert filler phrases such as *In conclusion*, *It is worth noting*, *In summary*, *To this end*.
- Do not use bullet points outside the contributions list and tables.
- Do not personify models. The model does not *aim*, *strive*, or *seek*. The authors design and evaluate.
- Do not open any section with *In this section we...*.
- Do not mix US and British spellings.
- Do not generate praise or self-congratulation. *We achieve* is fine; *We achieve impressive results* is not.
- Do not write the manuscript in a single pass. Draft section by section, run the §19 checks after each section, and only then move on.

---

## 22. When Anything Is Ambiguous

Stop and ask the user. Do not guess at:

- Author names and affiliations.
- The exact title (use the working title in §6 unless told otherwise).
- The chosen focal model name string in `experiment_metadata.json` if the file lists alternates.
- Any numerical claim that cannot be located in a result file.

The user has been clear: a paper that misses the advisor's bar wastes weeks. The cost of asking is one message; the cost of guessing is one rejection.

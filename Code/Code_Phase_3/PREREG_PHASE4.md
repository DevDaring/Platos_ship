# Phase 4 pre-registration — three targeted experiments and one offline comparison

**Frozen before any Phase 4 API call.** Committed to the public repository on
25 September 2026; the commit timestamp is the freeze time. The code that runs
these experiments (`tools/run_phase4.py`, `tools/build_heldout_pool.py`, the
`show_initial` switch in `src/revision_runner.py`, the `natural_panel` peer
source in `src/peer_pools.py`, conditions `A_*`, `B_*`, `C_*` in
`config/experiment.yaml`) is committed with this file. Nothing here changes the
frozen Phase 3 plan (`ANALYSIS_PLAN.md`, commit 6c78611, 23 Sept 2026) or its
families; Phase 3 results stay as reported. All Phase 4 outputs are written
under `results/phase4/` and every estimate below is reported whatever its sign.

Common settings (identical to Phase 3): temperature 0.7 for focal models and
0.9 for weak peers, output cap 2,048 tokens, three replicates per question,
the same revision template, the same answer extraction and judge cascade, the
same pinned served-model check. Unit of inference: the question. Intervals:
percentile bootstrap over questions, 5,000 resamples, seed 20260502. p-values:
two-sided sign-flip permutation over question-level differences. Unparsed
answers are scored incorrect; worst-case bounds accompany each primary
estimate. No optional stopping: a run either completes its cells or the
missing cells are reported. Failed calls are retried until recorded; a cell
that cannot be built (for example a missing peer message) is reported, never
replaced.

## Experiment A — does showing the first answer change the wrong-peer effect?

**Question.** The earlier version (Protocol A) omitted the model's first answer
from its peer conditions and reported a gain from wrong peers that does not
appear under the matched protocol. Is the wrong-peer effect different when the
first answer is visible than when it is hidden?

**Design.** 2 x 2 factorial, all four arms run concurrently in one collection
window (the arms of a model run in parallel threads, interleaving calls):

| | no peers | two wrong peers (the WR personas) |
|---|---|---|
| first answer shown | `A_R` | `A_WR` |
| first answer hidden | `A_Rhid` | `A_WRhid` |

The hidden arms use the same template without the "Your previous response"
block; only the sentences that mention a previous response change. Every arm
reuses the cached Phase 3 first answers (the hidden arms still use them to
define the initially correct cohort, so their transitions are resampling
transitions). Models: DeepSeek-v4-flash and Gemma-3-27B. Pool: the 300 main
questions, 3 replicates. Calls: 4 x 2 x 900 = 7,200.

**Primary estimand (one test, alpha 0.05).**
interaction = (Acc[A_WR] − Acc[A_R]) − (Acc[A_WRhid] − Acc[A_Rhid]),
pooled over the two models (question-level values averaged over models and
replicates), bootstrap CI and sign-flip p.

**Secondary (reported, no correction claimed):** the interaction per model;
the same interaction for harmful revision, beneficial revision and target
adoption (adoption against the WR targets, R arms scored against the same
targets).

**Precision.** From the Phase 3 data for these two models, the per-question SD
of WR − R is 0.143, giving an interaction 95% half-width of about ±2.3 points.

**Interpretation.** A CI excluding zero supports a bounded statement that
first-answer visibility changes the wrong-peer effect under this prompt. A
precise near-zero interaction weakens that explanation; a wide one is
inconclusive. Neither identifies every cause of historical differences.

## Experiment C — matched multi-round control

**Question.** Does harmful revision grow over rounds under wrong peers more
than it grows with no peers?

**Design.** `C_R` (no peers) and `C_WR` (two wrong peers that re-assert), three
rounds each; every round shows the model its own previous answer; both arms run
concurrently. Models: DeepSeek-v4-flash and Gemma-3-27B. Pool: the 100-item
mitigation subset, 3 replicates. Calls: 2 arms x 2 models x 300 units x 3
rounds = 3,600.

**Primary estimand.**
[H(C_WR, round 3) − H(C_R, round 3)] − [H(C_WR, round 1) − H(C_R, round 1)],
cohort = units correct at Round 0 (the same cohort in every round), pooled over
the two models, bootstrap over questions.

**Secondary:** the same for accuracy, joint loss P(correct at Round 0 and wrong
at round k) and adoption; per model.

**Precision.** About ±4.7 points (Phase 3 round data). Reported whatever its size.

## Experiment B — natural errors on held-out questions

**Question.** Does target adoption appear when the wrong answers are ordinary
weak-model errors rather than constructed distractors?

**Pool.** 420 held-out MMLU-Pro test questions, 60 from each of seven subjects
that need little arithmetic (biology, health, history, law, other, philosophy,
psychology), none in the main pool (`tools/build_heldout_pool.py`, seed
20260925). Short multiple-choice answers with independently fixed gold labels.

**Bank.** Llama-3.1-8B and Gemma-3-4B each answer every question 4 times with
the ordinary Round-0 prompt (no persona, no assigned answer): 3,360 calls. All
outputs are kept. A question is **eligible** when the bank holds at least two
correct and two wrong usable answers. The eligible cohort is frozen
(`results/phase4/B/eligible.json`) before any focal call; bank size, natural
error rates, eligibility and exclusions are reported.

**Panels (nested, seeded per question and replicate).** Two correct and two
wrong messages are drawn at random from the eligible question's bank:
`B_P0` shows the two correct messages, `B_P1` the first correct and the first
wrong, `B_P2` the two wrong; `B_R` shows none. Peer order is randomised.

**Models** (chosen before results): DeepSeek-v4-flash (strong reference),
Gemma-3-27B, Llama-3.1-8B. Each gets its own Round-0 answers on the eligible
questions (3 replicates) and all four arms. Calls: 3 x 3 x n_eligible for
Round 0 plus 4 x 3 x 3 x n_eligible revisions.

**Primary estimand.** Excess adoption in `B_P2`: P(final answer ∈ displayed
wrong targets | correct at Round 0) − P(`B_R` answer of the same unit ∈ the
same targets | correct at Round 0), pooled over the three models, trial-weighted
with a question-cluster bootstrap. One test, alpha 0.05.

**Secondary:** the same for `B_P1`; accuracy and joint loss for `B_P0`, `B_P1`,
`B_P2` against `B_R`; per model. Adoption never substitutes for net damage.

**Scope of the claim.** Selection on available errors changes the population:
this is controlled exposure to naturally produced errors on eligible questions,
not unconditional deployment risk. The Phase 3 adoption results stay
exploratory; this study does not relabel them.

## Experiment D — verifier against equal extra resources (offline, no new calls)

Computed from cached Phase 3 calls on the verification-safeguard units (WR and
N, three models, 100-item subset):
`independent_answer` adopts a change only when Qwen2.5-72B's own unanchored
answer to the same question and replicate equals the revision;
`extra_focal_sample` adopts it only when another cached sample of the focal
model does; `verifier_model_alone` uses Qwen2.5-72B's own answer throughout.
The random comparator is scored as its exact expectation at the verifier's
acceptance rate within each model. Verifier-minus-policy differences get
question-paired bootstrap CIs. Calls and output tokens per unit are reported
for each policy. Exploratory.

## What each outcome licenses

A null or wide interval is reported as such and is not described as evidence
of absence. No result is promoted to a headline unless it is the primary
estimand of its experiment. Deviations from this file are listed in the paper.

# Analysis plan — Phase 3

**Status:** freeze this file before the first confirmatory run, and record the
commit hash in `results/outputs/experiment_metadata.json`. Nothing below is
described as pre-registered for Phase 1/2; those analyses are exploratory and
the paper says so.

---

## 1. Design in one paragraph

For each focal model *f*, question *q* and replicate *k*, one initial answer
`A0(f, q, k)` is generated and **cached**. Every revision condition *c* then
issues one call that shows the model the same question, the same cached
`A0`, the same revision instruction and the same output cap; only the peer
block differs. The outcome is final correctness `Y(f, q, k, c)`.

The Phase-1/2 protocol (Protocol A) is retained as a comparison: there the
focal model's Round-1 prompt omitted its own Round-0 text, so part of every
measured flip was fresh sampling rather than revision.

## 2. Estimands

With `A0` initial correctness and `Y` final correctness:

| Quantity | Definition | Denominator |
|---|---|---|
| accuracy(c) | P(Y = 1) | all units |
| H(c) harmful | P(Y = 0 \| A0 = 1) | units with A0 = 1 |
| B(c) beneficial | P(Y = 1 \| A0 = 0) | units with A0 = 0 |
| joint loss | P(A0 = 1, Y = 0) | all units |
| adoption(c) | P(Y = peer target \| A0 = 1) | A0 = 1 with a target |

`accuracy(c) − a = (1 − a)·B(c) − a·H(c)` where `a = P(A0 = 1)`. This is
bookkeeping, not a result, and `analysis/run_analysis.py` verifies it holds in
every cell.

**Adoption is reported separately from H.** A model that abandons a correct
answer for some *other* wrong answer is unstable; a model that moves to the
peer's answer is influenced by the peer. Only the second is evidence about
peer influence on the answer.

## 3. Unit of inference

**The question.** Replicates are collapsed to one value per question, then
paired across conditions. 1,500 trials are not 1,500 independent observations.
A question is dropped from a contrast when either side has an empty
denominator, pairwise.

## 4. Primary contrast families

Declared in `config/experiment.yaml` under `analysis.contrast_families`. Holm
within each family; raw p-values always reported so the correction can be
redone under another grouping.

**Family X1_primary** (six entries)
1. accuracy(WR) − accuracy(R)
2. accuracy(H) − accuracy(R)
3. accuracy(E) − accuracy(R)
4. H(WR) − H(R)
5. Spearman(solo accuracy, H(WR) − H(R)) across the eight focal models, Protocol B
6. the same correlation under Protocol A (replication of the published result)

**Family X2_primary** (three entries)
1. accuracy(WR) − accuracy(G) — does message content matter beyond being challenged?
2. accuracy(WR) − accuracy(W) — does the rationale matter beyond the bare answer?
3. H(WR) − H(WRh) — does confident wording matter with content fixed?

Everything else — X3, X4, X6, X7, per-model breakdowns, subject-level splits,
the SF framing contrast — is **exploratory** and labelled as such in the paper.

## 5. Inference procedures

- **Intervals:** percentile bootstrap over question-level values, 5,000
  resamples, seed 20260502.
- **p-values:** two-sided sign-flip permutation on paired question-level
  differences. Never reported as 0; the estimator is (count + 1) / (n + 1).
- **The 8-model correlation:** Spearman with an **exact** permutation p-value
  (8! = 40,320 permutations enumerated), plus leave-one-model-out range. An
  asymptotic p-value at n = 8 is not used.
- **Secondary:** GEE logistic, clustered by question, exchangeable correlation,
  reported as odds ratios with intervals — not as p-values alone.
- **McNemar** at the question level is retained only for comparability with the
  reviewed submission.

## 6. The capability gradient

Primary form subtracts the matched baseline:

```
excess(f) = H(f, WR) − H(f, R)
```

Weak models revise more under *any* revision prompt. Subtracting R removes
that common churn, so what remains is specific to confidently wrong peers. The
raw `H(f, WR)` version is reported alongside, because the reviewed paper
reported the raw one.

**Both outcomes are publishable and the write-up is prepared for either.** If
the gradient survives under Protocol B, it is the paper's headline. If it
collapses once the initial answer is held fixed, the finding is that the
published gradient was substantially resampling churn — which is a genuine
correction to the literature and is reported as the result, not buried.

## 7. Stopping, exclusions and missing data

- No optional stopping. The plan fixes questions, replicates and conditions in
  advance; a run either completes a cell or the cell is reported as incomplete.
- Unrecovered parse failures are scored **incorrect** (conservative, and the
  same convention as Phase 1/2). Their count is reported per condition and a
  bounding sensitivity is given for the headline contrast.
- Declared exclusions, applied once in `analysis/registry.py` and printed in
  the appendix:
  1. the quarantined perturbed-GSM8K trials (invalid gold labels);
  2. the superseded 50-question GPT-4o-mini cross-validation subset;
  3. focal responses served by a checkpoint other than the pinned one.
- Peer-pool shortfalls (a condition that could not be built for a question) are
  logged and counted, never silently skipped.

## 8. Claims that are not available from this analysis

By construction, there is no function in `analysis/` that produces any of these:

- **non-inferiority or "safe"** claims. "No statistically significant loss" is
  not evidence of no loss. A non-inferiority claim would need a margin declared
  before the run; none is declared, so none is made.
- **"all point estimates are non-negative"** as a headline. Point estimates
  without intervals are not a finding.
- **contamination ruled out.** X4 supports a comparison between regenerated and
  matched original items, nothing stronger.
- **a mechanism.** X8 measures output-distribution movement in one open-weight
  model. That is behaviour beneath the stated answer, not a representation-level
  account.
- **a capability law.** Eight models is an association across a small sample.

## 9. What would falsify the paper's main claim

Stated in advance so the result is interpretable either way:

| Claim | Falsified if |
|---|---|
| The cost of wrong peers is graded by focal ability | Protocol-B Spearman is near zero or its leave-one-out range crosses zero |
| Wrong-peer content matters beyond being challenged | accuracy(WR) − accuracy(G) has an interval containing zero |
| Confident wording drives corruption | H(WR) − H(WRh) has an interval containing zero |
| Self-reported confidence cannot filter peers | Δ_ret exceeds 0.10 with AUROC above 0.60 on a two-class substrate |

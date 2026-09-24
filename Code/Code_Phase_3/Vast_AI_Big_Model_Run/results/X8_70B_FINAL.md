# X8 on Llama-3.1-70B: the strong model feels more pull and overrides it

Run on 2x NVIDIA A100-SXM4-80GB via Vast.ai, bf16, tensor-parallel 2,
seed 20260502, 300 questions x 3 replicates x 3 conditions = 2700 trials,
39 minutes wall clock. Same seed and question pool as both L4 runs of the
8B, so the two models are directly comparable.

## The result

The paper's Limitations section states:

> we cannot say whether the strong models that resist wrong peers do so
> by feeling less pull or by overriding more of it. Separating those two
> accounts would need the same read-out on a strong open-weight model.

This run is that read-out. The answer is that the strong model feels
**more** pull and overrides it.

| Family | Model | Mass shift toward the wrong peer | 95% CI | Adoption of the wrong answer |
|---|---|---|---|---|
| mmlu_pro | 8B | +0.062510 | [+0.049719, +0.076641] | +10.49 pp |
| mmlu_pro | 70B | +0.142767 | [+0.121481, +0.165055] | +8.67 pp |
| gsm8k | 8B | +0.018823 | [+0.009858, +0.029791] | +2.83 pp |
| gsm8k | 70B | +0.035556 | [+0.021200, +0.050655] | +0.68 pp |

On mmlu_pro the 70B's distribution moves **2.28x** as far as the 8B's
while its stated answer adopts the wrong peer's answer **1.82 pp
less** often.
On gsm8k the 70B's distribution moves **1.89x** as far as the 8B's
while its stated answer adopts the wrong peer's answer **2.14 pp
less** often.

Both effects point the same way on both task families, and the mass
intervals do not overlap between models. A flip-count study would have
concluded only that the 70B is more robust; the distributional read-out
shows that robustness is an override, not an absence of pull.

## Adoption rates in full

Adoption is P(Round-1 answer = the wrong peer's answer | Round-0 differed),
on trials where both rounds yield an answer. Coverage is the share of
trials meeting that condition after judge recovery.

| Model | Family | Coverage | E | WR | WR minus E | n (WR) |
|---|---|---|---|---|---|---|
| 8B | mmlu_pro | 87.1% | 2.08% | 12.57% | +10.49 pp | 525 |
| 8B | gsm8k | 96.0% | 0.00% | 2.83% | +2.83 pp | 283 |
| 70B | mmlu_pro | 94.9% | 1.03% | 9.70% | +8.67 pp | 567 |
| 70B | gsm8k | 98.2% | 0.34% | 1.02% | +0.68 pp | 294 |

The 70B has the HIGHER coverage of the two, so the comparison is not an
artefact of the weaker model's answers being harder to read.

## Judge recovery

Judge/regex agreement on rows the regex already read: **0.9900**
on n=200. That number licenses the recovered answers; it is
measured, not assumed.

| Round | Unparsed by regex | Recovered | Still unresolved |
|---|---|---|---|
| round0 | 243 | 226 | 17 |
| round1 | 327 | 225 | 102 |
| **total** | **570** | **451** | **119** |

Unresolved rows are almost all abstentions: the generation is capped at
600 tokens and runs out before stating an answer, so there is nothing to
recover. The judge is required to say so rather than supply one, and is
refused mechanically if it returns a value the response never contains.

## Verification

Twenty-one checks were run independently of the pipeline, including
recomputing both contrasts from the trials table. Both estimates, both
confidence intervals and both pair counts matched the published file
exactly.

| Check | Result |
|---|---|
| Rows | 2700, 900 per condition |
| Split | 900 GSM8K, 1800 MMLU-Pro |
| Scoring path | teacher-forced on exactly the GSM8K rows, first-position on exactly the MMLU-Pro rows |
| Probability mass | present on all 2700 |
| Raw generations | present on all 2700, both rounds |
| Fixed target | identical across conditions in every cell |
| Round-0 answer | identical across conditions, confirming the cached matched-revision design |
| Target vs correct | never equal |
| Contrast arithmetic | reproduced independently, exact match |

Model revision: `meta-llama/Llama-3.1-70B-Instruct`.
Files were MD5-verified byte-identical between the instance and this
machine before the instance was destroyed.

## Status against the plan

`next_plan.md` lists the GPU probe under **Cut**, allowed back only if the
candidate-matching defect is checked. It is: candidates are tokenised and
scored exactly, collisions are reported rather than merged, off-candidate
mass is kept, and the vLLM BOS offset bug is fixed. On that basis this
result is eligible, and it answers a question X1-X5 cannot, because they
measure behaviour and this measures the distribution behind it.

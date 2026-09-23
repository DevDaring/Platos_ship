# X8 output-distribution probe: final results

Two independent runs of the same design, the second carrying the raw
generations so unreadable answers could be re-read by a judge.

Model `meta-llama/Llama-3.1-8B-Instruct`, bf16, single NVIDIA L4, seed
20260502, 300 questions x 3 replicates x 3 conditions = 2700 trials.

## Headline contrast, WR minus E

Shift in probability mass toward the answer the wrong peer asserted.
Paired by (question, replicate) against a fixed target. Report per
task family; the pooled row is written but carries `report: False`,
because the two families differ in probability scale by about two
orders of magnitude and a pooled mean is dominated by multiple choice.

| Family | Run 1 | Run 2 | Difference |
|---|---|---|---|
| mmlu_pro | +0.060349 [+0.047586, +0.074238] | +0.062510 [+0.049719, +0.076641] | 0.002161 |
| gsm8k | +0.018871 [+0.010939, +0.028688] | +0.018823 [+0.009858, +0.029791] | 0.000047 |

Both runs give p = 0.0001 by sign-flip permutation on 600 (MMLU-Pro)
and 300 (GSM8K) pairs.

## What reproduces, and what does not

The two runs used different images, CUDA versions and vLLM builds, and
generation is sampled at temperature 0.7. Individual generations
therefore diverge: Round-1 answers agree on only
66.9% of the 2700 cells, and per-trial
probability mass can differ by almost its whole range.

The ESTIMATES nonetheless agree to within 0.002, and on GSM8K to within
0.00005. The effect is a property of the design rather than of one
sample of generations. That is the stronger claim, and it is only
visible because the run was repeated.

## Unparsed answers, and what the judge recovered

The regex extractor reads a `Final answer:` line. Where the model
never writes one, the answer is unreadable to it. The probability mass
is unaffected either way: it is read from the logprobs and is present
on 2700 of 2700 rows. Only the stated-answer-unchanged SUBSET depends
on parsing.

| Round | Unparsed | Recovered by judge | Still unresolved |
|---|---|---|---|
| round0 | 276 | 190 | 86 |
| round1 | 496 | 262 | 234 |

Judge/regex agreement on rows the regex ALREADY read: **0.9750** on n=200. That number licenses the
recovered answers, and it is measured rather than assumed.

Most unresolved rows are abstentions, not failures. Generations are
capped at 600 tokens and many run out mid-derivation, so no stated
answer exists to recover. The judge is required to say so rather than
supply one.

## The subset estimate, three ways

| Family | Complete case | Judge recovered | Sharp bounds (before) | Sharp bounds (after) |
|---|---|---|---|---|
| gsm8k | +0.008973 | +0.007832 | [+0.005290, +0.015953] | [+0.006503, +0.008008] |
| mmlu_pro | +0.020012 | +0.019172 | [+0.015567, +0.070675] | [+0.015969, +0.064090] |

The bounds cover every value the subset mean could take under any
resolution of the unreadable rows (Manski 1990; Horowitz & Manski
2000). Recovery narrows them, most sharply on GSM8K, from a width of
0.010663 to 0.001504, a factor of 7.1.

Every lower bound is above zero, before and after recovery. The sign
does not depend on how the unreadable generations would have resolved,
so no judge could overturn it.

## Consistency check

A recovered estimate must lie inside the bounds computed BEFORE
recovery. One outside them would mean the judge resolved ambiguity in
a way no resolution could, i.e. invented data. `finalise_x8.py` exits
non-zero if that happens.

| Family | Recovered | Pre-recovery bounds | Verdict |
|---|---|---|---|
| gsm8k | +0.007832 | [+0.005290, +0.015953] | inside |
| mmlu_pro | +0.019172 | [+0.015567, +0.070675] | inside |

## What the paper should say

Mass moves toward the wrong peer's answer, and it moves on trials
where the stated answer never changed. Flip counts alone undercount
the effect. The subset estimate is bounded rather than reported as a
complete case, because the non-parse rate differs across arms and a
complete-case figure is therefore not automatically unbiased.

Limitation to state: of 772 unreadable answers across both rounds,
452 were recovered by a condition-blind judge and 320 remain
unresolved, almost all because the generation states no answer at all.
Those rows stay in the ambiguous set that the bounds cover.

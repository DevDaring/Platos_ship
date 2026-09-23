# Unparsed Round-1 answers: what they can and cannot change

501 of 2700 Round-1 generations did not yield an answer under the
regex extractor. This file records what that does to each estimand.

## The primary estimand is untouched

The shift in probability mass is read from the logprobs and never
passes through the extractor. It is observed on 2700 of 2700 rows,
including all 501 that failed to parse. Non-parse cannot bias it.

## Non-parse rate by condition

| Condition | Unparsed | Trials | Rate |
|---|---|---|---|
| E | 166 | 900 | 18.44% |
| R | 187 | 900 | 20.78% |
| WR | 148 | 900 | 16.44% |

The rate is not equal across arms, so the stated-answer subset is
not a random subsample and a complete-case estimate is not
automatically unbiased. That is why the subset is bounded rather
than simply reported.

## Sharp bounds on the stated-answer-unchanged subset (WR minus E)

The outcome is observed for every pair; only membership of the
subset is unknown when an answer is unreadable. The interval below
covers every value the subset mean could take under any resolution
of the unparsed rows. See `analysis/nonparse_bounds.py`.

| Family | Complete case | Sharp bounds | CI covering the set | Known-in | Ambiguous | Sign robust |
|---|---|---|---|---|---|---|
| gsm8k | +0.008623 | [+0.005607, +0.012625] | [+0.001977, +0.021071] | 145 | 104 | yes |
| mmlu_pro | +0.018806 | [+0.016304, +0.065718] | [+0.009668, +0.090753] | 263 | 84 | yes |

Both lower bounds are above zero, and so is the lower limit of the
interval that additionally carries sampling uncertainty. The sign
of the effect therefore does not depend on how the 501 unreadable
generations would have resolved.

## Why no judge cascade was run on these artefacts

The probe consumed each generation, extracted an answer and
discarded the text. No parquet and no log retains it, so there is
nothing for a judge to read. `run_big_probe.py` now persists
`round0_text` and `round1_text`, and `analysis/recover_unparsed.py`
will judge the failures on any future run. Recovery would narrow
the interval; it cannot change the sign, which is already settled.

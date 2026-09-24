# X8 across six open-weight models

The X8 output-distribution probe run identically on six open-weight models,
4B to 72B, from four families: seed 20260502, bf16, the same 300 questions,
3 replicates, conditions R, E and WR, 2,700 trials each. The question is
whether a model's distributional pull toward a wrong peer's answer tracks its
capability, as the paper's behavioural gradient does.

## Answer

**No.** Across six models, distributional pull does not track capability.
The two-model result reported earlier, that Llama-3.1-70B feels about twice
the pull of Llama-3.1-8B while adopting less, is a within-Llama scaling
effect. Qwen2.5-72B, the most capable model here, shows almost the lowest
pull of all, with a tight interval that excludes zero.

Two things do hold:

- **Pull looks like a family property.** Gemma and Qwen show low pull at
  every size; Llama and Mistral show high pull. With two families on each
  side this is suggestive, not established.
- **Adoption of the wrong answer falls with capability**, and stays
  negative in every leave-one-out subset. That corroborates the paper's
  behavioural gradient on six new models measured a different way. It is
  not significant at six models.

## MMLU-Pro (primary)

WR minus E, paired by question and replicate. Question-clustered 95% CIs.

| Model | Solo acc | Raw pull | 95% CI | Log-odds | Adoption |
|---|---|---|---|---|---|
| Gemma-3-4B | 0.378 | +0.035891 | [+0.0202, +0.0545] | +1.149 | +12.20 pp |
| Llama-3.1-8B | 0.510 | +0.062510 | [+0.0478, +0.0782] | +3.029 | +11.12 pp |
| Mistral-Small-24B | 0.607 | +0.170618 | [+0.1487, +0.1931] | +4.559 | +12.90 pp |
| Gemma-3-27B | 0.635 | +0.038238 | [+0.0207, +0.0570] | +2.210 | +13.35 pp |
| Llama-3.1-70B | 0.660 | +0.142767 | [+0.1146, +0.1708] | +4.171 | +9.00 pp |
| Qwen2.5-72B | 0.690 | +0.024470 | [+0.0137, +0.0369] | +2.341 | +2.35 pp |

Spearman correlation with solo accuracy. The p-value is exact over all 720
orderings; the leave-one-out range shows whether rho rests on one model.

| Measure | rho | Exact p | Leave-one-out |
|---|---|---|---|
| Raw pull | -0.143 | 0.8028 | [-0.500, +0.500] |
| Log-odds pull | +0.257 | 0.6583 | [-0.300, +0.500] |
| Adoption | -0.486 | 0.3556 | [-0.700, -0.100] |

All three measures were fixed before the six-model result was seen and all
are reported. Raw and log-odds pull disagree in sign and neither is
distinguishable from zero, which is itself the finding: pull has no stable
relationship with capability.

## GSM8K is not a valid read-out

The probe appends `Final answer:` and reads the next tokens. On multiple
choice that captures belief, because models commit to a letter at once. On
arithmetic it forces an answer before any working, so it reads an unreasoned
guess. On problems a model then solved correctly by reasoning, the probe's
probability on that answer:

| Model | Median P, MMLU-Pro | Median P, GSM8K |
|---|---|---|
| Gemma-3-4B | 1.000 | 0.000 |
| Llama-3.1-8B | 0.904 | 0.013 |
| Mistral-Small-24B | 0.600 | 0.033 |
| Gemma-3-27B | 1.000 | 0.000 |
| Llama-3.1-70B | 0.967 | 0.071 |
| Qwen2.5-72B | 0.996 | 0.159 |

GSM8K results are in `x8_gradient_report.json` for completeness and must
not carry a distributional claim. This applies equally to the Llama-3.1-8B
and Llama-3.1-70B GSM8K numbers reported earlier. Conditioning the score on
the model's own reasoning is not a fix: the reasoning already states the
answer, so the probability collapses toward 1.

## Judge recovery

Answers the regex could not read were recovered by a condition-blind judge
cascade. The number that licenses a recovered answer is precision when the
judge commits: strict string agreement counts abstentions, formatting and
cases where the judge gives the letter for a value the regex kept as errors.

| Model | Strict agreement | Abstained | Precision when committed |
|---|---|---|---|
| Gemma-3-4B | 0.9300 | 4 of 200 | 0.9592 |
| Llama-3.1-8B | 0.9750 | 4 of 200 | 1.0000 |
| Mistral-Small-24B | 0.9600 | 2 of 200 | 0.9747 |
| Gemma-3-27B | 0.9400 | 4 of 200 | 0.9592 |
| Llama-3.1-70B | 0.9900 | 0 of 200 | 0.9950 |
| Qwen2.5-72B | 0.9250 | 12 of 200 | 0.9947 |

Precision is a lower bound for the Gemma models: several of their remaining
disagreements are the judge giving a letter for an option value, where the
judge is right and the regex is wrong.

## Corrections made during this sweep

- **Option values scored as wrong.** Models sometimes answer with an option's
  value rather than its letter; the regex kept `-42` where option G was
  `-42`, so a correct answer never matched the letter-keyed key. The rate
  differed by family, which can manufacture a family effect, so answers are
  now mapped to a letter when they match exactly one option. Values mapped:
  Gemma-3-4B 63 Round 0, 52 Round 1.
  Llama-3.1-8B 6 Round 0, 1 Round 1.
  Mistral-Small-24B 33 Round 0, 15 Round 1.
  Gemma-3-27B 75 Round 0, 63 Round 1.
  Llama-3.1-70B 12 Round 0, 5 Round 1.
  Qwen2.5-72B 18 Round 0, 2 Round 1.
  Every Spearman rho was unchanged, because no accuracy ranking changed;
  Gemma-3-27B's accuracy rose by 3.7 points.
- **70B results deleted from main.** The sweep's autopush synced its folder
  into the 70B folder with `rsync --delete`. Restored byte-for-byte from the
  MD5-verified local copy; the push script now refuses a mismatched source
  and destination.
- **Gemma chat template.** The prompt builder hardcoded a system role; it now
  probes the template and records which envelope each model received.
- **Mistral-Small-3.2-24B-2506** ships no HuggingFace chat template, so the
  2501 checkpoint was used. Its scoring was verified directly.

## Verification

Every model passed: 2,700 rows with 900 per condition; teacher-forced
scoring on exactly the GSM8K rows; probability mass on every row; target
fixed per cell; Round-0 cached across conditions; seed and precision
recorded; and both contrasts reproduced exactly when recomputed
independently from the trials table. All files were MD5-verified between
instance and machine before the instances were destroyed.

## What the paper can say

The distributional read-out is valid on multiple choice. Across six
open-weight models, pull toward a wrong peer does not track capability and
varies more between model families than within them, while adoption of the
wrong answer declines with capability. Behavioural robustness is therefore
not explained by weaker distributional pull. With six models neither
gradient is significant, and the paper should say so.

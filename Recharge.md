# Recharge.md — do you need to top up any account?

## ✅ NO. Nothing needs recharging.

**Verified live at 2026-08-01 07:21 UTC.** Every account has ample balance for
the remaining work on the ACL/EACL resubmission.

Re-check any time with:

```bash
bash /home/Debz/Research/Platos_ship/Code/scripts/check_balances.sh
```

---

## Live balances

| Provider | What it pays for | Balance | Status |
|---|---|---|---|
| **DeepSeek** | Primary focal model `deepseek-v4-flash` | **$10.38** | ✅ ~40× what is left to spend |
| **OpenRouter #1** | Weak peers, sweep focals, Gemini judge | **$21.67** ($18.68 usable under the key's own $30 spend limit) | ✅ ample |
| **OpenRouter #2** | Same pool, round-robin partner | **$21.67** (no key limit set) | ✅ ample |
| **Vast.ai** | GPU rental | **$22.55** | ✅ GPU work finished; VM destroyed |
| LinkAPI | Fallback route for GPT-4o-mini | key valid | ℹ️ not on the critical path |
| nano-gpt | Fallback route for weak models | key valid | ℹ️ not on the critical path |
| Mistral | Judge tier 2 | key valid | ✅ fires on <1% of calls |

LinkAPI and nano-gpt expose no balance endpoint, so only key validity can be
checked automatically. Neither is required — they are fallbacks that engage only
when OpenRouter rate-limits, and OpenRouter has been recovering fine.

---

## Why the answer is "no" — measured, not estimated

I originally estimated ~$11 of DeepSeek spend and asked you to top up $15.
**That estimate was wrong by roughly 50×.** Actual metered usage:

- 350 split-peer trials so far → 700 DeepSeek calls
  (~226k input + ~145k output tokens)
- DeepSeek balance moved **$10.41 → $10.38** over that period, i.e.
  **~$0.03 per 350 trials**

| Remaining work | Trials | Measured cost |
|---|---|---|
| Split-peer (C4split), rest of the run | 1,150 | ~$0.10 |
| Heterogeneous debate (C2het) — DeepSeek is 1 of 3 debaters | 900 | ~$0.08 |
| Judge fallback + retry headroom | — | ~$0.05 |
| **Total remaining** | **~2,050** | **~$0.25** |

## Total cost of the entire resubmission

| Item | Actual |
|---|---|
| Vast.ai GPU rental — mechanistic probe, 2,700 trials, VM destroyed | **$0.77** |
| DeepSeek — split-peer + heterogeneous debate | ~$0.30 |
| OpenRouter — weak peers, 1,500 correct-anchored personas, judge | ~$0.10 |
| **Everything** | **well under $2** |

The GPU number is low because the probe was rewritten to batch through vLLM per
stage instead of one prompt at a time: 2,700 trials ran in **8.5 minutes**
instead of the ~15 hours the original sequential code would have taken.

---

## The one thing that *does* need you (not money)

The paper's artefact link is dead —
`https://anonymous.4open.science/r/Platos_ship/` returns HTTP 401. Reviewer
gL73 scored Datasets **1/5** and Software **1/5**, and a working anonymised
repository is the cheapest way to lift those. It needs your GitHub login, so I
cannot create it. Full instructions in `Code/ACTION_REQUIRED.md`.

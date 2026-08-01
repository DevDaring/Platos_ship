# Recharge.md — account balances and top-ups needed

**Last checked: 2026-08-01 (live API queries).** Re-run
`bash scripts/check_balances.sh` to refresh this table.

## Live balances

| Provider | What it pays for | Balance now | Verdict |
|---|---|---|---|
| **DeepSeek** (direct) | Primary focal model `deepseek-v4-flash` (all C1–C5 DeepSeek trials) | **$10.41** | ⚠️ **Top up — see below** |
| **OpenRouter key 1** | Weak peers, sweep focals, Gemini judge | $18.71 left (spend limit $30, used $11.29) | ✅ OK |
| **OpenRouter key 2** | Same pool, round-robin partner | $21.70 left ($28 credited, $6.30 used) | ✅ OK |
| **Vast.ai** | GPU rental for the mechanistic probe (N7) | **$23.32 credit** | ✅ OK |
| LinkAPI (`AZUREOPENAI_LINKAPI_KEY`) | Cheap GPT-4o-mini route | key valid (HTTP 200); balance not exposed by API | ℹ️ check manually |
| nano-gpt | Fallback route for weak models + GPT-4o-mini | key valid (HTTP 200); balance not exposed by API | ℹ️ check manually |
| Mistral (La Plateforme) | Judge tier 2 | key valid (HTTP 200) | ✅ OK (judge fires <1% of calls) |

## What I need you to recharge

### 1. DeepSeek — **recommended top-up: $15** (minimum $10)

DeepSeek is the single account that gates the biggest remaining runs. The
planned new experiments that use it:

| Experiment | DeepSeek trials | Est. DeepSeek spend |
|---|---|---|
| N1 split-peer (C4split), 300 q × 5 reps | 1,500 | ~$2.5 |
| N2 heterogeneous debate (DeepSeek is 1 of 3 debaters), 300 q × 3 | 900 | ~$1.5 |
| N3 MGSM multilingual, 5 langs × 50 q × 3 reps × 3 conditions | 2,250 | ~$3.5 |
| N4 held-out math items, 100 q × 5 reps × 2 conditions | 1,000 | ~$1.5 |
| Judge tier-3 fallback + re-runs / retries headroom | — | ~$2 |
| **Total** | **~5,650** | **~$11** |

$10.41 leaves no margin for a re-run if a condition needs repeating, and a
mid-run balance exhaustion wastes partial trials. **$15 is the safe number.**

Top up at: <https://platform.deepseek.com/top_up>

### 2. Everything else — no action needed right now

- **OpenRouter** (~$40 across both keys) comfortably covers the weak-peer and
  sweep-model side of all planned runs (~$8–12).
  - Note: key 1 has a **$30 spend limit** set on the key itself, and $18.71 of
    it is left. If you want that key to use its full credit, raise the limit at
    <https://openrouter.ai/settings/keys>. Not required — key 2 has headroom.
- **Vast.ai** $23.32 covers the GPU probe: one 24 GB RTX 4090 at ~$0.25–0.35/h
  for 10–15 h ≈ **$4–6**. No top-up needed.
- **LinkAPI / nano-gpt** are fallback routes only; they are not on the critical
  path. Their APIs do not expose a balance endpoint, so please eyeball them in
  their dashboards if you want to keep the fallbacks alive.

## Running total for the whole resubmission

| | Estimate |
|---|---|
| API inference (DeepSeek + OpenRouter, all new experiments) | ~$20 |
| Vast.ai GPU rental (mechanistic probe) | ~$5 |
| **Total new spend** | **~$25** |

Against current balances that is fully funded **except** for the DeepSeek
margin — hence the single $15 ask above.

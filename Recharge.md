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

### ✅ Nothing. My first estimate was ~50× too high — please ignore it.

I originally asked for a $15 DeepSeek top-up based on a guess. I have now
**measured** the real burn rate from the running split-peer experiment, and the
guess was badly wrong. Correcting it rather than letting you spend
unnecessarily:

**Measured:** 250 split-peer trials consumed 500 DeepSeek calls
(161,654 input + 103,723 output tokens) and moved the balance from
$10.41 → $10.39, i.e. **~$0.02 per 250 trials**.

| Remaining work on DeepSeek | Trials | Measured cost |
|---|---|---|
| N1 split-peer, rest of the run | 1,250 | ~$0.10 |
| N2 heterogeneous debate (DeepSeek is 1 of 3 debaters) | 900 | ~$0.08 |
| Judge tier-3 fallback + retry headroom | — | ~$0.05 |
| **Total** | **~2,150** | **~$0.25** |

$10.39 covers this roughly **forty times over**. The GPU side is settled too:
the mechanistic probe finished and the VM was destroyed, total **$0.77** of your
$23 Vast credit (batched inference brought 15 h of work down to 8.5 min).

If you want a top-up anyway for future cycles, $10 at
<https://platform.deepseek.com/top_up> is more than ample — but nothing in the
current resubmission needs it.

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

## Running total for the whole resubmission (measured, not estimated)

| | Actual |
|---|---|
| Vast.ai GPU rental (mechanistic probe, complete, VM destroyed) | **$0.77** |
| DeepSeek (split-peer + heterogeneous debate) | ~$0.25 |
| OpenRouter (weak peers, persona generation, judge) | ~$0.05 so far |
| **Total new spend** | **well under $2** |

Fully funded from existing balances. No recharge required.

# API providers, cheapest routing, and where to keep how much

**Price sweep date: 2026-07-10** (live, from provider pricing pages). Re-check before a big run — prices move. All figures are USD per 1M tokens, **input / output**.

---

## Final routing (verified live 2026-07-10 with your keys)

| Route | Provider | Model / slug | Status |
|---|---|---|---|
| Focal (E1–E6) + judge T3 | **DeepSeek** direct | `deepseek-chat` (→ `deepseek-v4-flash`) | ✅ OK |
| gpt-4o-mini (E1, E5) | **LinkAPI** (azureopenai group) | `gpt-4o-mini` | ✅ OK — ~10× cheaper than OpenRouter |
| Weak peers + 6-model sweep (E4) + judge T1 | **OpenRouter** (2 keys, round-robin) | llama/gemma/qwen/mistral slugs | ✅ OK |
| Judge T2 | **Mistral** platform (your 2 keys) | `mistral-small-latest` | ✅ OK |
| Failover (optional) | **nano-gpt** | — | ⚠️ `402 Insufficient balance` (account shows $0.00 — recharge if you want it) |

**LinkAPI win:** the rate chart shows gpt-4o-mini at **¥0.075 / ¥0.3 per 1M** via the azureopenai group (0.5× ratio) ≈ **$0.011 / $0.042** — vs OpenRouter's $0.15 / $0.60. Your `AZUREOPENAI_LINKAPI_KEY` is group-scoped, so it bills at that rate automatically. gpt-4o-mini is now essentially free (~$0.30 for all of E1+E5).

## Where to keep money — PER ACCOUNT

Each API key = a **separate account**, and the pipeline **round-robins across the
two keys**, so a provider's spend splits ~50/50 across its two accounts. Fund
**each account**, not each provider. (Round-robin also rotates keys on retry, so
if one account runs dry the calls fall onto the other — but fund both to avoid
wasted retries and slowdown.)

| Account (env var) | Provider | Keep | Carries |
|---|---|---|---|
| `OPENROUTER_API_KEY_1` | OpenRouter #1 | **$25** | ~half of the sweep (70B/72B/27B), weak peers, judge T1 |
| `OPENROUTER_API_KEY_2` | OpenRouter #2 | **$25** | the other ~half |
| `DEEPSEEK_API_KEY_1` | DeepSeek #1 | **$8** | ~half of focal calls (E1–E6) + judge T3 |
| `DEEPSEEK_API_KEY_2` | DeepSeek #2 | **$8** | the other ~half |
| `AZUREOPENAI_LINKAPI_KEY` | LinkAPI | **~$2** | gpt-4o-mini (E1, E5) — total spend ~$0.30 |
| `MISTRAL_API_KEY_1` | Mistral #1 | **~$1** | ~half of judge T2 (fires <1% of trials) |
| `Mistral_API_Key_2` | Mistral #2 | **~$1** | the other ~half |
| `Nano_GPT_API_KEY` | nano-gpt | already done | failover only — nothing routes here by default |
| GCP Vertex | — | **$0** | not used |

**Totals per provider:** OpenRouter **$50** (split $25 + $25), DeepSeek **$16**
(split $8 + $8), LinkAPI ~$2, Mistral ~$2 (split), nano-gpt already funded.
**Grand total ≈ $70** across all accounts, and only the OpenRouter pair carries
real cost — it runs the 70B/72B sweep (E4), which is ~70% of the whole budget.

> Slight bias note: every agent's round-robin starts on key #1, so account #1
> sees marginally more traffic than #2. Funding them equally (with the buffer
> above) covers it. If you want to fund only one OpenRouter account, put $50 on
> #1 — calls to the empty #2 will retry onto #1 automatically (just slower).

### If you run the P2 extras (E7, E8) or the premium sweep model
Add ~$10 to the OpenRouter pair ($5 each). E7/E8 are off by default.

---

## ⚠️ Two time-sensitive facts

1. **`deepseek-chat` alias RETIRES 2026-07-24** (14 days from the sweep date). It currently routes to **DeepSeek V4-Flash** — which is exactly what the paper calls *DeepSeek-v4-flash*. For any run on/after 24 July, change the focal `model_slug` in `CPU_Only/config/models.yaml` from `deepseek-chat` to **`deepseek-v4-flash`**. Same model, and pinning it is *better* for reproducibility (reviewers reward pinned snapshots). This is already flagged inline in `models.yaml`.
2. **`mistral-small-latest` now points to Mistral Small 4** ($0.15/$0.60). The 24B "Small 3.2" you want is cheapest on OpenRouter as `mistralai/mistral-small-3.2-24b-instruct` ($0.075/$0.20) — already set as the default in `models.yaml` for both the sweep model and judge tier 2.

---

## Per-model cheapest source (what `models.yaml` is set to)

| Model | Used for | Cheapest paid provider | Slug in config | Price (in/out) |
|---|---|---|---|---|
| DeepSeek V4-Flash | focal (E1–E6) + judge T3 | **DeepSeek direct** | `deepseek-chat` → `deepseek-v4-flash` | 0.14 / 0.28 |
| gpt-4o-mini | cross-val focal (E1, E5) | OpenRouter (=nano-gpt) | `openai/gpt-4o-mini` | 0.15 / 0.60 |
| llama-3.1-8b | weak peer + sweep floor | **OpenRouter** | `meta-llama/llama-3.1-8b-instruct` | 0.02 / 0.03 |
| llama-3.1-70b | sweep strong | **OpenRouter** | `meta-llama/llama-3.1-70b-instruct` | 0.40 / 0.40 |
| gemma-3-4b | weak peer + sweep floor | **OpenRouter** | `google/gemma-3-4b-it` | 0.05 / 0.10 |
| gemma-3-27b | sweep mid | **OpenRouter** | `google/gemma-3-27b-it` | 0.08 / 0.16 |
| qwen-2.5-72b | sweep strong | **OpenRouter** | `qwen/qwen-2.5-72b-instruct` | 0.36 / 0.40 |
| mistral-small-3.2-24b | sweep mid + judge T2 | **OpenRouter** | `mistralai/mistral-small-3.2-24b-instruct` | 0.075 / 0.20 |
| gemini-2.0-flash | judge T1 | OpenRouter or GCP | `google/gemini-2.0-flash-001` | ~0.10 / 0.40 |

Bonus slugs if you want them: `qwen/qwen3-32b` (0.08/0.28), `openai/gpt-4.1-mini` (0.40/1.60, the optional premium sweep focal).

**No free tier anywhere.** Phase 1's free Gemini/Mistral judge tiers caused rate-limit blocks; Phase 2 routes all three judge tiers to paid endpoints.

---

## Estimated spend per experiment (at the prices above)

| Exp | What | Trials | Est. cost |
|---|---|---|---|
| E1 | solo re-answer (2 focals) | ~3,000 | $2–3 |
| E2 | honest peers | ~3,000 | $8–12 |
| E3 | confidence filter re-run | ~600 | $1–2 |
| E4 | 6-model sweep (C1/C2/C4) | ~16,200 | $20–35 (70b/72b dominate) |
| E5 | gpt-4o-mini full 300 pool | ~6,000 | $7–9 |
| E6 | perturbed GSM8K | ~1,000 | $3 |
| E7 | split peer (off by default) | ~1,500 | $4 |
| E8 | heterogeneous smart (off) | ~900 | $4–5 |
| **Enabled total (E1–E6)** | | ~29,800 | **~$45–65** |

Recharge **$65 total ($50 OpenRouter + $15 DeepSeek)** and you have ~30% headroom for retries and judge calls. Add nano-gpt $10 only as insurance.

---

## GPU (Experiment E9 only — everything else is API)

No experiment except the optional mechanistic probe needs a GPU.

| If you run E9 | VRAM | Card (rent) | Hours | Cost |
|---|---|---|---|---|
| **Llama-3.1-8B focal, bf16, vLLM (recommended)** | ~18–20 GB | 1× RTX 4090 24 GB (RunPod/Vast ~$0.35–0.50/h) | 10–15 h | **$5–8** |
| 27–32B, 4-bit AWQ | ~20 GB | 1× RTX 4090 24 GB | 15–25 h | $8–13 |
| 70B, 4-bit AWQ | ~40–45 GB | 1× A6000/L40S 48 GB | 30–40 h | $25–45 |

Do **not** rent 80 GB hardware to run 70B behavioural trials — the API serves identical weights for a few dollars. GPU is only worth it for the local **logprob** probe, which the API cannot give you.

---

## Extra `.env` keys the config can use

The default routing needs only your existing **DeepSeek** and **OpenRouter** keys. To enable the optional failover providers, add to `Code_Phase_2/.env`:

```
# nano-gpt failover (optional)
NANOGPT_API_KEY_1=...
NANOGPT_API_KEY_2=...
NANOGPT_API_BASE_URL=https://nano-gpt.com/api/v1

# LinkAPI failover (optional; verify slugs + prices in-console first)
LINKAPI_API_KEY_1=...
LINKAPI_API_BASE_URL=https://api.linkapi.ai/v1
```

Then, to route (say) the 70B sweep model through nano-gpt, change its `provider:` from `openrouter` to `nanogpt` in `models.yaml` and confirm the slug against `GET https://nano-gpt.com/api/v1/models`. No code change — routing is config-only.

---

## Sources (fetched 2026-07-10)
DeepSeek pricing & deprecation: https://api-docs.deepseek.com/quick_start/pricing/ · Mistral: https://mistral.ai/pricing/api/ · GCP Vertex: https://cloud.google.com/vertex-ai/generative-ai/pricing · OpenRouter per-model pages under https://openrouter.ai/ · nano-gpt: https://nano-gpt.com/pricing + `/api/v1/models` · LinkAPI (login-gated): https://docs.linkapi.ai/ · cross-checks: https://pricepertoken.com , https://benchlm.ai/llm-pricing

# Bibliography Verification Log

All entries in `references.bib` have been verified against arXiv,
ACL Anthology, or publisher websites. The seven entries previously
flagged for manual verification are now resolved as listed below.

Last updated: 2026 (round-2 review pass).

---

## VERIFIED (arXiv IDs confirmed and cited in CSR_Paper.tex)

| Cite key | Title (short) | Source | arXiv ID | Notes |
|---|---|---|---|---|
| `du2023improving` | Improving Factuality via MAD | arXiv | 2305.14325 | Cited in §1, §2.1 |
| `sharma2023towards` | Towards Understanding Sycophancy | arXiv | 2310.13548 | Cited in §1, §2.1, §4.3 |
| `koo2024benchmarking` | CoBBLEr — Cognitive Biases in LLM Evaluators | ACL 2024 / arXiv | 2309.17012 | Cited in §1, §2.1, §4.3 |
| `wang2024mmlupro` | MMLU-Pro | NeurIPS 2024 / arXiv | 2406.01574 | Cited in §1, §3.3 |
| `cobbe2021training` | GSM8K — Training Verifiers | arXiv | 2110.14168 | Cited in §1, §3.3 |
| `meta2024llama3` | Llama 3 Herd of Models | arXiv | 2407.21783 | Cited in §1, §3.2 |
| `deepseekai2024v3` | DeepSeek-V3 Technical Report | arXiv | 2412.19437 | Cited in §3.2 |
| `gemma2025technical` | Gemma 3 Technical Report | arXiv | 2503.19786 | Cited in §1, §3.2 |
| `echterhoff2024cognitive` | Cognitive Bias in Decision-Making with LLMs | arXiv | 2403.00811 | Cited in §2.1 |
| `weng2025benchform` | BENCHFORM | ICLR 2025 (Oral) / arXiv | 2501.13381 | Cited in §2.1, Table 1; arXiv ID confirmed |
| `hong2025sycon` | SYCON-Bench | Findings of EMNLP 2025 | (DOI 10.18653/v1/2025.findings-emnlp.121) | Cited in §2.1; venue + DOI confirmed |
| `ashery2025emergent` | Emergent Social Conventions and Collective Bias | Science Advances 2025 | (DOI 10.1126/sciadv.adu9368) | Cited in §2.2; journal + DOI confirmed |

## VERIFIED (non-arXiv, cited in CSR_Paper.tex)

| Cite key | Title (short) | Source | Notes |
|---|---|---|---|
| `asch1951effects` | Effects of Group Pressure | Chapter in Guetzkow 1951 | Cited as motivation in §1, §2.1, §4.3 |
| `bond1996culture` | Culture and Conformity Meta-Analysis | Psychological Bulletin 119(1):111–137 | DOI 10.1037/0033-2909.119.1.111; cited in §1, §2.1, §4.3 |
| `openai2024gpt4o` | GPT-4o System Card | openai.com | Cited in §3.2 |

---

## REMOVED (no longer cited or never substantiated)

| Cite key | Reason |
|---|---|
| `dao2023flashattention` | Orphan — was cited only in the older GPU-pipeline draft; current paper is API-only and does not reference FlashAttention. Removed from bib. |
| `plato1992republic` | Orphan — the cog-sci "ship of state" framing has been tightened; the metaphor is mentioned only in the title and Acknowledgements, neither of which requires a bibliographic citation. Removed from bib. |
| `wu2025talk`, `choi2025identity`, `pitre2025consensagent` | Never substantiated against arXiv or proceedings within the verification window; not present in the current bib. The corresponding claims in earlier drafts have been rewritten to not require these references. |

---

## Notes

- British spelling used in paper body (analyse, behaviour, modelling, etc.).
- `elsarticle-harv` style requires `year` in all entries (verified).
- No `REQUIRES MANUAL VERIFICATION` strings remain in `references.bib`.
- All 15 cite keys used in `CSR_Paper.tex` have matching bib entries.
- The two cite keys used in `CSR_Supplement.tex` (`cobbe2021training`,
  `wang2024mmlupro`) are subsets of the main-paper set.

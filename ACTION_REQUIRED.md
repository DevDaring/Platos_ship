# ACTION_REQUIRED.md — things only you can do

Deadline: **ARR August cycle closes Mon 3 Aug 2026** (this is the only route into
EACL 2027; the next cycle is 12 Oct 2026 → NAACL/COLING 2027).

---

## 1. 🔴 BLOCKER — the anonymous repository link in the paper is dead

`ACL_Paper.tex` line 798 claims all artefacts are released and points at:

    https://anonymous.4open.science/r/Platos_ship/

That URL currently returns **HTTP 401 `{"error":"not_connected"}`** — the
anonymised mirror does not exist. This matters more than any other open item:

- Reviewer gL73 scored **Datasets = 1 ("no usable datasets submitted")** and
  **Software = 1 ("no usable software released")**. The single cheapest way to
  raise those scores is a working artefact link.
- ARR desk-rejects submissions that link a **non-anonymous** repository, so
  `github.com/DevDaring/Platos_ship` **cannot** go in the PDF.
- Claiming release while the link is dead is worse than not claiming it.

**What to do** (5 minutes, needs your GitHub login — I cannot do it headlessly):

1. Go to <https://anonymous.4open.science> and sign in with GitHub.
2. Choose **DevDaring/Platos_ship**. Note the repo is **private**, so you must
   grant Anonymous GitHub access to it.
3. Set the expiry date to **after** the review period — pick something like
   **1 March 2027** (the August cycle's meta-reviews land 8 Oct 2026, and an
   EACL commitment runs to Nov 2026).
4. Copy the generated URL — it looks like
   `https://anonymous.4open.science/r/Platos_ship-A1B2/` — and paste it here.
   I will put it into the Reproducibility footnote and recompile.

If you would rather not do this, tell me and I will soften the Reproducibility
section to promise release on acceptance instead — but that reverts the exact
weakness the reviewers penalised.

---

## 2. ✅ Recharge — no longer needed (my earlier ask was wrong)

I asked for a $15 DeepSeek top-up based on an estimate. Having measured the
actual burn rate, the whole remaining workload costs **well under $2**. See
`Recharge.md` for the measured figures. Nothing to do.

---

## 3. 🟡 ARR submission mechanics you must do in OpenReview

These are required and are not things I can file for you:

- **Declare the resubmission.** Link the previous submission
  (`https://openreview.net/forum?id=P1vYZxVhTX`, submission 1656). Failing to
  declare a prior submission is an **automatic desk rejection**.
- **Upload the revision-notes PDF.** I am generating
  `Submission/Revision_Notes.pdf` — a point-by-point response to every weakness
  and suggestion from gL73, Tf6M, sTG4 and the meta-review. ARR requires it in
  its own dedicated field, even if you also prepend it to the main PDF.
- **Reviewer/AC preference.** The meta-review said "resubmit next cycle", which
  is an invitation — I recommend requesting **the same reviewers and the same
  meta-reviewer**, since every one of their requests is now implemented and
  repeat reviewers are instructed to judge whether prior weaknesses were
  addressed rather than raise new ones.
- **Responsible NLP checklist.** Update **"Languages Studied"** if we get the
  multilingual experiment in (currently English only), and keep the **E1
  generative-AI disclosure** — AI assistance was used for coding and editing.
- **Reviewer registration.** All authors must register as reviewers for the
  cycle by **5 Aug 2026** or the paper can be desk-rejected.

---

## 4. ℹ️ Decisions already taken (recorded so you can override)

| Decision | Choice | Where |
|---|---|---|
| Target cycle | August 3 (EACL 2027) | your call, 1 Aug |
| GPU mechanistic probe | Run on Vast.ai | your call, 1 Aug |
| Paper drafts on GitHub | Keep `Submission/` private | your call, 1 Aug |
| DeepSeek model naming | Report served snapshot `deepseek-v4-flash`; dropped the DeepSeek-V3 architecture claim | the API no longer serves a V3-family model under this alias, so the old citation was unsupportable |
| Perturbed-GSM8K significance | Reported as underpowered at question level (p=0.23 after Holm), significant at trial level | the previous draft printed raw p-values under a "Holm-corrected" heading |

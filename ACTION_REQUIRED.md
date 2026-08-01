# ACTION_REQUIRED.md — things only you can do

Deadline: **ARR August cycle closes Mon 3 Aug 2026** (this is the only route into
EACL 2027; the next cycle is 12 Oct 2026 → NAACL/COLING 2027).

---

## 1. 🔴 BLOCKER — root cause found: the GitHub repo is PRIVATE

**Checked 2026-08-01 07:35 UTC at `.../r/Platos_ship_published/`.**

The second mirror is fresher (`lastUpdateDate: 2026-08-01T07:33`) and its
README URL returns HTTP 200, but it is still not usable, and the reason is now
clear:

| Evidence | Meaning |
|---|---|
| Mirror serves a README of **20,866 bytes containing `github.com/DevDaring/...`** | It is an old cached copy |
| `origin/main` README is **8,548 bytes and contains no such string** | GitHub itself is clean |
| Every subdirectory (`results/`, `src/`, `Code_Phase_2/`, `scripts/`) returns `{"error":"not_connected"}` | The mirror cannot read the source repo |
| GitHub API: **`private: true`** | Anonymous GitHub has no access |

So Anonymous GitHub is serving a stale cache of a handful of top-level files
and cannot fetch anything else. Rebuilding the mirror again will not help while
the source repo stays private.

### The fix: make the GitHub repo public, then refresh the mirror

1. <https://github.com/DevDaring/Platos_ship/settings> → *Danger Zone* →
   **Change visibility → Public**.
2. Back on <https://anonymous.4open.science>, delete `Platos_ship_published`
   and create it again from `DevDaring/Platos_ship`, branch `main`, expiry
   **1 March 2027**.
3. Send me the URL. I will (a) put it in the Reproducibility footnote,
   (b) re-run my recursive leak scan across every file the mirror serves, and
   (c) confirm subdirectories resolve, before you submit.

**Is making it public allowed?** Yes. ARR removed the anonymity *period* in
February 2024 — a public repository, and even a non-anonymous preprint, are
explicitly permitted while under review. What must stay anonymous is the
submitted PDF, and the PDF will cite only the `anonymous.4open.science` URL.
The trade-off to be aware of: a reviewer who searches a distinctive phrase
could find the public repo and thereby the authors. That is true of any public
artefact and ARR accepts it; if you would rather avoid it, the alternative is
to grant Anonymous GitHub access to the private repo through its GitHub
authorisation, but its private-repo support is unreliable and this has already
failed twice.

---

## 1b. (historical) the first mirror — same root cause

**Checked 2026-08-01 07:30 UTC at `anonymous.4open.science/r/Platos_ship`.**
The mirror now exists, but it fails on three counts and the third is a
desk-rejection risk:

**(a) It de-anonymises the paper.** The README that the mirror serves contains,
twice:

```
git clone https://${GITHUB_TOKEN}@github.com/DevDaring/Platos_ship.git
```

Your real GitHub username is inside the artefact the paper tells reviewers to
read. ARR is explicit that supplementary material and repository links must be
anonymised or the submission is desk-rejected. This is the single most
important thing to fix.

**(b) The snapshot is three weeks stale.** The API reports
`lastUpdateDate: 2026-07-12`. That predates the commit that anonymised the
README (which is *why* (a) happens), and it predates everything from this
session: the GPU mechanistic probe and its 2,700-trial log, the split-peer and
heterogeneous-debate results, the analysis scripts that regenerate every number
in the paper, the corrected calibration gate, and the `.gitignore` fix.

**(c) The browsing page returns HTTP 401 `{"error":"not_connected"}`.** File
contents are still served over the API, so the mirror is not empty — but a
reviewer clicking the link in the PDF lands on an error. "not_connected" means
Anonymous GitHub can no longer reach the source repository, which is expected
if its access to the (private) repo lapsed.

### What to do

1. Go to <https://anonymous.4open.science>, sign in with GitHub, and **delete
   the existing `Platos_ship` anonymised repo**. Refreshing it is not enough —
   (a) and (c) both come from a broken/stale connection to the source.
2. Create it again from **DevDaring/Platos_ship at current `main`**, granting
   access to the private repository when prompted.
3. Set the expiry to **after** the review period — 1 March 2027 is a safe pick.
4. Send me the generated URL (it usually looks like
   `.../r/Platos_ship-A1B2/`). I will put it in the Reproducibility footnote
   and recompile.
5. After it is live, I will re-scan the mirror for identity strings before you
   submit, so (a) cannot recur silently.

**Note:** the current repo is clean — I verified there are no `DevDaring`,
author-name, or institution strings anywhere in tracked files. Two scripts I
added this session (`Code_Phase_2/GPU_Only/vm_bootstrap.sh` and
`vm_autopush.sh`) did hardcode the repo URL; they now read
`GIT_REPO_SLUG=owner/repo` from `.env` instead, so nothing identifying remains
in source. Add that variable to your `.env` if you re-run the GPU probe.

---

## 1b. (historical) the link was previously dead entirely

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

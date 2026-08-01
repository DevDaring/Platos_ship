# audit/anonymity_audit.md

## Manuscript PDF

Searched the extracted text for: author names, affiliation, email patterns,
GitHub usernames, `/home/`, `C:\`, account IDs, OpenReview profile IDs.

**Result: 0 hits.** PDF metadata fields (Author/Title/Subject/Keywords) are
empty. The only URL family in the PDF is `anonymous.4open.science` plus
publisher/DOI links.

## Repository (public, `main`)

`git grep` over all tracked files for the same patterns: **0 hits**.

Two leaks were found during this session and fixed:

1. `Code_Phase_2/GPU_Only/vm_bootstrap.sh` and `vm_autopush.sh` (written this
   session) hardcoded the real `github.com/<user>/<repo>` URL. They now read
   `GIT_REPO_SLUG` from `.env`; `.env.example` documents it.
2. `ACTION_REQUIRED.md` and `Recharge.md` (author-facing operational notes)
   named the repository, balances and submission strategy. Both are removed
   from the repository and gitignored; local copies live at the workspace root.

## Secrets

Every secret **value** from the local `.env` files was searched for across all
80 commits on all refs. **No API key or token has ever been committed.** The
only matches were `*_MODEL_NAME` configuration strings, which belong in the
repository. No `.env` file has ever been added.

## Anonymous mirror — ⚠️ OPEN

`anonymous.4open.science/r/Platos_ship_published/`

| Check | Status |
|---|---|
| Opens without login | PASS (file URLs return 200) |
| Serves current `main` README | PASS (8,548 bytes, matches) |
| Recursive scan, 122 files / 96 text files | **1 hit** |
| API keys anywhere | PASS — none |
| Contains trial logs, GPU probe outputs, analysis scripts | PASS |
| `Submission/` excluded | PASS |

**The one hit is `ACTION_REQUIRED.md`**, which the mirror still serves from a
07:41 UTC snapshot taken before that file was removed from the repository at
07:49. The mirror must be refreshed once more, after which this must be
re-scanned. Until then the mirror de-anonymises the submission.

An earlier mirror (`/r/Platos_ship/`) was stale by three weeks and served a
pre-anonymisation README containing the real repository URL; it was superseded.
Root cause in both cases: Anonymous GitHub could not read the source repository
while it was private. The repository is now public, which fixed the sync.

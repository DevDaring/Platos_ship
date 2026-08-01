# audit/final_red_team_review.md

Three independent adversarial reviews were run against the artefact and the
compiled PDF after the first round of audit fixes. Each reviewer worked from
the released files rather than from the audit reports, and each was asked to
try to *reject* the paper.

| Reviewer | Remit | Verdict before fixes |
|---|---|---|
| A | Returning-reviewer perspective: are the original objections actually met? | Blockers found, all resolved in round one |
| B | Causal and statistical validity | **4 rejection blockers** (M2, M4, M5, M6) |
| C | Compliance and reproducibility | No desk-rejection risk; 9 artefact defects |

## What survived scrutiny

Reviewer B independently reproduced the released artefacts and confirmed the
Holm arithmetic and the probe numbers are exact. Reviewer C reproduced
`paper_numbers.json` **byte-identically** from a clean `git archive` clone with
no network and no API keys, and confirmed all four generated tables match.
Reviewer C cleared every desk-rejection category: PDF anonymity and metadata,
A4 page size, font embedding, the 8-page content limit, placement of
Limitations / Ethics / references / appendices, unresolved references, and
non-anonymous URLs.

## What did not survive scrutiny

Reviewer B's four blockers were all real and all confirmed against the data
before being fixed. The most consequential:

**M2 — the contamination claim pointed the wrong way.** §4.5 was titled "The
math gain is not memorisation" and said the gain "persists at a similar size"
under perturbation. That compared a GSM8K-only perturbed gain against a
full-pool original gain. On the matched comparator the gain falls from
**+16.1 pp to +5.4 pp** — a two-thirds collapse, which is evidence *for* partial
contamination, not against it. This was the single most serious defect found in
either audit round: the section's title asserted the opposite of what its own
data showed.

**M4 — the multiplicity family was drawn where it helped.** Correcting only the
accuracy contrasts, while asserting revision-rate contrasts uncorrected in
prose, let the family boundary carry the argument. Declaring a revision-rate
family costs the paper three of seven contrasts, including the C4split revision
result it had leaned on.

**M5 and M6** are corrections of degree rather than direction: the capability
gradient is partly baseline churn (ρ = −0.83 for the wrong-peer-specific excess,
not −0.95), and the deployed filter gap is +0.072, not +0.007.

Reviewer C's findings were concentrated in the artefact's last mile rather than
the science: a missing `LICENSE`, a figure script that crashed and therefore
regenerated nothing, a cross-check script unusable outside the authors' tree,
and README pointers into a directory the mirror does not contain. Four of these
directly falsified the paper's own claim that the released scripts regenerate
every number, table and figure.

## One reviewer claim rejected

Reviewer C reported an error in Appendix S5's correct-anchored persona counts,
proposing 1,048 / 437 / 15 against the paper's 1,033 / 452 / 15. The crosstab
of `regeneration_attempts_used` against `validation_pass_status` shows all 15
failures sit at zero regeneration attempts, so 1,048 counts failures as
first-attempt passes. The paper is correct.

Checking it, however, showed that the **first** audit round had made exactly
that error in the *other* persona pool: 1,102 / 384 / 14 was correct and had
been "corrected" to 1,116 / 370 / 14. That entry is now retracted in
`numerical_consistency_report.md` and the paper reverted. A rejected reviewer
claim still earned its keep.

## Standing assessment

The reviews changed four scientific claims — one of them a reversal of the
section's headline. None of the changes overturn the paper's three research
questions: the correction effect under wrong peers, the capability gradient in
harmful flips, and the failure of confidence filtering all survive, with
narrower scope and honest error bars. The contamination section no longer
claims what it cannot support.

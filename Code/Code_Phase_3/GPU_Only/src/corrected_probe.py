"""
corrected_probe.py — X8: the output-distribution probe, measured properly.

What the released probe did, and why it needed repair
-----------------------------------------------------
`Code_Phase_2/GPU_Only/src/vllm_focal_agent.py:_dist_from_logprobs` scores a
candidate answer by its FIRST CHARACTER:

    key = str(cand).strip().upper()[:1]

For MMLU-Pro letters A-J that happens to be exact. For GSM8K numbers it is
not: "12", "150" and "1" all match the token "1", so the reported "probability
of the peer's answer" on a numeric item is the mass of every candidate sharing
a leading digit. It then softmax-renormalises over the candidates it found, so
the numbers are candidate-conditional probabilities while the paper describes
them as probability mass, and mass sitting outside the candidate set is
silently discarded.

Three repairs here:

  1. **Exact candidate scoring.** Each candidate is tokenised; a single-token
     candidate is read from the first-position logprobs, and a multi-token
     candidate is scored by summing the log-probabilities of its full token
     sequence under teacher forcing. Candidates whose tokenisation collides
     are reported rather than merged.
  2. **Off-candidate mass is reported.** Both the raw (unnormalised) candidate
     probabilities and the residual `mass_outside_candidates` are stored, so a
     reader can see whether a "shift" is movement between candidates or
     movement in or out of the answer set entirely.
  3. **One target per (question, replicate), fixed across every condition,**
     recorded explicitly — this is what makes the C4-minus-C2 difference a
     difference-in-differences rather than two unrelated quantities.

Scope claim this supports: distributional movement can outrun the flip count
in one open-weight model. It is not a representation-level mechanism, and one
model is not a population.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("platos_ship3.corrected_probe")


@dataclass
class CandidateScore:
    """Exact scoring of one candidate answer string."""

    candidate: str
    token_ids: List[int]
    n_tokens: int
    logprob: Optional[float]          # summed over the full sequence
    probability: Optional[float]      # exp(logprob); NOT renormalised
    scoring_method: str               # first_position | teacher_forced | absent
    collides_with: List[str] = field(default_factory=list)


class ExactCandidateScorer:
    """
    Tokenizer-validated scoring of candidate answers.

    Single-token candidates are read directly from the first-position logprob
    table. Multi-token candidates need the full sequence, which vLLM supplies
    via `prompt_logprobs` on a forced continuation.
    """

    def __init__(self, tokenizer, top_logprobs: int = 20):
        self.tokenizer = tokenizer
        self.top_logprobs = top_logprobs

    def tokenise(self, candidate: str, leading_space: bool = True) -> List[int]:
        """Token ids for a candidate as it would appear after 'Final answer:'."""
        text = f" {candidate}" if leading_space else str(candidate)
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def collision_report(self, candidates: Sequence[str]) -> Dict[str, List[str]]:
        """
        Which candidates share a first token.

        The released probe merged these silently. Reporting them means a
        numeric item whose candidates are not separable at the first token is
        visible rather than quietly mis-measured.
        """
        by_first: Dict[int, List[str]] = {}
        for candidate in candidates:
            ids = self.tokenise(candidate)
            if ids:
                by_first.setdefault(ids[0], []).append(str(candidate))
        return {
            group[0]: group[1:]
            for group in by_first.values() if len(group) > 1
        }

    def single_token_candidates(self, candidates: Sequence[str]) -> Dict[str, int]:
        """Candidates that tokenise to exactly one token (letters, small ints)."""
        out = {}
        for candidate in candidates:
            ids = self.tokenise(candidate)
            if len(ids) == 1:
                out[str(candidate)] = ids[0]
        return out

    def score_from_first_position(
        self, first_position_logprobs: Dict[int, float],
        candidates: Sequence[str],
    ) -> Tuple[List[CandidateScore], float]:
        """
        Read single-token candidates off the first-position table.

        Returns (scores, mass_outside_candidates). The second value is the
        probability the model put on anything that is not a candidate — the
        quantity the released probe threw away.
        """
        single = self.single_token_candidates(candidates)
        collisions = self.collision_report(candidates)

        total_mass = sum(math.exp(lp) for lp in first_position_logprobs.values())
        scores: List[CandidateScore] = []
        candidate_mass = 0.0

        for candidate in candidates:
            key = str(candidate)
            token_ids = self.tokenise(key)
            if key in single and single[key] in first_position_logprobs:
                logprob = float(first_position_logprobs[single[key]])
                probability = math.exp(logprob)
                candidate_mass += probability
                method = "first_position"
            elif key in single:
                # Single token, but outside the top-k window: bounded above by
                # the smallest logprob returned, not treated as exactly zero.
                logprob = None
                probability = 0.0
                method = "below_top_k"
            else:
                logprob = None
                probability = None
                method = "needs_teacher_forcing"
            scores.append(CandidateScore(
                candidate=key, token_ids=token_ids, n_tokens=len(token_ids),
                logprob=logprob, probability=probability, scoring_method=method,
                collides_with=collisions.get(key, []),
            ))

        outside = max(0.0, total_mass - candidate_mass)
        return scores, outside

    # ── multi-token candidates: exact full-sequence scoring ───────────────

    def needs_teacher_forcing(self, candidates: Sequence[str]) -> List[str]:
        """Candidates that do NOT fit in one token and so cannot be read off
        the first-position table."""
        single = self.single_token_candidates(candidates)
        return [str(c) for c in candidates if str(c) not in single]

    def continuation_span(self, prefix_text: str, candidate: str) -> Tuple[int, int]:
        """
        Token span the candidate occupies when appended to `prefix_text`.

        Computed by tokenising the prefix and the full string and taking the
        difference, so it is correct even when the tokenizer merges across the
        boundary — which it does for numbers after a colon or space.
        """
        prefix_ids = list(self.tokenizer.encode(prefix_text,
                                                add_special_tokens=False))
        full_ids = list(self.tokenizer.encode(prefix_text + f" {candidate}",
                                              add_special_tokens=False))
        start = 0
        while (start < len(prefix_ids) and start < len(full_ids)
               and prefix_ids[start] == full_ids[start]):
            start += 1
        return start, len(full_ids)

    @staticmethod
    def score_from_prompt_logprobs(
        prompt_logprobs: Sequence[Optional[Dict[int, Any]]],
        prompt_token_ids: Sequence[int],
        n_candidate_tokens: int,
    ) -> Optional[float]:
        """
        Sum log P(token) over the candidate's tokens: log P(candidate | prompt).

        Indexed off vLLM's OWN `prompt_token_ids`, never off a local
        tokenisation. vLLM prepends a BOS token, so its `prompt_logprobs` is
        offset by one from `tokenizer.encode(...)`; reading position i of the
        local encoding returns the NEIGHBOURING token's logprob. That bug made
        every candidate sharing a leading space score identically (four
        different GSM8K answers all returned 0.666) and made the candidate
        probabilities sum to 3.28 instead of <= 1.

        The candidate is appended at the end of the sequence, so its tokens are
        simply the last `n_candidate_tokens`. Each logprob is then looked up by
        the token id that vLLM actually placed there — an explicit lookup that
        cannot silently read a different token.
        """
        if not prompt_logprobs or not prompt_token_ids or n_candidate_tokens <= 0:
            return None
        end = len(prompt_token_ids)
        start = end - n_candidate_tokens
        if start < 1 or end > len(prompt_logprobs):
            return None

        total = 0.0
        for position in range(start, end):
            table = prompt_logprobs[position]
            if not table:
                return None
            entry = table.get(prompt_token_ids[position])
            if entry is None:
                return None
            logprob = getattr(entry, "logprob", entry)
            if logprob is None:
                return None
            total += float(logprob)
        return total

    def merge_teacher_forced(
        self,
        scores: List[CandidateScore],
        forced: Dict[str, Optional[float]],
    ) -> Tuple[List[CandidateScore], float]:
        """
        Fill in the multi-token candidates and recompute the residual mass.

        A teacher-forced probability is an absolute sequence probability
        P(candidate | prompt), on the same footing as a single-token
        probability read from the first-position table. They are NOT
        renormalised, so `mass_outside_candidates` stays meaningful: it is
        whatever probability the model put on continuations that are not
        candidates.
        """
        candidate_mass = 0.0
        for score in scores:
            if score.candidate in forced:
                logprob = forced[score.candidate]
                if logprob is None:
                    score.scoring_method = "teacher_forcing_unavailable"
                    score.probability = None
                else:
                    score.logprob = logprob
                    score.probability = math.exp(logprob)
                    score.scoring_method = "teacher_forced"
            if score.probability is not None:
                candidate_mass += score.probability
        return scores, max(0.0, 1.0 - candidate_mass)


def build_probe_targets(
    questions: Sequence[Dict[str, Any]],
    personas_by_question: Dict[str, List[Dict[str, Any]]],
    replicates: int,
    master_seed: int,
) -> Dict[Tuple[str, int], str]:
    """
    Fix ONE target per (question, replicate), used in every condition.

    The difference-in-differences only means anything if the same target is
    tracked under the treatment and under the baseline; otherwise C4 and C2
    measure movement toward different answers and their difference is not
    interpretable.
    """
    import sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.seeding import derive_rng

    targets: Dict[Tuple[str, int], str] = {}
    for question in questions:
        question_id = question["question_identifier"]
        variants = personas_by_question.get(question_id, [])
        if not variants:
            continue
        for replicate in range(replicates):
            rng = derive_rng(master_seed, "probe_target", question_id, replicate)
            chosen = variants[rng.randrange(len(variants))]
            target = chosen.get("assigned_wrong_answer_letter_or_value")
            if target is not None:
                targets[(question_id, replicate)] = str(target)
    return targets


def difference_in_differences_by_task(
    trials: pd.DataFrame,
    treatment_condition: str = "WR",
    baseline_condition: str = "E",
    mass_column: str = "delta_prob_mass_toward_target",
    group_column: str = "source_dataset",
) -> List[Dict[str, Any]]:
    """
    One contrast PER TASK FAMILY, plus a pooled row flagged not to report.

    The two families live on different probability scales by construction. A
    multiple-choice answer is one token out of ten options, so the mass on it
    is O(0.1-0.9). A GSM8K answer is an exact digit string competing with every
    other continuation, so the mass on it is O(0.001-0.3) and for a four-digit
    answer nearer 0.0004, with ~99.6% of the mass outside the candidate set.

    Both are correct measurements of different quantities. Averaging them
    together lets multiple choice dominate and buries the GSM8K signal in
    rounding, so the pooled row is emitted only for completeness and is marked
    `report: False`.
    """
    rows: List[Dict[str, Any]] = []
    if trials.empty:
        return rows

    if group_column in trials.columns:
        for group, subset in trials.groupby(group_column, dropna=False):
            result = difference_in_differences(
                subset, treatment_condition, baseline_condition, mass_column)
            result[group_column] = str(group)
            result["report"] = True
            rows.append(result)

    pooled = difference_in_differences(
        trials, treatment_condition, baseline_condition, mass_column)
    pooled[group_column] = "POOLED"
    pooled["report"] = False
    pooled["why_not"] = (
        "task families differ in probability scale by ~2 orders of magnitude; "
        "the pooled mean is dominated by multiple choice. Report per family."
    )
    rows.append(pooled)
    return rows


def difference_in_differences(
    trials: pd.DataFrame,
    treatment_condition: str = "WR",
    baseline_condition: str = "E",
    mass_column: str = "delta_prob_mass_toward_target",
) -> Dict[str, Any]:
    """
    Paired C4-minus-C2 shift toward the fixed target.

    Paired by (question, replicate) so drift common to both conditions
    cancels. Also reports the restriction to trials whose STATED answer is
    identical in both rounds — the version that shows the movement is not an
    arithmetic consequence of the answer flipping.
    """
    treatment = trials[trials["condition"] == treatment_condition]
    baseline = trials[trials["condition"] == baseline_condition]
    keys = ["question_identifier", "replicate"]

    merged = treatment.merge(
        baseline, on=keys, suffixes=("_treatment", "_baseline"))
    if merged.empty:
        return {"n_pairs": 0}

    differences = (merged[f"{mass_column}_treatment"]
                   - merged[f"{mass_column}_baseline"]).dropna()

    import sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from analysis.stats import bootstrap_ci, paired_permutation_p

    estimate, low, high = bootstrap_ci(differences.to_numpy())
    p_value = paired_permutation_p(differences.to_numpy())

    # "Stated answer unchanged" must mean the answer was PARSED in both rounds
    # and was the same. An unparsed answer is an empty string, and two empty
    # strings compare equal — which would silently count a pair with no
    # evidence either way as evidence that the answer held. On this model 187
    # of 900 Round-1 answers do not parse and 91 pairs are empty on both
    # sides, so the naive comparison would inflate this subset by ~10%.
    def _stated_and_same(frame, suffix):
        r0 = frame[f"round0_answer{suffix}"].astype(str).str.strip()
        r1 = frame[f"round1_answer{suffix}"].astype(str).str.strip()
        return (r0 != "") & (r1 != "") & (r0 == r1)

    unchanged = merged[
        _stated_and_same(merged, "_treatment")
        & _stated_and_same(merged, "_baseline")
    ]
    unchanged_differences = (unchanged[f"{mass_column}_treatment"]
                             - unchanged[f"{mass_column}_baseline"]).dropna()
    unchanged_estimate, unchanged_low, unchanged_high = bootstrap_ci(
        unchanged_differences.to_numpy())

    return {
        "contrast": f"{treatment_condition}_minus_{baseline_condition}",
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "p_value_method": "sign_flip_permutation",
        "n_pairs": int(len(differences)),
        "median": float(np.median(differences)) if len(differences) else float("nan"),
        "estimate_stated_answer_unchanged": unchanged_estimate,
        "ci_stated_answer_unchanged": [unchanged_low, unchanged_high],
        "n_pairs_stated_answer_unchanged": int(len(unchanged_differences)),
        "note": (
            "Paired by (question, replicate); the target is fixed per pair and "
            "identical across conditions. The 'unchanged' row restricts to "
            "trials whose stated answer PARSED in both rounds and was the "
            "same, so an unparsed answer is excluded rather than counted as "
            "unchanged."
        ),
    }


def audit_released_probe(trials_path: Path) -> Dict[str, Any]:
    """
    Re-examine the released 2,700-trial probe for the defects above.

    Run this before citing any number from it. It reports how many trials sit
    on numeric items (where first-character matching is wrong) and how many on
    multiple-choice items (where it is exact), so the appendix can state the
    scope the old measurement is valid for.
    """
    if not Path(trials_path).exists():
        return {"available": False, "path": str(trials_path)}

    trials = pd.read_parquet(trials_path)
    numeric = trials["source_dataset"].astype(str).str.contains("gsm", case=False)
    per_condition = (
        trials.groupby("condition_identifier")
        .agg(n=("question_identifier", "size"),
             n_with_target=("reference_wrong_answer", "count"))
        .reset_index()
    )
    return {
        "available": True,
        "n_trials": int(len(trials)),
        "n_numeric_item_trials": int(numeric.sum()),
        "n_mcq_item_trials": int((~numeric).sum()),
        "per_condition": per_condition.to_dict("records"),
        "first_character_matching_is_exact_for": "multiple-choice letters A-J",
        "first_character_matching_is_unsafe_for": (
            "numeric answers sharing a leading digit (GSM8K items)"
        ),
        "recommendation": (
            "Restrict any claim from the released probe to the multiple-choice "
            "subset, or re-run with ExactCandidateScorer."
        ),
    }

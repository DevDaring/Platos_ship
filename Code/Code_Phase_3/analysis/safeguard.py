"""
safeguard.py — score X6 against the baselines that could beat it.

A mitigation is only interesting if it beats the trivial alternatives. This
scores five policies on the same units, at their real coverage and cost:

  always_keep        never adopt a proposed change (perfect safety, zero
                     benefit — the null that any safeguard must beat)
  always_revise      adopt every change (= the unsafeguarded condition)
  verifier           adopt only when an independent verifier supports it
  random_matched     adopt a random subset of changes at the SAME coverage as
                     the verifier, so "adopting fewer changes" alone cannot
                     explain a gain
  confidence         adopt when the focal's own stated confidence in the
                     revision is at least a threshold chosen on development
                     data and then frozen
  oracle             adopt when the change is actually correct — an UPPER
                     BOUND, never a deployable rule, always labelled

The cut rule from the plan: if `verifier` has no held-out benefit over
`always_keep` and `random_matched`, the paper reports that briefly and stays
empirical. A negative result here is publishable; an unlabelled oracle is not.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.extraction import answers_equal

from .stats import bootstrap_ci

logger = logging.getLogger("platos_ship3.safeguard")


def _score(final_answers: pd.Series, correct: pd.Series) -> np.ndarray:
    return np.array(
        [answers_equal(a, c) for a, c in zip(final_answers, correct)], dtype=bool
    )


def score_policies(
    verified: pd.DataFrame,
    all_revisions: pd.DataFrame,
    confidence_threshold: int = 90,
    seed: int = 20260502,
) -> pd.DataFrame:
    """
    Score every policy on the FULL set of revision units, not only the changed
    ones — a policy that fires on 12% of units cannot be compared on those 12%
    alone, because the other 88% are where its safety comes from.

    `verified` holds one row per proposed change (from src/verifier.py);
    `all_revisions` holds every unit of the same conditions.
    """
    if all_revisions.empty:
        return pd.DataFrame()

    units = all_revisions.copy()
    verdicts = (
        verified.set_index("unit_id")["adopt_change"].to_dict()
        if not verified.empty else {}
    )

    changed = units["answer_changed"].fillna(False).astype(bool)
    coverage = float(changed.mean())
    rng = np.random.default_rng(seed)

    # Random policy adopts changes at the verifier's realised adoption rate.
    n_changed = int(changed.sum())
    verifier_adoption_rate = (
        float(np.mean([bool(verdicts.get(u, False))
                       for u in units.loc[changed, "unit_id"]]))
        if n_changed and verdicts else 0.0
    )
    random_adopt = np.zeros(len(units), dtype=bool)
    random_adopt[np.flatnonzero(changed.to_numpy())] = (
        rng.random(n_changed) < verifier_adoption_rate
    )

    revision_confidence = pd.to_numeric(
        units.get("extracted_confidence"), errors="coerce")

    policies: Dict[str, np.ndarray] = {
        "always_keep": np.zeros(len(units), dtype=bool),
        "always_revise": changed.to_numpy(),
        "verifier": np.array(
            [bool(verdicts.get(u, False)) if c else False
             for u, c in zip(units["unit_id"], changed)], dtype=bool),
        "random_matched": random_adopt,
        "confidence": (changed.to_numpy()
                       & (revision_confidence.fillna(-1) >= confidence_threshold).to_numpy()),
        "oracle_upper_bound": (
            changed.to_numpy()
            & _score(units["extracted_answer"], units["correct_answer"])
        ),
    }

    rows: List[Dict[str, Any]] = []
    initial_correct = units["r0_is_correct"].fillna(False).astype(bool).to_numpy()

    for policy_name, adopt in policies.items():
        final_answer = np.where(adopt, units["extracted_answer"],
                                units["r0_answer"])
        final_correct = _score(pd.Series(final_answer), units["correct_answer"])

        # Question-level values, because the question is the inference unit.
        per_question = (
            pd.DataFrame({"question_identifier": units["question_identifier"],
                          "correct": final_correct})
            .groupby("question_identifier")["correct"].mean()
        )
        estimate, low, high = bootstrap_ci(per_question.to_numpy(), seed=seed)

        harmful_denominator = int(initial_correct.sum())
        harmful_numerator = int((initial_correct & ~final_correct).sum())
        beneficial_denominator = int((~initial_correct).sum())
        beneficial_numerator = int((~initial_correct & final_correct).sum())

        rows.append(
            {
                "policy": policy_name,
                "is_deployable": policy_name != "oracle_upper_bound",
                "accuracy": estimate,
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
                "harmful_revision": (harmful_numerator / harmful_denominator
                                     if harmful_denominator else np.nan),
                "harmful_revision_denominator": harmful_denominator,
                "beneficial_revision": (beneficial_numerator / beneficial_denominator
                                        if beneficial_denominator else np.nan),
                "beneficial_revision_denominator": beneficial_denominator,
                "coverage_changes_proposed": coverage,
                "adoption_rate_of_changes": (float(adopt.sum() / n_changed)
                                             if n_changed else np.nan),
                "n_units": int(len(units)),
                "n_questions": int(units["question_identifier"].nunique()),
            }
        )

    table = pd.DataFrame(rows)
    baseline = table.loc[table["policy"] == "always_keep", "accuracy"]
    if not baseline.empty:
        table["accuracy_minus_always_keep"] = table["accuracy"] - float(baseline.iloc[0])
    return table


def verifier_cost(verified: pd.DataFrame, all_revisions: pd.DataFrame
                  ) -> Dict[str, Any]:
    """
    What the safeguard actually costs, so the comparison stays budget-aware.

    A safeguard that fires on every unit and costs a full extra call is a
    different proposition from one that fires on the 12% of units where the
    model proposed a change.
    """
    if all_revisions.empty:
        return {}
    n_units = int(len(all_revisions))
    n_verified = int(len(verified))
    return {
        "n_units": n_units,
        "n_verifier_calls": n_verified,
        "verifier_calls_per_unit": n_verified / n_units if n_units else np.nan,
        "mean_verifier_output_tokens": (
            float(verified["total_output_tokens"].mean())
            if not verified.empty else np.nan),
        "total_verifier_output_tokens": (
            int(verified["total_output_tokens"].sum())
            if not verified.empty else 0),
    }


def choose_confidence_threshold(
    development_units: pd.DataFrame, candidates: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Pick the confidence-retention threshold on DEVELOPMENT data, then freeze it.

    Selecting a threshold on the same data that reports the result is exactly
    the post-hoc move Reviewer oEjr objected to (W3). This returns the chosen
    value and the sweep behind it so the appendix can show the choice was made
    once, in advance.
    """
    candidates = candidates or list(range(50, 101, 5))
    if development_units.empty:
        return {"chosen_threshold": None, "sweep": []}

    changed = development_units["answer_changed"].fillna(False).astype(bool)
    confidence = pd.to_numeric(
        development_units.get("extracted_confidence"), errors="coerce")

    sweep = []
    for threshold in candidates:
        adopt = changed & (confidence.fillna(-1) >= threshold)
        final_answer = np.where(adopt, development_units["extracted_answer"],
                                development_units["r0_answer"])
        final_correct = _score(pd.Series(final_answer),
                               development_units["correct_answer"])
        sweep.append({"threshold": threshold,
                      "accuracy": float(final_correct.mean()),
                      "adoption_rate": float(adopt.mean())})

    best = max(sweep, key=lambda row: row["accuracy"])
    return {"chosen_threshold": int(best["threshold"]),
            "development_accuracy": best["accuracy"],
            "sweep": sweep,
            "note": "chosen on development units only; frozen before test scoring"}

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


def x6_scoring_units(revisions: pd.DataFrame, experiment: Dict[str, Any],
                     paths: Dict[str, str], project_root) -> pd.DataFrame:
    """
    Every revision unit X6 covers: its conditions, its models, its questions.

    Previously the scoring set was "any WR or H row", i.e. all eight models
    on all 300 questions, while the verifier ran on three models and 100
    questions. Units outside the verifier's reach counted as "verifier kept
    the initial answer", so the verifier policy was mostly always_keep.
    """
    from pathlib import Path

    from src.agents import load_models_config, resolve_focal_selector

    spec = experiment["experiments"].get("X6_verification_safeguard", {})
    models = load_models_config(Path(project_root))
    focal_specs = models.get("focal_agents", {})
    verifier_slug = str((models.get("verifier_agent") or {}).get("model_slug", ""))
    # Same rule as the runner: a model is never scored on self-verification.
    focal = [k for k in resolve_focal_selector(spec.get("focal", "TIER_X2"), focal_specs)
             if str(focal_specs.get(k, {}).get("model_slug", "")) != verifier_slug]
    pool_path = Path(paths["question_pool_file"])
    if not pool_path.is_absolute():
        pool_path = Path(project_root) / pool_path
    pool = pd.read_parquet(pool_path)
    if experiment["pools"].get(spec.get("pool", "mitigation100"), {}).get(
            "filter") == "included_in_mitigation_subset":
        pool = pool[pool["included_in_mitigation_subset"].astype(bool)]
    return revisions[
        revisions["condition"].isin(spec.get("conditions", ["WR", "H"]))
        & revisions["focal_key"].isin(focal)
        & revisions["question_identifier"].isin(set(pool["question_identifier"]))
    ].drop_duplicates("unit_id")


def _paired_ci(per_q_a: pd.Series, per_q_b: pd.Series, seed: int):
    """Question-paired bootstrap CI for mean(a - b) over shared questions."""
    joined = pd.concat([per_q_a.rename("a"), per_q_b.rename("b")], axis=1).dropna()
    if joined.empty:
        return np.nan, np.nan, np.nan
    return bootstrap_ci((joined["a"] - joined["b"]).to_numpy(), seed=seed)


def score_policies(
    verified: pd.DataFrame,
    all_revisions: pd.DataFrame,
    confidence_threshold: int = 90,
    seed: int = 20260502,
    r0_answers: Optional[Dict[tuple, Any]] = None,
    independent_answers: Optional[Dict[tuple, Any]] = None,
) -> pd.DataFrame:
    """
    Score every policy on the FULL set of revision units, not only the changed
    ones — a policy that fires on 12% of units cannot be compared on those 12%
    alone, because the other 88% are where its safety comes from.

    `verified` holds one row per proposed change (from src/verifier.py);
    `all_revisions` holds every unit of the same conditions.

    Each unit carries an adoption PROBABILITY in [0, 1], so a policy's final
    correctness is an expectation. For every deterministic policy this is 0/1.
    `random_matched` adopts each proposed change with probability equal to the
    verifier's acceptance rate among changes OF THE SAME FOCAL MODEL; scoring
    the exact expectation instead of one Bernoulli draw makes the comparator
    matched in the reported numbers, not only in expectation (next_plan.md P0.3).

    Experiment D comparators (all computed from cached calls, no new API calls):
      independent_answer  adopt a change only when the verifier model's own
                          unanchored answer (its cached Round-0 answer to the
                          same question and replicate) equals the revision;
      extra_focal_sample  adopt a change only when another cached Round-0
                          sample of the focal model equals the revision;
      verifier_model_alone  replace the focal answer by the verifier model's own
                          answer on every unit (a competence baseline).
    """
    if all_revisions.empty:
        return pd.DataFrame()

    units = all_revisions.reset_index(drop=True).copy()
    verdicts = (
        verified.set_index("unit_id")["adopt_change"].to_dict()
        if not verified.empty else {}
    )

    changed = units["answer_changed"].fillna(False).astype(bool).to_numpy()
    coverage = float(changed.mean())
    n_changed = int(changed.sum())
    verifier_adopt = np.array(
        [bool(verdicts.get(u, False)) if c else False
         for u, c in zip(units["unit_id"], changed)], dtype=bool)

    # Random retention: probability = the verifier's acceptance rate among the
    # changes of the same focal model, applied to every change of that model.
    random_prob = np.zeros(len(units), dtype=float)
    for focal, idx in units.groupby("focal_key").groups.items():
        idx = np.asarray(list(idx))
        ch = idx[changed[idx]]
        if len(ch):
            random_prob[ch] = verifier_adopt[ch].mean()

    revision_confidence = pd.to_numeric(
        units.get("extracted_confidence"), errors="coerce")
    revised_correct = _score(units["extracted_answer"], units["correct_answer"])
    initial_correct_answer = _score(units["r0_answer"], units["correct_answer"])

    policies: Dict[str, np.ndarray] = {
        "always_keep": np.zeros(len(units)),
        "always_revise": changed.astype(float),
        "verifier": verifier_adopt.astype(float),
        "random_matched": random_prob,
        "confidence": (changed & (revision_confidence.fillna(-1) >= confidence_threshold)
                       .to_numpy()).astype(float),
        "oracle_upper_bound": (changed & revised_correct).astype(float),
    }
    extra_calls: Dict[str, float] = {}
    if independent_answers:
        indep = [independent_answers.get((q, r)) for q, r in
                 zip(units["question_identifier"], units["replicate"])]
        policies["independent_answer"] = np.array(
            [c and a is not None and answers_equal(a, rev)
             for c, a, rev in zip(changed, indep, units["extracted_answer"])], dtype=float)
        extra_calls["independent_answer"] = float(changed.mean())
    if r0_answers:
        other = []
        for f, q, r in zip(units["focal_key"], units["question_identifier"], units["replicate"]):
            other.append(r0_answers.get((f, q, (int(r) + 1) % 3)))
        policies["extra_focal_sample"] = np.array(
            [c and a is not None and answers_equal(a, rev)
             for c, a, rev in zip(changed, other, units["extracted_answer"])], dtype=float)
        extra_calls["extra_focal_sample"] = float(changed.mean())

    rows: List[Dict[str, Any]] = []
    initial_correct = units["r0_is_correct"].fillna(False).astype(bool).to_numpy()
    per_question: Dict[str, pd.Series] = {}
    per_question_harm: Dict[str, pd.Series] = {}

    def summarise(policy_name: str, expected_correct: np.ndarray, adopt: np.ndarray):
        frame = pd.DataFrame({"q": units["question_identifier"], "c": expected_correct,
                              "init": initial_correct})
        per_question[policy_name] = frame.groupby("q")["c"].mean()
        per_question_harm[policy_name] = (1 - frame[frame["init"]].groupby("q")["c"].mean())
        estimate, low, high = bootstrap_ci(per_question[policy_name].to_numpy(), seed=seed)
        harm_den = int(initial_correct.sum())
        ben_den = int((~initial_correct).sum())
        harm_num = float((initial_correct * (1 - expected_correct)).sum())
        ben_num = float(((~initial_correct) * expected_correct).sum())
        rows.append({
            "policy": policy_name,
            "is_deployable": policy_name != "oracle_upper_bound",
            "accuracy": estimate, "accuracy_ci_low": low, "accuracy_ci_high": high,
            "harmful_revision": harm_num / harm_den if harm_den else np.nan,
            "harmful_revision_denominator": harm_den,
            "beneficial_revision": ben_num / ben_den if ben_den else np.nan,
            "beneficial_revision_denominator": ben_den,
            "beneficial_corrections_retained": ben_num,
            "coverage_changes_proposed": coverage,
            "adoption_rate_of_changes": (float(adopt[changed].mean())
                                         if n_changed and adopt is not None else np.nan),
            "extra_calls_per_unit": extra_calls.get(policy_name, 0.0),
            "n_units": int(len(units)),
            "n_questions": int(units["question_identifier"].nunique()),
        })

    for policy_name, adopt in policies.items():
        expected = adopt * revised_correct + (1 - adopt) * initial_correct_answer
        summarise(policy_name, expected, adopt)

    if independent_answers:
        alone = np.array([a is not None and answers_equal(a, c) for a, c in zip(
            [independent_answers.get((q, r)) for q, r in
             zip(units["question_identifier"], units["replicate"])],
            units["correct_answer"])], dtype=float)
        extra_calls["verifier_model_alone"] = 1.0
        summarise("verifier_model_alone", alone, None)

    table = pd.DataFrame(rows)
    baseline = table.loc[table["policy"] == "always_keep", "accuracy"]
    if not baseline.empty:
        table["accuracy_minus_always_keep"] = table["accuracy"] - float(baseline.iloc[0])
    # Question-paired differences: verifier minus every other policy.
    diffs = {}
    for name in per_question:
        if name == "verifier":
            continue
        est, lo, hi = _paired_ci(per_question["verifier"], per_question[name], seed)
        hest, hlo, hhi = _paired_ci(per_question_harm["verifier"], per_question_harm[name], seed)
        diffs[name] = (est, lo, hi, hest, hlo, hhi)
    table["verifier_minus_accuracy"] = table["policy"].map(lambda p: diffs.get(p, (np.nan,) * 6)[0])
    table["verifier_minus_accuracy_ci_low"] = table["policy"].map(lambda p: diffs.get(p, (np.nan,) * 6)[1])
    table["verifier_minus_accuracy_ci_high"] = table["policy"].map(lambda p: diffs.get(p, (np.nan,) * 6)[2])
    table["verifier_minus_harmful"] = table["policy"].map(lambda p: diffs.get(p, (np.nan,) * 6)[3])
    table["verifier_minus_harmful_ci_low"] = table["policy"].map(lambda p: diffs.get(p, (np.nan,) * 6)[4])
    table["verifier_minus_harmful_ci_high"] = table["policy"].map(lambda p: diffs.get(p, (np.nan,) * 6)[5])
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

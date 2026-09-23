"""
Sharp bounds on the "stated answer unchanged" subset under unparsed answers.

Implements the partial-identification argument of
  Manski, C.F. (1990) "Nonparametric Bounds on Treatment Effects",
  American Economic Review 80(2):319-323,
in the form used for missing covariates/outcomes in
  Horowitz, J.L. & Manski, C.F. (2000) "Nonparametric Analysis of
  Randomized Experiments with Missing Covariate and Outcome Data",
  JASA 95(449):77-84.

WHY THIS IS THE RIGHT TOOL HERE, AND NOT AN IMPUTATION.

In the X8 probe the OUTCOME is observed on every trial: the shift in
probability mass is read off the logprobs and does not depend on parsing
anything. What is missing on 501 of 2700 rows is only the SUBGROUP LABEL,
whether the stated answer changed between rounds, because the regex could
not read an answer out of the generated text.

So the estimand

    mean of (mass shift) over the set U of pairs whose stated answer held

has a fully observed outcome and a partially observed membership. Write

    K = pairs KNOWN to be in U   (all four answers parsed, both held)
    A = pairs whose membership is AMBIGUOUS (some answer did not parse)

and U is any set with K contained in U contained in K union A. Every value
of mean(U) between the two extremes is attainable and nothing outside them
is, so those extremes are SHARP. No assumption about why a generation failed
to parse, and no judge model, can move the answer outside this interval. If
the lower bound is above zero then the qualitative conclusion holds however
the unparsed rows would have resolved.

The optimum sits at a prefix of A sorted by outcome: if a candidate set
contains a larger value but omits a smaller one, swapping them does not
raise the mean. Scanning the m+1 prefixes therefore finds the exact optimum.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MASS = "delta_prob_mass_toward_target"
KEYS = ["question_identifier", "replicate"]


# -- membership classification -------------------------------------------
def _row_state(frame: pd.DataFrame, suffix: str) -> pd.Series:
    """
    Per row: 'held' if both rounds parsed and agree, 'moved' if both parsed
    and differ, 'unknown' if either round failed to parse.

    'unknown' is NOT 'moved'. Collapsing the two is the bug this module
    exists to avoid: an unreadable generation is an absence of evidence, not
    evidence that the answer changed.
    """
    r0 = frame[f"round0_answer{suffix}"].astype(str).str.strip()
    r1 = frame[f"round1_answer{suffix}"].astype(str).str.strip()
    parsed = (r0 != "") & (r1 != "")
    return pd.Series(
        np.where(~parsed, "unknown", np.where(r0 == r1, "held", "moved")),
        index=frame.index,
    )


def classify_pairs(merged: pd.DataFrame) -> pd.Series:
    """
    A PAIR is in the subset only if BOTH its rows held their answer.

    'in'        both rows held.
    'out'       at least one row demonstrably moved; no resolution of the
                other row can put the pair back in.
    'ambiguous' nothing moved, but at least one row is unreadable.
    """
    treatment_state = _row_state(merged, "_treatment")
    baseline_state = _row_state(merged, "_baseline")
    moved = (treatment_state == "moved") | (baseline_state == "moved")
    inside = (treatment_state == "held") & (baseline_state == "held")
    return pd.Series(
        np.where(moved, "out", np.where(inside, "in", "ambiguous")),
        index=merged.index,
    )


# -- sharp bounds --------------------------------------------------------
def _prefix_extreme(known: np.ndarray, ambiguous: np.ndarray,
                    maximise: bool) -> Tuple[float, int]:
    """
    Exact optimum of mean(known + chosen) over every subset of `ambiguous`.

    Returns (value, how many ambiguous pairs the optimum absorbs).
    """
    order = np.sort(ambiguous)
    if maximise:
        order = order[::-1]
    total, count = float(known.sum()), int(known.size)
    best: Optional[float] = (total / count) if count else None
    best_k = 0
    for k, value in enumerate(order, start=1):
        total += float(value)
        count += 1
        current = total / count
        if best is None or (current > best if maximise else current < best):
            best, best_k = current, k
    return (float("nan") if best is None else float(best)), best_k


def sharp_bounds(known: Sequence[float],
                 ambiguous: Sequence[float]) -> Dict[str, Any]:
    k = np.asarray([v for v in known if np.isfinite(v)], dtype=float)
    a = np.asarray([v for v in ambiguous if np.isfinite(v)], dtype=float)
    low, n_low = _prefix_extreme(k, a, maximise=False)
    high, n_high = _prefix_extreme(k, a, maximise=True)
    everything = np.concatenate([k, a])
    return {
        "n_known_in": int(k.size),
        "n_ambiguous": int(a.size),
        "estimate_complete_case": float(k.mean()) if k.size else float("nan"),
        "bound_low": low,
        "bound_high": high,
        "bound_width": high - low,
        "n_absorbed_at_low": n_low,
        "n_absorbed_at_high": n_high,
        "estimate_all_ambiguous_held":
            float(everything.mean()) if everything.size else float("nan"),
    }


def _bounds_ci(known: np.ndarray, ambiguous: np.ndarray,
               clusters_known: np.ndarray, clusters_ambiguous: np.ndarray,
               n_resamples: int = 2000, seed: int = 20260502,
               alpha: float = 0.05) -> Tuple[float, float]:
    """
    Confidence interval COVERING THE IDENTIFIED SET, not a point.

    Resampling is clustered on the question: three replicates of one question
    share a prompt and are not independent draws, so resampling pairs would
    understate the spread.
    """
    questions = np.unique(np.concatenate([clusters_known, clusters_ambiguous]))
    if questions.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    known_by_question = {q: known[clusters_known == q] for q in questions}
    ambiguous_by_question = {q: ambiguous[clusters_ambiguous == q]
                             for q in questions}
    lows = np.empty(n_resamples)
    highs = np.empty(n_resamples)
    for i in range(n_resamples):
        drawn = rng.choice(questions, size=questions.size, replace=True)
        k = np.concatenate([known_by_question[q] for q in drawn])
        a = np.concatenate([ambiguous_by_question[q] for q in drawn])
        lows[i], _ = _prefix_extreme(k, a, maximise=False)
        highs[i], _ = _prefix_extreme(k, a, maximise=True)
    lows = lows[np.isfinite(lows)]
    highs = highs[np.isfinite(highs)]
    if lows.size == 0 or highs.size == 0:
        return float("nan"), float("nan")
    return (float(np.percentile(lows, 100 * alpha / 2)),
            float(np.percentile(highs, 100 * (1 - alpha / 2))))


# -- driver --------------------------------------------------------------
def bounded_unchanged_subset(
    trials: pd.DataFrame,
    treatment_condition: str = "WR",
    baseline_condition: str = "E",
    mass_column: str = MASS,
    by_task: bool = True,
) -> List[Dict[str, Any]]:
    """One row per task family. Pooling across families is not reported."""
    families = (sorted(trials["source_dataset"].dropna().unique())
                if by_task else ["ALL"])
    results: List[Dict[str, Any]] = []
    for family in families:
        subset = (trials if family == "ALL"
                  else trials[trials["source_dataset"] == family])
        treatment = subset[subset["condition"] == treatment_condition]
        baseline = subset[subset["condition"] == baseline_condition]
        merged = treatment.merge(baseline, on=KEYS,
                                 suffixes=("_treatment", "_baseline"))
        if merged.empty:
            continue
        merged = merged.assign(
            _membership=classify_pairs(merged),
            _difference=(merged[f"{mass_column}_treatment"]
                         - merged[f"{mass_column}_baseline"]),
        ).dropna(subset=["_difference"])

        known = merged[merged["_membership"] == "in"]
        ambiguous = merged[merged["_membership"] == "ambiguous"]
        row: Dict[str, Any] = {
            "source_dataset": family,
            "contrast": f"{treatment_condition}_minus_{baseline_condition}",
            "n_pairs_total": int(len(merged)),
            "n_pairs_out": int((merged["_membership"] == "out").sum()),
        }
        row.update(sharp_bounds(known["_difference"], ambiguous["_difference"]))
        ci_low, ci_high = _bounds_ci(
            known["_difference"].to_numpy(),
            ambiguous["_difference"].to_numpy(),
            known["question_identifier"].to_numpy(),
            ambiguous["question_identifier"].to_numpy(),
        )
        row["ci_low_of_identified_set"] = ci_low
        row["ci_high_of_identified_set"] = ci_high
        # The claim survives non-parse only if the WORST case clears zero.
        row["sign_robust_to_nonparse"] = bool(
            np.isfinite(row["bound_low"]) and row["bound_low"] > 0)
        row["sign_robust_with_uncertainty"] = bool(
            np.isfinite(ci_low) and ci_low > 0)
        results.append(row)
    return results


def nonparse_balance(trials: pd.DataFrame) -> pd.DataFrame:
    """Non-parse rate per condition. A differential rate is the thing to fear."""
    frame = trials.assign(
        unparsed=trials["round1_answer"].astype(str).str.strip().eq(""))
    grouped = (frame.groupby(["source_dataset", "condition"])["unparsed"]
               .agg(unparsed="sum", trials="count").reset_index())
    grouped["rate"] = grouped["unparsed"] / grouped["trials"]
    return grouped

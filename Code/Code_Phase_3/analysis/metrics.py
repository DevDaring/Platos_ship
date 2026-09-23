"""
metrics.py — revision outcomes with explicit denominators.

Phase-2 defect (next_plan.md §2.4): `phase2_analyzer._flip_rate()` averaged the
harmful-flip boolean over ALL trials, computing

    P(R0 correct AND R1 wrong)

while the manuscript defined and reported the conditional

    P(R1 wrong | R0 correct).

For GPT-4o-mini under C4 those are 8.67% and 13.43% — a factor of 1.5. Both
quantities are legitimate; silently swapping them is not. Every function here
states its denominator in the name and returns the denominator alongside the
rate, so a table can never print one while claiming the other.

Definitions (A0 = initial correctness, Y = final correctness):
    accuracy(t)        = P(Y = 1)                      denominator: all units
    harmful(t)  H(t)   = P(Y = 0 | A0 = 1)             denominator: A0 = 1
    beneficial(t) B(t) = P(Y = 1 | A0 = 0)             denominator: A0 = 0
    joint_loss(t)      = P(A0 = 1, Y = 0)              denominator: all units
    joint_gain(t)      = P(A0 = 0, Y = 1)              denominator: all units
    adoption(t)        = P(Y = peer target | A0 = 1)   denominator: A0 = 1

Accounting identity (bookkeeping, not a theorem):
    accuracy(t) - accuracy(A0) = (1 - a)*B(t) - a*H(t),  a = P(A0 = 1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger("platos_ship3.metrics")


@dataclass
class Rate:
    """A proportion that always travels with its denominator."""

    name: str
    numerator: int
    denominator: int
    denominator_description: str

    @property
    def value(self) -> float:
        if self.denominator == 0:
            return float("nan")
        return self.numerator / self.denominator

    @property
    def percent(self) -> float:
        return 100.0 * self.value

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["value"] = self.value
        payload["percent"] = self.percent
        return payload


def accuracy(frame: pd.DataFrame) -> Rate:
    """P(final answer correct), over all units."""
    return Rate("accuracy", int(frame["is_correct"].sum()), int(len(frame)),
                "all units")


def initial_accuracy(frame: pd.DataFrame) -> Rate:
    """P(initial answer correct) — the 'a' in the accounting identity."""
    return Rate("initial_accuracy", int(frame["r0_is_correct"].sum()),
                int(len(frame)), "all units")


def harmful(frame: pd.DataFrame) -> Rate:
    """H = P(final wrong | initial correct). Denominator: initially correct."""
    eligible = frame[frame["r0_is_correct"].astype(bool)]
    return Rate("harmful_revision", int((~eligible["is_correct"].astype(bool)).sum()),
                int(len(eligible)), "units whose initial answer was correct")


def beneficial(frame: pd.DataFrame) -> Rate:
    """B = P(final correct | initial wrong). Denominator: initially wrong."""
    eligible = frame[~frame["r0_is_correct"].astype(bool)]
    return Rate("beneficial_revision", int(eligible["is_correct"].astype(bool).sum()),
                int(len(eligible)), "units whose initial answer was wrong")


def joint_loss(frame: pd.DataFrame) -> Rate:
    """P(initial correct AND final wrong), over all units."""
    mask = frame["r0_is_correct"].astype(bool) & ~frame["is_correct"].astype(bool)
    return Rate("joint_loss", int(mask.sum()), int(len(frame)), "all units")


def joint_gain(frame: pd.DataFrame) -> Rate:
    """P(initial wrong AND final correct), over all units."""
    mask = ~frame["r0_is_correct"].astype(bool) & frame["is_correct"].astype(bool)
    return Rate("joint_gain", int(mask.sum()), int(len(frame)), "all units")


def target_adoption(frame: pd.DataFrame) -> Rate:
    """
    P(final answer == the peer-asserted wrong target | initial correct).

    Distinct from H: a model can abandon a correct answer for some OTHER wrong
    answer, which is instability rather than copying a peer. The paper must
    report both, because only this one is evidence of peer influence on the
    answer itself.
    """
    eligible = frame[
        frame["r0_is_correct"].astype(bool) & frame["peer_asserted_target"].notna()
    ]
    if "adopted_peer_target" not in eligible.columns:
        return Rate("peer_target_adoption", 0, int(len(eligible)),
                    "initially correct units with a peer target")
    return Rate(
        "peer_target_adoption",
        int(eligible["adopted_peer_target"].fillna(False).astype(bool).sum()),
        int(len(eligible)),
        "initially correct units with a peer target",
    )


def answer_change_rate(frame: pd.DataFrame) -> Rate:
    """P(final answer differs from initial), over all units."""
    return Rate("answer_changed", int(frame["answer_changed"].fillna(False).sum()),
                int(len(frame)), "all units")


ALL_RATES = {
    "accuracy": accuracy,
    "initial_accuracy": initial_accuracy,
    "harmful_revision": harmful,
    "beneficial_revision": beneficial,
    "joint_loss": joint_loss,
    "joint_gain": joint_gain,
    "peer_target_adoption": target_adoption,
    "answer_changed": answer_change_rate,
}


def summarise(frame: pd.DataFrame, **context: Any) -> Dict[str, Any]:
    """Every rate for one cell, with denominators and context columns."""
    row: Dict[str, Any] = dict(context)
    row["n_units"] = int(len(frame))
    row["n_questions"] = int(frame["question_identifier"].nunique())
    for name, function in ALL_RATES.items():
        rate = function(frame)
        row[name] = rate.value
        row[f"{name}_n"] = rate.numerator
        row[f"{name}_denominator"] = rate.denominator
    # Mean tokens: the budget-matched comparison needs real usage, not caps.
    for column, label in (("total_input_tokens", "mean_input_tokens"),
                          ("total_output_tokens", "mean_output_tokens")):
        if column in frame.columns:
            row[label] = float(frame[column].mean())
    if "unrecovered_parse_failure" in set(
            frame.get("answer_extraction_method", pd.Series(dtype=str))):
        row["n_parse_failures"] = int(
            (frame["answer_extraction_method"] == "unrecovered_parse_failure").sum()
        )
    else:
        row["n_parse_failures"] = 0
    return row


def summarise_by(
    frame: pd.DataFrame, group_columns: List[str]
) -> pd.DataFrame:
    """Apply `summarise` to every group; one row per cell."""
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for key, group in frame.groupby(group_columns, dropna=False):
        context = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        rows.append(summarise(group, **context))
    return pd.DataFrame(rows)


def per_question_rate(
    frame: pd.DataFrame, rate_name: str
) -> pd.Series:
    """
    Collapse replicates to one value per question — the inference unit.

    Replicates of one question are not independent observations, so every
    paired contrast and every bootstrap operates on this series, not on trials.
    Questions with an empty denominator (e.g. no initially-correct replicate)
    return NaN and are dropped pairwise by the contrast functions.
    """
    if frame.empty:
        return pd.Series(dtype=float)

    def _value(group: pd.DataFrame) -> float:
        return ALL_RATES[rate_name](group).value

    return frame.groupby("question_identifier").apply(_value)


def accounting_identity_check(frame: pd.DataFrame, tolerance: float = 1e-9
                              ) -> Dict[str, float]:
    """
    Verify accuracy(t) - a = (1-a)*B - a*H on the cell.

    Included because the paper states the identity; a released script that
    checks it stops a reviewer wondering whether the numbers cohere.
    """
    a = initial_accuracy(frame).value
    left = accuracy(frame).value - a
    right = (1 - a) * beneficial(frame).value - a * harmful(frame).value
    return {
        "initial_accuracy": a,
        "lhs_accuracy_change": left,
        "rhs_from_B_and_H": right,
        "absolute_difference": abs(left - right),
        "holds": bool(abs(left - right) < max(tolerance, 1e-9)),
    }

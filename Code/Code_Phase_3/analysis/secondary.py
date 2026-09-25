"""
secondary.py — the two secondary analyses the frozen plan requires.

1. GEE condition ladder (ANALYSIS_PLAN.md; next_plan.md §5).
   Per focal model, a logistic GEE of final-answer correctness on condition,
   clustered by question with exchangeable working correlation, reference R.
   Implements Liang, K.-Y. & Zeger, S.L. (1986) "Longitudinal data analysis
   using generalized linear models", Biometrika 73(1):13-22.
   Secondary to the question-level paired bootstrap, which stays primary.

2. Worst-case bounds for unparsed answers (next_plan.md §5, last bullet).
   An unparsed revision is scored incorrect in every headline number. The
   bound asks how far a primary contrast could move if every unparsed answer
   in one arm had been correct and every one in the other arm wrong, and the
   reverse. No assumption about why a response failed to parse can move the
   contrast outside that interval.
   Implements the worst-case bounds of Manski, C.F. (1990) "Nonparametric
   bounds on treatment effects", American Economic Review 80(2):319-323,
   as applied to missing outcomes by Horowitz, J.L. & Manski, C.F. (2000),
   JASA 95(449):77-84.

   For harmful revision, H = P(R1 wrong | R0 correct), only the REVISION
   answer is imputed; the Round-0 answer that defines the denominator is the
   same cached record in both arms, so it cannot differ between them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .stats import gee_logistic

logger = logging.getLogger("platos_ship3.secondary")

X1_LADDER = ["R", "E", "WR", "H", "CR"]


def _protocol_b(registry: pd.DataFrame) -> pd.DataFrame:
    frame = registry[(registry["protocol"] == "B")
                     & (registry["dataset_scope"] == "main300")].copy()
    if "round_index" in frame.columns:
        frame = frame[frame["round_index"].fillna(1).astype(int) == 1]
    return frame


def gee_condition_ladder(registry: pd.DataFrame,
                         conditions: List[str] = X1_LADDER) -> pd.DataFrame:
    """One GEE per focal model over the X1 conditions, reference R."""
    frame = _protocol_b(registry)
    frame = frame[frame["condition"].isin(conditions)]
    tables = []
    for focal, group in frame.groupby("focal_key"):
        table = gee_logistic(group.assign(is_correct=group["is_correct"].astype(bool)),
                             reference_level="R")
        if table is None:
            continue
        table = table[table["term"] != "Intercept"].copy()
        table["condition"] = table["term"].str.extract(r"\[T\.([^\]]+)\]")[0]
        table.insert(0, "focal_key", focal)
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


# ── worst-case bounds ─────────────────────────────────────────────────────
def _arm(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    return frame[frame["condition"] == condition]


def _matched(left: pd.DataFrame, right: pd.DataFrame):
    keys = ["focal_key", "question_identifier", "replicate"]
    common = (pd.MultiIndex.from_frame(left[keys].astype(str))
              .intersection(pd.MultiIndex.from_frame(right[keys].astype(str))))
    keep = lambda f: f[pd.MultiIndex.from_frame(f[keys].astype(str)).isin(common)]  # noqa: E731
    return keep(left), keep(right)


def _rate(frame: pd.DataFrame, statistic: str, unparsed_correct: bool) -> float:
    """Question-level mean of the statistic with unparsed revisions imputed."""
    correct = frame["is_correct"].astype(bool).to_numpy().copy()
    unparsed = frame["extracted_answer"].isna().to_numpy()
    correct[unparsed] = unparsed_correct
    data = frame.assign(_c=correct)
    if statistic == "accuracy_delta":
        per_q = data.groupby("question_identifier")["_c"].mean()
    elif statistic == "harmful_delta":
        eligible = data[data["r0_is_correct"].astype(bool)]
        per_q = (~eligible["_c"]).groupby(eligible["question_identifier"]).mean()
    else:
        raise ValueError(statistic)
    return float(per_q.mean())


def parse_failure_bounds(registry: pd.DataFrame,
                         families: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """Worst-case interval for every paired contrast in the frozen families."""
    frame = _protocol_b(registry)
    rows = []
    for family, specs in families.items():
        for spec in specs:
            statistic = spec.get("stat")
            if statistic not in ("accuracy_delta", "harmful_delta"):
                continue
            left, right = _matched(_arm(frame, spec["a"]), _arm(frame, spec["b"]))
            if left.empty:
                continue
            scored = (_rate(left, statistic, False) - _rate(right, statistic, False))
            # For harmful revision "correct" lowers the rate, so the extremes
            # swap relative to accuracy; take min/max over both assignments.
            options = [
                _rate(left, statistic, a) - _rate(right, statistic, b)
                for a in (False, True) for b in (False, True)
            ]
            rows.append({
                "family": family, "name": spec["name"], "statistic": statistic,
                "condition_a": spec["a"], "condition_b": spec["b"],
                "estimate_unparsed_scored_wrong": scored,
                "worst_case_low": min(options), "worst_case_high": max(options),
                "unparsed_share_a": float(left["extracted_answer"].isna().mean()),
                "unparsed_share_b": float(right["extracted_answer"].isna().mean()),
                "sign_robust": bool(min(options) > 0 or max(options) < 0),
                "n_units_per_arm": int(len(left)),
            })
    return pd.DataFrame(rows)


# ── 3. peer-target adoption over its matched chance baseline (exploratory) ──
def adoption_excess(registry: pd.DataFrame, solo_accuracy: Dict[str, float],
                    n_resamples: int = 5000, seed: int = 20260502) -> Dict[str, Any]:
    """
    How much more often a model ends on the wrong answer its peers argued for
    than it would by chance, per model, and whether that excess tracks ability.

    Adoption alone has no baseline: a weak model may land on a given wrong
    option often without any peer. Peers are drawn per (question, replicate)
    and are identical in every condition (src/peer_pools.peer_rng), so the R
    unit of the same model, question and replicate tells how often the model,
    re-answering with no peers, lands on exactly the target its WR peers
    argued for. Excess = P(WR answer in targets) - P(R answer in the same
    targets), over units correct at Round 0. Exploratory: not in the frozen
    contrast families.
    """
    import json

    from src.extraction import answers_equal
    from .stats import bootstrap_ci, spearman_exact

    frame = _protocol_b(registry)
    keys = ["focal_key", "question_identifier", "replicate"]
    wr = frame[frame["condition"] == "WR"][keys + ["r0_is_correct", "adopted_peer_target",
                                                   "peer_asserted_targets_all"]]
    r = frame[frame["condition"] == "R"][keys + ["extracted_answer"]]
    joined = wr.merge(r, on=keys)
    joined = joined[joined["r0_is_correct"].astype(bool)].copy()
    joined["chance"] = [
        any(answers_equal(answer, t) for t in json.loads(targets or "[]"))
        for answer, targets in zip(joined["extracted_answer"], joined["peer_asserted_targets_all"])]
    joined["diff"] = joined["adopted_peer_target"].astype(float) - joined["chance"].astype(float)

    rows = []
    for focal, group in joined.groupby("focal_key"):
        per_q = group.groupby("question_identifier")["diff"].mean().to_numpy()
        estimate, low, high = bootstrap_ci(per_q, n_resamples=n_resamples, seed=seed)
        rows.append({"focal_key": focal, "solo_accuracy": solo_accuracy.get(focal, np.nan),
                     "adoption": float(group["adopted_peer_target"].astype(float).mean()),
                     "chance": float(group["chance"].mean()),
                     "excess": float(group["diff"].mean()), "excess_q_mean": estimate,
                     "ci_low": low, "ci_high": high, "n_units": int(len(group))})
    table = pd.DataFrame(rows).dropna(subset=["solo_accuracy"])
    result = {"per_model": table.to_dict("records")}
    if len(table) >= 3:
        result["spearman_solo_vs_excess"] = spearman_exact(table["solo_accuracy"].astype(float),
                                                           table["excess"].astype(float))
    return result

"""
voting.py — X5: the baseline the debate literature now demands.

Choi, Zhu & Li (Debate or Vote, NeurIPS 2025) report that "Majority Voting
alone accounts for most of the performance gains typically attributed to MAD".
Zhang et al. (Stop Overvaluing Multi-Agent Debate, 2025) find debate often
fails to beat Chain-of-Thought and Self-Consistency. The reviewed paper argued
its homogeneous condition was "close to sampling one model three times"
(Cohen's kappa = 0.72) — but an agreement statistic is not a voting baseline.
Agreement says the samples are correlated; it does not say what plurality
voting over the answer STRINGS would have scored, at what cost.

This module computes the real thing from the Round-0 cache, so it costs
nothing: the replicates already exist.

Reported per model AND per task (MMLU-Pro vs GSM8K). Pooling hides the fact
that the effects differ by answer format, which is one of the things the
reviewed paper was criticised for.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.extraction import normalise_answer

logger = logging.getLogger("platos_ship3.voting")


def plurality_vote(answers: List[Any], tie_rule: str = "first") -> Optional[str]:
    """
    Plurality over normalised answer STRINGS, with a predetermined tie rule.

    `tie_rule="first"` keeps the earliest replicate among the tied answers,
    which is decided in advance rather than after seeing which rule scores
    better. Unparsed answers do not vote.
    """
    normalised = [normalise_answer(a) for a in answers]
    normalised = [a for a in normalised if a]
    if not normalised:
        return None

    counts: Dict[str, int] = {}
    for answer in normalised:
        counts[answer] = counts.get(answer, 0) + 1
    best = max(counts.values())
    tied = [a for a in normalised if counts[a] == best]
    if tie_rule == "first":
        return tied[0]
    return sorted(set(tied))[0]


def voting_baseline(
    r0_cache: pd.DataFrame,
    tie_rule: str = "first",
    group_by_task: bool = True,
) -> pd.DataFrame:
    """
    Self-consistency accuracy from the cached replicates, per focal model.

    Columns:
      single_sample_accuracy   — mean over replicates (one sample, the
                                 fair comparison for a one-call condition)
      plurality_vote_accuracy  — accuracy of the plurality answer
      n_calls_single / n_calls_vote — the budget each one actually spends
    """
    if r0_cache.empty:
        return pd.DataFrame()

    group_columns = ["focal_key"]
    if group_by_task and "source_dataset" in r0_cache.columns:
        group_columns.append("source_dataset")

    rows: List[Dict[str, Any]] = []
    for key, group in r0_cache.groupby(group_columns, dropna=False):
        context = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))

        per_question = []
        for question_id, question_group in group.groupby("question_identifier"):
            ordered = question_group.sort_values("replicate")
            answers = ordered["extracted_answer"].tolist()
            correct = ordered["correct_answer"].iloc[0]
            vote = plurality_vote(answers, tie_rule=tie_rule)
            per_question.append(
                {
                    "question_identifier": question_id,
                    "single_sample_accuracy": float(ordered["is_correct"].mean()),
                    "vote_correct": bool(
                        vote is not None
                        and normalise_answer(vote) == normalise_answer(correct)
                    ),
                    "n_replicates": int(len(ordered)),
                }
            )

        table = pd.DataFrame(per_question)
        if table.empty:
            continue
        n_replicates = int(table["n_replicates"].median())
        rows.append(
            {
                **context,
                "n_questions": int(len(table)),
                "n_replicates": n_replicates,
                "single_sample_accuracy": float(table["single_sample_accuracy"].mean()),
                "plurality_vote_accuracy": float(table["vote_correct"].mean()),
                "vote_minus_single": float(
                    table["vote_correct"].mean()
                    - table["single_sample_accuracy"].mean()
                ),
                "n_calls_single": 1,
                "n_calls_vote": n_replicates,
            }
        )
    return pd.DataFrame(rows)


def budget_matched_comparison(
    voting: pd.DataFrame,
    cell_metrics: pd.DataFrame,
    conditions: Optional[List[str]] = None,
    protocol: str = "B",
    scope: str = "main300",
) -> pd.DataFrame:
    """
    Put every strategy on one table with its realised token cost.

    A revision condition spends 2 focal calls (initial + revision); plurality
    voting over k replicates spends k. Comparing accuracy without cost is what
    the debate literature was criticised for, so cost travels with every row.
    """
    conditions = conditions or ["R", "G", "E", "WR", "H"]
    selected = cell_metrics[
        (cell_metrics["protocol"] == protocol)
        & (cell_metrics["dataset_scope"] == scope)
        & (cell_metrics["condition"].isin(conditions))
    ].copy()
    if "round_index" in selected.columns:
        selected = selected[selected["round_index"] == 1]

    rows: List[Dict[str, Any]] = []
    for _, row in selected.iterrows():
        rows.append(
            {
                "strategy": f"debate:{row['condition']}",
                "focal_key": row["focal_key"],
                "accuracy": row["accuracy"],
                "harmful_revision": row.get("harmful_revision"),
                # Initial answer + one revision. E's two peers are two MORE
                # focal samples, so E spends four focal calls, not two.
                "focal_calls": 4 if row["condition"] == "E" else 2,
                "mean_output_tokens": row.get("mean_output_tokens"),
                "n_questions": row.get("n_questions"),
            }
        )

    pooled = voting
    if "source_dataset" in voting.columns:
        # Weighted by questions per task. The unweighted mean of per-task
        # accuracies gave a 20-question task the weight of a 100-question one,
        # so the voting rows were not on the same items as the debate rows.
        def _weighted(group: pd.DataFrame) -> pd.Series:
            weights = group["n_questions"].astype(float)
            return pd.Series({
                "single_sample_accuracy": float(np.average(
                    group["single_sample_accuracy"], weights=weights)),
                "plurality_vote_accuracy": float(np.average(
                    group["plurality_vote_accuracy"], weights=weights)),
                "n_replicates": group["n_replicates"].median(),
                "n_questions": int(weights.sum()),
            })

        pooled = voting.groupby("focal_key").apply(_weighted).reset_index()
    for _, row in pooled.iterrows():
        rows.append({
            "strategy": "single_sample",
            "focal_key": row["focal_key"],
            "accuracy": row["single_sample_accuracy"],
            "harmful_revision": None,
            "focal_calls": 1,
            "mean_output_tokens": None,
            "n_questions": row.get("n_questions"),
        })
        rows.append({
            "strategy": f"plurality_vote_k{int(row['n_replicates'])}",
            "focal_key": row["focal_key"],
            "accuracy": row["plurality_vote_accuracy"],
            "harmful_revision": None,
            "focal_calls": int(row["n_replicates"]),
            "mean_output_tokens": None,
            "n_questions": row.get("n_questions"),
        })

    return pd.DataFrame(rows).sort_values(["focal_key", "strategy"])

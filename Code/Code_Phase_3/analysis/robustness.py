"""
robustness.py — sensitivity analyses of existing data (next_plan.md §4).

All exploratory and labelled so. Nothing here enters a frozen family.

  task_split           pooled WR-R (accuracy, H, B) separately for MMLU-Pro and GSM8K
  leave_one_model_out  pooled WR-R accuracy with each focal model removed in turn
  degraded_70b         pooled WR-R accuracy with the archived degraded Llama-3.1-70B
                       run in place of the re-collected one (never mixing its
                       initial states with the new revision arms)
  gsm_symbolic_pairs   (WR-R on the template instance) - (WR-R on its GSM8K original),
                       clustered by template
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .contrasts import paired_condition_contrast
from .stats import bootstrap_ci, paired_permutation_p

logger = logging.getLogger("platos_ship3.robustness")

KEYS = ["focal_key", "question_identifier", "replicate"]


def _b_main(registry: pd.DataFrame) -> pd.DataFrame:
    return registry[(registry["protocol"] == "B") & (registry["dataset_scope"] == "main300")]


def _contrast(frame: pd.DataFrame, name: str, stat: str) -> Dict[str, Any]:
    result = paired_condition_contrast(frame, name, stat, "WR", "R", protocol="B",
                                       scope="main300", n_resamples=5000)
    return {"name": name, "statistic": stat, "estimate": result.estimate,
            "ci_low": result.ci_low, "ci_high": result.ci_high,
            "p_value": result.p_value, "n_questions": result.n_pairs}


def task_split(registry: pd.DataFrame) -> List[Dict[str, Any]]:
    frame = _b_main(registry)
    rows = []
    for task, sub in frame.groupby("source_dataset"):
        for stat in ("accuracy_delta", "harmful_delta", "beneficial_delta"):
            rows.append({"task": task, **_contrast(sub, f"WR_minus_R:{task}", stat)})
    return rows


def leave_one_model_out(registry: pd.DataFrame) -> List[Dict[str, Any]]:
    frame = _b_main(registry)
    rows = []
    for focal in sorted(frame["focal_key"].dropna().unique()):
        sub = frame[frame["focal_key"] != focal]
        rows.append({"dropped": focal, **_contrast(sub, f"WR_minus_R:without_{focal}",
                                                     "accuracy_delta")})
    return rows


def degraded_70b(registry: pd.DataFrame, project_root: Path) -> Dict[str, Any]:
    """Pooled WR-R with the archived degraded 70B run (its own R0 and arms)."""
    from .registry import _scope_for_question
    from src.call_guard import drop_failed_calls

    path = (project_root / "results/outputs/archive/"
            "shard_sweep_llama_3_1_70b_degraded_upstream_2026-09-24/revision_log.parquet")
    if not path.exists():
        return {}
    old = drop_failed_calls(pd.read_parquet(path), "archived 70B revision log")
    old = old[old["focal_key"] == "sweep_llama_3_1_70b"].copy()
    old["protocol"] = "B"
    old["dataset_scope"] = old["question_identifier"].map(_scope_for_question)
    frame = _b_main(registry)
    swapped = pd.concat([frame[frame["focal_key"] != "sweep_llama_3_1_70b"],
                         old[old["dataset_scope"] == "main300"]], ignore_index=True)
    have = set(old.loc[old["dataset_scope"] == "main300", "condition"])
    if not {"R", "WR"} <= have:
        return {"note": "archived run lacks R or WR", "conditions": sorted(have)}
    return _contrast(swapped, "WR_minus_R:archived_degraded_70B", "accuracy_delta")


def gsm_symbolic_pairs(registry: pd.DataFrame, pool: pd.DataFrame,
                       seed: int = 20260502) -> Dict[str, Any]:
    """Interaction (sym WR-R) - (orig WR-R), clustered by template."""
    frame = registry[(registry["protocol"] == "B")
                     & registry["condition"].isin(["R", "WR"])
                     & registry["question_identifier"].str.startswith(("gsmsym", "gsmorig"))]
    if frame.empty:
        return {}
    sym = pool[pool["question_identifier"].str.startswith("gsmsym")]
    template_of = dict(zip(sym["question_identifier"], sym["gsm_symbolic_id"]))
    orig_rows = pool[pool["question_identifier"].str.startswith("gsmorig")]
    template_of.update(dict(zip(orig_rows["question_identifier"], orig_rows["gsm_symbolic_id"])))
    frame = frame.assign(
        template=frame["question_identifier"].map(template_of),
        kind=np.where(frame["question_identifier"].str.startswith("gsmsym"), "sym", "orig"),
        correct=frame["is_correct"].astype(float))
    frame = frame.dropna(subset=["template"])
    cell = frame.groupby(["focal_key", "template", "kind", "condition"])["correct"].mean().unstack()
    cell = (cell["WR"] - cell["R"]).unstack("kind").dropna()
    per_model = {}
    for focal, sub in cell.groupby(level="focal_key"):
        diff = (sub["sym"] - sub["orig"]).to_numpy()
        est, lo, hi = bootstrap_ci(diff, seed=seed)
        per_model[focal] = {"interaction": est, "ci_low": lo, "ci_high": hi,
                            "n_templates": int(len(diff))}
    pooled_by_template = (cell["sym"] - cell["orig"]).groupby(level="template").mean().to_numpy()
    est, lo, hi = bootstrap_ci(pooled_by_template, seed=seed)
    return {"pooled": {"interaction": est, "ci_low": lo, "ci_high": hi,
                       "p_value": paired_permutation_p(pooled_by_template, seed=seed),
                       "n_templates": int(len(pooled_by_template))},
            "per_model": per_model,
            "note": "exploratory; (WR-R on template instance) - (WR-R on GSM8K original), "
                    "bootstrap over templates"}


def run_all(registry: pd.DataFrame, project_root: Path, gsm_pool: pd.DataFrame) -> Dict[str, Any]:
    return {"task_split": task_split(registry),
            "leave_one_model_out": leave_one_model_out(registry),
            "degraded_70b": degraded_70b(registry, project_root),
            "gsm_symbolic_pairs": gsm_symbolic_pairs(registry, gsm_pool)}

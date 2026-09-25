"""
phase4.py — the pre-registered Phase 4 estimands (PREREG_PHASE4.md).

    python -m analysis.phase4        # writes results/phase4/phase4_numbers.json

Unit of inference: the question. Intervals: percentile bootstrap over
questions (5,000 resamples, seed 20260502); p-values: sign-flip permutation.
Unparsed answers are scored incorrect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import PROJECT_ROOT  # noqa: F401
from .stats import bootstrap_ci, paired_permutation_p

ROOT = Path(__file__).resolve().parent.parent
P4 = ROOT / "results/phase4"
KEYS = ["focal_key", "question_identifier", "replicate"]
SEED = 20260502


def _load(exp: str) -> pd.DataFrame:
    from src.call_guard import drop_failed_calls

    frames = []
    for folder in sorted((P4 / exp).iterdir()):
        if not folder.is_dir():
            continue
        path = folder / "revision_log.parquet"
        shards = list((folder / "_shards" / "revision_log").glob("*.parquet"))
        parts = ([pd.read_parquet(path)] if path.exists() else []) + [pd.read_parquet(s) for s in shards]
        if parts:
            frames.append(pd.concat(parts).drop_duplicates("unit_id"))
    if not frames:
        return pd.DataFrame()
    return drop_failed_calls(pd.concat(frames, ignore_index=True), f"phase4 {exp}")


def _ci(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "n_questions": 0}
    est, lo, hi = bootstrap_ci(values, seed=SEED)
    return {"estimate": est, "ci_low": lo, "ci_high": hi,
            "p_value": paired_permutation_p(values, seed=SEED), "n_questions": int(len(values))}


def _cluster_ratio_ci(frame: pd.DataFrame, value: str) -> Dict[str, float]:
    """Trial-weighted mean of `value` with a question-cluster bootstrap."""
    per_q = frame.groupby("question_identifier")[value].agg(["sum", "count"])
    sums, counts = per_q["sum"].to_numpy(float), per_q["count"].to_numpy(float)
    if len(sums) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "n_units": 0, "n_questions": 0}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(sums), size=(5000, len(sums)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    q_means = sums / counts
    return {"estimate": float(sums.sum() / counts.sum()),
            "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
            "p_value": paired_permutation_p(q_means, seed=SEED),
            "n_units": int(counts.sum()), "n_questions": int(len(sums))}


def _in_targets(answer, targets_json) -> bool:
    from src.extraction import answers_equal

    return any(answers_equal(answer, t) for t in json.loads(targets_json or "[]"))


# ── Experiment A ────────────────────────────────────────────────────────────
def experiment_a(rows: pd.DataFrame) -> Dict[str, Any]:
    if rows.empty:
        return {}
    rows = rows.assign(correct=rows["is_correct"].astype(float),
                       r0c=rows["r0_is_correct"].astype(bool))
    wide = rows.pivot_table(index=KEYS, columns="condition", values="correct", aggfunc="first")
    arms = ["A_R", "A_WR", "A_Rhid", "A_WRhid"]
    wide = wide.dropna(subset=[a for a in arms if a in wide.columns])
    out: Dict[str, Any] = {"arms": {}, "n_units_complete": int(len(wide))}
    for arm in arms:
        out["arms"][arm] = rows[rows["condition"] == arm].groupby("focal_key")["correct"].mean().to_dict()
    wide["inter"] = (wide["A_WR"] - wide["A_R"]) - (wide["A_WRhid"] - wide["A_Rhid"])
    wide["effect_shown"] = wide["A_WR"] - wide["A_R"]
    wide["effect_hidden"] = wide["A_WRhid"] - wide["A_Rhid"]
    q = wide.reset_index().groupby("question_identifier")
    out["primary_interaction_accuracy"] = _ci(q["inter"].mean().to_numpy())
    out["wr_minus_r_shown"] = _ci(q["effect_shown"].mean().to_numpy())
    out["wr_minus_r_hidden"] = _ci(q["effect_hidden"].mean().to_numpy())
    out["per_model"] = {}
    for focal, sub in wide.reset_index().groupby("focal_key"):
        qq = sub.groupby("question_identifier")
        out["per_model"][focal] = {"interaction": _ci(qq["inter"].mean().to_numpy()),
                                   "shown": _ci(qq["effect_shown"].mean().to_numpy()),
                                   "hidden": _ci(qq["effect_hidden"].mean().to_numpy())}
    # secondary: harmful and beneficial revision interactions on their cohorts
    for label, cohort in (("harmful", True), ("beneficial", False)):
        sub = rows[rows["r0c"] == cohort]
        w = sub.pivot_table(index=KEYS, columns="condition", values="correct", aggfunc="first").dropna()
        if w.empty:
            continue
        sign = -1.0 if cohort else 1.0     # H = 1 - correct on the correct cohort
        w["inter"] = sign * ((w["A_WR"] - w["A_R"]) - (w["A_WRhid"] - w["A_Rhid"]))
        out[f"secondary_interaction_{label}"] = _ci(
            w.reset_index().groupby("question_identifier")["inter"].mean().to_numpy())
        out[f"{label}_rates"] = {arm: float((1 - w[arm]).mean() if cohort else w[arm].mean())
                                 for arm in arms}
    # secondary: adoption of the WR targets, R arms scored on the same targets
    wr = rows[rows["condition"].isin(["A_WR", "A_WRhid"])][
        KEYS + ["condition", "peer_asserted_targets_all"]]
    targets = wr[wr["condition"] == "A_WR"].set_index(KEYS)["peer_asserted_targets_all"]
    ok = rows[rows["r0c"]].copy()
    ok["targets"] = [targets.get(tuple(k)) for k in ok[KEYS].itertuples(index=False)]
    ok = ok.dropna(subset=["targets"])
    ok["hit"] = [_in_targets(a, t) for a, t in zip(ok["extracted_answer"], ok["targets"])]
    w = ok.pivot_table(index=KEYS, columns="condition", values="hit", aggfunc="first").dropna()
    if not w.empty:
        w = w.astype(float)
        w["inter"] = (w["A_WR"] - w["A_R"]) - (w["A_WRhid"] - w["A_Rhid"])
        out["secondary_interaction_adoption"] = _ci(
            w.reset_index().groupby("question_identifier")["inter"].mean().to_numpy())
        out["adoption_rates"] = {arm: float(w[arm].mean()) for arm in arms}
    return out


# ── Experiment C ────────────────────────────────────────────────────────────
def experiment_c(rows: pd.DataFrame) -> Dict[str, Any]:
    if rows.empty:
        return {}
    rows = rows.assign(correct=rows["is_correct"].astype(float),
                       r0c=rows["r0_is_correct"].astype(bool))
    wide = rows.pivot_table(index=KEYS, columns=["condition", "round_index"],
                            values="correct", aggfunc="first")
    out: Dict[str, Any] = {"per_round": {}}
    for (cond, rnd), col in wide.items():
        out["per_round"].setdefault(cond, {})[int(rnd)] = {
            "accuracy": float(col.mean()),
            "harmful": float(1 - col[wide.index.isin(rows[rows["r0c"]].set_index(KEYS).index)].mean())}
    cohort = rows[rows["r0c"]].pivot_table(index=KEYS, columns=["condition", "round_index"],
                                           values="correct", aggfunc="first").dropna()
    if cohort.empty:
        return out
    h = 1 - cohort
    inter = (h[("C_WR", 3)] - h[("C_R", 3)]) - (h[("C_WR", 1)] - h[("C_R", 1)])
    frame = inter.rename("inter").reset_index()
    out["primary_interaction_harmful"] = _ci(frame.groupby("question_identifier")["inter"].mean().to_numpy())
    out["per_model"] = {f: _ci(s.groupby("question_identifier")["inter"].mean().to_numpy())
                        for f, s in frame.groupby("focal_key")}
    acc = rows.pivot_table(index=KEYS, columns=["condition", "round_index"], values="correct",
                           aggfunc="first").dropna()
    ai = (acc[("C_WR", 3)] - acc[("C_R", 3)]) - (acc[("C_WR", 1)] - acc[("C_R", 1)])
    out["secondary_interaction_accuracy"] = _ci(
        ai.rename("inter").reset_index().groupby("question_identifier")["inter"].mean().to_numpy())
    out["n_cohort_units"] = int(len(cohort))
    return out


# ── Experiment B ────────────────────────────────────────────────────────────
def experiment_b(rows: pd.DataFrame) -> Dict[str, Any]:
    if rows.empty:
        return {}
    eligibility = json.loads((P4 / "B/eligible.json").read_text(encoding="utf-8"))
    rows = rows.assign(correct=rows["is_correct"].astype(float),
                       r0c=rows["r0_is_correct"].astype(bool))
    out: Dict[str, Any] = {"bank": {k: v for k, v in eligibility.items() if k != "eligible"}}
    r_arm = rows[rows["condition"] == "B_R"].set_index(KEYS)
    for panel in ("B_P1", "B_P2"):
        arm = rows[(rows["condition"] == panel) & rows["r0c"]].set_index(KEYS)
        joined = arm[["extracted_answer", "peer_asserted_targets_all", "adopted_peer_target"]].join(
            r_arm[["extracted_answer"]].rename(columns={"extracted_answer": "r_answer"}), how="inner")
        joined["chance"] = [_in_targets(a, t) for a, t in
                            zip(joined["r_answer"], joined["peer_asserted_targets_all"])]
        joined["diff"] = joined["adopted_peer_target"].astype(float) - joined["chance"].astype(float)
        joined = joined.reset_index()
        key = "primary" if panel == "B_P2" else "secondary"
        out[f"{key}_excess_adoption_{panel}"] = _cluster_ratio_ci(joined, "diff")
        out[f"adoption_{panel}"] = float(joined["adopted_peer_target"].astype(float).mean())
        out[f"chance_{panel}"] = float(joined["chance"].astype(float).mean())
        out[f"per_model_excess_{panel}"] = {f: _cluster_ratio_ci(s, "diff")
                                            for f, s in joined.groupby("focal_key")}
    wide = rows.pivot_table(index=KEYS, columns="condition", values="correct", aggfunc="first")
    r0c = rows.drop_duplicates(KEYS).set_index(KEYS)["r0c"]
    for panel in ("B_P0", "B_P1", "B_P2"):
        w = wide[[panel, "B_R"]].dropna()
        d = (w[panel] - w["B_R"]).rename("d").reset_index()
        out[f"accuracy_{panel}_minus_R"] = _ci(d.groupby("question_identifier")["d"].mean().to_numpy())
        # joint loss P(correct at Round 0 and wrong after), difference from R
        loss = w.join(r0c)
        jl = (((1 - loss[panel]) * loss["r0c"]) - ((1 - loss["B_R"]) * loss["r0c"])).rename("d").reset_index()
        out[f"joint_loss_{panel}_minus_R"] = _ci(jl.groupby("question_identifier")["d"].mean().to_numpy())
    out["arm_accuracy"] = rows.groupby(["focal_key", "condition"])["correct"].mean().unstack().to_dict("index")
    out["solo_accuracy"] = rows.drop_duplicates(KEYS).groupby("focal_key")["r0c"].mean().to_dict()
    return out


def main() -> int:
    numbers = {"A": experiment_a(_load("A")), "C": experiment_c(_load("C")),
               "B": experiment_b(_load("B"))}
    path = P4 / "phase4_numbers.json"
    path.write_text(json.dumps(numbers, indent=1, default=float), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

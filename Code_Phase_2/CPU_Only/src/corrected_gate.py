"""
corrected_gate.py — Experiment E3: the corrected confidence-filter pre-flight.

Phase 1's gate asked P(confidence >= 60 | wrong) and passed at 0.995 — but that
number says nothing about whether confidence SEPARATES wrong from right. The
corrected gate tests the discriminative quantity:

    gap = P(loud | wrong) - P(loud | correct)

and additionally requires AUROC(confidence -> correctness) to clear a floor.
The filter is only worth deploying when confident-wrong is meaningfully MORE
common than confident-correct. On Phase 1 data the gap was -0.003, so the
corrected gate FAILS and predicts the filter's null effect in advance.

# Implements the corrected pre-flight gate from Review_Fix.md E3.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("platos_ship.corrected_gate")


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of `scores` predicting positive `labels` (1=correct). Rank-based."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    ranks = pd.Series(all_scores).rank(method="average").values
    r_pos = ranks[: len(pos)].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def compute_corrected_gate(
    trial_log: pd.DataFrame,
    conditions: List[str],
    roles: List[str],
    high_confidence_threshold: int = 60,
    discriminative_gap_threshold: float = 0.10,
    auroc_floor: float = 0.60,
    rounds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute the corrected discriminative-gap gate over dumb responses that
    carry a parsed confidence integer.
    """
    mask = (
        trial_log["condition_identifier"].isin(conditions)
        & trial_log["responding_agent_role"].isin(roles)
        & trial_log["extracted_self_reported_confidence_integer"].notna()
    )
    if rounds is not None and "debate_round_index" in trial_log.columns:
        mask &= trial_log["debate_round_index"].isin(rounds)
    df = trial_log[mask].copy()

    n = len(df)
    if n == 0:
        return {
            "n_responses": 0,
            "gate_decision": "insufficient_data",
            "loud_when_wrong": None,
            "loud_when_correct": None,
            "discriminative_gap": None,
            "auroc_confidence_vs_correct": None,
        }

    conf = df["extracted_self_reported_confidence_integer"].astype(float).values
    correct = df["extracted_answer_matches_ground_truth"].astype(bool).values

    loud = conf >= high_confidence_threshold
    wrong = ~correct
    loud_when_wrong = float(loud[wrong].mean()) if wrong.any() else 0.0
    loud_when_correct = float(loud[correct].mean()) if correct.any() else 0.0
    gap = loud_when_wrong - loud_when_correct

    auroc = _auroc(conf, correct.astype(int))

    passed = (gap > discriminative_gap_threshold) and (
        np.isnan(auroc) or auroc > auroc_floor
    )
    decision = "passed" if passed else "failed"

    return {
        "n_responses": int(n),
        "conditions": conditions,
        "roles": roles,
        "rounds": rounds,
        "high_confidence_threshold": high_confidence_threshold,
        "loud_when_wrong": round(loud_when_wrong, 4),
        "loud_when_correct": round(loud_when_correct, 4),
        "discriminative_gap": round(gap, 4),
        "discriminative_gap_threshold": discriminative_gap_threshold,
        "auroc_confidence_vs_correct": None if np.isnan(auroc) else round(auroc, 4),
        "auroc_floor": auroc_floor,
        "gate_decision": decision,
    }


def run_corrected_gate(project_root: Path, trial_log: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Run and persist the corrected gate. Reports both the original and E3 substrates."""
    with open(project_root / "config" / "experiment.yaml") as f:
        exp_config = yaml.safe_load(f)
    with open(project_root / "config" / "paths.yaml") as f:
        paths = yaml.safe_load(f)

    if trial_log is None:
        tl_path = paths["trial_log_file"]
        if not Path(tl_path).is_absolute():
            tl_path = project_root / tl_path
        trial_log = pd.read_parquet(str(tl_path))

    gate_cfg = exp_config["calibration_gate"]
    corrected_cfg = gate_cfg.get("corrected_gate", {})
    thr = gate_cfg.get("high_confidence_threshold", 60)
    gap_thr = corrected_cfg.get("discriminative_gap_threshold", 0.10)
    auroc_floor = corrected_cfg.get("also_require_auroc_above", 0.60)

    reports = []

    # (a) Original Phase-1 substrate: dumb R1 confidence in C3/C4
    reports.append({
        "substrate": "phase1_C3C4_dumb_round1",
        **compute_corrected_gate(
            trial_log, gate_cfg["condition_subset_to_analyse"], ["dumb"],
            thr, gap_thr, auroc_floor, rounds=[1],
        ),
    })

    # (b) E3 substrate: anchored-with-confidence peers (C5R) — Round 0 now parses
    reports.append({
        "substrate": "E3_C5R_anchored_round0",
        **compute_corrected_gate(
            trial_log, ["C5R_anchored_with_confidence_filter"], ["dumb"],
            thr, gap_thr, auroc_floor, rounds=[0],
        ),
    })

    # (c) E3 substrate: honest peers (C5H) — real confidence variation
    reports.append({
        "substrate": "E3_C5H_honest_round0",
        **compute_corrected_gate(
            trial_log, ["C5H_honest_with_confidence_filter"], ["dumb"],
            thr, gap_thr, auroc_floor, rounds=[0],
        ),
    })

    report_df = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in reports])
    out_path = paths["corrected_calibration_gate_report_file"]
    if not Path(out_path).is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_parquet(str(out_path), index=False)

    for r in reports:
        logger.info(
            f"Corrected gate [{r['substrate']}]: gap={r.get('discriminative_gap')}, "
            f"AUROC={r.get('auroc_confidence_vs_correct')}, decision={r.get('gate_decision')} "
            f"(n={r.get('n_responses')})"
        )
    return {"reports": reports, "report_path": str(out_path)}

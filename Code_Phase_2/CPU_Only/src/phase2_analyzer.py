"""
phase2_analyzer.py — Analysis for the reviewer-driven experiments.

Produces the numbers the resubmission needs:

  E4  Capability sweep: per focal model, solo (C1) accuracy vs C4 outcome, and
      the across-model Spearman trend (solo accuracy -> debate benefit / harm).
  E1  Solo re-answer contrast: C1R vs C4 (is the gain peer-driven or second-chance?).
  E2  Honest-vs-anchored contrast: C4H vs C4, C3H vs C3.

Reads final_answers.parquet (one focal row per trial) and writes
capability_sweep_summary.parquet + capability_sweep_analysis.json.

# Implements the analysis plan from Review_Fix.md sections E1, E2, E4, W5.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .statistical_analyzer import mcnemar_paired_test

logger = logging.getLogger("platos_ship.phase2_analyzer")


def _acc(fa: pd.DataFrame, cond: str, focal: str, round_col: str = "round_one_answer_was_correct") -> Optional[float]:
    sub = fa[(fa["condition_identifier"] == cond) & (fa["focal_smart_agent_name"] == focal)]
    if sub.empty:
        return None
    return float(sub[round_col].mean())


def _flip_rate(fa: pd.DataFrame, cond: str, focal: str, col: str) -> Optional[float]:
    sub = fa[(fa["condition_identifier"] == cond) & (fa["focal_smart_agent_name"] == focal)]
    if sub.empty:
        return None
    return float(sub[col].mean())


def _spearman(x: List[float], y: List[float]) -> Tuple[Optional[float], int]:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    xs = pd.Series([p[0] for p in pairs]).rank().values
    ys = pd.Series([p[1] for p in pairs]).rank().values
    if np.std(xs) == 0 or np.std(ys) == 0:
        return None, len(pairs)
    rho = float(np.corrcoef(xs, ys)[0, 1])
    return rho, len(pairs)


def run_phase2_analysis(project_root: Path) -> Dict[str, Any]:
    with open(project_root / "config" / "paths.yaml") as f:
        paths = yaml.safe_load(f)

    fa_path = paths["final_answers_file"]
    if not Path(fa_path).is_absolute():
        fa_path = project_root / fa_path
    if not Path(fa_path).exists():
        logger.warning("final_answers.parquet not found; skipping Phase 2 analysis.")
        return {}
    fa = pd.read_parquet(str(fa_path))

    focals = sorted(fa["focal_smart_agent_name"].dropna().unique().tolist())

    C1, C2, C4 = "C1_smart_solo", "C2_three_smart", "C4_one_smart_two_dumb"
    C1R = "C1R_solo_reanswer"
    C3, C3H = "C3_two_smart_one_dumb", "C3H_two_smart_one_honest"
    C4H = "C4H_one_smart_two_honest"

    # ── E4: capability sweep table ──
    sweep_rows = []
    for focal in focals:
        solo = _acc(fa, C1, focal)
        c4 = _acc(fa, C4, focal)
        c2 = _acc(fa, C2, focal)
        row = {
            "focal_smart_agent_name": focal,
            "solo_accuracy_C1": solo,
            "homogeneous_accuracy_C2": c2,
            "two_wrong_peers_accuracy_C4": c4,
            "C4_minus_C1_accuracy_change": (None if (solo is None or c4 is None) else round(c4 - solo, 4)),
            "C4_correct_to_incorrect_flip_rate": _flip_rate(fa, C4, focal, "focal_agent_flipped_correct_to_incorrect"),
            "C4_incorrect_to_correct_flip_rate": _flip_rate(fa, C4, focal, "focal_agent_flipped_incorrect_to_correct"),
        }
        sweep_rows.append(row)
    sweep_df = pd.DataFrame(sweep_rows)

    solo_vals = [r["solo_accuracy_C1"] for r in sweep_rows]
    delta_vals = [r["C4_minus_C1_accuracy_change"] for r in sweep_rows]
    c2i_vals = [r["C4_correct_to_incorrect_flip_rate"] for r in sweep_rows]

    rho_delta, n_delta = _spearman(solo_vals, delta_vals)
    rho_c2i, n_c2i = _spearman(solo_vals, c2i_vals)

    # ── E1: solo re-answer contrast (per focal) ──
    e1 = {}
    for focal in focals:
        c1r = _acc(fa, C1R, focal)
        c4 = _acc(fa, C4, focal)
        c1 = _acc(fa, C1, focal)
        if c1r is not None:
            e1[focal] = {
                "C1_solo": c1,
                "C1R_solo_reanswer": c1r,
                "C4_two_wrong_peers": c4,
                "C4_minus_C1R": (None if (c4 is None) else round(c4 - c1r, 4)),
                "interpretation_hint": (
                    "peer_driven_if_C4_gt_C1R" if (c4 is not None and c1r is not None and c4 - c1r > 0.02)
                    else "self_reconsideration_if_C4_approx_C1R"
                ),
            }

    # ── E2: honest vs anchored contrast (per focal) ──
    e2 = {}
    for focal in focals:
        entry = {}
        for anchored, honest, label in [(C4, C4H, "C4_vs_C4H"), (C3, C3H, "C3_vs_C3H")]:
            a = _acc(fa, anchored, focal)
            h = _acc(fa, honest, focal)
            if a is not None and h is not None:
                entry[label] = {
                    "anchored_accuracy": a,
                    "honest_accuracy": h,
                    "honest_minus_anchored": round(h - a, 4),
                }
        if entry:
            e2[focal] = entry

    analysis = {
        "focal_models_analysed": focals,
        "E4_capability_sweep": {
            "spearman_solo_accuracy_vs_C4_accuracy_change": rho_delta,
            "n_models_in_solo_vs_delta": n_delta,
            "spearman_solo_accuracy_vs_C4_correct_to_incorrect": rho_c2i,
            "n_models_in_solo_vs_flip": n_c2i,
            "note": "Small-n trend (<=8 models). Report with CIs, not as a law (Review_Fix W8).",
        },
        "E1_solo_reanswer_contrast": e1,
        "E2_honest_vs_anchored_contrast": e2,
    }

    # Persist
    sum_path = paths["capability_sweep_summary_file"]
    if not Path(sum_path).is_absolute():
        sum_path = project_root / sum_path
    Path(sum_path).parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_parquet(str(sum_path), index=False)

    ana_path = paths["capability_sweep_analysis_file"]
    if not Path(ana_path).is_absolute():
        ana_path = project_root / ana_path
    with open(str(ana_path), "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    logger.info(f"Phase 2 analysis written: {sum_path.name}, {Path(ana_path).name}")
    logger.info(
        f"E4 sweep: Spearman(solo, C4-C1)={rho_delta} (n={n_delta}); "
        f"Spearman(solo, C->I)={rho_c2i} (n={n_c2i})"
    )
    return analysis


def run_phase2_statistics(project_root: Path) -> pd.DataFrame:
    """
    Paired McNemar tests for the new contrasts, using the SAME machinery and
    Bonferroni family logic as Phase 1 (statistical_analyzer.mcnemar_paired_test)
    so the numbers merge into the same paper's statistical_tests table.
    """
    with open(project_root / "config" / "paths.yaml") as f:
        paths = yaml.safe_load(f)
    with open(project_root / "config" / "experiment.yaml") as f:
        exp = yaml.safe_load(f)

    fa_path = paths["final_answers_file"]
    if not Path(fa_path).is_absolute():
        fa_path = project_root / fa_path
    if not Path(fa_path).exists():
        logger.warning("final_answers.parquet not found; skipping Phase 2 statistics.")
        return pd.DataFrame()
    fa = pd.read_parquet(str(fa_path))

    focals = sorted(fa["focal_smart_agent_name"].dropna().unique().tolist())
    present = set(zip(fa["condition_identifier"], fa["focal_smart_agent_name"]))

    # (contrast) pairs, evaluated only where both conditions exist for a focal
    candidate_pairs = [
        ("C1_smart_solo", "C1R_solo_reanswer"),                 # E1: does re-answering alone help?
        ("C1R_solo_reanswer", "C4_one_smart_two_dumb"),         # E1: does peer exposure add beyond re-answering?
        ("C4H_one_smart_two_honest", "C4_one_smart_two_dumb"),  # E2: does anchoring matter?
        ("C3H_two_smart_one_honest", "C3_two_smart_one_dumb"),  # E2
        ("C1_smart_solo", "C4H_one_smart_two_honest"),          # E2: honest peers help/hurt vs solo?
        ("C1_smart_solo", "C4_one_smart_two_dumb"),             # sweep symmetry with Phase 1 headline
        ("C4_one_smart_two_dumb", "C5R_anchored_with_confidence_filter"),  # E3
        ("C4_one_smart_two_dumb", "C4split_one_wrong_one_correct"),        # E7
        ("C2_three_smart", "C2het_three_distinct_smart"),       # E8
    ]

    rows = []
    for focal in focals:
        active = [(a, b) for (a, b) in candidate_pairs
                  if (a, focal) in present and (b, focal) in present]
        n_comp = max(len(active), 1)
        corrected_alpha = 0.05 / n_comp
        for cond_a, cond_b in active:
            res = mcnemar_paired_test(fa, cond_a, cond_b, focal)
            res["focal_smart_agent_name"] = focal
            res["bonferroni_family_size"] = n_comp
            res["bonferroni_corrected_p_value"] = round(min(res["raw_p_value"] * n_comp, 1.0), 6)
            res["is_significant_at_corrected_alpha"] = res["raw_p_value"] < corrected_alpha
            rows.append(res)

    df = pd.DataFrame(rows)
    out = paths["phase2_statistical_tests_file"]
    if not Path(out).is_absolute():
        out = project_root / out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out), index=False)
    logger.info(f"Phase 2 statistics: {len(df)} paired McNemar tests -> {Path(out).name}")
    return df

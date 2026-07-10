"""
metrics_calculator.py — Computes all metrics from §11.

Accuracy rates, flip rates, Asch conformity index, bandwagon dose-response,
bootstrap CIs, and C5-specific mitigation metrics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("platos_ship.metrics")


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 10000, seed: int = 42) -> tuple:
    """Compute 95% bootstrap CI using percentile method."""
    if len(values) < 2:
        return (float(np.mean(values)), float(np.mean(values)))
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    return (
        round(float(np.percentile(boot_means, 2.5)), 6),
        round(float(np.percentile(boot_means, 97.5)), 6),
    )


def compute_metrics_summary(
    final_answers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-(condition, focal_agent) metrics.
    Returns metrics_summary DataFrame.
    """
    rows = []

    for (cond_id, focal_name), group in final_answers.groupby(
        ["condition_identifier", "focal_smart_agent_name"]
    ):
        n = len(group)
        agg_rule = group["aggregation_rule_applied"].iloc[0] if n > 0 else "unknown"

        r0_acc = group["round_zero_answer_was_correct"].mean()
        r1_acc = group["round_one_answer_was_correct"].mean()
        delta = r1_acc - r0_acc

        # Flip rates
        r0_correct = group[group["round_zero_answer_was_correct"] == True]
        r0_incorrect = group[group["round_zero_answer_was_correct"] == False]

        flip_c2i = 0.0
        if len(r0_correct) > 0:
            flip_c2i = r0_correct["focal_agent_flipped_correct_to_incorrect"].mean()

        flip_i2c = 0.0
        if len(r0_incorrect) > 0:
            flip_i2c = r0_incorrect["focal_agent_flipped_incorrect_to_correct"].mean()

        # Asch conformity index (only for C3, C4, C5)
        asch = None
        if cond_id in ["C3_two_smart_one_dumb", "C4_one_smart_two_dumb",
                        "C5_one_smart_two_dumb_confidence_weighted"]:
            unanimous_wrong = group[group["dumb_peer_consensus_status"] == "unanimous_wrong"]
            split_peers = group[group["dumb_peer_consensus_status"] == "split"]

            flip_unanimous = 0.0
            if len(unanimous_wrong) > 0:
                uw_correct = unanimous_wrong[unanimous_wrong["round_zero_answer_was_correct"] == True]
                if len(uw_correct) > 0:
                    flip_unanimous = uw_correct["focal_agent_flipped_correct_to_incorrect"].mean()

            flip_split = 0.0
            if len(split_peers) > 0:
                sp_correct = split_peers[split_peers["round_zero_answer_was_correct"] == True]
                if len(sp_correct) > 0:
                    flip_split = sp_correct["focal_agent_flipped_correct_to_incorrect"].mean()

            asch = flip_unanimous - flip_split

        # Bootstrap CI for flip rate
        flip_values = group["focal_agent_flipped_correct_to_incorrect"].values.astype(float)
        ci_lower, ci_upper = bootstrap_ci(flip_values)

        # Mean confidence
        r0_conf = group["round_zero_focal_self_reported_confidence_integer"].dropna()
        r1_conf = group["round_one_focal_self_reported_confidence_integer"].dropna()

        rows.append({
            "condition_identifier": cond_id,
            "focal_smart_agent_name": focal_name,
            "aggregation_rule_applied": agg_rule,
            "total_trial_count": n,
            "round_zero_accuracy_rate": round(r0_acc, 6),
            "round_one_accuracy_rate": round(r1_acc, 6),
            "round_one_minus_round_zero_accuracy_delta": round(delta, 6),
            "flip_rate_correct_to_incorrect": round(flip_c2i, 6),
            "flip_rate_incorrect_to_correct": round(flip_i2c, 6),
            "asch_conformity_index": round(asch, 6) if asch is not None else None,
            "bandwagon_dose_response_indicator": round(flip_c2i, 6),
            "bootstrap_confidence_interval_lower_95_percent": ci_lower,
            "bootstrap_confidence_interval_upper_95_percent": ci_upper,
            "mean_round_zero_focal_self_reported_confidence": round(r0_conf.mean(), 2) if len(r0_conf) > 0 else None,
            "mean_round_one_focal_self_reported_confidence": round(r1_conf.mean(), 2) if len(r1_conf) > 0 else None,
        })

    return pd.DataFrame(rows)


def compute_mitigation_summary(
    final_answers: pd.DataFrame,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute C5-specific mitigation metrics comparing C4 vs C5."""

    # Get the mitigation subset questions (those that appear in C5)
    c5_data = final_answers[final_answers["condition_identifier"] == "C5_one_smart_two_dumb_confidence_weighted"]

    if c5_data.empty:
        logger.warning("No C5 data found for mitigation summary")
        return pd.DataFrame()

    c5_questions = set(c5_data["question_identifier"].unique())

    # C4 restricted to same questions
    c4_data = final_answers[
        (final_answers["condition_identifier"] == "C4_one_smart_two_dumb") &
        (final_answers["question_identifier"].isin(c5_questions))
    ]

    c4_acc = c4_data["round_one_answer_was_correct"].mean() if len(c4_data) > 0 else 0
    c5_acc = c5_data["round_one_answer_was_correct"].mean()
    delta = c5_acc - c4_acc

    # Bootstrap CI for delta
    rng = np.random.RandomState(seed)
    boot_deltas = []

    c4_vals = c4_data["round_one_answer_was_correct"].values.astype(float)
    c5_vals = c5_data["round_one_answer_was_correct"].values.astype(float)

    for _ in range(n_bootstrap):
        if len(c4_vals) > 0 and len(c5_vals) > 0:
            c4_sample = rng.choice(c4_vals, size=len(c4_vals), replace=True)
            c5_sample = rng.choice(c5_vals, size=len(c5_vals), replace=True)
            boot_deltas.append(c5_sample.mean() - c4_sample.mean())

    if boot_deltas:
        ci_lower = float(np.percentile(boot_deltas, 2.5))
        ci_upper = float(np.percentile(boot_deltas, 97.5))
    else:
        ci_lower, ci_upper = 0.0, 0.0

    # McNemar p-value (per-trial comparison)
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        # Merge C4 and C5 by question for paired comparison
        c4_by_q = c4_data.groupby("question_identifier")["round_one_answer_was_correct"].mean()
        c5_by_q = c5_data.groupby("question_identifier")["round_one_answer_was_correct"].mean()
        common_qs = list(set(c4_by_q.index) & set(c5_by_q.index))
        if len(common_qs) >= 2:
            c4_correct = (c4_by_q[common_qs] >= 0.5).astype(int).values
            c5_correct = (c5_by_q[common_qs] >= 0.5).astype(int).values

            # Contingency table
            a = ((c4_correct == 1) & (c5_correct == 1)).sum()
            b = ((c4_correct == 1) & (c5_correct == 0)).sum()
            c = ((c4_correct == 0) & (c5_correct == 1)).sum()
            d = ((c4_correct == 0) & (c5_correct == 0)).sum()
            table = [[a, b], [c, d]]
            result = mcnemar(table, exact=True)
            mcnemar_p = result.pvalue
        else:
            mcnemar_p = 1.0
    except Exception as e:
        logger.warning(f"McNemar test failed: {e}")
        mcnemar_p = 1.0

    # C5 filtering stats
    mean_filtered = c5_data["c5_count_of_peer_messages_filtered_out"].dropna().mean() if "c5_count_of_peer_messages_filtered_out" in c5_data.columns else 0
    prop_zero_peers = 0.0
    if "c5_count_of_peer_messages_filtered_out" in c5_data.columns:
        # All peers filtered = dumb_count peers filtered
        total_dumb = c5_data["condition_dumb_agent_count"].iloc[0] if len(c5_data) > 0 else 2
        prop_zero_peers = (c5_data["c5_count_of_peer_messages_filtered_out"] >= total_dumb).mean()

    # C5 flip rate
    c5_r0_correct = c5_data[c5_data["round_zero_answer_was_correct"] == True]
    c5_flip = c5_r0_correct["focal_agent_flipped_correct_to_incorrect"].mean() if len(c5_r0_correct) > 0 else 0

    c4_r0_correct = c4_data[c4_data["round_zero_answer_was_correct"] == True]
    c4_flip = c4_r0_correct["focal_agent_flipped_correct_to_incorrect"].mean() if len(c4_r0_correct) > 0 else 0

    row = {
        "comparison_label": "C4_versus_C5_on_mitigation_subset",
        "c4_round_one_accuracy_on_mitigation_subset": round(c4_acc, 6),
        "c5_round_one_accuracy": round(c5_acc, 6),
        "c5_minus_c4_accuracy_delta_percentage_points": round(delta * 100, 4),
        "c5_minus_c4_bootstrap_confidence_interval_lower_95_percent": round(ci_lower, 6),
        "c5_minus_c4_bootstrap_confidence_interval_upper_95_percent": round(ci_upper, 6),
        "c5_minus_c4_mcnemar_p_value": round(mcnemar_p, 6),
        "c5_minus_c4_mcnemar_p_value_bonferroni_corrected": round(min(mcnemar_p * 7, 1.0), 6),
        "mean_count_of_peer_messages_filtered_out_per_c5_trial": round(float(mean_filtered), 4),
        "proportion_of_c5_trials_with_zero_peer_messages_after_filtering": round(prop_zero_peers, 4),
        "c5_focal_agent_flip_rate_correct_to_incorrect": round(c5_flip, 6),
        "c4_focal_agent_flip_rate_correct_to_incorrect_on_mitigation_subset": round(c4_flip, 6),
    }

    return pd.DataFrame([row])


def compute_and_save_metrics(project_root: Path) -> Dict[str, pd.DataFrame]:
    """Load final answers and compute all metrics."""
    with open(project_root / "config" / "paths.yaml") as f:
        paths = yaml.safe_load(f)

    fa_path = paths["final_answers_file"]
    if not Path(fa_path).is_absolute():
        fa_path = project_root / fa_path
    final_answers = pd.read_parquet(str(fa_path))

    # Main metrics
    metrics_df = compute_metrics_summary(final_answers)
    ms_path = paths["metrics_summary_file"]
    if not Path(ms_path).is_absolute():
        ms_path = project_root / ms_path
    Path(ms_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_parquet(str(ms_path), index=False)
    logger.info(f"Metrics summary saved: {len(metrics_df)} rows to {ms_path}")

    # Mitigation metrics (if C5 data exists)
    mit_df = compute_mitigation_summary(final_answers)
    if not mit_df.empty:
        mit_path = paths["mitigation_summary_file"]
        if not Path(mit_path).is_absolute():
            mit_path = project_root / mit_path
        mit_df.to_parquet(str(mit_path), index=False)
        logger.info(f"Mitigation summary saved to {mit_path}")

    return {"metrics_summary": metrics_df, "mitigation_summary": mit_df}

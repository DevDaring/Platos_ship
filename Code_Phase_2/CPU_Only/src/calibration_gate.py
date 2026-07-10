"""
calibration_gate.py — Stage 2: Decides whether C5 mitigation runs.

Computes P(confidence >= 60 | wrong, dumb, C3/C4) and checks against threshold.
Gate passes if metric >= 0.40.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("platos_ship.calibration_gate")


def compute_calibration_metric(
    trial_log: pd.DataFrame,
    high_confidence_threshold: int = 60,
    conditions: list = None,
) -> Dict[str, Any]:
    """
    Compute P(confidence >= threshold | wrong, dumb agent, specified conditions).
    
    Returns dict with metric values and breakdowns.
    """
    if conditions is None:
        conditions = ["C3_two_smart_one_dumb", "C4_one_smart_two_dumb"]
    
    # Filter to dumb agent responses in C3/C4
    mask = (
        (trial_log["condition_identifier"].isin(conditions)) &
        (trial_log["responding_agent_role"] == "dumb") &
        (trial_log["extracted_self_reported_confidence_integer"].notna())
    )
    dumb_responses = trial_log[mask].copy()
    
    if len(dumb_responses) == 0:
        logger.warning("No dumb agent responses with parsed confidence found")
        return {
            "precondition_metric_value": 0.0,
            "partner_metric_value": 0.0,
            "total_responses_used": 0,
            "breakdown_by_model": {},
        }
    
    # Wrong answers
    wrong_mask = dumb_responses["extracted_answer_matches_ground_truth"] == False
    wrong_responses = dumb_responses[wrong_mask]
    
    # Correct answers
    correct_mask = dumb_responses["extracted_answer_matches_ground_truth"] == True
    correct_responses = dumb_responses[correct_mask]
    
    # P(high confidence | wrong)
    if len(wrong_responses) > 0:
        high_conf_wrong = (wrong_responses["extracted_self_reported_confidence_integer"] >= high_confidence_threshold).sum()
        p_high_conf_given_wrong = high_conf_wrong / len(wrong_responses)
    else:
        p_high_conf_given_wrong = 0.0
    
    # P(high confidence | correct) — partner metric
    if len(correct_responses) > 0:
        high_conf_correct = (correct_responses["extracted_self_reported_confidence_integer"] >= high_confidence_threshold).sum()
        p_high_conf_given_correct = high_conf_correct / len(correct_responses)
    else:
        p_high_conf_given_correct = 0.0
    
    # Breakdown by model
    breakdown = {}
    for model_name, group in dumb_responses.groupby("responding_agent_model_name"):
        wrong_g = group[group["extracted_answer_matches_ground_truth"] == False]
        if len(wrong_g) > 0:
            p = (wrong_g["extracted_self_reported_confidence_integer"] >= high_confidence_threshold).sum() / len(wrong_g)
        else:
            p = 0.0
        breakdown[model_name] = {
            "p_high_conf_given_wrong": round(p, 4),
            "total_wrong_responses": len(wrong_g),
            "total_responses": len(group),
        }
    
    return {
        "precondition_metric_value": round(p_high_conf_given_wrong, 4),
        "partner_metric_value": round(p_high_conf_given_correct, 4),
        "total_responses_used": len(dumb_responses),
        "total_wrong_responses": len(wrong_responses),
        "total_correct_responses": len(correct_responses),
        "breakdown_by_model": breakdown,
    }


def bootstrap_confidence_interval(
    trial_log: pd.DataFrame,
    high_confidence_threshold: int = 60,
    conditions: list = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for the precondition metric."""
    if conditions is None:
        conditions = ["C3_two_smart_one_dumb", "C4_one_smart_two_dumb"]
    
    mask = (
        (trial_log["condition_identifier"].isin(conditions)) &
        (trial_log["responding_agent_role"] == "dumb") &
        (trial_log["extracted_self_reported_confidence_integer"].notna()) &
        (trial_log["extracted_answer_matches_ground_truth"] == False)
    )
    wrong_dumb = trial_log[mask].copy()
    
    if len(wrong_dumb) < 2:
        return 0.0, 1.0
    
    rng = np.random.RandomState(seed)
    confidences = wrong_dumb["extracted_self_reported_confidence_integer"].values
    
    boot_metrics = []
    for _ in range(n_bootstrap):
        sample = rng.choice(confidences, size=len(confidences), replace=True)
        p = (sample >= high_confidence_threshold).sum() / len(sample)
        boot_metrics.append(p)
    
    lower = np.percentile(boot_metrics, 2.5)
    upper = np.percentile(boot_metrics, 97.5)
    return round(lower, 4), round(upper, 4)


def run_calibration_gate(
    project_root: Path,
    trial_log: pd.DataFrame = None,
) -> Dict[str, Any]:
    """
    Run the calibration gate analysis (Stage 2).
    
    Returns the gate decision and writes calibration_gate_report.parquet.
    """
    with open(project_root / "config" / "experiment.yaml") as f:
        exp_config = yaml.safe_load(f)
    with open(project_root / "config" / "paths.yaml") as f:
        paths = yaml.safe_load(f)
    
    gate_config = exp_config["calibration_gate"]
    conditions = gate_config["condition_subset_to_analyse"]
    threshold = gate_config["high_confidence_threshold"]
    metric_threshold = gate_config["precondition_metric_threshold"]
    
    # Load trial log if not provided
    if trial_log is None:
        tl_path = paths["trial_log_file"]
        if not Path(tl_path).is_absolute():
            tl_path = project_root / tl_path
        trial_log = pd.read_parquet(str(tl_path))
    
    # Compute metric
    metrics = compute_calibration_metric(trial_log, threshold, conditions)
    
    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_confidence_interval(
        trial_log, threshold, conditions
    )
    
    # Gate decision
    metric_value = metrics["precondition_metric_value"]
    gate_passed = metric_value >= metric_threshold
    gate_decision = "passed" if gate_passed else "failed"
    
    logger.info(f"Calibration gate metric: {metric_value:.4f} (threshold: {metric_threshold})")
    logger.info(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    logger.info(f"Gate decision: {gate_decision}")
    
    # Build report DataFrame
    report_data = {
        "precondition_metric_name": "probability_high_confidence_given_wrong_answer_for_dumb_agents",
        "precondition_metric_value": metric_value,
        "precondition_metric_bootstrap_confidence_interval_lower_95_percent": ci_lower,
        "precondition_metric_bootstrap_confidence_interval_upper_95_percent": ci_upper,
        "precondition_metric_threshold_for_pass": metric_threshold,
        "gate_decision": gate_decision,
        "partner_metric_probability_high_confidence_given_correct_answer": metrics["partner_metric_value"],
        "loud_wrong_minus_loud_right_difference": round(
            metric_value - metrics["partner_metric_value"], 4
        ),
        "total_dumb_agent_responses_with_parsed_confidence_used_in_analysis": metrics["total_responses_used"],
        "breakdown_by_dumb_model_name": json.dumps(metrics["breakdown_by_model"]),
    }
    
    report_df = pd.DataFrame([report_data])
    
    # Save
    report_path = paths["calibration_gate_report_file"]
    if not Path(report_path).is_absolute():
        report_path = project_root / report_path
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    report_df.to_parquet(str(report_path), index=False)
    
    # Log decision
    if gate_passed:
        logger.info("CALIBRATION_GATE_PASSED — proceeding to C5")
    else:
        logger.info("CALIBRATION_GATE_FAILED — skipping C5; mitigation hypothesis not supported")
    
    return {
        "gate_decision": gate_decision,
        "metric_value": metric_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "report": report_data,
    }

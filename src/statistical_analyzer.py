"""
statistical_analyzer.py — McNemar's tests, Bonferroni correction,
and logistic regression for bandwagon dose-response.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("platos_ship.statistical_analyzer")


def mcnemar_paired_test(
    fa: pd.DataFrame,
    cond_a: str,
    cond_b: str,
    focal_agent: str,
    metric_col: str = "round_one_answer_was_correct",
) -> Dict[str, Any]:
    """Run McNemar's test comparing two conditions on shared questions."""
    from statsmodels.stats.contingency_tables import mcnemar

    a_data = fa[(fa["condition_identifier"] == cond_a) & (fa["focal_smart_agent_name"] == focal_agent)]
    b_data = fa[(fa["condition_identifier"] == cond_b) & (fa["focal_smart_agent_name"] == focal_agent)]

    # Aggregate per question (majority vote across trials)
    a_by_q = a_data.groupby("question_identifier")[metric_col].mean() >= 0.5
    b_by_q = b_data.groupby("question_identifier")[metric_col].mean() >= 0.5

    common = sorted(set(a_by_q.index) & set(b_by_q.index))

    if len(common) < 5:
        return {
            "comparison_label": f"{cond_a}_vs_{cond_b}",
            "test_name": "mcnemar",
            "test_statistic": None,
            "raw_p_value": 1.0,
            "effect_size_estimate": 0.0,
            "notes": f"Too few shared questions ({len(common)})",
        }

    a_vals = a_by_q[common].astype(int).values
    b_vals = b_by_q[common].astype(int).values

    # 2x2 contingency table
    a11 = ((a_vals == 1) & (b_vals == 1)).sum()  # both correct
    a12 = ((a_vals == 1) & (b_vals == 0)).sum()  # a correct, b wrong
    a21 = ((a_vals == 0) & (b_vals == 1)).sum()  # a wrong, b correct
    a22 = ((a_vals == 0) & (b_vals == 0)).sum()  # both wrong

    table = [[a11, a12], [a21, a22]]

    try:
        result = mcnemar(table, exact=True)
        statistic = result.statistic
        p_value = result.pvalue
    except Exception as e:
        logger.warning(f"McNemar failed for {cond_a} vs {cond_b}: {e}")
        statistic = None
        p_value = 1.0

    # Effect size: difference in proportions
    acc_a = a_vals.mean()
    acc_b = b_vals.mean()
    effect = acc_a - acc_b

    return {
        "comparison_label": f"{cond_a}_vs_{cond_b}",
        "test_name": "mcnemar_exact",
        "test_statistic": float(statistic) if statistic is not None else None,
        "raw_p_value": float(p_value),
        "effect_size_estimate": round(float(effect), 6),
        "notes": f"n_questions={len(common)}, contingency=[[{a11},{a12}],[{a21},{a22}]]",
    }


def bandwagon_dose_response(
    fa: pd.DataFrame,
    focal_agent: str,
) -> Dict[str, Any]:
    """
    Logistic regression of round_one_answer_was_correct against
    condition_dumb_agent_count (C2, C3, C4 only).
    """
    try:
        import statsmodels.api as sm

        subset = fa[
            (fa["condition_identifier"].isin([
                "C2_three_smart", "C3_two_smart_one_dumb", "C4_one_smart_two_dumb"
            ])) &
            (fa["focal_smart_agent_name"] == focal_agent)
        ].copy()

        if len(subset) < 10:
            return {
                "test_name": "logistic_regression_dose_response",
                "notes": f"Insufficient data ({len(subset)} rows)",
            }

        y = subset["round_one_answer_was_correct"].astype(int).values
        X = sm.add_constant(subset["condition_dumb_agent_count"].values)

        model = sm.Logit(y, X)
        result = model.fit(disp=0)

        coef = result.params[1]
        p_val = result.pvalues[1]

        return {
            "comparison_label": "bandwagon_dose_response",
            "test_name": "logistic_regression",
            "test_statistic": round(float(coef), 6),
            "raw_p_value": round(float(p_val), 6),
            "effect_size_estimate": round(float(coef), 6),
            "notes": f"Coefficient on dumb_agent_count, n={len(subset)}, AIC={result.aic:.1f}",
        }
    except Exception as e:
        logger.warning(f"Dose-response regression failed: {e}")
        return {
            "comparison_label": "bandwagon_dose_response",
            "test_name": "logistic_regression",
            "test_statistic": None,
            "raw_p_value": 1.0,
            "effect_size_estimate": 0.0,
            "notes": f"Failed: {e}",
        }


def run_statistical_analysis(
    project_root: Path,
    final_answers: pd.DataFrame = None,
    include_c5: bool = False,
) -> pd.DataFrame:
    """Run all statistical tests and save results."""
    with open(project_root / "config" / "paths.yaml") as f:
        paths = yaml.safe_load(f)
    with open(project_root / "config" / "experiment.yaml") as f:
        exp_config = yaml.safe_load(f)

    if final_answers is None:
        fa_path = paths["final_answers_file"]
        if not Path(fa_path).is_absolute():
            fa_path = project_root / fa_path
        final_answers = pd.read_parquet(str(fa_path))

    focal_agent = exp_config["focal_smart_agent_assignment"]

    # Define comparison pairs
    pairs = [
        ("C1_smart_solo", "C2_three_smart"),
        ("C1_smart_solo", "C3_two_smart_one_dumb"),
        ("C1_smart_solo", "C4_one_smart_two_dumb"),
        ("C2_three_smart", "C3_two_smart_one_dumb"),
        ("C2_three_smart", "C4_one_smart_two_dumb"),
        ("C3_two_smart_one_dumb", "C4_one_smart_two_dumb"),
    ]

    if include_c5:
        pairs.append(("C4_one_smart_two_dumb", "C5_one_smart_two_dumb_confidence_weighted"))

    n_comparisons = len(pairs)
    corrected_alpha = 0.05 / n_comparisons

    results = []
    for cond_a, cond_b in pairs:
        test_result = mcnemar_paired_test(final_answers, cond_a, cond_b, focal_agent)
        test_result["bonferroni_corrected_p_value"] = round(
            min(test_result["raw_p_value"] * n_comparisons, 1.0), 6
        )
        test_result["is_significant_at_corrected_alpha"] = (
            test_result["raw_p_value"] < corrected_alpha
        )
        results.append(test_result)

    # Dose-response
    dose_result = bandwagon_dose_response(final_answers, focal_agent)
    dose_result["bonferroni_corrected_p_value"] = round(
        min(dose_result.get("raw_p_value", 1.0) * n_comparisons, 1.0), 6
    )
    dose_result["is_significant_at_corrected_alpha"] = (
        dose_result.get("raw_p_value", 1.0) < corrected_alpha
    )
    results.append(dose_result)

    df = pd.DataFrame(results)

    # Save
    st_path = paths["statistical_tests_file"]
    if not Path(st_path).is_absolute():
        st_path = project_root / st_path
    Path(st_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(st_path), index=False)
    logger.info(f"Statistical tests saved: {len(df)} tests to {st_path}")

    return df

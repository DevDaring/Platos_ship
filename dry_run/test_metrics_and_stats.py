"""
test_metrics_and_stats.py — Tests metrics calculator and statistical analyzer
with synthetic trial data.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


def create_synthetic_data():
    """Create synthetic trial_log and final_answers data for testing."""
    np.random.seed(42)

    conditions = ["C1_smart_solo", "C2_three_smart", "C3_two_smart_one_dumb",
                   "C4_one_smart_two_dumb", "C5_one_smart_two_dumb_confidence_weighted"]

    # Create final_answers data
    rows = []
    for q_idx in range(20):  # 20 questions
        qid = f"test_q_{q_idx:03d}"
        for cond in conditions:
            dumb_count = {"C1_smart_solo": 0, "C2_three_smart": 0,
                          "C3_two_smart_one_dumb": 1, "C4_one_smart_two_dumb": 2,
                          "C5_one_smart_two_dumb_confidence_weighted": 2}[cond]

            agg_rule = {"C1_smart_solo": "none", "C2_three_smart": "standard_debate",
                        "C3_two_smart_one_dumb": "standard_debate",
                        "C4_one_smart_two_dumb": "standard_debate",
                        "C5_one_smart_two_dumb_confidence_weighted": "confidence_weighted"}[cond]

            for trial in range(3):
                r0_correct = np.random.random() > 0.3  # 70% R0 accuracy
                # More flips with more dumb agents
                flip_prob = 0.05 + dumb_count * 0.1
                if r0_correct:
                    flipped = np.random.random() < flip_prob
                    r1_correct = not flipped
                else:
                    r1_correct = np.random.random() < 0.2

                consensus = "not_applicable"
                if dumb_count > 0:
                    consensus = np.random.choice(["unanimous_wrong", "split", "unanimous_correct"],
                                                 p=[0.6, 0.3, 0.1])

                rows.append({
                    "question_identifier": qid,
                    "condition_identifier": cond,
                    "trial_replication_index": trial,
                    "focal_smart_agent_name": "deepseek_primary",
                    "aggregation_rule_applied": agg_rule,
                    "round_zero_independent_answer": "A" if r0_correct else "B",
                    "round_zero_answer_was_correct": r0_correct,
                    "round_zero_focal_self_reported_confidence_integer": np.random.randint(50, 100),
                    "round_one_post_debate_answer": "A" if r1_correct else "C",
                    "round_one_answer_was_correct": r1_correct,
                    "round_one_focal_self_reported_confidence_integer": np.random.randint(40, 95),
                    "focal_agent_flipped_correct_to_incorrect": r0_correct and not r1_correct,
                    "focal_agent_flipped_incorrect_to_correct": not r0_correct and r1_correct,
                    "dumb_peer_consensus_status": consensus,
                    "condition_dumb_agent_count": dumb_count,
                    "c5_count_of_peer_messages_filtered_out": (
                        np.random.randint(0, 3) if cond == "C5_one_smart_two_dumb_confidence_weighted" else None
                    ),
                })

    fa_df = pd.DataFrame(rows)

    # Create trial_log data (simplified)
    trial_rows = []
    for _, fa_row in fa_df.iterrows():
        # Focal smart agent R0
        trial_rows.append({
            "trial_universal_unique_identifier": f"uuid_{fa_row['question_identifier']}_{fa_row['trial_replication_index']}",
            "question_identifier": fa_row["question_identifier"],
            "condition_identifier": fa_row["condition_identifier"],
            "trial_replication_index": fa_row["trial_replication_index"],
            "focal_smart_agent_name": "deepseek_primary",
            "responding_agent_identifier": "Agent_Alpha",
            "responding_agent_role": "smart_focal",
            "responding_agent_model_name": "deepseek-chat",
            "responding_agent_provider": "deepseek",
            "debate_round_index": 0,
            "aggregation_rule_applied": fa_row["aggregation_rule_applied"],
            "raw_response_text": "test response",
            "extracted_final_answer": fa_row["round_zero_independent_answer"],
            "extracted_self_reported_confidence_integer": fa_row["round_zero_focal_self_reported_confidence_integer"],
            "confidence_parse_status": "success",
            "answer_extraction_method": "regex_success",
            "extracted_answer_matches_ground_truth": fa_row["round_zero_answer_was_correct"],
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "wall_clock_latency_seconds": 1.0,
            "error_status": "success",
            "retry_attempts_used": 0,
            "timestamp_utc": "2026-05-03T00:00:00",
            "random_seed_used_for_this_trial": 42,
            "peer_messages_seen_in_context": None,
            "peer_messages_filtered_out_count": None,
            "peer_messages_ordering_seed": None,
            "injected_dumb_persona_identifier": None,
        })

        # Add dumb agent responses
        dumb_count = fa_row["condition_dumb_agent_count"]
        for d in range(dumb_count):
            trial_rows.append({
                "trial_universal_unique_identifier": f"uuid_{fa_row['question_identifier']}_{fa_row['trial_replication_index']}",
                "question_identifier": fa_row["question_identifier"],
                "condition_identifier": fa_row["condition_identifier"],
                "trial_replication_index": fa_row["trial_replication_index"],
                "focal_smart_agent_name": "deepseek_primary",
                "responding_agent_identifier": f"Agent_Dumb_{d+1}",
                "responding_agent_role": "dumb",
                "responding_agent_model_name": "mock-dumb",
                "responding_agent_provider": "local_huggingface",
                "debate_round_index": 0,
                "aggregation_rule_applied": fa_row["aggregation_rule_applied"],
                "raw_response_text": "wrong answer text",
                "extracted_final_answer": "C",
                "extracted_self_reported_confidence_integer": np.random.randint(40, 100),
                "confidence_parse_status": "success",
                "answer_extraction_method": "regex_success",
                "extracted_answer_matches_ground_truth": False,
                "total_input_tokens": 50,
                "total_output_tokens": 30,
                "wall_clock_latency_seconds": 0.5,
                "error_status": "success",
                "retry_attempts_used": 0,
                "timestamp_utc": "2026-05-03T00:00:00",
                "random_seed_used_for_this_trial": 42,
                "peer_messages_seen_in_context": None,
                "peer_messages_filtered_out_count": None,
                "peer_messages_ordering_seed": None,
                "injected_dumb_persona_identifier": None,
            })

    tl_df = pd.DataFrame(trial_rows)
    return fa_df, tl_df


def test_metrics_calculator():
    """Test metrics computation with synthetic data."""
    from src.metrics_calculator import compute_metrics_summary, compute_mitigation_summary

    fa_df, _ = create_synthetic_data()

    print("  Computing metrics summary...")
    metrics = compute_metrics_summary(fa_df)
    print(f"  Metrics rows: {len(metrics)}")
    print(f"  Columns: {list(metrics.columns)}")

    required_cols = [
        "condition_identifier", "focal_smart_agent_name",
        "round_zero_accuracy_rate", "round_one_accuracy_rate",
        "flip_rate_correct_to_incorrect",
        "bootstrap_confidence_interval_lower_95_percent",
        "bootstrap_confidence_interval_upper_95_percent",
    ]
    for col in required_cols:
        assert col in metrics.columns, f"Missing column: {col}"

    assert len(metrics) >= 5, f"Expected at least 5 condition rows, got {len(metrics)}"
    print("  ✓ Metrics summary schema verified")

    # Mitigation summary
    print("\n  Computing mitigation summary...")
    mit = compute_mitigation_summary(fa_df)
    if not mit.empty:
        print(f"  Mitigation rows: {len(mit)}")
        print(f"  C5-C4 delta: {mit.iloc[0].get('c5_minus_c4_accuracy_delta_percentage_points', 'N/A')}pp")
        print("  ✓ Mitigation summary computed")
    else:
        print("  ⚠ No mitigation data (expected if C5 empty)")

    return True


def test_statistical_analyzer():
    """Test statistical tests with synthetic data."""
    from src.statistical_analyzer import mcnemar_paired_test, bandwagon_dose_response

    fa_df, _ = create_synthetic_data()

    print("  Running McNemar tests...")
    pairs = [
        ("C1_smart_solo", "C2_three_smart"),
        ("C1_smart_solo", "C4_one_smart_two_dumb"),
    ]

    for a, b in pairs:
        result = mcnemar_paired_test(fa_df, a, b, "deepseek_primary")
        print(f"    {a} vs {b}: p={result.get('raw_p_value', 'N/A'):.4f}")
    print("  ✓ McNemar tests complete")

    print("\n  Running dose-response regression...")
    dose = bandwagon_dose_response(fa_df, "deepseek_primary")
    print(f"    Coefficient: {dose.get('test_statistic', 'N/A')}")
    print(f"    p-value: {dose.get('raw_p_value', 'N/A')}")
    print("  ✓ Dose-response regression complete")

    return True


def test_calibration_gate():
    """Test calibration gate with synthetic data."""
    from src.calibration_gate import compute_calibration_metric

    _, tl_df = create_synthetic_data()

    print("  Computing calibration metric...")
    metrics = compute_calibration_metric(tl_df, high_confidence_threshold=60)
    print(f"    P(high_conf | wrong, dumb): {metrics['precondition_metric_value']:.4f}")
    print(f"    Total responses used: {metrics['total_responses_used']}")
    print(f"    Breakdown: {json.dumps(metrics.get('breakdown_by_model', {}), indent=2)}")
    print("  ✓ Calibration gate metric computed")

    return True


def run_all():
    project_root = Path(__file__).parent.parent.resolve()
    load_dotenv(project_root / ".env")
    sys.path.insert(0, str(project_root))

    print("=" * 60)
    print("METRICS AND STATISTICS TESTS")
    print("=" * 60)

    all_passed = True
    tests = [
        ("Metrics Calculator", test_metrics_calculator),
        ("Statistical Analyzer", test_statistical_analyzer),
        ("Calibration Gate", test_calibration_gate),
    ]

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_fn()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print(f"\nRESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_all())

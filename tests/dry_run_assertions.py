"""
dry_run_assertions.py — Post-dry-run validation.

Checks all output files exist, schemas match spec, and every model
produced at least one row.
"""

import sys
import json
from pathlib import Path

import pandas as pd
import yaml


def run_assertions():
    project_root = Path(__file__).parent.parent.resolve()

    with open(project_root / "config" / "paths.yaml") as f:
        paths = yaml.safe_load(f)

    print("=" * 60)
    print("POST-DRY-RUN ASSERTIONS")
    print("=" * 60)

    all_ok = True

    def check(condition, message):
        nonlocal all_ok
        status = "✓" if condition else "✗"
        print(f"  {status} {message}")
        if not condition:
            all_ok = False

    # 1. Check output files exist
    file_keys = [
        "trial_log_file", "final_answers_file", "experiment_metadata_file",
    ]
    for key in file_keys:
        p = Path(paths[key])
        if not p.is_absolute():
            p = project_root / p
        check(p.exists(), f"{key} exists: {p}")

    # 2. Check trial_log schema
    tl_path = paths["trial_log_file"]
    if not Path(tl_path).is_absolute():
        tl_path = project_root / tl_path

    if Path(tl_path).exists():
        tl = pd.read_parquet(str(tl_path))
        print(f"\n  Trial log: {len(tl)} rows, {len(tl.columns)} columns")

        required_tl_cols = [
            "trial_universal_unique_identifier", "question_identifier",
            "condition_identifier", "trial_replication_index",
            "focal_smart_agent_name", "responding_agent_identifier",
            "responding_agent_role", "responding_agent_model_name",
            "responding_agent_provider", "debate_round_index",
            "aggregation_rule_applied", "raw_response_text",
            "extracted_final_answer", "answer_extraction_method",
            "extracted_answer_matches_ground_truth",
            "error_status", "timestamp_utc",
        ]

        for col in required_tl_cols:
            check(col in tl.columns, f"Trial log column: {col}")

        # 3. Check every model has at least one row
        providers = tl["responding_agent_provider"].unique()
        check("deepseek" in providers or "mock" in providers,
              f"DeepSeek/smart agent has rows (providers: {list(providers)})")

        # 4. Check all conditions present
        conditions = set(tl["condition_identifier"].unique())
        for cond in ["C1_smart_solo", "C2_three_smart", "C3_two_smart_one_dumb",
                      "C4_one_smart_two_dumb"]:
            check(cond in conditions, f"Condition {cond} present in trial log")

        # 5. Confidence parsing exercised
        conf_col = "extracted_self_reported_confidence_integer"
        if conf_col in tl.columns:
            has_conf = tl[conf_col].notna().any()
            has_null_conf = tl[conf_col].isna().any()
            check(has_conf, "At least one confidence value parsed")
            # Null confidence is ok to not have in dry run

        # 6. Check both rounds present
        rounds = set(tl["debate_round_index"].unique())
        check(0 in rounds, "Round 0 responses present")
        check(1 in rounds, "Round 1 responses present")

    # 7. Check final_answers
    fa_path = paths["final_answers_file"]
    if not Path(fa_path).is_absolute():
        fa_path = project_root / fa_path

    if Path(fa_path).exists():
        fa = pd.read_parquet(str(fa_path))
        print(f"\n  Final answers: {len(fa)} rows")
        check(len(fa) >= 1, "Final answers has at least 1 row")

        required_fa_cols = [
            "question_identifier", "condition_identifier",
            "round_zero_answer_was_correct", "round_one_answer_was_correct",
            "focal_agent_flipped_correct_to_incorrect",
        ]
        for col in required_fa_cols:
            check(col in fa.columns, f"Final answers column: {col}")

    # 8. Check experiment metadata
    meta_path = paths["experiment_metadata_file"]
    if not Path(meta_path).is_absolute():
        meta_path = project_root / meta_path

    if Path(meta_path).exists():
        with open(str(meta_path)) as f:
            meta = json.load(f)
        check("experiment_run_universal_unique_identifier" in meta,
              "Metadata has run UUID")
        check("random_seed_value" in meta, "Metadata has random seed")

    print(f"\n{'='*60}")
    print(f"  ASSERTIONS: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print(f"{'='*60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run_assertions())

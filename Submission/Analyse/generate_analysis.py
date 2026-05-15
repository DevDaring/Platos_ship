"""
generate_analysis.py
────────────────────
Run this script once to print all numeric results from the experiment.
Output is used to populate analysis.md with real data.

Usage:
    python generate_analysis.py
"""

import json
import pathlib
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("Run:  pip install pandas pyarrow numpy")

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT  = ROOT / "results" / "data" / "outputs"
PROC = ROOT / "results" / "data" / "processed"
LOGS = ROOT / "results" / "logs"

SEP = "=" * 70

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

# ── Load all artefacts ────────────────────────────────────────────────────────
trial_log   = pd.read_parquet(OUT / "trial_log.parquet")
final_ans   = pd.read_parquet(OUT / "final_answers.parquet")
metrics     = pd.read_parquet(OUT / "metrics_summary.parquet")
stats       = pd.read_parquet(OUT / "statistical_tests.parquet")
mitigation  = pd.read_parquet(OUT / "mitigation_summary.parquet")
cal_gate    = pd.read_parquet(OUT / "calibration_gate_report.parquet")
completed   = pd.read_parquet(OUT / "completed_trials.parquet")
qpool       = pd.read_parquet(PROC / "question_pool.parquet")
personas    = pd.read_parquet(PROC / "dumb_personas.parquet")

with open(OUT / "experiment_metadata.json") as f:
    meta = json.load(f)

# ── 0. Schemas ────────────────────────────────────────────────────────────────
section("0. DATA SCHEMAS")
for name, df in [
    ("trial_log", trial_log),
    ("final_answers", final_ans),
    ("metrics_summary", metrics),
    ("statistical_tests", stats),
    ("mitigation_summary", mitigation),
    ("calibration_gate_report", cal_gate),
    ("completed_trials", completed),
    ("question_pool", qpool),
    ("dumb_personas", personas),
]:
    print(f"\n--- {name} ---")
    print(f"  Shape : {df.shape}")
    print(f"  Cols  : {list(df.columns)}")

# ── 1. Dataset composition ────────────────────────────────────────────────────
section("1. DATASET & QUESTION POOL")
print(f"Total questions in pool: {len(qpool)}")
if "source" in qpool.columns:
    print("\nBy source:")
    print(qpool["source"].value_counts().to_string())
if "difficulty" in qpool.columns:
    print("\nBy difficulty:")
    print(qpool["difficulty"].value_counts().sort_index().to_string())
if "category" in qpool.columns:
    print("\nBy MMLU-Pro category (top 15):")
    print(qpool["category"].value_counts().head(15).to_string())
if "subject" in qpool.columns:
    print("\nBy subject:")
    print(qpool["subject"].value_counts().to_string())

# ── 2. Personas ───────────────────────────────────────────────────────────────
section("2. PERSONA GENERATION")
print(f"Total personas: {len(personas)}")
if "reasoning_style" in personas.columns:
    print("\nBy reasoning style:")
    print(personas["reasoning_style"].value_counts().to_string())
if "model" in personas.columns:
    print("\nBy model:")
    print(personas["model"].value_counts().to_string())

# ── 3. Trial execution summary ────────────────────────────────────────────────
section("3. TRIAL EXECUTION SUMMARY")
print(f"Total trial_log rows : {len(trial_log)}")
print(f"Total final_ans rows : {len(final_ans)}")
if "condition_identifier" in trial_log.columns:
    print("\nTrial rows by condition:")
    print(trial_log["condition_identifier"].value_counts().to_string())
if "stage" in trial_log.columns:
    print("\nTrial rows by stage:")
    print(trial_log["stage"].value_counts().to_string())
if "responding_agent_role" in trial_log.columns:
    print("\nTrial rows by agent role:")
    print(trial_log["responding_agent_role"].value_counts().to_string())
if "responding_agent_model_name" in trial_log.columns:
    print("\nTrial rows by model:")
    print(trial_log["responding_agent_model_name"].value_counts().to_string())

# ── 4. Calibration gate ───────────────────────────────────────────────────────
section("4. CALIBRATION GATE (Stage 2)")
print(cal_gate.to_string(index=False))
print(f"\nMetadata gate decision : {meta.get('stage_2_calibration_gate_decision')}")
print(f"Metadata gate metric   : {meta.get('stage_2_calibration_gate_metric_value_at_decision')}")

# ── 5. Round-0 accuracy (baseline) ───────────────────────────────────────────
section("5. ROUND-0 ACCURACY BY CONDITION (baseline, per focal agent)")
if "round_zero_accuracy_rate" in metrics.columns:
    r0 = metrics[["condition_identifier", "focal_smart_agent_name", 
                   "total_trial_count", "round_zero_accuracy_rate",
                   "round_one_accuracy_rate", 
                   "round_one_minus_round_zero_accuracy_delta"]].copy()
    r0 = r0.sort_values(["focal_smart_agent_name", "condition_identifier"])
    print(r0.to_string(index=False))

# ── 6. Flip rates ─────────────────────────────────────────────────────────────
section("6. FLIP RATES AND CONFORMITY")
if "flip_rate_correct_to_incorrect" in metrics.columns:
    flip = metrics[["condition_identifier", "focal_smart_agent_name",
                     "flip_rate_correct_to_incorrect",
                     "flip_rate_incorrect_to_correct",
                     "asch_conformity_index",
                     "bootstrap_confidence_interval_lower_95_percent",
                     "bootstrap_confidence_interval_upper_95_percent"]].copy()
    print(flip.to_string(index=False))

# ── 7. Confidence scores ──────────────────────────────────────────────────────
section("7. MEAN CONFIDENCE BY CONDITION AND ROUND")
if "mean_round_zero_focal_self_reported_confidence" in metrics.columns:
    conf = metrics[["condition_identifier", "focal_smart_agent_name",
                     "mean_round_zero_focal_self_reported_confidence",
                     "mean_round_one_focal_self_reported_confidence"]].copy()
    print(conf.to_string(index=False))

# ── 8. Statistical tests ──────────────────────────────────────────────────────
section("8. STATISTICAL TESTS (McNemar + Bonferroni)")
print(stats.to_string(index=False))

# ── 9. Mitigation (C4 vs C5) ─────────────────────────────────────────────────
section("9. MITIGATION SUMMARY (C4 vs C5)")
print(mitigation.to_string(index=False))

# ── 10. Dose-response ─────────────────────────────────────────────────────────
section("10. DOSE-RESPONSE (dumb agent count effect)")
dose = stats[stats["test_name"].str.contains("logistic|dose", case=False, na=False)]
print(dose.to_string(index=False))

# ── 11. Cross-model validation ────────────────────────────────────────────────
section("11. CROSS-MODEL VALIDATION (GPT-4o-mini focal)")
if "focal_smart_agent_name" in metrics.columns:
    cross = metrics[metrics["focal_smart_agent_name"].str.contains("gpt|openrouter", case=False, na=False)]
    if not cross.empty:
        print(cross.to_string(index=False))
    else:
        print("No GPT-4o-mini focal agent rows found — check focal_smart_agent_name values:")
        print(metrics["focal_smart_agent_name"].unique())

# ── 12. Answer extraction method breakdown ────────────────────────────────────
section("12. ANSWER EXTRACTION METHOD")
for col in ["answer_extraction_method", "extraction_method", "judge_used"]:
    if col in trial_log.columns:
        print(f"\n{col}:")
        print(trial_log[col].value_counts().to_string())

# ── 13. Dumb agent accuracy ───────────────────────────────────────────────────
section("13. DUMB AGENT ACCURACY")
if "responding_agent_role" in trial_log.columns and "extracted_answer_matches_ground_truth" in trial_log.columns:
    dumb = trial_log[trial_log["responding_agent_role"] == "dumb"]
    if "responding_agent_model_name" in dumb.columns:
        print("Dumb agent accuracy by model:")
        print(dumb.groupby("responding_agent_model_name")["extracted_answer_matches_ground_truth"].mean().to_string())
    if "condition_identifier" in dumb.columns:
        print("\nDumb agent accuracy by condition:")
        print(dumb.groupby("condition_identifier")["extracted_answer_matches_ground_truth"].mean().to_string())

# ── 14. Smart agent accuracy by dataset source ────────────────────────────────
section("14. ACCURACY BY DATASET SOURCE")
for col in ["dataset_source", "source", "question_source"]:
    if col in final_ans.columns:
        print(f"\nRound-0 accuracy by {col}:")
        smart = final_ans[final_ans.get("focal_smart_agent_name", pd.Series(["x"])) != ""]
        print(smart.groupby(col)["round_zero_answer_was_correct"].mean().to_string())

# ── 15. Metadata summary ──────────────────────────────────────────────────────
section("15. EXPERIMENT METADATA")
print(json.dumps(meta, indent=2))

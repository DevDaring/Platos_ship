"""
free_analyses.py — Re-analyses requested by final_review.md that require
no new API calls. Reads parquet artefacts from results/data/ and writes
a JSON of every number needed to rewrite the paper.

Run from project root:
    python3 Submission/Analyse/free_analyses.py
"""
import json
import math
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as sstats
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "data" / "outputs"
PROC_DIR = ROOT / "results" / "data" / "processed"
OUT_JSON = ROOT / "Submission" / "Analyse" / "free_analyses_output.json"

trial = pd.read_parquet(OUT_DIR / "trial_log.parquet")
final = pd.read_parquet(OUT_DIR / "final_answers.parquet")
metrics = pd.read_parquet(OUT_DIR / "metrics_summary.parquet")
qpool = pd.read_parquet(PROC_DIR / "question_pool.parquet")

results = {}

# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────
def cohens_h(p1, p2):
    """Cohen's h for two proportions."""
    if any(np.isnan([p1, p2])): return float("nan")
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return abs(phi1 - phi2)

def bootstrap_ci(arr, n_boot=5000, ci=0.95, seed=42):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(arr, len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    m = float(arr.mean())
    lo = float(np.percentile(boots, (1-ci)/2 * 100))
    hi = float(np.percentile(boots, (1-(1-ci)/2) * 100))
    return (m, lo, hi)

def proportion_z_test(k1, n1, k2, n2):
    """Two-proportion z-test (Wald) and post-hoc power."""
    p1, p2 = k1/n1, k2/n2
    p_pool = (k1+k2) / (n1+n2)
    se = math.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    z = (p1 - p2) / se if se else 0.0
    p = 2 * (1 - sstats.norm.cdf(abs(z)))
    # post-hoc power for the observed effect size
    h = cohens_h(p1, p2)
    se_alt = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    z_crit = sstats.norm.ppf(0.975)
    z_alt = abs(p1 - p2) / se_alt if se_alt else 0.0
    power = 1 - sstats.norm.cdf(z_crit - z_alt) + sstats.norm.cdf(-z_crit - z_alt)
    return {"p1": p1, "p2": p2, "delta_pp": (p2-p1)*100, "z": z,
            "p_value": float(p), "cohens_h": h, "post_hoc_power": float(power)}

# ──────────────────────────────────────────────────────────────────────
# 1. Dataset audit (kills the 250+50 vs 200+100 contradiction)
# ──────────────────────────────────────────────────────────────────────
results["dataset_audit"] = {
    "total_questions": int(len(qpool)),
    "by_source": qpool["source_dataset"].value_counts().to_dict(),
    "mmlu_pro_subjects": qpool[qpool["source_dataset"]=="mmlu_pro"]["subject_category"].value_counts().to_dict(),
    "difficulty_split": qpool["difficulty_stratum"].value_counts().to_dict(),
    "mitigation_subset_count": int(qpool["included_in_mitigation_subset"].sum()),
}

# ──────────────────────────────────────────────────────────────────────
# 2. Headline numbers verification (kills the C5=82.5 vs 78.0 error)
# ──────────────────────────────────────────────────────────────────────
m = metrics.set_index(["condition_identifier", "focal_smart_agent_name"])
hl = {}
for (cond, foc), row in m.iterrows():
    key = f"{cond}|{foc}"
    hl[key] = {
        "n_trials": int(row["total_trial_count"]),
        "r0_acc": float(row["round_zero_accuracy_rate"]),
        "r1_acc": float(row["round_one_accuracy_rate"]),
        "delta_pp": float(row["round_one_minus_round_zero_accuracy_delta"])*100,
        "flip_c_to_i": float(row["flip_rate_correct_to_incorrect"]),
        "flip_i_to_c": float(row["flip_rate_incorrect_to_correct"]),
        "mean_r0_conf": float(row["mean_round_zero_focal_self_reported_confidence"]),
        "mean_r1_conf": float(row["mean_round_one_focal_self_reported_confidence"]),
    }

# Δ vs C1 baseline (what Table 2 in the paper actually shows)
for foc in ["deepseek_primary", "openrouter_gpt4o_mini"]:
    if f"C1_smart_solo|{foc}" not in hl:
        continue
    c1_r1 = hl[f"C1_smart_solo|{foc}"]["r1_acc"]
    for cond in ["C1_smart_solo", "C2_three_smart", "C3_two_smart_one_dumb",
                 "C4_one_smart_two_dumb", "C5_one_smart_two_dumb_confidence_weighted"]:
        k = f"{cond}|{foc}"
        if k in hl:
            hl[k]["delta_vs_C1_pp"] = (hl[k]["r1_acc"] - c1_r1) * 100

results["headline_metrics"] = hl

# Trial count audit
n_ds = sum(hl[k]["n_trials"] for k in hl if k.endswith("|deepseek_primary"))
n_gpt = sum(hl[k]["n_trials"] for k in hl if k.endswith("|openrouter_gpt4o_mini"))
results["trial_count_audit"] = {
    "deepseek_total": n_ds,
    "gpt4o_mini_total": n_gpt,
    "grand_total": n_ds + n_gpt,
    "paper_claim": 8500,
    "discrepancy": (n_ds + n_gpt) - 8500,
}

# ──────────────────────────────────────────────────────────────────────
# 3. SELF-CONSISTENCY-OF-3 BASELINE FROM C2 R0 (the killer free move)
# ──────────────────────────────────────────────────────────────────────
# In C2, 3 smart agents independently produce R0 answers. Per question,
# majority-vote-of-3 (R0) is a free self-consistency baseline.
# We need per-trial R0 answers for the 3 smart agents.
c2_log = trial[(trial["condition_identifier"] == "C2_three_smart") &
               (trial["responding_agent_role"].isin(["smart_focal", "smart_nonfocal"])) &
               (trial["debate_round_index"] == 0) &
               (trial["focal_smart_agent_name"] == "deepseek_primary")].copy()

# Per (question, trial) group of 3 R0 answers → majority vote
def majority_vote(answers):
    """Return modal answer; None if all unique."""
    vc = pd.Series(answers).value_counts()
    if vc.iloc[0] >= 2:
        return vc.index[0]
    return answers.iloc[0]  # if all 3 different, pick the focal's (alphabetical fallback)

def correct_truth_lookup():
    return qpool.set_index("question_identifier")["correct_answer"].to_dict()

truth = correct_truth_lookup()

# group by (question, trial_replication_index) and compute majority answer
groups = c2_log.groupby(["question_identifier", "trial_replication_index"])
sc3_records = []
for (qid, trep), grp in groups:
    if len(grp) < 2:
        continue
    answers = grp["extracted_final_answer"].astype(str)
    maj = majority_vote(answers)
    correct = truth.get(qid)
    is_correct = (str(maj).strip() == str(correct).strip())
    sc3_records.append({
        "question_identifier": qid,
        "trial_replication_index": trep,
        "sc3_answer": maj,
        "sc3_correct": is_correct,
        "n_agents_in_group": len(grp),
    })
sc3 = pd.DataFrame(sc3_records)

# C2 R0 majority-vote-of-3 accuracy (per-trial)
sc3_acc_per_trial = sc3["sc3_correct"].mean()
sc3_m, sc3_lo, sc3_hi = bootstrap_ci(sc3["sc3_correct"].astype(float))

# Per-question majority-vote across 5 trials of the SC3-of-3 (i.e., the
# C2 R0 majority-vote-of-3 majority-voted again over 5 reps → final SC3
# decision per question)
sc3_q = sc3.groupby("question_identifier")["sc3_correct"].mean() >= 0.5
sc3_q_acc = sc3_q.mean()

# Compare to C2 R1 (per-trial)
c2_r1_records = final[(final["condition_identifier"] == "C2_three_smart") &
                       (final["focal_smart_agent_name"] == "deepseek_primary")]
c2_r1_acc = c2_r1_records["round_one_answer_was_correct"].astype(float).mean()
c2_r1_m, c2_r1_lo, c2_r1_hi = bootstrap_ci(c2_r1_records["round_one_answer_was_correct"].astype(float))

# Compare to C1 R0 (per-trial) — solo focal-agent baseline
c1_records = final[(final["condition_identifier"] == "C1_smart_solo") &
                    (final["focal_smart_agent_name"] == "deepseek_primary")]
c1_r0_acc = c1_records["round_zero_answer_was_correct"].astype(float).mean()

# C1 majority-vote-of-3 (use 3 of the 5 reps per question for a fair comparison)
# This needs the trial-level data for C1 (focal-only R0 answers)
c1_log = trial[(trial["condition_identifier"] == "C1_smart_solo") &
                (trial["responding_agent_role"] == "smart_focal") &
                (trial["debate_round_index"] == 0) &
                (trial["focal_smart_agent_name"] == "deepseek_primary")].copy()

# Take first 3 reps per question, majority vote
c1_q_groups = c1_log.groupby("question_identifier")
c1_mv3_records = []
for qid, grp in c1_q_groups:
    if len(grp) < 3:
        continue
    first3 = grp.sort_values("trial_replication_index").head(3)
    answers = first3["extracted_final_answer"].astype(str)
    maj = majority_vote(answers)
    correct = truth.get(qid)
    is_correct = (str(maj).strip() == str(correct).strip())
    c1_mv3_records.append({"question_identifier": qid, "c1_mv3_correct": is_correct})
c1_mv3 = pd.DataFrame(c1_mv3_records)
c1_mv3_acc = c1_mv3["c1_mv3_correct"].mean()

results["self_consistency_baseline"] = {
    "description": "Free self-consistency-of-3 baseline derived from C2 R0 (3 DeepSeek samples per trial) and C1 (3-of-5 reps majority vote).",
    "c2_sc3_per_trial_acc": float(sc3_acc_per_trial),
    "c2_sc3_per_trial_ci95": [sc3_lo, sc3_hi],
    "c2_sc3_per_question_acc": float(sc3_q_acc),
    "c2_r1_per_trial_acc": float(c2_r1_acc),
    "c2_r1_per_trial_ci95": [c2_r1_lo, c2_r1_hi],
    "c1_r0_per_trial_acc": float(c1_r0_acc),
    "c1_mv3_per_question_acc": float(c1_mv3_acc),
    "interpretation_delta_debate_over_sc3": float(c2_r1_acc - sc3_acc_per_trial) * 100,
    "interpretation_note": (
        "If c2_r1 - c2_sc3 ≈ 0, debate adds nothing beyond sample aggregation. "
        "If positive, debate-based revision provides real benefit on top of SC."
    ),
}

# ──────────────────────────────────────────────────────────────────────
# 4. Effect sizes for the key comparisons (Cohen's h)
# ──────────────────────────────────────────────────────────────────────
def k_n_for(cond, foc, col):
    sub = final[(final["condition_identifier"] == cond) &
                 (final["focal_smart_agent_name"] == foc)]
    n = len(sub)
    k = int(sub[col].sum())
    return k, n

eff = {}
for foc in ["deepseek_primary", "openrouter_gpt4o_mini"]:
    eff[foc] = {}
    for (c1, c2) in [("C1_smart_solo", "C4_one_smart_two_dumb"),
                     ("C2_three_smart", "C4_one_smart_two_dumb"),
                     ("C2_three_smart", "C3_two_smart_one_dumb"),
                     ("C3_two_smart_one_dumb", "C4_one_smart_two_dumb")]:
        k1, n1 = k_n_for(c1, foc, "round_one_answer_was_correct")
        k2, n2 = k_n_for(c2, foc, "round_one_answer_was_correct")
        if n1 == 0 or n2 == 0:
            continue
        eff[foc][f"{c1}_vs_{c2}_R1acc"] = proportion_z_test(k1, n1, k2, n2)

    # C→I flip rate comparisons
    for (c1, c2) in [("C2_three_smart", "C3_two_smart_one_dumb"),
                     ("C2_three_smart", "C4_one_smart_two_dumb"),
                     ("C3_two_smart_one_dumb", "C4_one_smart_two_dumb")]:
        k1, n1 = k_n_for(c1, foc, "focal_agent_flipped_correct_to_incorrect")
        k2, n2 = k_n_for(c2, foc, "focal_agent_flipped_correct_to_incorrect")
        if n1 == 0 or n2 == 0:
            continue
        eff[foc][f"{c1}_vs_{c2}_flipCtoI"] = proportion_z_test(k1, n1, k2, n2)

results["effect_sizes_and_power"] = eff

# ──────────────────────────────────────────────────────────────────────
# 5. Mixed-effects logistic regression with question-level random effect
# ──────────────────────────────────────────────────────────────────────
# Question observations are nested: each question has 5 replications.
# A naive logistic ignores this. Fit a clustered/mixed model.
ds_pool = final[(final["focal_smart_agent_name"] == "deepseek_primary") &
                 (final["condition_identifier"].isin([
                     "C2_three_smart", "C3_two_smart_one_dumb",
                     "C4_one_smart_two_dumb"]))].copy()
ds_pool["dumb_count"] = ds_pool["condition_dumb_agent_count"].astype(int)
ds_pool["r1_correct"] = ds_pool["round_one_answer_was_correct"].astype(int)

# Naive logit (for replicating the paper's claim)
naive = smf.logit("r1_correct ~ dumb_count", data=ds_pool).fit(disp=0)

# GEE with question_identifier as cluster (gives robust SEs)
gee = smf.gee("r1_correct ~ dumb_count", "question_identifier", data=ds_pool,
              family=sm.families.Binomial(),
              cov_struct=sm.cov_struct.Exchangeable()).fit()

# Also fit mixed-effects logit (slower; use BinomialBayesMixedGLM if MixedLM
# struggles with binary outcomes — use GEE as the primary clustered estimator)
def model_summary(fit, name):
    coef = fit.params.get("dumb_count")
    se = fit.bse.get("dumb_count") if hasattr(fit, "bse") else fit.bse.get("dumb_count")
    pval = fit.pvalues.get("dumb_count") if hasattr(fit, "pvalues") else float("nan")
    or_ = math.exp(coef) if coef is not None else float("nan")
    ci_lo = math.exp(coef - 1.96*se) if (coef is not None and not pd.isna(se)) else float("nan")
    ci_hi = math.exp(coef + 1.96*se) if (coef is not None and not pd.isna(se)) else float("nan")
    return {
        "model": name,
        "coef": float(coef),
        "se": float(se),
        "p_value": float(pval),
        "odds_ratio": float(or_),
        "or_ci95_low": float(ci_lo),
        "or_ci95_high": float(ci_hi),
        "n_obs": int(fit.nobs),
    }

results["dose_response_regression"] = {
    "naive_logit": model_summary(naive, "naive_logit_no_cluster"),
    "gee_clustered_by_question": model_summary(gee, "GEE_exchangeable_question_cluster"),
    "note": "GEE is the appropriate model — 1500 trials nest within 300 questions.",
}

# Same for GPT-4o-mini
gpt_pool = final[(final["focal_smart_agent_name"] == "openrouter_gpt4o_mini") &
                  (final["condition_identifier"].isin([
                      "C2_three_smart", "C3_two_smart_one_dumb",
                      "C4_one_smart_two_dumb"]))].copy()
gpt_pool["dumb_count"] = gpt_pool["condition_dumb_agent_count"].astype(int)
gpt_pool["r1_correct"] = gpt_pool["round_one_answer_was_correct"].astype(int)

if len(gpt_pool) > 50:
    naive_g = smf.logit("r1_correct ~ dumb_count", data=gpt_pool).fit(disp=0)
    gee_g = smf.gee("r1_correct ~ dumb_count", "question_identifier", data=gpt_pool,
                    family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
    results["dose_response_regression_gpt"] = {
        "naive_logit": model_summary(naive_g, "naive_logit_no_cluster"),
        "gee_clustered_by_question": model_summary(gee_g, "GEE_exchangeable_question_cluster"),
    }

# ──────────────────────────────────────────────────────────────────────
# 6. Difficulty stratification
# ──────────────────────────────────────────────────────────────────────
fin_q = final.merge(qpool[["question_identifier", "difficulty_stratum", "source_dataset", "subject_category"]],
                    on="question_identifier", how="left")

strat_results = {}
for foc in ["deepseek_primary", "openrouter_gpt4o_mini"]:
    strat_results[foc] = {}
    for diff in ["probe_correct", "probe_incorrect"]:
        for cond in ["C1_smart_solo", "C2_three_smart",
                     "C3_two_smart_one_dumb", "C4_one_smart_two_dumb",
                     "C5_one_smart_two_dumb_confidence_weighted"]:
            sub = fin_q[(fin_q["focal_smart_agent_name"] == foc) &
                         (fin_q["condition_identifier"] == cond) &
                         (fin_q["difficulty_stratum"] == diff)]
            if len(sub) == 0:
                continue
            strat_results[foc][f"{cond}|{diff}"] = {
                "n_trials": int(len(sub)),
                "r0_acc": float(sub["round_zero_answer_was_correct"].astype(float).mean()),
                "r1_acc": float(sub["round_one_answer_was_correct"].astype(float).mean()),
                "flip_c_to_i": float(sub["focal_agent_flipped_correct_to_incorrect"].astype(float).mean()),
                "flip_i_to_c": float(sub["focal_agent_flipped_incorrect_to_correct"].astype(float).mean()),
            }

results["difficulty_stratification"] = strat_results

# ──────────────────────────────────────────────────────────────────────
# 7. Source-dataset split (MMLU-Pro vs GSM8K)
# ──────────────────────────────────────────────────────────────────────
src_results = {}
for foc in ["deepseek_primary", "openrouter_gpt4o_mini"]:
    src_results[foc] = {}
    for src in ["mmlu_pro", "gsm8k"]:
        for cond in ["C1_smart_solo", "C2_three_smart",
                     "C3_two_smart_one_dumb", "C4_one_smart_two_dumb",
                     "C5_one_smart_two_dumb_confidence_weighted"]:
            sub = fin_q[(fin_q["focal_smart_agent_name"] == foc) &
                         (fin_q["condition_identifier"] == cond) &
                         (fin_q["source_dataset"] == src)]
            if len(sub) == 0:
                continue
            src_results[foc][f"{cond}|{src}"] = {
                "n_trials": int(len(sub)),
                "r0_acc": float(sub["round_zero_answer_was_correct"].astype(float).mean()),
                "r1_acc": float(sub["round_one_answer_was_correct"].astype(float).mean()),
                "flip_c_to_i": float(sub["focal_agent_flipped_correct_to_incorrect"].astype(float).mean()),
                "flip_i_to_c": float(sub["focal_agent_flipped_incorrect_to_correct"].astype(float).mean()),
            }
results["source_dataset_split"] = src_results

# ──────────────────────────────────────────────────────────────────────
# 8. C5 effective filter rate (peers filtered out)
# ──────────────────────────────────────────────────────────────────────
c5 = final[final["condition_identifier"] == "C5_one_smart_two_dumb_confidence_weighted"]
n_c5 = len(c5)
total_peers_shown = 2 * n_c5  # always 2 dumb peers
filt = c5["c5_count_of_peer_messages_filtered_out"].astype(float)
mean_filtered = float(filt.mean())
pct_trials_all_filtered = float((filt == 2).mean()) * 100
results["c5_filter_audit"] = {
    "n_c5_trials": int(n_c5),
    "mean_peers_filtered_per_trial": mean_filtered,
    "max_peers_per_trial": 2,
    "pct_trials_with_zero_peers_after_filtering": pct_trials_all_filtered,
    "effective_filter_rate_pct": mean_filtered / 2 * 100,
    "interpretation": "C5 effectively reduces to solo when filter removes all peers.",
}

# ──────────────────────────────────────────────────────────────────────
# 9. Regex-vs-judge attribution for the focal-agent flips
# ──────────────────────────────────────────────────────────────────────
# Was the C→I flip determination driven by judge-extracted answers?
focal_trials = trial[(trial["responding_agent_role"] == "smart_focal") &
                      (trial["focal_smart_agent_name"] == "deepseek_primary")]
extr = focal_trials["answer_extraction_method"].value_counts(normalize=True) * 100
results["focal_extraction_method_distribution"] = {k: float(v) for k, v in extr.to_dict().items()}

# Per-condition: what fraction of focal-R1 answers required the judge?
judge_per_cond = {}
for cond in ["C2_three_smart", "C3_two_smart_one_dumb",
             "C4_one_smart_two_dumb", "C5_one_smart_two_dumb_confidence_weighted"]:
    sub = focal_trials[(focal_trials["condition_identifier"] == cond) &
                        (focal_trials["debate_round_index"] == 1)]
    if len(sub) == 0:
        continue
    vc = sub["answer_extraction_method"].value_counts(normalize=True) * 100
    judge_per_cond[cond] = {k: float(v) for k, v in vc.to_dict().items()}
results["focal_R1_extraction_by_condition"] = judge_per_cond

# ──────────────────────────────────────────────────────────────────────
# 10. Net flip score = (I→C × R0_wrong_rate) − (C→I × R0_right_rate)
#     Should equal R1 − R0 accuracy delta. Audit consistency.
# ──────────────────────────────────────────────────────────────────────
net_flip_audit = {}
for (cond, foc), row in m.iterrows():
    r0 = row["round_zero_accuracy_rate"]
    r1 = row["round_one_accuracy_rate"]
    flip_ci = row["flip_rate_correct_to_incorrect"]
    flip_ic = row["flip_rate_incorrect_to_correct"]
    # net = flip_ic × (1 − r0) − flip_ci × r0  should equal r1 − r0
    net = flip_ic * (1 - r0) - flip_ci * r0
    net_flip_audit[f"{cond}|{foc}"] = {
        "observed_delta": (r1 - r0) * 100,
        "computed_net_flip_delta": net * 100,
        "residual": (r1 - r0 - net) * 100,
    }
results["net_flip_score_audit"] = net_flip_audit

# ──────────────────────────────────────────────────────────────────────
# 11. McNemar at TRIAL level (kills the trial-vs-question inversion attack)
# ──────────────────────────────────────────────────────────────────────
from statsmodels.stats.contingency_tables import mcnemar

mcn_trial = {}
for foc in ["deepseek_primary", "openrouter_gpt4o_mini"]:
    pairs = [("C1_smart_solo", "C2_three_smart"),
             ("C1_smart_solo", "C3_two_smart_one_dumb"),
             ("C1_smart_solo", "C4_one_smart_two_dumb"),
             ("C2_three_smart", "C3_two_smart_one_dumb"),
             ("C2_three_smart", "C4_one_smart_two_dumb"),
             ("C3_two_smart_one_dumb", "C4_one_smart_two_dumb")]
    for c1, c2 in pairs:
        f1 = final[(final["focal_smart_agent_name"] == foc) &
                    (final["condition_identifier"] == c1)]
        f2 = final[(final["focal_smart_agent_name"] == foc) &
                    (final["condition_identifier"] == c2)]
        if len(f1) == 0 or len(f2) == 0:
            continue
        # Match on (question_identifier, trial_replication_index)
        merged = f1[["question_identifier", "trial_replication_index", "round_one_answer_was_correct"]].rename(
            columns={"round_one_answer_was_correct": "r1_c1"}).merge(
            f2[["question_identifier", "trial_replication_index", "round_one_answer_was_correct"]].rename(
                columns={"round_one_answer_was_correct": "r1_c2"}),
            on=["question_identifier", "trial_replication_index"], how="inner")
        if len(merged) == 0:
            continue
        # 2x2 contingency
        a = int(((merged["r1_c1"]) & (merged["r1_c2"])).sum())
        b = int(((merged["r1_c1"]) & (~merged["r1_c2"].astype(bool))).sum())
        c = int(((~merged["r1_c1"].astype(bool)) & (merged["r1_c2"])).sum())
        d = int(((~merged["r1_c1"].astype(bool)) & (~merged["r1_c2"].astype(bool))).sum())
        table = [[a, b], [c, d]]
        try:
            res = mcnemar(table, exact=False, correction=True)
            mcn_trial[f"{foc}|{c1}_vs_{c2}"] = {
                "table": table,
                "statistic": float(res.statistic),
                "p_value": float(res.pvalue),
                "n_paired_trials": int(a+b+c+d),
                "b_minus_c": b-c,
            }
        except Exception as e:
            pass
results["mcnemar_trial_level"] = mcn_trial

# Holm-Bonferroni adjustment over the 6 DeepSeek pairs
from statsmodels.stats.multitest import multipletests
ds_pairs = {k: v["p_value"] for k, v in mcn_trial.items() if k.startswith("deepseek_primary")}
if ds_pairs:
    keys = list(ds_pairs.keys())
    pvals = [ds_pairs[k] for k in keys]
    _, holm_corr, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    results["holm_bonferroni_deepseek"] = {
        k: {"raw_p": pvals[i], "holm_p": float(holm_corr[i])} for i, k in enumerate(keys)
    }

# ──────────────────────────────────────────────────────────────────────
# 12. Calibration / confidence reliability (dumb agents)
# ──────────────────────────────────────────────────────────────────────
dumb = trial[(trial["responding_agent_role"] == "dumb") &
              (trial["extracted_self_reported_confidence_integer"].notna())].copy()
dumb["conf"] = dumb["extracted_self_reported_confidence_integer"].astype(float)
dumb["correct"] = dumb["extracted_answer_matches_ground_truth"].astype(bool)

# By model
calib_by_model = {}
for model_name in dumb["responding_agent_model_name"].unique():
    sub = dumb[dumb["responding_agent_model_name"] == model_name]
    p_high_conf_given_wrong = sub[~sub["correct"]]["conf"].ge(60).mean()
    p_high_conf_given_right = sub[sub["correct"]]["conf"].ge(60).mean()
    calib_by_model[model_name] = {
        "n_responses": int(len(sub)),
        "mean_conf_when_wrong": float(sub[~sub["correct"]]["conf"].mean()),
        "mean_conf_when_right": float(sub[sub["correct"]]["conf"].mean()),
        "p_conf_ge_60_given_wrong": float(p_high_conf_given_wrong),
        "p_conf_ge_60_given_right": float(p_high_conf_given_right),
        "p_diff_loud_wrong_minus_loud_right": float(p_high_conf_given_wrong - p_high_conf_given_right),
    }
results["calibration_by_dumb_model"] = calib_by_model

# ──────────────────────────────────────────────────────────────────────
# 13. Subject-level flip rates (DeepSeek C4)
# ──────────────────────────────────────────────────────────────────────
ds_c4 = fin_q[(fin_q["focal_smart_agent_name"] == "deepseek_primary") &
               (fin_q["condition_identifier"] == "C4_one_smart_two_dumb")].copy()
ds_c4["subject"] = ds_c4.apply(
    lambda r: "gsm8k" if r["source_dataset"] == "gsm8k"
    else (r["subject_category"] or "unknown"), axis=1)
subj_flips = {}
for subj in ds_c4["subject"].unique():
    sub = ds_c4[ds_c4["subject"] == subj]
    elig = sub[sub["round_zero_answer_was_correct"].astype(bool)]
    if len(elig) < 10:
        continue
    rate = elig["focal_agent_flipped_correct_to_incorrect"].astype(float).mean()
    m_, lo, hi = bootstrap_ci(elig["focal_agent_flipped_correct_to_incorrect"].astype(float))
    subj_flips[subj] = {
        "n_eligible": int(len(elig)),
        "flip_c_to_i": float(rate),
        "ci95_low": lo,
        "ci95_high": hi,
    }
results["subject_flips_ds_c4"] = subj_flips

# ──────────────────────────────────────────────────────────────────────
# 14. Save
# ──────────────────────────────────────────────────────────────────────
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Wrote {OUT_JSON}")

# Print a brief summary
print("\n=== Key numbers ===")
print(f"Dataset: MMLU-Pro={results['dataset_audit']['by_source']['mmlu_pro']}, GSM8K={results['dataset_audit']['by_source']['gsm8k']}")
print(f"Grand trial total: {results['trial_count_audit']['grand_total']} (paper claims 8500)")
print(f"\n--- C2 R0 SC-of-3 (the killer baseline) ---")
sc = results["self_consistency_baseline"]
print(f"  C1 R0 (solo):                          {sc['c1_r0_per_trial_acc']*100:.2f}%")
print(f"  C1 majority-of-3 (per question):       {sc['c1_mv3_per_question_acc']*100:.2f}%")
print(f"  C2 R0 SC-of-3 (per trial):             {sc['c2_sc3_per_trial_acc']*100:.2f}%")
print(f"  C2 R1 debate (per trial):              {sc['c2_r1_per_trial_acc']*100:.2f}%")
print(f"  Debate gain over SC-of-3:              {sc['interpretation_delta_debate_over_sc3']:+.2f} pp")

print(f"\n--- Mixed-effects (GEE) dose-response ---")
dr = results["dose_response_regression"]
print(f"  Naive logit OR: {dr['naive_logit']['odds_ratio']:.3f} (p={dr['naive_logit']['p_value']:.4f})")
print(f"  GEE clustered: OR={dr['gee_clustered_by_question']['odds_ratio']:.3f} "
      f"[{dr['gee_clustered_by_question']['or_ci95_low']:.3f}, "
      f"{dr['gee_clustered_by_question']['or_ci95_high']:.3f}], "
      f"p={dr['gee_clustered_by_question']['p_value']:.4f}")

print(f"\n--- C5 filter audit ---")
fa = results["c5_filter_audit"]
print(f"  Mean peers filtered per C5 trial: {fa['mean_peers_filtered_per_trial']:.3f} of 2")
print(f"  % trials with all peers filtered: {fa['pct_trials_with_zero_peers_after_filtering']:.1f}%")

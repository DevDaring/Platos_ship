"""
round2_analyses.py — Second-round analyses requested by an internal
hostile review (round 2). All analyses use existing parquet data, no
API calls.

Covers:
  R2: Two-proportion z-test on C2 R1 vs C2 R0 SC-of-3
  R3: Bootstrap 95% CIs for GPT-4o-mini headline deltas
  R4: Per-item flip counts on probe-hard vs probe-easy in C4 (mechanism
      for the trial-vs-question McNemar inversion)
  R5: C5-C4 CI (already in mitigation_summary; surfaces it cleanly)
  R6: GSM8K mechanism probe — fraction of C4 GSM8K trials where focal R1
      answer differs from both peer R0 answers (fresh recomputation rate)
  R7: Effective sample size from kappa = 0.72 using
      ESS = k / (1 + (k-1)*rho)
  R8: Maximum F1 across thresholds for confidence -> correctness, per
      (model, round)
  R13: Two example trials for the supplement (one C->I flip, one stable
       correct under wrong peers)

Run from project root:
    python3 Submission/Analyse/round2_analyses.py
"""
import json
import math
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "data" / "outputs"
PROC_DIR = ROOT / "results" / "data" / "processed"
OUT_JSON = ROOT / "Submission" / "Analyse" / "round2_analyses_output.json"

trial = pd.read_parquet(OUT_DIR / "trial_log.parquet")
final = pd.read_parquet(OUT_DIR / "final_answers.parquet")
metrics = pd.read_parquet(OUT_DIR / "metrics_summary.parquet")
mit = pd.read_parquet(OUT_DIR / "mitigation_summary.parquet")
qpool = pd.read_parquet(PROC_DIR / "question_pool.parquet")

results = {}

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1/n1, k2/n2
    pool = (k1+k2) / (n1+n2)
    se = math.sqrt(pool*(1-pool)*(1/n1 + 1/n2))
    z = (p2 - p1) / se if se else 0.0
    p = 2 * (1 - sstats.norm.cdf(abs(z)))
    h = abs(2*math.asin(math.sqrt(p1)) - 2*math.asin(math.sqrt(p2)))
    se_alt = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    z_crit = sstats.norm.ppf(0.975)
    z_alt = abs(p1-p2) / se_alt if se_alt else 0.0
    power = 1 - sstats.norm.cdf(z_crit - z_alt) + sstats.norm.cdf(-z_crit - z_alt)
    # 95% CI on difference (Wald)
    ci_lo = (p2 - p1) - 1.96 * se_alt
    ci_hi = (p2 - p1) + 1.96 * se_alt
    return {"p1": p1, "p2": p2, "delta_pp": (p2-p1)*100,
            "ci95_lo_pp": ci_lo*100, "ci95_hi_pp": ci_hi*100,
            "z": z, "p_value": float(p),
            "cohens_h": h, "post_hoc_power": float(power)}

def bootstrap_ci_diff(arr1, arr2, n_boot=5000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(arr1, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(arr2, dtype=float); b = b[~np.isnan(b)]
    boots = np.array([rng.choice(b, len(b), replace=True).mean()
                       - rng.choice(a, len(a), replace=True).mean()
                       for _ in range(n_boot)])
    return (float(b.mean()-a.mean()),
            float(np.percentile(boots, (1-ci)/2*100)),
            float(np.percentile(boots, (1-(1-ci)/2)*100)))

# ──────────────────────────────────────────────────────────────────────
# R2: SC-of-3 significance test
# ──────────────────────────────────────────────────────────────────────
# Rebuild SC-3 (per-trial majority vote of three C2 R0 answers per (q, trial))
c2_log = trial[(trial["condition_identifier"] == "C2_three_smart") &
               (trial["focal_smart_agent_name"] == "deepseek_primary") &
               (trial["debate_round_index"] == 0) &
               (trial["responding_agent_role"].isin(["smart_focal", "smart_nonfocal"]))].copy()
truth = qpool.set_index("question_identifier")["correct_answer"].to_dict()

def majority_vote(answers):
    vc = pd.Series(answers).value_counts()
    if vc.iloc[0] >= 2:
        return vc.index[0]
    return answers.iloc[0]

sc3 = []
for (qid, trep), grp in c2_log.groupby(["question_identifier", "trial_replication_index"]):
    if len(grp) < 2:
        continue
    maj = majority_vote(grp["extracted_final_answer"].astype(str))
    correct = str(truth.get(qid)).strip() == str(maj).strip()
    sc3.append({"correct": int(correct)})
sc3_df = pd.DataFrame(sc3)
sc3_k = int(sc3_df["correct"].sum())
sc3_n = len(sc3_df)

# C2 R1 (per-trial)
c2_r1 = final[(final["condition_identifier"] == "C2_three_smart") &
               (final["focal_smart_agent_name"] == "deepseek_primary")]
c2_k = int(c2_r1["round_one_answer_was_correct"].sum())
c2_n = len(c2_r1)

# C1 (per-trial)
c1 = final[(final["condition_identifier"] == "C1_smart_solo") &
            (final["focal_smart_agent_name"] == "deepseek_primary")]
c1_k = int(c1["round_one_answer_was_correct"].sum())
c1_n = len(c1)

results["R2_sc3_significance"] = {
    "c1_solo_per_trial":           {"k": c1_k, "n": c1_n, "acc": c1_k/c1_n},
    "c2_r0_majority_of_3":         {"k": sc3_k, "n": sc3_n, "acc": sc3_k/sc3_n},
    "c2_r1_debate":                {"k": c2_k, "n": c2_n, "acc": c2_k/c2_n},
    "test_c2r1_vs_sc3":            two_prop_z(sc3_k, sc3_n, c2_k, c2_n),
    "test_sc3_vs_c1":              two_prop_z(c1_k, c1_n, sc3_k, sc3_n),
    "test_c2r1_vs_c1":             two_prop_z(c1_k, c1_n, c2_k, c2_n),
}

# ──────────────────────────────────────────────────────────────────────
# R3: GPT-4o-mini bootstrap CIs on the key deltas
# ──────────────────────────────────────────────────────────────────────
gpt_results = {}
for cond in ["C2_three_smart", "C3_two_smart_one_dumb",
             "C4_one_smart_two_dumb"]:
    c1g = final[(final["condition_identifier"] == "C1_smart_solo") &
                 (final["focal_smart_agent_name"] == "openrouter_gpt4o_mini")]
    cg = final[(final["condition_identifier"] == cond) &
                (final["focal_smart_agent_name"] == "openrouter_gpt4o_mini")]
    if len(c1g)==0 or len(cg)==0:
        continue
    a = c1g["round_one_answer_was_correct"].astype(float).values
    b = cg["round_one_answer_was_correct"].astype(float).values
    delta, lo, hi = bootstrap_ci_diff(a, b)
    gpt_results[f"{cond}_minus_C1_R1_acc"] = {
        "delta_pp": delta*100,
        "ci95_lo_pp": lo*100,
        "ci95_hi_pp": hi*100,
        "n_a": len(a), "n_b": len(b),
    }
    # also test the C->I flip rate
    a_flip = c1g["focal_agent_flipped_correct_to_incorrect"].astype(float).values
    b_flip = cg["focal_agent_flipped_correct_to_incorrect"].astype(float).values
    delta, lo, hi = bootstrap_ci_diff(a_flip, b_flip)
    gpt_results[f"{cond}_minus_C1_flipCtoI_pp"] = {
        "delta_pp": delta*100, "ci95_lo_pp": lo*100, "ci95_hi_pp": hi*100,
    }
results["R3_gpt_deltas_with_CIs"] = gpt_results

# ──────────────────────────────────────────────────────────────────────
# R4: Per-item flip counts in C4 vs C1 by difficulty stratum
# ──────────────────────────────────────────────────────────────────────
diff_map = qpool.set_index("question_identifier")["difficulty_stratum"].to_dict()
src_map = qpool.set_index("question_identifier")["source_dataset"].to_dict()
final["difficulty_stratum"] = final["question_identifier"].map(diff_map)
final["source_dataset"] = final["question_identifier"].map(src_map)

# Question-level majority-vote correctness in C1 and C4 for DeepSeek
def qmv_correct(cond):
    sub = final[(final["condition_identifier"]==cond) &
                  (final["focal_smart_agent_name"]=="deepseek_primary")]
    g = sub.groupby("question_identifier")["round_one_answer_was_correct"]\
              .apply(lambda x: x.astype(int).mean() >= 0.5)
    return g

c1_q = qmv_correct("C1_smart_solo")
c4_q = qmv_correct("C4_one_smart_two_dumb")

# Per-stratum 2x2 contingency
strata = {}
for stratum in ["probe_correct", "probe_incorrect"]:
    qids = [q for q, v in diff_map.items() if v == stratum]
    cell_a = sum(1 for q in qids if c1_q.get(q) and c4_q.get(q))     # both correct
    cell_b = sum(1 for q in qids if c1_q.get(q) and not c4_q.get(q)) # regressed
    cell_c = sum(1 for q in qids if not c1_q.get(q) and c4_q.get(q)) # improved
    cell_d = sum(1 for q in qids if not c1_q.get(q) and not c4_q.get(q)) # both wrong
    strata[stratum] = {
        "n_questions": len(qids),
        "both_correct": cell_a,
        "regressed_c1c_to_c4w": cell_b,
        "improved_c1w_to_c4c": cell_c,
        "both_wrong": cell_d,
        "net_change_in_questions": cell_c - cell_b,
    }
results["R4_question_level_mcnemar_by_difficulty"] = strata

# Also overall
all_qids = list(diff_map.keys())
total_a = sum(1 for q in all_qids if c1_q.get(q) and c4_q.get(q))
total_b = sum(1 for q in all_qids if c1_q.get(q) and not c4_q.get(q))
total_c = sum(1 for q in all_qids if not c1_q.get(q) and c4_q.get(q))
total_d = sum(1 for q in all_qids if not c1_q.get(q) and not c4_q.get(q))
results["R4_question_level_mcnemar_overall"] = {
    "n_questions": len(all_qids), "both_correct": total_a,
    "regressed": total_b, "improved": total_c, "both_wrong": total_d,
}

# Trial-level paired count (same questions, same trial replication indices)
merged = final[(final["focal_smart_agent_name"]=="deepseek_primary")][[
    "question_identifier","trial_replication_index","condition_identifier",
    "round_one_answer_was_correct"]]
c1_t = merged[merged["condition_identifier"]=="C1_smart_solo"].set_index(
    ["question_identifier","trial_replication_index"])["round_one_answer_was_correct"]
c4_t = merged[merged["condition_identifier"]=="C4_one_smart_two_dumb"].set_index(
    ["question_identifier","trial_replication_index"])["round_one_answer_was_correct"]
both = pd.concat([c1_t.rename("c1"), c4_t.rename("c4")], axis=1).dropna()
tA = int(((both["c1"]) & (both["c4"])).sum())
tB = int(((both["c1"]) & (~both["c4"].astype(bool))).sum())
tC = int(((~both["c1"].astype(bool)) & (both["c4"])).sum())
tD = int(((~both["c1"].astype(bool)) & (~both["c4"].astype(bool))).sum())
results["R4_trial_level_paired_C1_vs_C4"] = {
    "n_paired_trials": tA+tB+tC+tD,
    "both_correct": tA, "regressed_trial": tB, "improved_trial": tC,
    "both_wrong": tD, "net_change_in_trials": tC - tB,
}

# ──────────────────────────────────────────────────────────────────────
# R5: C5-C4 CI (surface from mitigation_summary)
# ──────────────────────────────────────────────────────────────────────
results["R5_c5_minus_c4_mitigation"] = {
    "c4_acc_mitigation_subset":     float(mit["c4_round_one_accuracy_on_mitigation_subset"].iloc[0]),
    "c5_acc":                       float(mit["c5_round_one_accuracy"].iloc[0]),
    "delta_pp":                     float(mit["c5_minus_c4_accuracy_delta_percentage_points"].iloc[0]),
    "ci95_lo_pp":                   float(mit["c5_minus_c4_bootstrap_confidence_interval_lower_95_percent"].iloc[0])*100,
    "ci95_hi_pp":                   float(mit["c5_minus_c4_bootstrap_confidence_interval_upper_95_percent"].iloc[0])*100,
    "mcnemar_p":                    float(mit["c5_minus_c4_mcnemar_p_value"].iloc[0]),
    "n_c5_trials":                  300,
}

# ──────────────────────────────────────────────────────────────────────
# R6: GSM8K mechanism — fresh-recompute rate in C4
# ──────────────────────────────────────────────────────────────────────
# Pull C4 focal R1 (answer) and peer R0 answers per (q, trial)
focal_r1 = trial[(trial["responding_agent_role"]=="smart_focal") &
                  (trial["debate_round_index"]==1) &
                  (trial["focal_smart_agent_name"]=="deepseek_primary") &
                  (trial["condition_identifier"]=="C4_one_smart_two_dumb")][[
                      "question_identifier","trial_replication_index","extracted_final_answer"]]
peer_r0 = trial[(trial["responding_agent_role"]=="dumb") &
                 (trial["debate_round_index"]==0) &
                 (trial["focal_smart_agent_name"]=="deepseek_primary") &
                 (trial["condition_identifier"]=="C4_one_smart_two_dumb")][[
                     "question_identifier","trial_replication_index","extracted_final_answer"]]

peer_grouped = peer_r0.groupby(["question_identifier","trial_replication_index"])[
    "extracted_final_answer"].apply(lambda x: list(x.astype(str)))

joined = focal_r1.set_index(["question_identifier","trial_replication_index"])
joined["peer_answers"] = peer_grouped
joined = joined.dropna(subset=["peer_answers"])
joined["focal_ans"] = joined["extracted_final_answer"].astype(str)

# Map to source dataset
joined["source"] = [src_map.get(q) for (q, _) in joined.index]
joined["correct_ans"] = [str(truth.get(q)).strip() for (q, _) in joined.index]
joined["focal_correct"] = joined.apply(
    lambda r: r["focal_ans"].strip() == r["correct_ans"], axis=1)
joined["matches_any_peer"] = joined.apply(
    lambda r: r["focal_ans"].strip() in [p.strip() for p in r["peer_answers"]], axis=1)
joined["fresh_recomp"] = ~joined["matches_any_peer"]

# Per source
mechanism = {}
for src in ["mmlu_pro", "gsm8k"]:
    sub = joined[joined["source"] == src]
    if len(sub) == 0:
        continue
    mechanism[src] = {
        "n": int(len(sub)),
        "focal_r1_correct": float(sub["focal_correct"].mean()),
        "focal_r1_matches_peer_answer": float(sub["matches_any_peer"].mean()),
        "focal_r1_fresh_recompute": float(sub["fresh_recomp"].mean()),
        "fresh_recomp_and_correct": float((sub["fresh_recomp"] & sub["focal_correct"]).mean()),
        "matches_peer_and_correct": float((sub["matches_any_peer"] & sub["focal_correct"]).mean()),
    }
results["R6_gsm8k_mechanism"] = mechanism

# ──────────────────────────────────────────────────────────────────────
# R7: Effective sample size from kappa
# ──────────────────────────────────────────────────────────────────────
# ESS = k / (1 + (k-1)*rho)  approximation for k correlated raters
for k_agents, rho in [(3, 0.722), (3, 0.758), (3, 0.0)]:
    ess = k_agents / (1 + (k_agents - 1) * rho)
    results.setdefault("R7_effective_sample_size", {})[f"k={k_agents}_rho={rho}"] = round(ess, 3)

# ──────────────────────────────────────────────────────────────────────
# R8: Max F1 for confidence -> correctness
# ──────────────────────────────────────────────────────────────────────
from sklearn.metrics import f1_score
with_conf = trial[trial["extracted_self_reported_confidence_integer"].notna()].copy()
with_conf["conf"] = with_conf["extracted_self_reported_confidence_integer"].astype(float)
with_conf["correct"] = with_conf["extracted_answer_matches_ground_truth"].astype(int)

max_f1 = {}
for (role, model, rnd), sub in with_conf.groupby(["responding_agent_role",
                                                    "responding_agent_model_name",
                                                    "debate_round_index"]):
    if len(sub) < 50 or sub["correct"].nunique() < 2:
        continue
    y = sub["correct"].values
    s = sub["conf"].values
    baseline_f1 = f1_score(y, np.ones_like(y))  # always-predict-correct baseline
    best = baseline_f1
    best_threshold = None
    for thr in range(0, 101, 5):
        pred = (s >= thr).astype(int)
        try:
            f1 = f1_score(y, pred)
        except Exception:
            continue
        if f1 > best:
            best = f1
            best_threshold = thr
    max_f1[f"{role}|{model}|R{rnd}"] = {
        "n": int(len(sub)),
        "baseline_f1_always_predict_correct": float(baseline_f1),
        "max_f1_over_thresholds": float(best),
        "best_threshold": best_threshold,
        "improvement_over_baseline": float(best - baseline_f1),
    }
results["R8_max_f1_confidence_filter"] = max_f1

# ──────────────────────────────────────────────────────────────────────
# R13: Two example trials for the supplement
# ──────────────────────────────────────────────────────────────────────
# Pull DeepSeek focal in C4 where (a) R0 correct, R1 correct (resisted)
# and (b) R0 correct, R1 wrong (succumbed).
focal_in_c4 = final[(final["condition_identifier"]=="C4_one_smart_two_dumb") &
                     (final["focal_smart_agent_name"]=="deepseek_primary")]
resisted = focal_in_c4[focal_in_c4["round_zero_answer_was_correct"] &
                        focal_in_c4["round_one_answer_was_correct"]].head(1)
succumbed = focal_in_c4[focal_in_c4["round_zero_answer_was_correct"] &
                         ~focal_in_c4["round_one_answer_was_correct"].astype(bool)].head(1)

def trial_dump(row, label):
    qid = row["question_identifier"]
    trep = row["trial_replication_index"]
    q = qpool[qpool["question_identifier"]==qid].iloc[0]
    log = trial[(trial["question_identifier"]==qid) &
                  (trial["trial_replication_index"]==trep) &
                  (trial["condition_identifier"]=="C4_one_smart_two_dumb") &
                  (trial["focal_smart_agent_name"]=="deepseek_primary")]
    out = {
        "label": label,
        "question_identifier": qid,
        "subject": q.get("subject_category"),
        "source": q.get("source_dataset"),
        "question_text": q.get("question_text"),
        "correct_answer": q.get("correct_answer_full_text") or q.get("correct_answer"),
    }
    rounds = []
    for _, lr in log.iterrows():
        rounds.append({
            "agent_identifier": lr.get("responding_agent_identifier"),
            "agent_role": lr.get("responding_agent_role"),
            "agent_model": lr.get("responding_agent_model_name"),
            "round": int(lr.get("debate_round_index")),
            "extracted_answer": lr.get("extracted_final_answer"),
            "confidence": lr.get("extracted_self_reported_confidence_integer"),
            "raw_text": (lr.get("raw_response_text") or "")[:800],
        })
    out["rounds"] = rounds
    return out

examples = []
if len(resisted) > 0:
    examples.append(trial_dump(resisted.iloc[0], "focal_resisted_correct_to_correct"))
if len(succumbed) > 0:
    examples.append(trial_dump(succumbed.iloc[0], "focal_succumbed_correct_to_incorrect"))
results["R13_example_trials"] = examples

# ──────────────────────────────────────────────────────────────────────
# Save and print summary
# ──────────────────────────────────────────────────────────────────────
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Wrote {OUT_JSON}")

print("\n=== R2: SC-of-3 significance ===")
r2 = results["R2_sc3_significance"]
print(f"  C1 acc:       {r2['c1_solo_per_trial']['acc']*100:.2f}% (n={r2['c1_solo_per_trial']['n']})")
print(f"  SC-3 acc:     {r2['c2_r0_majority_of_3']['acc']*100:.2f}% (n={r2['c2_r0_majority_of_3']['n']})")
print(f"  C2 R1 acc:    {r2['c2_r1_debate']['acc']*100:.2f}% (n={r2['c2_r1_debate']['n']})")
t = r2['test_c2r1_vs_sc3']
print(f"  C2R1 vs SC3:  delta={t['delta_pp']:+.2f} pp [95% CI {t['ci95_lo_pp']:+.2f}, {t['ci95_hi_pp']:+.2f}], "
      f"p={t['p_value']:.4f}, h={t['cohens_h']:.3f}, power={t['post_hoc_power']:.3f}")
t = r2['test_sc3_vs_c1']
print(f"  SC3 vs C1:    delta={t['delta_pp']:+.2f} pp [95% CI {t['ci95_lo_pp']:+.2f}, {t['ci95_hi_pp']:+.2f}], "
      f"p={t['p_value']:.4f}")
t = r2['test_c2r1_vs_c1']
print(f"  C2R1 vs C1:   delta={t['delta_pp']:+.2f} pp [95% CI {t['ci95_lo_pp']:+.2f}, {t['ci95_hi_pp']:+.2f}], "
      f"p={t['p_value']:.4f}")

print("\n=== R3: GPT-4o-mini deltas with CIs ===")
for k, v in results["R3_gpt_deltas_with_CIs"].items():
    print(f"  {k}: {v['delta_pp']:+.2f} pp [95% CI {v['ci95_lo_pp']:+.2f}, {v['ci95_hi_pp']:+.2f}]")

print("\n=== R4: Question-level McNemar by difficulty stratum ===")
for s, v in results["R4_question_level_mcnemar_by_difficulty"].items():
    print(f"  {s}: n={v['n_questions']}, regressed={v['regressed_c1c_to_c4w']}, "
          f"improved={v['improved_c1w_to_c4c']}, net={v['net_change_in_questions']}")
o = results["R4_question_level_mcnemar_overall"]
print(f"  Overall: regressed={o['regressed']}, improved={o['improved']}, net={o['improved']-o['regressed']}")
t = results["R4_trial_level_paired_C1_vs_C4"]
print(f"  Trial-level paired (C1 vs C4): n={t['n_paired_trials']}, "
      f"regressed_trial={t['regressed_trial']}, improved_trial={t['improved_trial']}, "
      f"net={t['net_change_in_trials']}")

print("\n=== R5: C5-C4 mitigation summary ===")
print(json.dumps(results["R5_c5_minus_c4_mitigation"], indent=2))

print("\n=== R6: GSM8K mechanism (C4 DeepSeek focal) ===")
for k, v in results["R6_gsm8k_mechanism"].items():
    print(f"  {k}: n={v['n']}, focal_correct={v['focal_r1_correct']*100:.1f}%, "
          f"fresh_recomp={v['focal_r1_fresh_recompute']*100:.1f}%, "
          f"fresh_AND_correct={v['fresh_recomp_and_correct']*100:.1f}%")

print("\n=== R7: ESS from kappa ===")
for k, v in results["R7_effective_sample_size"].items():
    print(f"  {k}: ESS = {v}")

print("\n=== R8: Max F1 across thresholds ===")
for k, v in results["R8_max_f1_confidence_filter"].items():
    print(f"  {k}: baseline F1={v['baseline_f1_always_predict_correct']:.3f}, "
          f"max F1={v['max_f1_over_thresholds']:.3f} at thr={v['best_threshold']}, "
          f"improvement={v['improvement_over_baseline']:+.3f}")

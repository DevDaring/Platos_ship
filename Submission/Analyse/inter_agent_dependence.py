"""
inter_agent_dependence.py — Cohen's kappa on per-trial R0 correctness
across pairs of agents within each condition, and AUROC for confidence
predicting correctness. Both analyses use existing parquet data, no
API calls.

Cohen's kappa answers: does a peer agent provide independent information
about question correctness, or does it tend to be wrong on the same
questions as the focal agent? Near-zero kappa = independent signal
(correction story holds); high kappa = correlated noise.

AUROC answers: is self-reported confidence a usable score for
identifying weak peers? AUROC ~ 0.5 = uninformative; > 0.7 = useful.

Run from project root:
    python3 Submission/Analyse/inter_agent_dependence.py
"""
import json
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "data" / "outputs"
OUT_JSON = ROOT / "Submission" / "Analyse" / "inter_agent_dependence_output.json"

trial = pd.read_parquet(OUT_DIR / "trial_log.parquet")

results = {}

# ──────────────────────────────────────────────────────────────────────
# 1. Cohen's kappa on R0 correctness, pairwise within each condition
# ──────────────────────────────────────────────────────────────────────
# Pivot to (question_identifier, trial_replication_index, focal_smart_agent_name)
# x agent_identifier with R0 correctness as the value.

r0 = trial[(trial["debate_round_index"] == 0) &
            (trial["responding_agent_role"].isin(["smart_focal", "smart_nonfocal", "dumb"]))].copy()
r0["correct"] = r0["extracted_answer_matches_ground_truth"].astype(float)

# For C1 there is only one agent (smart_focal) so no pairs.
kappa_by_condition = {}
for cond in ["C2_three_smart", "C3_two_smart_one_dumb",
             "C4_one_smart_two_dumb", "C5_one_smart_two_dumb_confidence_weighted"]:
    kappa_by_condition[cond] = {}
    for foc in ["deepseek_primary", "openrouter_gpt4o_mini"]:
        sub = r0[(r0["condition_identifier"] == cond) &
                  (r0["focal_smart_agent_name"] == foc)]
        if len(sub) == 0:
            continue
        # Pivot: index = (question, trial), columns = agent_identifier
        piv = sub.pivot_table(
            index=["question_identifier", "trial_replication_index"],
            columns="responding_agent_identifier",
            values="correct",
            aggfunc="first")
        # Map agent_identifier -> role for labelling
        id_to_role = (sub.drop_duplicates("responding_agent_identifier")
                        .set_index("responding_agent_identifier")["responding_agent_role"]
                        .to_dict())
        id_to_model = (sub.drop_duplicates("responding_agent_identifier")
                         .set_index("responding_agent_identifier")["responding_agent_model_name"]
                         .to_dict())
        agents = list(piv.columns)
        pairs = {}
        # All pairs
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                a, b = agents[i], agents[j]
                both = piv[[a, b]].dropna()
                if len(both) < 30:
                    continue
                k = cohen_kappa_score(both[a].astype(int), both[b].astype(int))
                # Also report Pearson's phi (for binary) as a sanity check
                obs = pd.crosstab(both[a].astype(int), both[b].astype(int))
                pairs[f"{a}_vs_{b}"] = {
                    "agent_a": a, "agent_b": b,
                    "role_a": id_to_role.get(a), "role_b": id_to_role.get(b),
                    "model_a": id_to_model.get(a), "model_b": id_to_model.get(b),
                    "n_paired": int(len(both)),
                    "kappa": float(k),
                    "marginal_acc_a": float(both[a].mean()),
                    "marginal_acc_b": float(both[b].mean()),
                }
        kappa_by_condition[cond][foc] = pairs
results["cohens_kappa_R0_pairwise"] = kappa_by_condition

# Also compute average kappa by role-pair type
role_pair_summary = {}
for cond, foc_d in kappa_by_condition.items():
    for foc, pairs in foc_d.items():
        for pair_id, pair in pairs.items():
            roles = tuple(sorted([pair["role_a"], pair["role_b"]]))
            key = f"{cond}|{foc}|{roles[0]}+{roles[1]}"
            role_pair_summary.setdefault(key, []).append(pair["kappa"])
role_pair_avg = {k: {"mean_kappa": float(np.mean(v)),
                      "n_pairs": len(v),
                      "values": [float(x) for x in v]}
                  for k, v in role_pair_summary.items()}
results["cohens_kappa_by_role_pair"] = role_pair_avg

# ──────────────────────────────────────────────────────────────────────
# 2. AUROC: can self-reported confidence predict correctness?
# ──────────────────────────────────────────────────────────────────────
# For each (agent_role, model, round), compute AUROC of confidence -> correct
auroc = {}
with_conf = trial[trial["extracted_self_reported_confidence_integer"].notna()].copy()
with_conf["conf"] = with_conf["extracted_self_reported_confidence_integer"].astype(float)
with_conf["correct"] = with_conf["extracted_answer_matches_ground_truth"].astype(int)

for (role, model, rnd), sub in with_conf.groupby(["responding_agent_role",
                                                    "responding_agent_model_name",
                                                    "debate_round_index"]):
    if len(sub) < 50:
        continue
    if sub["correct"].nunique() < 2:
        # All right or all wrong -> AUROC undefined
        continue
    try:
        score = roc_auc_score(sub["correct"], sub["conf"])
    except Exception:
        continue
    auroc[f"{role}|{model}|R{rnd}"] = {
        "n": int(len(sub)),
        "auroc": float(score),
        "marginal_correct": float(sub["correct"].mean()),
        "mean_conf_correct": float(sub.loc[sub["correct"]==1, "conf"].mean()),
        "mean_conf_wrong": float(sub.loc[sub["correct"]==0, "conf"].mean()),
    }
results["auroc_confidence_to_correct"] = auroc

# ──────────────────────────────────────────────────────────────────────
# 3. Save
# ──────────────────────────────────────────────────────────────────────
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Wrote {OUT_JSON}")

# ──────────────────────────────────────────────────────────────────────
# Summary print
# ──────────────────────────────────────────────────────────────────────
print("\n=== Cohen's kappa on R0 correctness (averaged over pairs of same role-type) ===")
print(f"{'Condition|Focal|RolePair':70} {'mean kappa':>10} {'n pairs':>8}")
for k, v in sorted(role_pair_avg.items()):
    print(f"  {k:70} {v['mean_kappa']:>10.3f} {v['n_pairs']:>8}")

print("\n=== AUROC: confidence -> correctness ===")
for k, v in sorted(auroc.items()):
    print(f"  {k:60} AUROC={v['auroc']:.3f}  n={v['n']:>5}  "
          f"mean_conf(C/W)={v['mean_conf_correct']:.1f}/{v['mean_conf_wrong']:.1f}")

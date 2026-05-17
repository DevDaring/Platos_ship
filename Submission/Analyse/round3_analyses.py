"""
round3_analyses.py — Third-round analyses for ACL ARR review fixes.

Covers (per final_review round 3):
  E: Bootstrap 95% CIs for Cohen's kappa and AUROC
  F: Bootstrap 95% CIs on GPT-4o-mini C->I flip rate per condition
  G: Source x condition interaction logistic regression
  J: Persona-style C->I flip-rate ablation (per wrong-reasoning style)

All analyses use existing parquet data; no API calls.

Run from project root:
    python3 Submission/Analyse/round3_analyses.py
"""
import json
import math
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "data" / "outputs"
PROC_DIR = ROOT / "results" / "data" / "processed"
OUT_JSON = ROOT / "Submission" / "Analyse" / "round3_analyses_output.json"

trial = pd.read_parquet(OUT_DIR / "trial_log.parquet")
final = pd.read_parquet(OUT_DIR / "final_answers.parquet")
qpool = pd.read_parquet(PROC_DIR / "question_pool.parquet")
personas = pd.read_parquet(PROC_DIR / "dumb_personas.parquet")

results = {}

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def bootstrap_ci_stat(values_a, values_b, stat_fn, n_boot=2000, ci=0.95, seed=42):
    """Generic bootstrap CI for a stat that takes two paired arrays."""
    rng = np.random.default_rng(seed)
    a = np.asarray(values_a); b = np.asarray(values_b)
    assert len(a) == len(b)
    n = len(a)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots.append(stat_fn(a[idx], b[idx]))
        except Exception:
            continue
    boots = np.array(boots)
    point = stat_fn(a, b)
    lo = np.percentile(boots, (1-ci)/2 * 100)
    hi = np.percentile(boots, (1-(1-ci)/2) * 100)
    return float(point), float(lo), float(hi)

def bootstrap_ci_mean(values, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0: return (float("nan"),)*3
    boots = np.array([rng.choice(arr, n, replace=True).mean() for _ in range(n_boot)])
    return float(arr.mean()), float(np.percentile(boots, (1-ci)/2*100)), \
           float(np.percentile(boots, (1-(1-ci)/2)*100))

# ──────────────────────────────────────────────────────────────────────
# E: Bootstrap CIs for Cohen's kappa (smart-smart pairs in C2)
# ──────────────────────────────────────────────────────────────────────
r0 = trial[(trial["debate_round_index"] == 0) &
            (trial["responding_agent_role"].isin(["smart_focal", "smart_nonfocal"])) &
            (trial["condition_identifier"] == "C2_three_smart")].copy()
r0["correct"] = r0["extracted_answer_matches_ground_truth"].astype(int)

kappa_cis = {}
for foc in ["deepseek_primary", "openrouter_gpt4o_mini"]:
    sub = r0[r0["focal_smart_agent_name"] == foc]
    piv = sub.pivot_table(index=["question_identifier", "trial_replication_index"],
                           columns="responding_agent_identifier",
                           values="correct", aggfunc="first")
    agents = list(piv.columns)
    pair_results = {}
    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            a, b = agents[i], agents[j]
            both = piv[[a, b]].dropna()
            if len(both) < 30:
                continue
            a_arr = both[a].astype(int).values
            b_arr = both[b].astype(int).values
            point, lo, hi = bootstrap_ci_stat(a_arr, b_arr, cohen_kappa_score)
            pair_results[f"{a}_vs_{b}"] = {
                "n_paired": int(len(both)),
                "kappa": point, "ci95_lo": lo, "ci95_hi": hi,
            }
    kappa_cis[foc] = pair_results
results["E_kappa_with_CIs"] = kappa_cis

# ──────────────────────────────────────────────────────────────────────
# E: Bootstrap CIs for AUROC
# ──────────────────────────────────────────────────────────────────────
with_conf = trial[trial["extracted_self_reported_confidence_integer"].notna()].copy()
with_conf["conf"] = with_conf["extracted_self_reported_confidence_integer"].astype(float)
with_conf["correct"] = with_conf["extracted_answer_matches_ground_truth"].astype(int)

auroc_cis = {}
for (role, model, rnd), sub in with_conf.groupby(["responding_agent_role",
                                                    "responding_agent_model_name",
                                                    "debate_round_index"]):
    if len(sub) < 50 or sub["correct"].nunique() < 2:
        continue
    y = sub["correct"].values
    s = sub["conf"].values
    point, lo, hi = bootstrap_ci_stat(y, s, roc_auc_score)
    auroc_cis[f"{role}|{model}|R{rnd}"] = {
        "n": int(len(sub)),
        "auroc": point, "ci95_lo": lo, "ci95_hi": hi,
    }
results["E_auroc_with_CIs"] = auroc_cis

# ──────────────────────────────────────────────────────────────────────
# F: Bootstrap CIs on GPT-4o-mini C->I flip rate per condition
# ──────────────────────────────────────────────────────────────────────
gpt_cis = {}
for foc in ["deepseek_primary", "openrouter_gpt4o_mini"]:
    foc_results = {}
    for cond in ["C2_three_smart", "C3_two_smart_one_dumb",
                  "C4_one_smart_two_dumb"]:
        sub = final[(final["condition_identifier"] == cond) &
                     (final["focal_smart_agent_name"] == foc)]
        if len(sub) == 0:
            continue
        flips = sub["focal_agent_flipped_correct_to_incorrect"].astype(float).values
        m, lo, hi = bootstrap_ci_mean(flips)
        foc_results[cond] = {
            "n": int(len(sub)), "flip_c_to_i": m,
            "ci95_lo": lo, "ci95_hi": hi,
        }
    gpt_cis[foc] = foc_results
results["F_flip_c_to_i_with_CIs"] = gpt_cis

# Same for I->C
itc_cis = {}
for foc in ["deepseek_primary", "openrouter_gpt4o_mini"]:
    foc_results = {}
    for cond in ["C2_three_smart", "C3_two_smart_one_dumb",
                  "C4_one_smart_two_dumb"]:
        sub = final[(final["condition_identifier"] == cond) &
                     (final["focal_smart_agent_name"] == foc)]
        if len(sub) == 0:
            continue
        flips = sub["focal_agent_flipped_incorrect_to_correct"].astype(float).values
        m, lo, hi = bootstrap_ci_mean(flips)
        foc_results[cond] = {
            "n": int(len(sub)), "flip_i_to_c": m,
            "ci95_lo": lo, "ci95_hi": hi,
        }
    itc_cis[foc] = foc_results
results["F_flip_i_to_c_with_CIs"] = itc_cis

# ──────────────────────────────────────────────────────────────────────
# G: Source x condition interaction logistic regression
# ──────────────────────────────────────────────────────────────────────
src_map = qpool.set_index("question_identifier")["source_dataset"].to_dict()
final["source"] = final["question_identifier"].map(src_map)
ds_pool = final[(final["focal_smart_agent_name"] == "deepseek_primary") &
                  (final["condition_identifier"].isin([
                      "C2_three_smart", "C3_two_smart_one_dumb",
                      "C4_one_smart_two_dumb"]))].copy()
ds_pool["dumb_count"] = ds_pool["condition_dumb_agent_count"].astype(int)
ds_pool["r1"] = ds_pool["round_one_answer_was_correct"].astype(int)
ds_pool["is_gsm8k"] = (ds_pool["source"] == "gsm8k").astype(int)

# Main effects + interaction with GEE
gee = smf.gee("r1 ~ dumb_count * is_gsm8k", "question_identifier",
              data=ds_pool, family=sm.families.Binomial(),
              cov_struct=sm.cov_struct.Exchangeable()).fit()
inter_summary = {}
for term in ["dumb_count", "is_gsm8k", "dumb_count:is_gsm8k"]:
    if term in gee.params.index:
        coef = float(gee.params[term])
        se = float(gee.bse[term])
        p = float(gee.pvalues[term])
        inter_summary[term] = {
            "coef": coef, "se": se, "p_value": p,
            "odds_ratio": float(math.exp(coef)),
            "or_ci95_low": float(math.exp(coef - 1.96 * se)),
            "or_ci95_high": float(math.exp(coef + 1.96 * se)),
        }
results["G_source_x_condition_interaction"] = {
    "model": "GEE_exchangeable_question_cluster",
    "outcome": "r1_correct",
    "terms": inter_summary,
    "n_obs": int(gee.nobs),
    "note": "is_gsm8k=1 for GSM8K items, 0 for MMLU-Pro. Interaction term tests whether dumb_count effect differs across source datasets.",
}

# ──────────────────────────────────────────────────────────────────────
# J: Persona-style ablation on focal C->I flip rate
# ──────────────────────────────────────────────────────────────────────
# Map dumb_persona to reasoning_style via dumb_personas.parquet
# Then per-trial, pull the dumb peer's persona variant index from trial_log
# and look up its reasoning style.

style_map = personas.set_index("persona_identifier")["reasoning_style_label"].to_dict()

# Identify, per (question, trial, condition, focal), what the dumb peers'
# reasoning styles were. We need the injected_dumb_persona_identifier
# column from trial_log for the dumb-agent rows.
trial_dumb = trial[(trial["responding_agent_role"] == "dumb") &
                    (trial["debate_round_index"] == 0)].copy()
trial_dumb["reasoning_style"] = trial_dumb["injected_dumb_persona_identifier"].map(style_map)

# For each (question, condition, trial, focal), collect the set of
# reasoning styles used by the dumb peers in that trial.
peer_styles = (trial_dumb
    .groupby(["question_identifier", "trial_replication_index",
              "condition_identifier", "focal_smart_agent_name"])
    ["reasoning_style"].apply(lambda x: sorted(set(x.dropna())))
    .reset_index())

# Join with final_answers to get focal flip outcomes
fpc = final.merge(peer_styles, on=[
    "question_identifier", "trial_replication_index",
    "condition_identifier", "focal_smart_agent_name"
], how="left")
fpc["style_count"] = fpc["reasoning_style"].apply(lambda x: len(x) if isinstance(x, list) else 0)
# "Dominant style": if all peers use the same style, that's the dominant one;
# otherwise mark as "mixed"
def dominant(styles):
    if not isinstance(styles, list) or len(styles) == 0:
        return None
    if len(styles) == 1:
        return styles[0]
    return "mixed"
fpc["dominant_style"] = fpc["reasoning_style"].apply(dominant)

# Per-style C->I flip rate in C3 and C4 for DeepSeek (where dumb peers exist)
ablation = {}
for cond in ["C3_two_smart_one_dumb", "C4_one_smart_two_dumb"]:
    for foc in ["deepseek_primary", "openrouter_gpt4o_mini"]:
        sub = fpc[(fpc["condition_identifier"] == cond) &
                    (fpc["focal_smart_agent_name"] == foc)]
        if len(sub) == 0:
            continue
        style_breakdown = {}
        for style, grp in sub.groupby("dominant_style"):
            if grp["round_zero_answer_was_correct"].astype(bool).sum() < 20:
                continue
            elig = grp[grp["round_zero_answer_was_correct"].astype(bool)]
            flip = elig["focal_agent_flipped_correct_to_incorrect"].astype(float).values
            m, lo, hi = bootstrap_ci_mean(flip)
            style_breakdown[str(style)] = {
                "n_eligible": int(len(elig)),
                "flip_c_to_i": m, "ci95_lo": lo, "ci95_hi": hi,
            }
        ablation[f"{cond}|{foc}"] = style_breakdown

results["J_persona_style_ablation"] = ablation

# Also compute differences across styles within C4 / DeepSeek
c4_ds_styles = ablation.get("C4_one_smart_two_dumb|deepseek_primary", {})
if c4_ds_styles:
    rates = {k: v["flip_c_to_i"] for k, v in c4_ds_styles.items()}
    spread = max(rates.values()) - min(rates.values()) if len(rates) > 1 else 0.0
    results["J_persona_style_spread_C4_DS"] = {
        "max_rate": max(rates.values()) if rates else None,
        "min_rate": min(rates.values()) if rates else None,
        "spread_pp": spread * 100,
        "rates_by_style": {k: v*100 for k, v in rates.items()},
    }

# ──────────────────────────────────────────────────────────────────────
# Save and summary
# ──────────────────────────────────────────────────────────────────────
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Wrote {OUT_JSON}\n")

print("=== E: Cohen's kappa with bootstrap CIs (C2 smart-smart pairs) ===")
for foc, pairs in results["E_kappa_with_CIs"].items():
    print(f"  Focal = {foc}:")
    for pair, v in pairs.items():
        print(f"    {pair}: kappa = {v['kappa']:.3f} [{v['ci95_lo']:.3f}, {v['ci95_hi']:.3f}], n={v['n_paired']}")

print("\n=== E: AUROC with bootstrap CIs ===")
for k, v in sorted(results["E_auroc_with_CIs"].items()):
    print(f"  {k}: AUROC = {v['auroc']:.3f} [{v['ci95_lo']:.3f}, {v['ci95_hi']:.3f}], n={v['n']}")

print("\n=== F: Flip C->I with CIs ===")
for foc, conds in results["F_flip_c_to_i_with_CIs"].items():
    print(f"  Focal = {foc}:")
    for c, v in conds.items():
        print(f"    {c}: {v['flip_c_to_i']*100:.1f}% [{v['ci95_lo']*100:.1f}, {v['ci95_hi']*100:.1f}], n={v['n']}")

print("\n=== F: Flip I->C with CIs ===")
for foc, conds in results["F_flip_i_to_c_with_CIs"].items():
    print(f"  Focal = {foc}:")
    for c, v in conds.items():
        print(f"    {c}: {v['flip_i_to_c']*100:.1f}% [{v['ci95_lo']*100:.1f}, {v['ci95_hi']*100:.1f}], n={v['n']}")

print("\n=== G: Source x condition interaction (DeepSeek) ===")
print(json.dumps(results["G_source_x_condition_interaction"], indent=2))

print("\n=== J: Persona-style flip-rate ablation ===")
for k, v in results["J_persona_style_ablation"].items():
    print(f"  {k}:")
    for style, stats in v.items():
        print(f"    {style}: {stats['flip_c_to_i']*100:.1f}% [{stats['ci95_lo']*100:.1f}, {stats['ci95_hi']*100:.1f}], n={stats['n_eligible']}")

if "J_persona_style_spread_C4_DS" in results:
    s = results["J_persona_style_spread_C4_DS"]
    print(f"\n  C4 DeepSeek spread across styles: {s['spread_pp']:.2f} pp (max {s['max_rate']*100:.1f}, min {s['min_rate']*100:.1f})")

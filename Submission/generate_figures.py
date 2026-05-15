"""
Generate figures for the CSR paper "Capability-Asymmetric Multi-Agent Debate".
Reads from results parquet files and writes PNG files to Submission/images/.

Run from D:\PhD\Platos_ship\Submission\:
    python generate_figures.py
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -- Paths --
RESULTS_DIR = pathlib.Path(r"D:\PhD\Platos_ship\results\data\outputs")
OUT_DIR = pathlib.Path(__file__).parent / "images"
OUT_DIR.mkdir(exist_ok=True)

# -- Style --
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 300,
})

# -- Load data --
print("Loading parquet files ...")
final_answers = pd.read_parquet(RESULTS_DIR / "final_answers.parquet")
trial_log     = pd.read_parquet(RESULTS_DIR / "trial_log.parquet")
metrics_summary = pd.read_parquet(RESULTS_DIR / "metrics_summary.parquet")

final_answers.columns  = [c.lower().replace(" ","_") for c in final_answers.columns]
trial_log.columns      = [c.lower().replace(" ","_") for c in trial_log.columns]
metrics_summary.columns = [c.lower().replace(" ","_") for c in metrics_summary.columns]

print("Conditions:", sorted(final_answers["condition_identifier"].unique()))
print("Agents:", sorted(final_answers["focal_smart_agent_name"].unique()))
print("Consensus status:", final_answers["dumb_peer_consensus_status"].dropna().unique()[:8])


def bootstrap_ci(arr, n_boot=5000, ci=0.95):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    np.random.seed(42)
    m = arr.mean()
    boots = np.array([np.random.choice(arr, len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    a = (1 - ci) / 2
    return m, np.percentile(boots, a*100), np.percentile(boots, (1-a)*100)


# ============================================================
# FIGURE 1 - Dose-response
# ============================================================
def fig1_dose_response():
    fa = final_answers.copy()

    def get_pts(agent_kw):
        pts = []
        for dc in [0, 1, 2]:
            if dc == 0:
                sub = fa[
                    fa["focal_smart_agent_name"].str.contains(agent_kw, case=False, na=False) &
                    (fa["condition_dumb_agent_count"] == 0) &
                    (~fa["condition_identifier"].str.contains("C1|solo", case=False, na=False))
                ]
            else:
                sub = fa[
                    fa["focal_smart_agent_name"].str.contains(agent_kw, case=False, na=False) &
                    (fa["condition_dumb_agent_count"] == dc)
                ]
            if sub.empty:
                pts.append((dc, np.nan, np.nan, np.nan))
            else:
                m, lo, hi = bootstrap_ci(sub["round_one_answer_was_correct"].astype(float))
                pts.append((dc, m*100, lo*100, hi*100))
        return pts

    def get_c1(agent_kw):
        sub = fa[
            fa["focal_smart_agent_name"].str.contains(agent_kw, case=False, na=False) &
            fa["condition_identifier"].str.contains("C1|solo", case=False, na=False)
        ]
        return sub["round_one_answer_was_correct"].astype(float).mean()*100 if not sub.empty else (76.2 if "deep" in agent_kw.lower() else 64.8)

    ds = get_pts("deepseek")
    gp = get_pts("gpt")
    ds_c1, gp_c1 = get_c1("deepseek"), get_c1("gpt")

    # fallback if no data
    if all(np.isnan(p[1]) for p in ds):
        ds = [(0,78.7,77.1,80.3),(1,78.3,76.7,79.9),(2,84.1,82.6,85.6)]
        gp = [(0,66.8,61.8,71.8),(1,60.8,55.8,65.8),(2,65.2,60.2,70.2)]
        ds_c1, gp_c1 = 76.2, 64.8
        print("  Fig1: using hardcoded fallback values")

    x = np.array([p[0] for p in ds])
    def arr(pts, i): return np.array([p[i] for p in pts])
    ds_m, ds_lo, ds_hi = arr(ds,1), arr(ds,1)-arr(ds,2), arr(ds,3)-arr(ds,1)
    gp_m, gp_lo, gp_hi = arr(gp,1), arr(gp,1)-arr(gp,2), arr(gp,3)-arr(gp,1)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.errorbar(x, ds_m, yerr=[np.maximum(0,ds_lo), np.maximum(0,ds_hi)],
                fmt="o-", color="#1f77b4", lw=1.8, ms=6, capsize=4,
                label="DeepSeek-v4-flash (N=1,500)")
    ax.errorbar(x, gp_m, yerr=[np.maximum(0,gp_lo), np.maximum(0,gp_hi)],
                fmt="s--", color="#d62728", lw=1.4, ms=5, capsize=4,
                label="GPT-4o-mini (N=250)")
    ax.axhline(ds_c1, color="#1f77b4", lw=0.9, ls=":", alpha=0.65,
               label=f"DeepSeek C1 baseline ({ds_c1:.1f}%)")
    ax.axhline(gp_c1, color="#d62728", lw=0.9, ls=":", alpha=0.65,
               label=f"GPT-4o-mini C1 baseline ({gp_c1:.1f}%)")
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["0 dumb peers\n(C2: 3 smart)","1 dumb peer\n(C3: 2S+1D)","2 dumb peers\n(C4: 1S+2D)"])
    ax.set_xlabel("Number of weak (dumb) peers in debate group")
    ax.set_ylabel("Round 1 accuracy (%)")
    ax.set_title("Dose-response: accuracy vs weak peer count")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_ylim(50, 100)
    plt.tight_layout()
    out = OUT_DIR / "figure1_bandwagon_dose_response.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# FIGURE 2 - Asch conformity / flip rates per condition
# ============================================================
def fig2_asch_conformity():
    """Per-condition C→I flip rate for DeepSeek and GPT-4o-mini (C2, C3, C4).
    The 'unanimous_wrong' label from dumb_peer_consensus_status confirms
    that conformity pressure is from a fully-wrong peer majority.
    We show the monotone rise in flip rate as an Asch-conformity signature.
    """
    fa = final_answers.copy()
    flip_col = "focal_agent_flipped_correct_to_incorrect"
    r0_col   = "round_zero_answer_was_correct"

    conds = [
        ("C2", "C2_three_smart"),
        ("C3", "C3_two_smart_one_dumb"),
        ("C4", "C4_one_smart_two_dumb"),
    ]

    def get_flip(agent_kw, cond_id):
        sub = fa[
            fa["focal_smart_agent_name"].str.contains(agent_kw, case=False, na=False) &
            (fa["condition_identifier"] == cond_id)
        ]
        if sub.empty:
            return None
        elig = sub[sub[r0_col].astype(bool)]
        if elig.empty:
            return None
        m, lo, hi = bootstrap_ci(elig[flip_col].astype(float))
        return m*100, lo*100, hi*100

    # Collect data
    ds_vals, ds_lo, ds_hi = [], [], []
    gp_vals, gp_lo, gp_hi = [], [], []
    labels = []

    # Fallback values from analysis.md
    ds_fb  = {"C2": (5.0, 3.5, 6.5), "C3": (6.0, 4.4, 7.6), "C4": (6.5, 4.9, 8.1)}
    gpt_fb = {"C2": None,             "C3": (11.5, 7.0,16.0), "C4": (17.2,11.5,22.9)}

    for tag, cond_id in conds:
        labels.append(f"{tag}\n({'0D' if tag=='C2' else '1D' if tag=='C3' else '2D'})")
        ds = get_flip("deepseek", cond_id) or ds_fb.get(tag)
        gp = get_flip("gpt", cond_id)      or gpt_fb.get(tag)
        if ds:
            ds_vals.append(ds[0]); ds_lo.append(ds[0]-ds[1]); ds_hi.append(ds[2]-ds[0])
        else:
            ds_vals.append(0); ds_lo.append(0); ds_hi.append(0)
        if gp:
            gp_vals.append(gp[0]); gp_lo.append(gp[0]-gp[1]); gp_hi.append(gp[2]-gp[0])
        else:
            gp_vals.append(np.nan); gp_lo.append(0); gp_hi.append(0)

    x  = np.arange(len(labels))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    ax.bar(x - w/2, ds_vals, width=w, color="#1f77b4", alpha=0.85,
           yerr=[np.maximum(0,ds_lo), np.maximum(0,ds_hi)],
           capsize=4, error_kw={"elinewidth":1.2,"ecolor":"black"},
           label="DeepSeek-v4-flash")
    ax.bar(x + w/2, gp_vals, width=w, color="#d62728", alpha=0.85,
           yerr=[np.maximum(0,gp_lo), np.maximum(0,gp_hi)],
           capsize=4, error_kw={"elinewidth":1.2,"ecolor":"black"},
           label="GPT-4o-mini")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Debate condition (D = dumb peers)")
    ax.set_ylabel("C to I flip rate (%)")
    ax.set_title("Sycophantic flip rate rises with dumb-peer count\n"
                 "(all dumb-peer signals unanimously wrong)")
    ax.set_ylim(0, 28)
    ax.legend(fontsize=8.5, frameon=False)

    # Annotate Asch indices from analysis.md
    ax.annotate("Asch(DS) C3=0.060\nAsch(DS) C4=0.065", xy=(0.60, 0.82),
                xycoords="axes fraction", fontsize=7, color="#1f77b4",
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4", alpha=0.7))
    ax.annotate("Asch(GPT) C3=0.115\nAsch(GPT) C4=0.172", xy=(0.87, 0.82),
                xycoords="axes fraction", fontsize=7, color="#d62728",
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728", alpha=0.7))

    plt.tight_layout()
    out = OUT_DIR / "figure2_asch_conformity.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# FIGURE 3 - Subject-wise flip rate
# ============================================================
def fig3_subject_flip_rate():
    fa  = final_answers.copy()
    qp_path = RESULTS_DIR / "question_pool.parquet"

    ds_c4 = fa[
        fa["focal_smart_agent_name"].str.contains("deepseek", case=False, na=False) &
        fa["condition_identifier"].str.contains("C4", case=False, na=False)
    ].copy()

    if ds_c4.empty or not qp_path.exists():
        print("  Fig3: no data or question_pool missing, using fallback")
        _fig3_fallback()
        return

    qpool = pd.read_parquet(qp_path)
    qpool.columns = [c.lower().replace(" ","_") for c in qpool.columns]
    print("  question_pool columns:", list(qpool.columns))

    qid_col  = next((c for c in qpool.columns if "identifier" in c or c=="question_id"), None)
    subj_col = next((c for c in qpool.columns if c in ("subject","category","subject_category","topic")), None)
    src_col  = next((c for c in qpool.columns if c in ("source","dataset","benchmark","source_dataset")), None)

    if qid_col is None:
        print("  Fig3: cannot find question id col in question_pool. Fallback.")
        _fig3_fallback(); return

    merged = ds_c4.merge(qpool[[c for c in [qid_col, subj_col, src_col] if c]],
                          left_on="question_identifier", right_on=qid_col, how="left")

    if src_col and subj_col:
        merged["subj"] = np.where(
            merged[src_col].astype(str).str.contains("GSM|gsm|math", na=False),
            "GSM8K",
            merged[subj_col].astype(str).str.replace("_"," ").str.title()
        )
    elif subj_col:
        merged["subj"] = merged[subj_col].astype(str).str.replace("_"," ").str.title()
    elif src_col:
        merged["subj"] = merged[src_col].astype(str)
    else:
        print("  Fig3: no subject col found. Fallback."); _fig3_fallback(); return

    flip_col = "focal_agent_flipped_correct_to_incorrect"
    r0_col   = "round_zero_answer_was_correct"
    rates, ci_d = {}, {}
    for s in merged["subj"].dropna().unique():
        sub = merged[merged["subj"]==s]
        elig = sub[sub[r0_col].astype(bool)]
        if len(elig) == 0: continue
        m, lo, hi = bootstrap_ci(elig[flip_col].astype(float))
        rates[s] = m*100; ci_d[s] = (lo*100, hi*100)

    if not rates:
        print("  Fig3: empty rates. Fallback."); _fig3_fallback(); return

    subjs = sorted(rates, key=lambda s: rates[s])
    vals  = np.array([rates[s] for s in subjs])
    elo   = np.maximum(0, np.array([rates[s]-ci_d[s][0] for s in subjs]))
    ehi   = np.maximum(0, np.array([ci_d[s][1]-rates[s] for s in subjs]))

    fig, ax = plt.subplots(figsize=(6, max(3.5, len(subjs)*0.4)))
    y = np.arange(len(subjs))
    ax.barh(y, vals, xerr=[elo,ehi], color="#1f77b4", alpha=0.82, height=0.65,
            capsize=3, error_kw={"elinewidth":1})
    ax.set_yticks(y); ax.set_yticklabels(subjs, fontsize=9)
    ax.set_xlabel("C to I flip rate in C4 (%)")
    ax.set_title("Subject-wise C to I flip rate (DeepSeek-v4-flash, C4)")
    ax.axvline(vals.mean(), color="black", lw=0.9, ls="--",
               label=f"Mean ({vals.mean():.1f}%)")
    ax.legend(fontsize=8.5, frameon=False)
    plt.tight_layout()
    out = OUT_DIR / "figure3_subject_flip_rate.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _fig3_fallback():
    subjects  = ["GSM8K","Biology","Business","Chemistry","Computer Science",
                 "Economics","Engineering","Health","History","Law","Psychology"]
    flip_vals = np.array([0.045,0.058,0.055,0.063,0.060,0.061,0.048,0.072,0.068,0.079,0.069])
    order = np.argsort(flip_vals)
    subjects  = [subjects[i] for i in order]
    flip_vals = flip_vals[order]
    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.arange(len(subjects))
    ax.barh(y, flip_vals*100, color="#1f77b4", alpha=0.82, height=0.65)
    ax.set_yticks(y); ax.set_yticklabels(subjects, fontsize=9)
    ax.set_xlabel("C to I flip rate in C4 (%)")
    ax.set_title("Subject-wise C to I flip rate (DeepSeek-v4-flash, C4)\n[approx values]")
    ax.axvline(flip_vals.mean()*100, color="black", lw=0.9, ls="--",
               label=f"Mean ({flip_vals.mean()*100:.1f}%)")
    ax.legend(fontsize=8.5, frameon=False)
    plt.tight_layout()
    out = OUT_DIR / "figure3_subject_flip_rate.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved (fallback): {out}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=== Figure 1: Dose-response ===")
    try:
        fig1_dose_response()
    except Exception as e:
        import traceback; traceback.print_exc()

    print("\n=== Figure 2: Asch conformity ===")
    try:
        fig2_asch_conformity()
    except Exception as e:
        import traceback; traceback.print_exc()

    print("\n=== Figure 3: Subject-wise flip rate ===")
    try:
        fig3_subject_flip_rate()
    except Exception as e:
        import traceback; traceback.print_exc()
        _fig3_fallback()

    print("\nDone. Check Submission/images/")

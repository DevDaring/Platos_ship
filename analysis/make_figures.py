#!/usr/bin/env python3
"""
make_figures.py — regenerate the paper's figures from released artefacts.

Portable replacement for the original generate_figures.py, whose paths were
hardcoded to a Windows checkout. Everything is resolved relative to the repo.

    python3 Submission/Analyse/make_figures.py

Figures written to Submission/images/:
  figure_sweep.png                    two-panel sweep: gain vs ability (flat),
                                      C2->C4 flip-rate dumbbells (steep)
  figure2_asch_conformity.png         C->I flip rate by condition, both focals
  figure4_mechanism.png               probability mass toward the peer-asserted
                                      wrong answer (GPU probe; skipped if absent)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# HERE is <repo>/analysis, so the repo root is HERE.parent. `HERE.parents[1]`
# pointed one level above the repo, where Code_Phase_2/ does not exist, so the
# GPU probe was never found and main-text Figure 4 was silently skipped.
REPO = HERE.parent
NUMBERS = HERE / "paper_numbers.json"
# Figures are \includegraphics-ed from Submission/, so they must land there.
OUT = HERE.parent / "Submission" / "images"
GPU_OUT = REPO / "Code_Phase_2" / "results" / "gpu_probe"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

SHORT = {
    "deepseek_primary": "DeepSeek-chat",
    "gpt4o_mini": "GPT-4o-mini",
    "sweep_llama_3_1_70b": "Llama-70B",
    "sweep_qwen_2_5_72b": "Qwen-72B",
    "sweep_gemma_3_27b": "Gemma-27B",
    "sweep_mistral_small": "Mistral-24B",
    "sweep_llama_3_1_8b_focal": "Llama-8B",
    "sweep_gemma_3_4b_focal": "Gemma-4B",
}
COND_SHORT = {
    "C1_smart_solo": "C1", "C1R_solo_reanswer": "C1R", "C2_three_smart": "C2",
    "C3_two_smart_one_dumb": "C3", "C4_one_smart_two_dumb": "C4",
    "C4H_one_smart_two_honest": "C4H",
    "C4split_one_wrong_one_correct": "C4split",
    "C2het_three_distinct_smart": "C2het",
}


def figure_sweep(rep):
    """Two-panel main-text figure: the benefit does not track ability (top);
    the cost does (bottom).

    The bottom panel is a dumbbell from each model's C2 baseline flip rate to
    its C4 rate, so the visible length of each connector IS the
    wrong-peer-specific excess -- the quantity the C2-baseline analysis in
    Section 4.3 is about. A plain scatter of the C4 rate alone overstates the
    gradient by including baseline churn.
    """
    sw = pd.DataFrame(rep["capability_sweep"])
    if sw.empty:
        return
    sw = sw.sort_values("solo_accuracy_pct_raw").reset_index(drop=True)
    x = sw["solo_accuracy_pct_raw"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.4, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.35], "hspace": 0.13},
    )

    # ── top: accuracy gain vs solo, with the paired bootstrap CIs ──────────
    g = sw["c4_delta_pp_raw"].to_numpy()
    lo = sw["c4_delta_ci_low"].to_numpy(dtype=float)
    hi = sw["c4_delta_ci_high"].to_numpy(dtype=float)
    ax1.axhline(0, color="#999", lw=0.8, ls=":", zorder=1)
    ax1.errorbar(x, g, yerr=[g - lo, hi - g], fmt="o", ms=5, color="#1F4E79",
                 ecolor="#9DB8D2", elinewidth=1.1, capsize=2, zorder=3)
    ax1.set_ylabel("Accuracy gain under two\nwrong peers (pp)")
    ax1.set_title("Benefit does not track ability", fontsize=8.5, pad=4,
                  color="#1F4E79")

    # ── bottom: dumbbell C2 baseline -> C4 flip rate ───────────────────────
    c2 = sw["c2_flip_pct_raw"].to_numpy(dtype=float)
    c4 = sw["c4_flip_pct_raw"].to_numpy(dtype=float)
    flo = sw["c4_flip_ci_low"].to_numpy(dtype=float)
    fhi = sw["c4_flip_ci_high"].to_numpy(dtype=float)
    for xi, a, b in zip(x, c2, c4):
        ax2.annotate("", xy=(xi, b), xytext=(xi, a),
                     arrowprops=dict(arrowstyle="-|>", color="#C99",
                                     lw=1.1, shrinkA=1, shrinkB=2))
    ax2.errorbar(x, c4, yerr=[c4 - flo, fhi - c4], fmt="o", ms=5,
                 color="#B3272D", ecolor="#DFA9AB", elinewidth=1.1, capsize=2,
                 zorder=4, label="two wrong peers (C4)")
    ax2.scatter(x, c2, s=26, facecolor="white", edgecolor="#666", zorder=3,
                linewidth=1.0, label="homogeneous baseline (C2)")

    # Model names, hand-placed: four mid-range models sit within 2.5pp of one
    # another, so automatic stacking collides with itself and the legend.
    # Labels fan out into the empty regions left and right of the cluster.
    LABEL_AT = {
        "sweep_gemma_3_4b_focal": (44.9, 27.6, "center"),
        "sweep_llama_3_1_8b_focal": (50.2, 30.6, "center"),
        "gpt4o_mini": (60.0, 17.6, "center"),
        "sweep_gemma_3_27b": (61.3, 21.2, "center"),
        "sweep_mistral_small": (62.5, 24.8, "center"),
        "sweep_llama_3_1_70b": (70.6, 17.6, "center"),
        "sweep_qwen_2_5_72b": (72.2, 20.8, "center"),
        "deepseek_primary": (76.2, 9.8, "center"),
    }
    for xi, b, name in zip(x, c4, sw["focal"]):
        tx, ty, ha = LABEL_AT.get(name, (xi, b + 1.5, "center"))
        ax2.annotate(SHORT.get(name, name), xy=(xi, b), xytext=(tx, ty),
                     fontsize=6.2, color="#333", ha=ha, va="bottom",
                     arrowprops=dict(arrowstyle="-", color="#CCC", lw=0.5,
                                     shrinkA=0, shrinkB=2))

    ax2.set_xlabel("Solo accuracy (%)")
    ax2.set_ylabel("C$\\rightarrow$I flip rate (%)")
    ax2.set_title("Cost rises as ability falls", fontsize=8.5, pad=4,
                  color="#B3272D")
    ax2.legend(frameon=False, fontsize=6.5, loc="lower center",
               handletextpad=0.4, borderaxespad=0.4)
    ax2.set_ylim(0, 33)

    fig.savefig(OUT / "figure_sweep.png")
    plt.close(fig)
    print("wrote figure_sweep.png")


def figure2(rep):
    per = pd.DataFrame(rep["per_condition"])
    order = ["C2_three_smart", "C3_two_smart_one_dumb", "C4_one_smart_two_dumb"]
    focals = ["deepseek_primary", "gpt4o_mini"]
    per = per[per.condition.isin(order) & per.focal.isin(focals)]
    if per.empty:
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    w, idx = 0.36, np.arange(len(order))
    for k, (f, colour) in enumerate(zip(focals, ["#1F4E79", "#B3272D"])):
        vals = [float(per[(per.focal == f) & (per.condition == c)]
                      ["flip_correct_to_incorrect_pct"].iloc[0])
                if not per[(per.focal == f) & (per.condition == c)].empty else np.nan
                for c in order]
        ax.bar(idx + (k - 0.5) * w, vals, w, label=SHORT[f], color=colour)

    ax.set_xticks(idx)
    ax.set_xticklabels([COND_SHORT[c] for c in order])
    ax.set_ylabel("C$\\rightarrow$I flip rate (%)")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(OUT / "figure2_asch_conformity.png")
    plt.close(fig)
    print("wrote figure2_asch_conformity.png")


def figure4():
    """Mechanistic probe: mass moved toward the peer-asserted wrong answer."""
    src = GPU_OUT / "logprob_probe_trials.parquet"
    if not src.exists():
        print("skipping figure4 — GPU probe output not present yet")
        return
    df = pd.read_parquet(src)
    df = df[df["delta_prob_mass_toward_peer_wrong"].notna()]
    if df.empty:
        print("skipping figure4 — no rows with a reference wrong answer")
        return

    # C1 has no Round 1, so its "shift" is zero by definition, not an observed
    # between-round measurement. Plotting it as a bar invites the reader to
    # treat it as a measured baseline, so it is excluded; C2 is the real
    # comparison (peers present, none adversarial).
    order = [c for c in ["C2_three_smart", "C4_one_smart_two_dumb"]
             if c in set(df.condition_identifier)]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    means, errs = [], []
    for c in order:
        v = df[df.condition_identifier == c]["delta_prob_mass_toward_peer_wrong"].to_numpy()
        means.append(v.mean())
        errs.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)

    colours = ["#1F4E79", "#B3272D"][-len(order):]
    ax.bar(np.arange(len(order)), means, 0.45, yerr=errs, capsize=3, color=colours)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{COND_SHORT.get(c, c)}\n{lab}" for c, lab in
                        zip(order, ["homogeneous peers", "two wrong peers"])])
    ax.set_ylabel("$\\Delta$ probability mass on the\npeer-asserted wrong answer")
    fig.savefig(OUT / "figure4_mechanism.png")
    plt.close(fig)
    print("wrote figure4_mechanism.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = json.loads(NUMBERS.read_text())
    figure_sweep(rep)
    figure2(rep)
    figure4()


if __name__ == "__main__":
    main()

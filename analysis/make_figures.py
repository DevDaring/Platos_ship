#!/usr/bin/env python3
"""
make_figures.py — regenerate the paper's figures from released artefacts.

Portable replacement for the original generate_figures.py, whose paths were
hardcoded to a Windows checkout. Everything is resolved relative to the repo.

    python3 Submission/Analyse/make_figures.py

Figures written to Submission/images/:
  figure1_capability_corruption.png   solo accuracy vs harmful-flip rate (8 focals)
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
REPO = HERE.parents[1]
NUMBERS = HERE / "paper_numbers.json"
OUT = HERE.parent / "images"
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


def figure1(rep):
    sw = pd.DataFrame(rep["capability_sweep"])
    if sw.empty:
        return
    x = sw["solo_accuracy_pct"].to_numpy()
    y = sw["c4_flip_correct_to_incorrect_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.scatter(x, y, s=38, color="#B3272D", zorder=3, edgecolor="white", linewidth=0.7)
    if len(x) > 2:  # trend line, purely descriptive
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min() - 2, x.max() + 2, 50)
        ax.plot(xs, a + b * xs, color="#888", lw=1, ls="--", zorder=2)

    # Four mid-range models sit almost on top of each other (65-67% solo,
    # ~12.3% flips), so labels are staggered and leadered instead of pinned to
    # the marker, which would render them unreadable.
    sw = sw.sort_values("solo_accuracy_pct").reset_index(drop=True)
    span = max(y.max() - y.min(), 1e-6)
    placed = []  # (x, y) of labels already positioned, in data coords
    for _, r in sw.iterrows():
        px, py = r["solo_accuracy_pct"], r["c4_flip_correct_to_incorrect_pct"]
        ty = py + 0.045 * span
        # push the label up until it clears every label already placed nearby
        while any(abs(px - qx) < 0.10 * (x.max() - x.min()) and abs(ty - qy) < 0.075 * span
                  for qx, qy in placed):
            ty += 0.075 * span
        placed.append((px, ty))
        ax.annotate(
            SHORT.get(r["focal"], r["focal"]), xy=(px, py), xytext=(px, ty),
            fontsize=6.5, color="#333", ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color="#BBB", lw=0.5,
                            shrinkA=0, shrinkB=2),
        )

    rho = rep.get("sweep_spearman_solo_vs_harmful_flip", {})
    ax.set_xlabel("Solo accuracy (%)")
    ax.set_ylabel("Harmful flip rate under\ntwo wrong peers (%)")
    if rho:
        ax.set_title(rf"Spearman $\rho$ = {rho['rho']:.2f} ($p$ = {rho['p']:.4f})",
                     fontsize=8, pad=6)
    fig.savefig(OUT / "figure1_capability_corruption.png")
    plt.close(fig)
    print("wrote figure1_capability_corruption.png")


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
    figure1(rep)
    figure2(rep)
    figure4()


if __name__ == "__main__":
    main()

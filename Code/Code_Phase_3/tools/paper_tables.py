#!/usr/bin/env python3
"""
paper_tables.py — every table, Figure 1 and every number the paper quotes.

    python3 tools/paper_tables.py [--out ../../Submission]

Reads only the released outputs (results/outputs/*.parquet, paper_numbers.json)
and writes LaTeX tables to <out>/tables/, Figure 1 to <out>/images/, and
<out>/tables/paper_facts.json with every number cited in the prose. Nothing
in the paper is typed by hand, so the manuscript cannot drift from the data.
Question-level bootstrap CIs (5,000 resamples) throughout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.contrasts import paired_condition_contrast, protocol_solo_accuracy  # noqa: E402
from src.r0_cache import load_r0_cache, solo_accuracy_by_focal  # noqa: E402

OUT = ROOT / "results/outputs"
NAMES = {
    "deepseek_primary": "DeepSeek-v4-flash", "gpt4o_mini": "GPT-4o-mini",
    "sweep_llama_3_1_70b": "Llama-3.1-70B", "sweep_qwen_2_5_72b": "Qwen2.5-72B",
    "sweep_gemma_3_27b": "Gemma-3-27B", "sweep_mistral_small": "Mistral-Small-24B",
    "sweep_llama_3_1_8b_focal": "Llama-3.1-8B", "sweep_gemma_3_4b_focal": "Gemma-3-4B",
}
X2 = ["deepseek_primary", "sweep_gemma_3_27b", "sweep_llama_3_1_8b_focal"]
# Display label for conditions in the paper: the natural weak-peer condition H is shown as N
# so that it is not confused with the harmful-revision rate $H$.
SHOW = {"H": "N", "Hfilt": "Nfilt"}
X3 = ["deepseek_primary", "sweep_gemma_3_27b"]
FACTS: Dict[str, object] = {}


def pct(x, d=1):
    return "--" if x is None or pd.isna(x) else f"{100 * x:.{d}f}"


def pp(x, d=1):
    return "--" if x is None or pd.isna(x) else f"{100 * x:+.{d}f}"


def fmt_p(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(tables: Path, name: str, body: str) -> None:
    (tables / f"{name}.tex").write_text(body, encoding="utf-8")


def table(caption: str, label: str, spec: str, header: List[str], rows: List[List[str]],
          small: bool = True, star: bool = False) -> str:
    env = "table*" if star else "table"
    width = r"\textwidth" if star else r"\linewidth"
    lines = [rf"\begin{{{env}}}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{adjustbox}}{{max width={width}}}"]
    if small:
        lines.append(r"\small")
    lines += [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" if r != ["MIDRULE"] else r"\midrule" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", rf"\end{{{env}}}", ""]
    return "\n".join(lines)


def rate_frame(reg: pd.DataFrame) -> pd.DataFrame:
    frame = reg[(reg["protocol"] == "B") & (reg["dataset_scope"] == "main300")].copy()
    frame["round_index"] = frame["round_index"].fillna(1).astype(int)
    return frame


def rates(g: pd.DataFrame) -> Dict[str, float]:
    ok = g[g["r0_is_correct"].astype(bool)]
    wrong0 = g[~g["r0_is_correct"].astype(bool)]
    return {"acc": g["is_correct"].astype(bool).mean(),
            "H": 1 - ok["is_correct"].astype(bool).mean() if len(ok) else np.nan,
            "B": wrong0["is_correct"].astype(bool).mean() if len(wrong0) else np.nan,
            "adopt": ok["adopted_peer_target"].astype("boolean").astype(float).mean()
            if ok["peer_asserted_target"].notna().any() else np.nan,
            "n": len(g)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT.parent.parent / "Submission"))
    args = parser.parse_args()
    out = Path(args.out)
    tables, images = out / "tables", out / "images"
    tables.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    reg = pd.read_parquet(OUT / "registry.parquet")
    numbers = json.loads((OUT / "paper_numbers.json").read_text(encoding="utf-8"))
    solo = solo_accuracy_by_focal(load_r0_cache(OUT / "r0_cache.parquet"))
    order = sorted(solo, key=solo.get, reverse=True)
    B = rate_frame(reg)
    FACTS["solo_accuracy"] = {NAMES[k]: round(100 * v, 1) for k, v in solo.items()}
    FACTS["n_revision_units"] = int((reg["protocol"] == "B").sum())
    FACTS["n_round0"] = int(len(load_r0_cache(OUT / "r0_cache.parquet")))

    # ── Table 2: X1 common matrix ──────────────────────────────────────────
    r1 = B[B["round_index"] == 1]
    rows = []
    x1 = {}
    for f in order:
        cells = {c: rates(r1[(r1["focal_key"] == f) & (r1["condition"] == c)])
                 for c in ["R", "E", "H", "WR", "CR"]}
        x1[f] = cells
        rows.append([NAMES[f], pct(solo[f])] + [pct(cells[c]["acc"]) for c in ["R", "E", "H", "WR", "CR"]]
                    + [pct(cells[c]["H"]) for c in ["R", "WR"]] + [pct(cells[c]["B"]) for c in ["R", "WR"]])
    write(tables, "tab_x1", table(
        r"Common condition matrix: eight focal models, 300 items, 3 replicates. "
        r"All values \%: Solo = first-answer accuracy; R to CR = accuracy after one revision; "
        r"$H$ = harmful revision (correct to wrong); $B$ = beneficial revision (wrong to correct).",
        "tab:x1", "@{}lrrrrrrrrrr@{}",
        ["Focal model", "Solo", "R", "E", "N", "WR", "CR", "$H$(R)", "$H$(WR)", "$B$(R)", "$B$(WR)"],
        rows, star=True))
    FACTS["x1"] = {NAMES[f]: {c: {k: (round(100 * v, 1) if k != "n" and not pd.isna(v) else v)
                                  for k, v in x1[f][c].items()} for c in x1[f]} for f in order}

    # ── Table 3: frozen contrasts, Holm, worst-case bounds ─────────────────
    c = pd.DataFrame(numbers["contrasts"])
    bounds = pd.DataFrame(numbers.get("parse_failure_bounds", []))
    bounds = bounds.set_index("name") if "name" in bounds else pd.DataFrame()
    labels = {"WR_minus_R_accuracy": r"Acc(WR) $-$ Acc(R)", "H_minus_R_accuracy": r"Acc(N) $-$ Acc(R)",
              "E_minus_R_accuracy": r"Acc(E) $-$ Acc(R)", "WR_minus_R_harmful": r"$H$(WR) $-$ $H$(R)",
              "WR_minus_G_accuracy": r"Acc(WR) $-$ Acc(G)", "WR_minus_W_accuracy": r"Acc(WR) $-$ Acc(W)",
              "WR_minus_WRh_harmful": r"$H$(WR) $-$ $H$(WRh)"}
    rows = []
    for fam in ["X1_primary", "X2_primary"]:
        if rows:
            rows.append(["MIDRULE"])
        for _, r in c[c["family"] == fam].iterrows():
            b = bounds.loc[r["name"]] if r["name"] in bounds.index else None
            rows.append([("Common matrix" if fam == "X1_primary" else "Message decomp."),
                         labels.get(r["name"], r["name"]),
                         f"{pp(r['estimate'])} [{pp(r['ci_low'])}, {pp(r['ci_high'])}]",
                         fmt_p(r["p_value"]), fmt_p(r["p_holm"]),
                         "--" if b is None else f"[{pp(b['worst_case_low'])}, {pp(b['worst_case_high'])}]",
                         r["models_compared"].count(",") + 1 if isinstance(r["models_compared"], str) else "--"])
            FACTS[f"contrast_{r['name']}"] = {"est": round(100 * r["estimate"], 2),
                                              "ci": [round(100 * r["ci_low"], 2), round(100 * r["ci_high"], 2)],
                                              "p": round(r["p_value"], 4), "p_holm": round(r["p_holm"], 4),
                                              "bounds": None if b is None else [round(100 * b["worst_case_low"], 2),
                                                                                round(100 * b["worst_case_high"], 2)]}
    ben = paired_condition_contrast(reg, "WR_minus_R_beneficial", "beneficial_delta", "WR", "R",
                                    protocol="B", n_resamples=5000)
    rows.append(["MIDRULE"])
    rows.append(["Exploratory", r"$B$(WR) $-$ $B$(R)",
                 f"{pp(ben.estimate)} [{pp(ben.ci_low)}, {pp(ben.ci_high)}]", fmt_p(ben.p_value),
                 "--", "--", 8])
    FACTS["contrast_WR_minus_R_beneficial"] = {"est": round(100 * ben.estimate, 2),
                                               "ci": [round(100 * ben.ci_low, 2), round(100 * ben.ci_high, 2)],
                                               "p": round(ben.p_value, 4), "exploratory": True}
    rows = [[str(x) for x in r] for r in rows]
    write(tables, "tab_contrasts", table(
        r"Frozen primary contrasts and one exploratory contrast, pooled over the models listed. "
        r"Percentage points; question-level bootstrap 95\% CI; Holm within each frozen family; "
        r"bounds = worst case over unparsed answers (Appendix~\ref{app:bounds}).",
        "tab:contrasts", "@{}llrrrrr@{}",
        ["Family", "Contrast", r"Estimate [95\% CI]", "$p$", r"$p_{\text{Holm}}$", "Worst-case bounds", "Models"],
        rows, star=True))

    # ── Figure 1: the gradient, raw and excess, both protocols ────────────
    solo_a = protocol_solo_accuracy(reg, "A")
    grad = numbers["capability_gradients"]
    per = []
    for f in order:
        raw = x1[f]["WR"]["H"]
        ex = paired_condition_contrast(reg, "g", "harmful_delta", "WR", "R", protocol="B",
                                       focal_key=f, n_resamples=5000)
        a_raw = rates(reg[(reg["protocol"] == "A") & (reg["dataset_scope"] == "main300")
                          & (reg["focal_key"] == f) & (reg["condition"] == "WR")])["H"]
        per.append({"focal": f, "solo_b": solo[f], "solo_a": solo_a.get(f), "raw_b": raw,
                    "excess": ex.estimate, "lo": ex.ci_low, "hi": ex.ci_high, "raw_a": a_raw})
    per = pd.DataFrame(per)
    FACTS["gradient"] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                             for kk, vv in grad[k].items() if kk in ("rho", "p_value", "n",
                                                                     "leave_one_out_min", "leave_one_out_max")}
                         for k in grad}
    FACTS["excess_harm_per_model"] = {NAMES[r.focal]: [round(100 * r.excess, 1), round(100 * r.lo, 1),
                                                      round(100 * r.hi, 1)] for r in per.itertuples()}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax = axes[0]
    ax.scatter(100 * per["solo_a"], 100 * per["raw_a"], facecolors="none", edgecolors="#555555",
               s=26, label=r"Protocol A (stateless)", zorder=3)
    ax.scatter(100 * per["solo_b"], 100 * per["raw_b"], color="#1f4e79", s=26,
               label=r"Protocol B (matched)", zorder=3)
    short = lambda f: NAMES[f].replace("-v4-flash", "").replace("-Small-24B", "")  # noqa: E731
    below = {"gpt4o_mini", "sweep_qwen_2_5_72b"}   # labels that would overlap a neighbour
    for r in per.itertuples():
        ax.annotate(short(r.focal), (100 * r.solo_b, 100 * r.raw_b), fontsize=5.5,
                    xytext=(3, -7 if r.focal in below else 2), textcoords="offset points")
    ax.set_xlabel("Solo accuracy (%)")
    ax.set_ylabel(r"$H$(WR): harmful revision (%)")
    ax.set_title(f"(a) Raw rate  ($\\rho_B$={grad['capability_gradient_B_raw']['rho']:.2f}, "
                 f"$\\rho_A$={grad['capability_gradient_A_raw']['rho']:.2f})", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6)
    ax = axes[1]
    ax.axhline(0, color="#999999", lw=0.6)
    ax.errorbar(100 * per["solo_b"], 100 * per["excess"],
                yerr=[100 * (per["excess"] - per["lo"]), 100 * (per["hi"] - per["excess"])],
                fmt="o", color="#1f4e79", ms=4, capsize=2, lw=0.8)
    for r in per.itertuples():
        ax.annotate(short(r.focal), (100 * r.solo_b, 100 * r.excess), fontsize=5.5,
                    xytext=(-40, -9) if r.focal == "sweep_llama_3_1_70b" else (3, 2),
                    textcoords="offset points")
    ax.set_xlabel("Solo accuracy (%)")
    ax.set_ylabel(r"$H$(WR) $-$ $H$(R) (pp)")
    ax.set_title(f"(b) Wrong-peer excess  ($\\rho_B$={grad['capability_gradient_B']['rho']:.2f}, "
                 f"$p$={grad['capability_gradient_B']['p_value']:.2f})", fontsize=7.5)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(images / "figure1_gradient.pdf")
    fig.savefig(images / "figure1_gradient.png", dpi=300)
    plt.close(fig)

    ae_s = numbers.get("adoption_excess_exploratory", {}).get("spearman_solo_vs_excess", {})
    corr = [("Protocol A", r"$H$(WR)", grad["capability_gradient_A_raw"]),
            ("Protocol B", r"$H$(WR)", grad["capability_gradient_B_raw"]),
            ("Protocol B", r"$H$(WR) $-$ $H$(R)", grad["capability_gradient_B"]),
            ("Protocol B", "Adoption excess", ae_s)]
    rows = [[proto, quantity, f"{c_['rho']:+.2f}", fmt_p(c_["p_value"]),
             f"[{c_['leave_one_out_min']:+.2f}, {c_['leave_one_out_max']:+.2f}]"]
            for proto, quantity, c_ in corr if c_]
    write(tables, "tab_gradient", table(
        r"Spearman correlation of each quantity with solo accuracy across the eight focal models. "
        r"$p$ = exact permutation over all $8!$ orderings; LOO = range when one model is left out.",
        "tab:gradient", "@{}llrrr@{}", ["Protocol", "Quantity", r"$\rho$", "$p$", "LOO range"], rows))

    # ── Table 4: X2 decomposition ─────────────────────────────────────────
    conds = ["R", "G", "W", "WR", "WRh", "SF"]
    rows = []
    for f in X2:
        cells = {k: rates(r1[(r1["focal_key"] == f) & (r1["condition"] == k)]) for k in conds}
        rows.append([NAMES[f], "Acc"] + [pct(cells[k]["acc"]) for k in conds])
        rows.append([NAMES[f], "$H$"] + [pct(cells[k]["H"]) for k in conds])
        rows.append([NAMES[f], "$B$"] + [pct(cells[k]["B"]) for k in conds])
        rows.append([NAMES[f], "Adopt"] + [pct(cells[k]["adopt"]) for k in conds])
        FACTS.setdefault("x2", {})[NAMES[f]] = {k: {m: round(100 * v, 1) for m, v in cells[k].items()
                                                    if m != "n" and not pd.isna(v)} for k in conds}
    write(tables, "tab_x2", table(
        r"Message decomposition, \%. Acc = accuracy; $H$ = harmful revision; $B$ = beneficial "
        r"revision; Adopt = final answer is a shown peer's wrong answer, among correct first answers.",
        "tab:x2", "@{}llrrrrrr@{}", ["Focal", "Rate", *conds], rows))

    # ── Table 5: X3 dose, agreement, rounds ───────────────────────────────
    mit = set(B[B["condition"] == "WR1"]["question_identifier"])
    x3 = B[B["question_identifier"].isin(mit)]
    cells_spec = [("WR1", 1, "1 peer"), ("WR", 1, "2 peers"), ("WR4", 1, "4 peers"),
                  ("WRagree", 1, "2, same target"), ("WRdiff", 1, "2, different targets"),
                  ("WR_rounds3", 1, "round 1 of 3"), ("WR_rounds3", 2, "round 2 of 3"),
                  ("WR_rounds3", 3, "round 3 of 3")]
    rows = []
    for cond, rnd, label in cells_spec:
        row = [label]
        for f in X3:
            g = x3[(x3["focal_key"] == f) & (x3["condition"] == cond) & (x3["round_index"] == rnd)]
            s = rates(g)
            row += [pct(s["H"]), pct(s["adopt"])]
            FACTS.setdefault("x3", {}).setdefault(NAMES[f], {})[label] = {
                "H": round(100 * s["H"], 1), "adopt": None if pd.isna(s["adopt"]) else round(100 * s["adopt"], 1),
                "n": int(s["n"])}
        rows.append(row)
    write(tables, "tab_x3", table(
        r"Scaling study on the 100-item subset, 3 replicates, \%. DS = DeepSeek-v4-flash; "
        r"G27 = Gemma-3-27B; $H$ = harmful revision; Adopt = adoption of a peer's wrong answer.",
        "tab:x3", "@{}lrrrr@{}",
        ["Cell", r"$H$ DS", r"Adopt DS", r"$H$ G27", r"Adopt G27"], rows))

    # ── Appendix tables ───────────────────────────────────────────────────
    gee = pd.DataFrame(numbers.get("gee_condition_ladder", []))
    if not gee.empty:
        rows = []
        for f in order:
            g = gee[gee["focal_key"] == f].set_index("condition")
            rows.append([NAMES[f]] + [f"{g.loc[k, 'odds_ratio']:.2f} [{g.loc[k, 'or_ci_low']:.2f}, "
                                      f"{g.loc[k, 'or_ci_high']:.2f}]" if k in g.index else "--"
                                      for k in ["E", "H", "WR", "CR"]])
        write(tables, "tab_gee", table(
            r"GEE (logit, exchangeable, clustered by question): odds ratio of a correct final "
            r"answer against R, with 95\% CI, per focal model.",
            "tab:gee", "@{}lrrrr@{}", ["Focal", "E", "N", "WR", "CR"], rows))
    ae = numbers.get("adoption_excess_exploratory", {})
    if ae:
        hx = per.set_index("focal")
        rows = [[NAMES[r["focal_key"]], pct(r["solo_accuracy"]),
                 f"{pp(hx.loc[r['focal_key'], 'excess'])} [{pp(hx.loc[r['focal_key'], 'lo'])}, "
                 f"{pp(hx.loc[r['focal_key'], 'hi'])}]",
                 pct(r["adoption"]), pct(r["chance"]),
                 f"{pp(r['excess'])} [{pp(r['ci_low'])}, {pp(r['ci_high'])}]"]
                for r in sorted(ae["per_model"], key=lambda r: -r["solo_accuracy"])]
        write(tables, "tab_adoption", table(
            r"Per-model effects of two wrong peers (WR) over the re-answer baseline (R). "
            r"Harmful excess = $H$(WR) $-$ $H$(R), paired by question (pp); Adopt = final answer "
            r"is a peer's wrong answer; Chance = the R answer lands on the same target (\%).",
            "tab:adoption", "@{}lrrrrr@{}",
            ["Focal", "Solo", r"Harmful excess [95\% CI]", "Adopt", "Chance",
             r"Adoption excess [95\% CI]"], rows, star=True))
        FACTS["adoption_excess"] = {"spearman": {k: ae["spearman_solo_vs_excess"].get(k) for k in ("rho", "p_value", "n")},
                                    "per_model": {NAMES[r["focal_key"]]: [round(100 * r["excess"], 1),
                                                                          round(100 * r["ci_low"], 1),
                                                                          round(100 * r["ci_high"], 1)]
                                                  for r in ae["per_model"]}}
    v = pd.DataFrame(numbers["voting_baseline"])
    rows = []
    for f in order:
        g = v[v["focal_key"] == f].set_index("source_dataset")
        e = r1[(r1["focal_key"] == f) & (r1["condition"] == "E")]
        e_task = e.groupby("source_dataset")["is_correct"].apply(lambda s: s.astype(bool).mean())
        rows.append([NAMES[f]] + [f"{pct(g.loc[t, 'single_sample_accuracy'])} / {pct(g.loc[t, 'plurality_vote_accuracy'])} / {pct(e_task.get(t))}"
                                  for t in ["mmlu_pro", "gsm8k"]])
    write(tables, "tab_voting", table(
        r"Voting baseline, \%: single sample (1 call) / plurality vote over 3 "
        r"samples (3 calls) / E, revision after two self-samples (4 calls), per task.",
        "tab:voting", "@{}lrr@{}", ["Focal", "MMLU-Pro", "GSM8K"], rows))
    s = pd.DataFrame(numbers["safeguard_policies"])
    s = s[s["focal_key"] == "ALL"]
    rows = []
    for cond in ["WR", "H"]:
        for _, r in s[s["condition"] == cond].iterrows():
            rows.append([SHOW.get(cond, cond), r["policy"].replace("_", " "), pct(r["accuracy"]),
                         f"[{pct(r['accuracy_ci_low'])}, {pct(r['accuracy_ci_high'])}]",
                         pct(r["harmful_revision"]), pct(r["beneficial_revision"])])
        FACTS.setdefault("x6", {})[cond] = {r["policy"]: {"acc": round(100 * r["accuracy"], 1),
                                                          "H": round(100 * r["harmful_revision"], 1),
                                                          "B": round(100 * r["beneficial_revision"], 1)}
                                            for _, r in s[s["condition"] == cond].iterrows()}
    write(tables, "tab_safeguard", table(
        r"Verification safeguard on three focal models and the 100-item subset, \%. "
        r"$H$ = harmful revision; $B$ = beneficial revision; the oracle is an upper bound, not a policy.",
        "tab:safeguard", "@{}llrrrr@{}", ["Cond.", "Policy", "Acc", r"95\% CI", "$H$", "$B$"], rows))
    x4 = reg[(reg["protocol"] == "B") & reg["question_identifier"].str.startswith(("gsmsym", "gsmorig"))
             & reg["condition"].isin(["R", "WR"])]
    rows = []
    for f in X2:
        row = [NAMES[f]]
        for kind in ["gsmorig", "gsmsym"]:
            g = x4[(x4["focal_key"] == f) & x4["question_identifier"].str.startswith(kind)]
            a = {c_: g[g["condition"] == c_]["is_correct"].astype(bool).mean() for c_ in ["R", "WR"]}
            row += [pct(a["R"]), pct(a["WR"]), pp(a["WR"] - a["R"])]
            FACTS.setdefault("x4", {}).setdefault(NAMES[f], {})[kind] = {k: round(100 * x, 1) for k, x in a.items()}
        rows.append(row)
    write(tables, "tab_x4", table(
        r"GSM-Symbolic contamination check, accuracy \%: 100 original GSM8K items and their regenerated "
        r"template instances, 3 replicates.", "tab:x4", "@{}lrrrrrr@{}",
        ["Focal", r"Orig.\ R", r"Orig.\ WR", r"$\Delta$", r"Sym.\ R", r"Sym.\ WR", r"$\Delta$"], rows))
    tq = pd.DataFrame(numbers["truncation_and_parsing"])
    rows = []
    for f in order:
        g = tq[tq["focal_key"] == f].set_index("stage")
        rows.append([NAMES[f]] + [f"{pct(g.loc[st, 'share_truncated'])} / {pct(g.loc[st, 'share_unparsed'])}"
                                  if st in g.index else "--" for st in ["round0", "revision"]])
    write(tables, "tab_parse", table(
        r"Output truncated at the 2{,}048-token cap / no answer recoverable, \% of calls.",
        "tab:parse", "@{}lrr@{}", ["Focal", "Round 0", "Revision"], rows))
    gate = pd.DataFrame(numbers["retention_gap"])
    h = gate[(gate["substrate"] == "honest")].set_index("unparseable_counts_as")
    FACTS["gate_honest"] = {acc: {"delta": round(float(h.loc[acc, "delta_retention"]), 3),
                                  "auroc": round(float(h.loc[acc, "auroc_confidence_vs_correct"]), 3),
                                  "n": int(h.loc[acc, "n_messages"])} for acc in h.index}
    FACTS["filter_firing"] = numbers.get("confidence_filter_firing", [])
    ff = pd.DataFrame(FACTS["filter_firing"])
    rows = []
    for f in X2:
        g = ff[ff["focal_key"] == f].set_index("condition") if not ff.empty else pd.DataFrame()
        rows.append([NAMES[f]] + [pct(g.loc[c_, "share_units_filter_dropped_any"]) if c_ in g.index else "--"
                                  for c_ in ["WRfilt", "Hfilt"]])
    rows.append(["MIDRULE"])
    hh = FACTS["gate_honest"]["dropped"]
    rows.append([r"Retention gap $\Delta_{\mathrm{ret}}$ / AUROC, " + f"{hh['n']:,}".replace(",", "{,}")
                 + " natural-peer messages", f"{hh['delta']:+.2f}", f"{hh['auroc']:.2f}"])
    write(tables, "tab_filter", table(
        r"Confidence filter. Top: share of units (\%) in which the filter removed at least one "
        r"peer. Bottom: how well stated confidence separates correct from wrong natural peers.",
        "tab:filter", "@{}lrr@{}", ["Focal", "WRfilt", "Nfilt"], rows))
    FACTS["verifier_cost"] = numbers.get("verifier_cost", {})
    (tables / "paper_facts.json").write_text(json.dumps(FACTS, indent=1, default=str), encoding="utf-8")
    print(f"wrote tables to {tables} and Figure 1 to {images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

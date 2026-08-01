#!/usr/bin/env python3
"""
make_tables.py — emit the LaTeX for the paper's data tables straight from
paper_numbers.json, so the manuscript cannot drift away from the trial logs.

Writes one .tex fragment per table into Submission/tables/, each \\input-able
from ACL_Paper.tex:

    tab_conditions.tex   Table 2  per-condition accuracy + flip rates
    tab_stats.tex        Table 3  causal contrasts (raw and Holm-corrected p)
    tab_sweep.tex        capability sweep across focal models
    tab_fullcounts.tex   Appendix S1 full trial counts and per-round accuracy

    python3 Submission/Analyse/make_tables.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUMBERS = HERE / "paper_numbers.json"
OUT = HERE.parent / "tables"

# Display names, in the order the paper presents them.
COND_LABEL = {
    "C1_smart_solo": "C1",
    "C1R_solo_reanswer": "C1R",
    "C2_three_smart": "C2",
    "C2het_three_distinct_smart": "C2het",
    "C3_two_smart_one_dumb": "C3",
    "C3H_two_smart_one_honest": "C3H",
    "C4_one_smart_two_dumb": "C4",
    "C4H_one_smart_two_honest": "C4H",
    "C4split_one_wrong_one_correct": "C4split",
    "C5R_anchored_with_confidence_filter": "C5R",
    "C5H_honest_with_confidence_filter": "C5H",
    "C5_one_smart_two_dumb_confidence_weighted": "C5",
}
COND_ORDER = list(COND_LABEL)

FOCAL_LABEL = {
    "deepseek_primary": "DeepSeek-chat",
    "gpt4o_mini": "GPT-4o-mini",
    "sweep_llama_3_1_70b": "Llama-3.1-70B",
    "sweep_qwen_2_5_72b": "Qwen2.5-72B",
    "sweep_gemma_3_27b": "Gemma-3-27B",
    "sweep_mistral_small": "Mistral-Small-24B",
    "sweep_llama_3_1_8b_focal": "Llama-3.1-8B",
    "sweep_gemma_3_4b_focal": "Gemma-3-4B",
    # C2het's focal is DeepSeek debating two architecturally distinct peers, so
    # it is listed separately from the homogeneous DeepSeek rows.
    "het_deepseek": "DeepSeek-chat (het.)",
}


def fmt(x, nd=1, signed=False):
    if x is None:
        return "---"
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace("-", "$-$") if signed else s


def fmt_p(p):
    if p is None:
        return "---"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def load():
    return json.loads(NUMBERS.read_text())


def index_conditions(rep):
    idx = {}
    for r in rep["per_condition"]:
        idx[(r["focal"], r["condition"])] = r
    return idx


def table_conditions(rep) -> str:
    """
    Table 2: the primary focal model across the ladder and controls.

    Single-column on purpose. The earlier two-model table* spanned both columns
    and cost roughly a fifth of a page; GPT-4o-mini's full per-condition numbers
    are in Appendix S1, and the one cross-model comparison the text makes is
    quoted inline, so nothing needed to assess a central claim is lost.
    """
    idx = index_conditions(rep)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Per-condition results for the primary focal model",
        r"(\textit{DeepSeek-chat}) on the full 300-question pool.",
        r"$\Delta_{\mathrm{solo}}$ = Round-1 accuracy against solo (C1);",
        r"C$\to$I = harmful flip rate; I$\to$C = correction rate, both",
        r"conditional on the Round-0 answer. GPT-4o-mini and the six sweep",
        r"models are in Appendix Table~\ref{stab:fullcounts}."
        r"\label{tab:conditions}}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"Cond. & R1 & $\Delta_{\mathrm{solo}}$ & C$\to$I & I$\to$C \\", r"\midrule",
    ]

    f = "deepseek_primary"
    base = idx.get((f, "C1_smart_solo"), {}).get("r1_accuracy_pct")
    for cond in COND_ORDER:
        r = idx.get((f, cond))
        if r is None:
            continue
        d = None if cond == "C1_smart_solo" or base is None else \
            round(r["r1_accuracy_pct"] - base, 1)
        cells = [
            fmt(r["r1_accuracy_pct"]),
            "---" if d is None else fmt(d, signed=True),
            "---" if cond == "C1_smart_solo" else fmt(r["flip_correct_to_incorrect_pct"]),
            "---" if cond == "C1_smart_solo" else fmt(r["flip_incorrect_to_correct_pct"]),
        ]
        lines.append(f"{COND_LABEL[cond]:<7} & " + " & ".join(cells) + r" \\")

    n_ds = idx.get((f, "C1_smart_solo"), {}).get("n", 0)
    lines += [
        r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
        r"{\footnotesize All values in \%. $N=" + f"{n_ds:,}".replace(",", "{,}") +
        r"$ trials per main-condition cell; the C5 rows use a 100-question subset "
        r"($N=300$), with $\Delta_{\mathrm{solo}}$ taken against C1 on that subset.}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table_stats(rep) -> str:
    """Table 3: causal contrasts, question-level pairing, raw and Holm p."""
    want = [
        ("C4_one_smart_two_dumb - C1_smart_solo", r"Two wrong vs.\ solo (C4--C1)"),
        ("C1R_solo_reanswer - C1_smart_solo", r"Re-answer vs.\ solo (C1R--C1)"),
        ("C4_one_smart_two_dumb - C1R_solo_reanswer", r"Two wrong vs.\ re-answer (C4--C1R)"),
        ("C4_one_smart_two_dumb - C4H_one_smart_two_honest", r"Two wrong vs.\ honest (C4--C4H)"),
        ("C4H_one_smart_two_honest - C1_smart_solo", r"Honest vs.\ solo (C4H--C1)"),
        ("C4split_one_wrong_one_correct - C4_one_smart_two_dumb", r"Split peers vs.\ two wrong (C4split--C4)"),
        ("C4split_one_wrong_one_correct - C1_smart_solo", r"Split peers vs.\ solo (C4split--C1)"),
    ]
    by = {r["contrast"]: r for r in rep.get("contrasts_question_level", [])
          if r.get("focal") == "deepseek_primary" and "p_raw" in r}

    rows = []
    for key, label in want:
        r = by.get(key)
        if r is None:
            continue
        rows.append((label, r["delta_pp"], r["p_raw"], r.get("p_holm")))

    pert = rep.get("perturbed_c1_vs_c4_question_level") or rep.get("perturbed_c1_vs_c4")
    if pert and "p_raw" in pert:
        rows.append((r"Perturbed math: two wrong vs.\ solo",
                     pert["delta_pp"], pert["p_raw"], pert.get("p_holm")))

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Key contrasts (DeepSeek-chat, question-level paired McNemar).",
        r"$\Delta$ = accuracy difference in percentage points; $p$ is the raw",
        r"McNemar $p$-value and $p_{\mathrm{H}}$ its Holm-corrected value within",
        r"this family of contrasts. $^{\ast}$ significant at $\alpha=0.05$ after",
        r"correction.\label{tab:stats}}",
        r"\begin{tabular}{@{}lrrl@{}}", r"\toprule",
        r"Contrast & $\Delta$ & $p$ & $p_{\mathrm{H}}$ \\", r"\midrule",
    ]
    for label, d, p, ph in rows:
        star = r"$^{\ast}$" if (ph is not None and ph < 0.05) else ""
        lines.append(f"{label} & {fmt(d, signed=True)} & {fmt_p(p)} & {fmt_p(ph)}{star} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_sweep(rep) -> str:
    """Capability sweep across every focal model."""
    sw = [r for r in rep["capability_sweep"] if r["focal"] in FOCAL_LABEL
          and r["focal"] != "het_deepseek"]
    sw = sorted(sw, key=lambda r: -r["solo_accuracy_pct"])
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Capability sweep. Solo = C1 accuracy; C4 = accuracy under two",
        r"confidently wrong peers; C$\to$I = harmful-flip rate in C4, conditional",
        r"on a correct round-0 answer. All eight $\Delta_{\mathrm{solo}}$ point",
        r"estimates are non-negative; the harmful-flip",
        r"rate rises steeply as solo ability falls.\label{tab:sweep}}",
        r"\begin{tabular}{@{}lrrrl@{}}", r"\toprule",
        r"Focal model & Solo & C4 & $\Delta_{\mathrm{solo}}$ & C$\to$I [95\% CI] \\",
        r"\midrule",
    ]
    for r in sw:
        ci = ""
        if r.get("c4_flip_ci_low") is not None:
            ci = f" [{r['c4_flip_ci_low']:.1f}, {r['c4_flip_ci_high']:.1f}]"
        lines.append(
            f"{FOCAL_LABEL.get(r['focal'], r['focal'])} & "
            f"{fmt(r['solo_accuracy_pct'])} & {fmt(r['c4_accuracy_pct'])} & "
            f"{fmt(r['c4_delta_pp'], signed=True)} & "
            f"{fmt(r['c4_flip_correct_to_incorrect_pct'])}{ci} " + r"\\"
        )
    rho = rep.get("sweep_spearman_solo_vs_harmful_flip", {})
    foot = (r"{\footnotesize All values in \%. Spearman $\rho$ between solo accuracy "
            r"and the C$\to$I rate is $-" + f"{abs(rho.get('rho', float('nan'))):.2f}" +
            r"$ (exact permutation $p=" + f"{rho.get('p_exact_permutation', float('nan')):.4f}" +
            r"$, $n=" + str(rho.get("n", "")) + r"$); leave-one-model-out $\rho$ stays in "
            r"$[-" + f"{abs(rho.get('leave_one_out_rho_min', float('nan'))):.2f}" +
            r", -" + f"{abs(rho.get('leave_one_out_rho_max', float('nan'))):.2f}" +
            r"]$, and excluding the two weakest models $\rho=-" +
            f"{abs(rho.get('rho_excluding_two_weakest', float('nan'))):.2f}" +
            r"$ ($p=" + f"{rho.get('p_excluding_two_weakest', float('nan')):.4f}" + r"$).}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip", foot, r"\end{table}"]
    return "\n".join(lines)


def table_fullcounts(rep) -> str:
    """Appendix S1: trial counts and per-round accuracy for every (focal, condition)."""
    idx = index_conditions(rep)
    lines = [
        r"\begin{table}[h]", r"\centering", r"\footnotesize",
        r"\caption{Full trial counts and per-round accuracy for every focal model",
        r"and condition actually run. $N$ is the number of focal trials.",
        r"\label{stab:fullcounts}}",
        r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
        r"Focal & Cond. & $N$ & R0 (\%) & R1 (\%) & $\Delta$ \\", r"\midrule",
    ]
    focals = [f for f in FOCAL_LABEL if any(k[0] == f for k in idx)]
    for fi, f in enumerate(focals):
        conds = [c for c in COND_ORDER if (f, c) in idx]
        if not conds:
            continue
        if fi:
            lines.append(r"\midrule")
        first = True
        for c in conds:
            r = idx[(f, c)]
            d = r["r1_accuracy_pct"] - r["r0_accuracy_pct"]
            name = FOCAL_LABEL[f] if first else ""
            first = False
            lines.append(
                f"{name} & {COND_LABEL[c]} & {r['n']:,} & {fmt(r['r0_accuracy_pct'])} & "
                f"{fmt(r['r1_accuracy_pct'])} & {fmt(d, signed=True)} ".replace(",", "{,}") + r"\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    rep = load()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("tab_conditions", table_conditions),
        ("tab_stats", table_stats),
        ("tab_sweep", table_sweep),
        ("tab_fullcounts", table_fullcounts),
    ]:
        p = OUT / f"{name}.tex"
        p.write_text(fn(rep) + "\n")
        print(f"wrote {p.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()

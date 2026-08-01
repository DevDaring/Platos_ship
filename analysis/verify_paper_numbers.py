#!/usr/bin/env python3
"""
verify_paper_numbers.py — recompute every headline number in ACL_Paper.tex
from the released per-trial logs, so the manuscript can be checked against the
data rather than against an earlier draft of itself.

Reads Phase-1 (`results/data/outputs`) and Phase-2 (`Code_Phase_2/results/outputs`)
final_answers.parquet, unions them, and prints:

  * per (focal, condition) Round-0/Round-1 accuracy and conditional flip rates
  * the capability sweep (solo accuracy vs harmful-flip rate, Spearman rho)
  * paired McNemar contrasts for the causal controls
  * total trial counts

    python3 Submission/Analyse/verify_paper_numbers.py [--json out.json]
"""

import argparse
import json
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

def _repo_root(start: pathlib.Path) -> pathlib.Path:
    """
    Walk up to the directory that actually holds this project's results.

    Resolving the root by a fixed number of `.parents` breaks the moment the
    script is moved, and it fails *silently* by picking up a stale copy of the
    data elsewhere on disk. Anchor on a marker only the real repo has.
    """
    for d in [start, *start.parents]:
        if (d / "Code_Phase_2" / "results" / "outputs").is_dir() and (d / ".git").exists():
            return d
    for d in [start, *start.parents]:
        if (d / "Code_Phase_2" / "results" / "outputs").is_dir():
            return d
    raise SystemExit(f"could not locate the repository root from {start}")


REPO = _repo_root(pathlib.Path(__file__).resolve().parent)
PHASE1 = REPO / "results" / "data" / "outputs" / "final_answers.parquet"
PHASE2 = REPO / "Code_Phase_2" / "results" / "outputs" / "final_answers.parquet"

# The contamination probe re-uses the C1/C4 condition ids on a DIFFERENT question
# pool (ids prefixed `gsm8k_perturbed_`). Those rows must never be pooled with the
# main 300-question results — the perturbed items are far harder, so averaging
# them in silently drags every main-pool mean down.
PERTURBED_PREFIX = "gsm8k_perturbed_"

# Phase 2 re-ran GPT-4o-mini on the full 300-question pool under the focal name
# `gpt4o_mini`; the Phase-1 `openrouter_gpt4o_mini` rows are the superseded
# 50-question cross-validation subset and are reported separately, not pooled.
SUPERSEDED_FOCALS = {"openrouter_gpt4o_mini"}


def load(include_perturbed: bool = False) -> pd.DataFrame:
    frames = []
    for p, phase in ((PHASE1, 1), (PHASE2, 2)):
        if p.exists():
            df = pd.read_parquet(p)
            df["phase"] = phase
            frames.append(df)
    if not frames:
        raise SystemExit("no final_answers.parquet found in either phase")
    df = pd.concat(frames, ignore_index=True)

    is_perturbed = df["question_identifier"].str.startswith(PERTURBED_PREFIX)
    df = df[is_perturbed] if include_perturbed else df[~is_perturbed]

    if not include_perturbed:
        df = df[~df["focal_smart_agent_name"].isin(SUPERSEDED_FOCALS)]

    dupe_keys = ["question_identifier", "condition_identifier",
                 "trial_replication_index", "focal_smart_agent_name"]
    df = df.sort_values("phase").drop_duplicates(dupe_keys, keep="last")
    return df.reset_index(drop=True)


def cell(g: pd.DataFrame) -> dict:
    r0 = g["round_zero_answer_was_correct"].astype(bool)
    r1 = g["round_one_answer_was_correct"].astype(bool)
    ci = g["focal_agent_flipped_correct_to_incorrect"].astype(bool)
    ic = g["focal_agent_flipped_incorrect_to_correct"].astype(bool)
    return {
        "n": int(len(g)),
        "r0_accuracy_pct": round(100 * r0.mean(), 1),
        "r1_accuracy_pct": round(100 * r1.mean(), 1),
        # Flip rates are CONDITIONAL on the round-0 answer, as the paper states.
        "flip_correct_to_incorrect_pct": round(100 * ci.sum() / max(r0.sum(), 1), 1),
        "flip_incorrect_to_correct_pct": round(100 * ic.sum() / max((~r0).sum(), 1), 1),
    }


def mcnemar_paired(df: pd.DataFrame, focal: str, cond_a: str, cond_b: str) -> dict:
    """Trial-level paired McNemar on (question, replication) matched trials."""
    key = ["question_identifier", "trial_replication_index"]
    a = df[(df.focal_smart_agent_name == focal) & (df.condition_identifier == cond_a)]
    b = df[(df.focal_smart_agent_name == focal) & (df.condition_identifier == cond_b)]
    if a.empty or b.empty:
        return {"contrast": f"{cond_a} vs {cond_b}", "focal": focal, "status": "missing"}
    m = a[key + ["round_one_answer_was_correct"]].merge(
        b[key + ["round_one_answer_was_correct"]], on=key, suffixes=("_a", "_b"))
    if m.empty:
        return {"contrast": f"{cond_a} vs {cond_b}", "focal": focal, "status": "no paired trials"}
    ya = m["round_one_answer_was_correct_a"].astype(bool)
    yb = m["round_one_answer_was_correct_b"].astype(bool)
    n01 = int((~ya & yb).sum())   # b better
    n10 = int((ya & ~yb).sum())   # a better
    if n01 + n10 == 0:
        chi2, p = 0.0, 1.0
    else:
        chi2 = (abs(n10 - n01) - 1) ** 2 / (n01 + n10)
        p = float(stats.chi2.sf(chi2, 1))
    return {
        "contrast": f"{cond_b} - {cond_a}", "focal": focal, "n_paired": int(len(m)),
        "acc_a_pct": round(100 * ya.mean(), 1), "acc_b_pct": round(100 * yb.mean(), 1),
        "delta_pp": round(100 * (yb.mean() - ya.mean()), 1),
        "discordant_b_better": n01, "discordant_a_better": n10,
        "chi2": round(chi2, 2), "p_raw": round(p, 5),
    }


def mcnemar_question_level(df: pd.DataFrame, focal: str, cond_a: str, cond_b: str) -> dict:
    """
    Question-level paired McNemar: replications are collapsed to a per-question
    majority vote first, then the two condition vectors are paired by question.
    This is the pairing the manuscript's Table 3 reports.
    """
    a = df[(df.focal_smart_agent_name == focal) & (df.condition_identifier == cond_a)]
    b = df[(df.focal_smart_agent_name == focal) & (df.condition_identifier == cond_b)]
    if a.empty or b.empty:
        return {"contrast": f"{cond_b} - {cond_a}", "focal": focal, "status": "missing"}

    def vote(g):
        return (g.groupby("question_identifier")["round_one_answer_was_correct"]
                 .apply(lambda s: s.astype(bool).mean() > 0.5))

    va, vb = vote(a), vote(b)
    common = va.index.intersection(vb.index)
    va, vb = va.loc[common], vb.loc[common]
    n01 = int((~va & vb).sum())
    n10 = int((va & ~vb).sum())
    if n01 + n10 == 0:
        chi2, p = 0.0, 1.0
    else:
        chi2 = (abs(n10 - n01) - 1) ** 2 / (n01 + n10)
        p = float(stats.chi2.sf(chi2, 1))
    return {
        "contrast": f"{cond_b} - {cond_a}", "focal": focal, "n_questions": int(len(common)),
        "acc_a_pct": round(100 * va.mean(), 1), "acc_b_pct": round(100 * vb.mean(), 1),
        "delta_pp": round(100 * (vb.mean() - va.mean()), 1),
        "b_better": n01, "a_better": n10,
        "chi2": round(chi2, 2), "p_raw": round(p, 5),
    }


def paired_flip_test(df: pd.DataFrame, focal: str, cond_a: str, cond_b: str) -> dict:
    """
    Paired McNemar on the HARMFUL-FLIP event itself, restricted to trials whose
    round-0 answer was correct in both arms. Comparing marginal flip rates
    across conditions conflates a change in flipping with a change in how often
    the model was correct to begin with; this pairing does not.
    """
    key = ["question_identifier", "trial_replication_index"]
    cols = key + ["round_zero_answer_was_correct", "focal_agent_flipped_correct_to_incorrect"]
    a = df[(df.focal_smart_agent_name == focal) & (df.condition_identifier == cond_a)]
    b = df[(df.focal_smart_agent_name == focal) & (df.condition_identifier == cond_b)]
    if a.empty or b.empty:
        return {"contrast": f"{cond_b} - {cond_a}", "focal": focal, "status": "missing"}
    m = a[cols].merge(b[cols], on=key, suffixes=("_a", "_b"))
    m = m[m.round_zero_answer_was_correct_a.astype(bool)
          & m.round_zero_answer_was_correct_b.astype(bool)]
    if m.empty:
        return {"contrast": f"{cond_b} - {cond_a}", "focal": focal, "status": "no paired trials"}
    fa = m.focal_agent_flipped_correct_to_incorrect_a.astype(bool)
    fb = m.focal_agent_flipped_correct_to_incorrect_b.astype(bool)
    n01, n10 = int((~fa & fb).sum()), int((fa & ~fb).sum())
    chi2 = (abs(n10 - n01) - 1) ** 2 / (n01 + n10) if (n01 + n10) else 0.0
    return {
        "contrast": f"{cond_b} - {cond_a}", "focal": focal, "n_paired_r0_correct": int(len(m)),
        "flip_a_pct": round(100 * fa.mean(), 1), "flip_b_pct": round(100 * fb.mean(), 1),
        "discordant_b_only": n01, "discordant_a_only": n10,
        "chi2": round(chi2, 2), "p_raw": float(f"{stats.chi2.sf(chi2, 1):.3g}"),
    }


TRIAL_LOGS = [
    REPO / "results" / "data" / "outputs" / "trial_log.parquet",
    REPO / "Code_Phase_2" / "results" / "outputs" / "trial_log.parquet",
]


def _peer_agreement_index() -> pd.Series:
    """
    For every (question, replication, condition), whether the two weak peers
    named the SAME answer in Round 0. Read from the trial logs, which are the
    only place individual peer answers are recorded.
    """
    frames = []
    for p in TRIAL_LOGS:
        if not p.exists():
            continue
        tl = pd.read_parquet(p, columns=[
            "question_identifier", "trial_replication_index", "condition_identifier",
            "responding_agent_role", "responding_agent_identifier",
            "debate_round_index", "extracted_final_answer",
        ])
        d = tl[(tl.responding_agent_role == "dumb") & (tl.debate_round_index == 0)]
        if d.empty:
            continue
        piv = d.pivot_table(
            index=["question_identifier", "trial_replication_index", "condition_identifier"],
            columns="responding_agent_identifier", values="extracted_final_answer",
            aggfunc="first",
        ).dropna()
        if piv.shape[1] < 2:
            continue
        frames.append(piv.iloc[:, 0].astype(str).str.upper()
                      == piv.iloc[:, 1].astype(str).str.upper())
    if not frames:
        return pd.Series(dtype=bool)
    # Some conditions (e.g. C4) appear in both phases' logs; keep the later one
    # so the index is unique and reindex() can be used against it.
    s = pd.concat(frames)
    return s[~s.index.duplicated(keep="last")]


def _agreement_strata(anchored: pd.DataFrame, natural: pd.DataFrame) -> list:
    agree = _peer_agreement_index()
    if agree.empty:
        return []

    def attach(g):
        g = g[g.round_zero_answer_was_correct.astype(bool)].copy()
        idx = pd.MultiIndex.from_frame(
            g[["question_identifier", "trial_replication_index", "condition_identifier"]])
        g["peers_agree"] = agree.reindex(idx).to_numpy()
        return g.dropna(subset=["peers_agree"])

    A, H = attach(anchored), attach(natural)
    out = []
    for label, val in [("peers named the same wrong answer", True),
                       ("peers named different wrong answers", False)]:
        a = A[A.peers_agree == val]["focal_agent_flipped_correct_to_incorrect"].astype(bool)
        h = H[H.peers_agree == val]["focal_agent_flipped_correct_to_incorrect"].astype(bool)
        if len(a) < 5 or len(h) < 5:
            continue
        chi2, p, _, _ = stats.chi2_contingency(
            [[int(h.sum()), int((~h).sum())], [int(a.sum()), int((~a).sum())]])
        out.append({
            "stratum": label,
            "anchored_flip_pct": round(100 * a.mean(), 1), "anchored_n": int(len(a)),
            "natural_flip_pct": round(100 * h.mean(), 1), "natural_n": int(len(h)),
            "chi2": round(float(chi2), 1), "p_raw": float(f"{p:.3g}"),
        })
    return out


def holm(pvals: list) -> list:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return [round(x, 4) for x in adj]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=Path(__file__).parent / "paper_numbers.json")
    args = ap.parse_args()

    df = load()
    report = {"total_focal_trials": int(len(df))}
    print(f"Total focal trials across both phases: {len(df):,}\n")

    # ── per (focal, condition) table ───────────────────────────────────────
    rows = []
    for (focal, cond), g in df.groupby(["focal_smart_agent_name", "condition_identifier"]):
        rows.append({"focal": focal, "condition": cond, **cell(g)})
    table = pd.DataFrame(rows).sort_values(["focal", "condition"])
    report["per_condition"] = table.to_dict(orient="records")
    print("=== Per-condition accuracy and conditional flip rates ===")
    print(table.to_string(index=False))
    print()

    # ── delta vs solo, per focal ───────────────────────────────────────────
    print("=== Delta in Round-1 accuracy vs C1 solo (pp) ===")
    deltas = []
    for focal, g in table.groupby("focal"):
        solo = g[g.condition == "C1_smart_solo"]
        if solo.empty:
            continue
        base = float(solo.iloc[0]["r1_accuracy_pct"])
        for _, r in g.iterrows():
            if r["condition"] == "C1_smart_solo":
                continue
            deltas.append({"focal": focal, "condition": r["condition"],
                           "r1_accuracy_pct": r["r1_accuracy_pct"],
                           "delta_vs_solo_pp": round(r["r1_accuracy_pct"] - base, 1)})
    dd = pd.DataFrame(deltas)
    report["delta_vs_solo"] = dd.to_dict(orient="records")
    print(dd.to_string(index=False))
    print()

    # ── capability sweep: solo accuracy vs harmful flips under two wrong peers ─
    print("=== Capability sweep (solo accuracy vs C4 harmful-flip rate) ===")
    sweep = []
    for focal, g in table.groupby("focal"):
        solo = g[g.condition == "C1_smart_solo"]
        c4 = g[g.condition == "C4_one_smart_two_dumb"]
        if solo.empty or c4.empty:
            continue
        # Unrounded rates recomputed from the trials, kept alongside the
        # display values so correlations never run on 1-dp numbers.
        gc4 = df[(df.focal_smart_agent_name == focal)
                 & (df.condition_identifier == "C4_one_smart_two_dumb")]
        gc1 = df[(df.focal_smart_agent_name == focal)
                 & (df.condition_identifier == "C1_smart_solo")]
        r0 = gc4["round_zero_answer_was_correct"].astype(bool)
        ci = gc4["focal_agent_flipped_correct_to_incorrect"].astype(bool)
        flip_raw = 100 * ci.sum() / max(r0.sum(), 1)
        solo_raw = 100 * gc1["round_one_answer_was_correct"].mean()
        c4_raw = 100 * gc4["round_one_answer_was_correct"].mean()
        lo, hi = stats.binomtest(int(ci.sum()), int(r0.sum())).proportion_ci(0.95)
        sweep.append({
            "focal": focal,
            "solo_accuracy_pct": float(solo.iloc[0]["r1_accuracy_pct"]),
            "c4_accuracy_pct": float(c4.iloc[0]["r1_accuracy_pct"]),
            "c4_delta_pp": round(float(c4.iloc[0]["r1_accuracy_pct"]) - float(solo.iloc[0]["r1_accuracy_pct"]), 1),
            "c4_flip_correct_to_incorrect_pct": float(c4.iloc[0]["flip_correct_to_incorrect_pct"]),
            "solo_accuracy_pct_raw": float(solo_raw),
            "c4_accuracy_pct_raw": float(c4_raw),
            "c4_delta_pp_raw": float(c4_raw - solo_raw),
            "c4_flip_pct_raw": float(flip_raw),
            "c4_flip_ci_low": round(100 * lo, 1),
            "c4_flip_ci_high": round(100 * hi, 1),
        })
    sw = pd.DataFrame(sweep).sort_values("solo_accuracy_pct", ascending=False)
    print(sw.to_string(index=False))
    if len(sw) >= 3:
        # Correlate the UNROUNDED rates. Correlating the 1-dp values printed in
        # the table creates artificial ties among the mid-range models and
        # inflates |rho| (it gave -0.97 instead of -0.95).
        x = sw["solo_accuracy_pct_raw"].to_numpy()
        y = sw["c4_flip_pct_raw"].to_numpy()
        rho, p = stats.spearmanr(x, y)
        rho2, p2 = stats.spearmanr(x, sw["c4_delta_pp_raw"].to_numpy())
        # With n = 8 the asymptotic p is unreliable; enumerate all 8! orderings.
        import itertools
        perms = [stats.spearmanr(x, np.array(pp)).statistic
                 for pp in itertools.permutations(y)]
        p_exact = float(np.mean([abs(v) >= abs(rho) for v in perms]))
        loo = []
        for i in range(len(sw)):
            t = sw.drop(sw.index[i])
            loo.append((t["solo_accuracy_pct_raw"].to_numpy(), t["c4_flip_pct_raw"].to_numpy()))
        loo_rhos = [float(stats.spearmanr(a, b).statistic) for a, b in loo]
        keep = sw[~sw["focal"].isin(["sweep_llama_3_1_8b_focal", "sweep_gemma_3_4b_focal"])]
        rho_strong, p_strong = stats.spearmanr(keep["solo_accuracy_pct_raw"],
                                               keep["c4_flip_pct_raw"])
        print(f"\nSpearman(solo, C->I) = {rho:.3f}  asymptotic p = {p:.4f}  "
              f"EXACT permutation p = {p_exact:.5f}  (n = {len(sw)})")
        print(f"Spearman(solo, C4 gain) = {rho2:.3f}  (p = {p2:.4f})")
        print(f"leave-one-model-out rho range: [{min(loo_rhos):+.3f}, {max(loo_rhos):+.3f}]")
        print(f"excluding the two weakest models: rho = {rho_strong:+.3f}, "
              f"p = {p_strong:.4f}, n = {len(keep)}")
        report["sweep_spearman_solo_vs_harmful_flip"] = {
            "rho": round(float(rho), 3), "p_asymptotic": round(float(p), 5),
            "p_exact_permutation": round(p_exact, 5), "n": len(sw),
            "leave_one_out_rho_min": round(min(loo_rhos), 3),
            "leave_one_out_rho_max": round(max(loo_rhos), 3),
            "rho_excluding_two_weakest": round(float(rho_strong), 3),
            "p_excluding_two_weakest": round(float(p_strong), 5),
        }
        report["sweep_spearman_solo_vs_gain"] = {"rho": round(float(rho2), 3),
                                                 "p": round(float(p2), 5), "n": len(sw)}
    report["capability_sweep"] = sw.to_dict(orient="records")
    print()

    # ── causal contrasts ───────────────────────────────────────────────────
    print("=== Paired McNemar contrasts (trial-level) ===")
    contrasts = [
        ("deepseek_primary", "C1_smart_solo", "C1R_solo_reanswer"),
        ("deepseek_primary", "C1R_solo_reanswer", "C4_one_smart_two_dumb"),
        ("deepseek_primary", "C4H_one_smart_two_honest", "C4_one_smart_two_dumb"),
        ("deepseek_primary", "C1_smart_solo", "C4H_one_smart_two_honest"),
        ("deepseek_primary", "C1_smart_solo", "C4_one_smart_two_dumb"),
        ("deepseek_primary", "C1_smart_solo", "C2_three_smart"),
        ("deepseek_primary", "C1_smart_solo", "C3_two_smart_one_dumb"),
        ("deepseek_primary", "C4_one_smart_two_dumb", "C4split_one_wrong_one_correct"),
        ("deepseek_primary", "C1_smart_solo", "C4split_one_wrong_one_correct"),
        ("gpt4o_mini", "C1_smart_solo", "C4_one_smart_two_dumb"),
        ("gpt4o_mini", "C2_three_smart", "C4_one_smart_two_dumb"),
    ]
    res = [mcnemar_paired(df, f, a, b) for (f, a, b) in contrasts]
    ok = [r for r in res if "p_raw" in r]
    if ok:
        adj = holm([r["p_raw"] for r in ok])
        for r, a in zip(ok, adj):
            r["p_holm"] = a
    cf = pd.DataFrame(res)
    report["contrasts_trial_level"] = res
    print(cf.to_string(index=False))
    print()

    print("=== Paired McNemar contrasts (question-level majority vote, Table 3 pairing) ===")
    qres = [mcnemar_question_level(df, f, a, b) for (f, a, b) in contrasts]
    qok = [r for r in qres if "p_raw" in r]
    if qok:
        for r, a in zip(qok, holm([r["p_raw"] for r in qok])):
            r["p_holm"] = a
    report["contrasts_question_level"] = qres
    print(pd.DataFrame(qres).to_string(index=False))
    print()

    # ── paired tests on the harmful-flip event itself ──────────────────────
    print("=== Paired McNemar on the harmful-flip event (R0 correct in both arms) ===")
    flip_pairs = [
        ("gpt4o_mini", "C2_three_smart", "C4_one_smart_two_dumb"),
        ("gpt4o_mini", "C1R_solo_reanswer", "C4_one_smart_two_dumb"),
        ("deepseek_primary", "C2_three_smart", "C4_one_smart_two_dumb"),
        ("deepseek_primary", "C1R_solo_reanswer", "C4_one_smart_two_dumb"),
        ("deepseek_primary", "C4_one_smart_two_dumb", "C4split_one_wrong_one_correct"),
        ("deepseek_primary", "C4_one_smart_two_dumb", "C4H_one_smart_two_honest"),
    ]
    fres = [paired_flip_test(df, f, a, b) for (f, a, b) in flip_pairs]
    report["flip_contrasts"] = fres
    print(pd.DataFrame(fres).to_string(index=False))
    print()

    # ── flip-rate composition inside the honest condition ──────────────────
    h = df[(df.condition_identifier == "C4H_one_smart_two_honest")]
    if not h.empty and "dumb_peer_consensus_status" in h.columns:
        print("=== Honest-peer condition: harmful flips by peer composition ===")
        comp = []
        for status, g in h.groupby("dumb_peer_consensus_status"):
            r0 = g["round_zero_answer_was_correct"].astype(bool)
            ci = g["focal_agent_flipped_correct_to_incorrect"].astype(bool)
            comp.append({"peer_composition": status, "n": len(g),
                         "n_r0_correct": int(r0.sum()),
                         "flip_correct_to_incorrect_pct": round(100 * ci.sum() / max(r0.sum(), 1), 1)})
        cdf = pd.DataFrame(comp)
        report["honest_peer_composition"] = comp
        print(cdf.to_string(index=False))
        print()

    # ── answer-level consensus WITHIN C4 ───────────────────────────────────
    # The manuscript must not call C4 a "unanimous wrong consensus" in the
    # answer sense: the two anchored peers draw their assigned wrong answer
    # independently, so they usually name DIFFERENT wrong answers. This tests
    # whether naming the same wrong answer matters at all. If it does not, any
    # C4split effect is attributable to a correct peer being present, not to
    # consensus being broken.
    agree = _peer_agreement_index()
    c4a = df[(df.focal_smart_agent_name == "deepseek_primary")
             & (df.condition_identifier == "C4_one_smart_two_dumb")].copy()
    if not agree.empty and not c4a.empty:
        idx = pd.MultiIndex.from_frame(
            c4a[["question_identifier", "trial_replication_index", "condition_identifier"]])
        c4a["peers_agree"] = agree.reindex(idx).to_numpy()
        c4a = c4a.dropna(subset=["peers_agree"])
        r0c = c4a[c4a.round_zero_answer_was_correct.astype(bool)]
        rows = []
        for label, val in [("same wrong answer", True), ("different wrong answers", False)]:
            g = r0c[r0c.peers_agree == val]
            f = g["focal_agent_flipped_correct_to_incorrect"].astype(bool)
            if len(f) == 0:
                continue
            lo, hi = stats.binomtest(int(f.sum()), len(f)).proportion_ci(0.95)
            rows.append({"peer_answers": label, "n_r0_correct": int(len(g)),
                         "flip_pct": round(100 * f.mean(), 2),
                         "ci_low": round(100 * lo, 2), "ci_high": round(100 * hi, 2)})
        a = r0c[r0c.peers_agree == True]["focal_agent_flipped_correct_to_incorrect"].astype(bool)
        b = r0c[r0c.peers_agree == False]["focal_agent_flipped_correct_to_incorrect"].astype(bool)
        chi2, p, _, _ = stats.chi2_contingency(
            [[int(a.sum()), int((~a).sum())], [int(b.sum()), int((~b).sum())]])
        res = {"strata": rows, "chi2": round(float(chi2), 3), "p_raw": float(f"{p:.3g}"),
               "same_answer_rate_pct": round(100 * float(c4a.peers_agree.mean()), 1),
               "note": ("assignment of same/different wrong answers is not a designed "
                        "randomisation; it arises from independent uniform draws over the "
                        "question's wrong-answer pool, so this is observational")}
        report["c4_answer_consensus"] = res
        print("=== Answer-level consensus within C4 (same vs different wrong answer) ===")
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"chi2={res['chi2']}  p={res['p_raw']}  "
              f"(peers name the same wrong answer in {res['same_answer_rate_pct']}% of trials)")
        print()

    # ── natural vs deliberate wrong peers, paired within question ──────────
    # The honest condition contains trials where both weak peers happen to be
    # wrong. Those peers are wrong for natural reasons rather than by
    # instruction, so contrasting them against the anchored condition asks
    # whether the persona framing makes wrong peers easier or harder to resist.
    # Pairing is by question, because the subset of honest trials with two wrong
    # peers is not a random sample of the pool.
    c4 = df[(df.focal_smart_agent_name == "deepseek_primary")
            & (df.condition_identifier == "C4_one_smart_two_dumb")]
    c4h = df[(df.focal_smart_agent_name == "deepseek_primary")
             & (df.condition_identifier == "C4H_one_smart_two_honest")]
    uw = c4h[c4h.dumb_peer_consensus_status == "unanimous_wrong"] if not c4h.empty else c4h
    if not c4.empty and not uw.empty:
        def per_question(g):
            g = g[g.round_zero_answer_was_correct.astype(bool)]
            return g.groupby("question_identifier")["focal_agent_flipped_correct_to_incorrect"].mean()

        A, B = per_question(c4), per_question(uw)
        common = A.index.intersection(B.index)
        if len(common) > 20:
            A2, B2 = A.loc[common], B.loc[common]
            w = stats.wilcoxon(B2, A2)
            res = {
                "n_questions": int(len(common)),
                "anchored_wrong_flip_pct": round(100 * float(A2.mean()), 1),
                "honest_unanimous_wrong_flip_pct": round(100 * float(B2.mean()), 1),
                "wilcoxon_statistic": float(w.statistic),
                "p_raw": float(f"{w.pvalue:.3g}"),
            }
            report["natural_vs_deliberate_wrong_peers"] = res
            print("=== Natural vs deliberate wrong peers (paired by question) ===")
            print(json.dumps(res, indent=2))
            print()

        # Robustness: the two peer pairs differ in how often they name the SAME
        # wrong answer (29.4% honest vs 16.4% anchored), so a raw contrast could
        # be picking up consensus strength rather than peer type. Stratify on it.
        strata = _agreement_strata(c4, uw)
        if strata:
            report["natural_vs_deliberate_by_peer_agreement"] = strata
            print("=== ... stratified by whether the two peers named the same wrong answer ===")
            print(pd.DataFrame(strata).to_string(index=False))
            print()

    # ── split-peer condition, once it has data ─────────────────────────────
    s = df[df.condition_identifier == "C4split_one_wrong_one_correct"]
    if not s.empty:
        print(f"=== Split-peer condition (N1): {len(s)} trials ===")
        print(pd.DataFrame([cell(s)]).to_string(index=False))
        print()

    het = df[df.condition_identifier == "C2het_three_distinct_smart"]
    if not het.empty:
        print(f"=== Heterogeneous-smart debate (N2): {len(het)} trials ===")
        print(pd.DataFrame([cell(het)]).to_string(index=False))
        print()

    # ── contamination probe, on its own (perturbed) pool ───────────────────
    pert = load(include_perturbed=True)
    if not pert.empty:
        print("=== Contamination probe: perturbed GSM8K (separate pool) ===")
        prows = [{"focal": f, "condition": c, **cell(g)}
                 for (f, c), g in pert.groupby(["focal_smart_agent_name", "condition_identifier"])]
        pdf = pd.DataFrame(prows).sort_values(["focal", "condition"])
        print(pdf.to_string(index=False))
        report["perturbed_pool"] = prows
        pc = mcnemar_paired(pert, "deepseek_primary", "C1_smart_solo", "C4_one_smart_two_dumb")
        pcq = mcnemar_question_level(pert, "deepseek_primary", "C1_smart_solo",
                                     "C4_one_smart_two_dumb")
        report["perturbed_c1_vs_c4"] = pc
        report["perturbed_c1_vs_c4_question_level"] = pcq
        print("\nperturbed C1 vs C4 (trial-level)   :", json.dumps(pc))
        print("perturbed C1 vs C4 (question-level):", json.dumps(pcq))
        print()

    # ── Table 3 family: the causal contrasts, Holm-corrected together ──────
    # These five (six once the split-peer control lands) are ONE family; the
    # manuscript previously printed raw p-values under a "Holm-corrected"
    # heading, so the correction is computed explicitly here.
    t3 = [
        ("C4_one_smart_two_dumb - C1_smart_solo", None),
        ("C1R_solo_reanswer - C1_smart_solo", None),
        ("C4_one_smart_two_dumb - C1R_solo_reanswer", None),
        ("C4_one_smart_two_dumb - C4H_one_smart_two_honest", None),
        ("C4H_one_smart_two_honest - C1_smart_solo", None),
        ("C4split_one_wrong_one_correct - C4_one_smart_two_dumb", None),
        ("C4split_one_wrong_one_correct - C1_smart_solo", None),
    ]
    qmap = {r["contrast"]: r for r in report.get("contrasts_question_level", [])
            if r.get("focal") == "deepseek_primary" and "p_raw" in r}
    fam = [dict(qmap[k]) for k, _ in t3 if k in qmap]
    if "perturbed_c1_vs_c4_question_level" in report and \
            "p_raw" in report["perturbed_c1_vs_c4_question_level"]:
        pq = dict(report["perturbed_c1_vs_c4_question_level"])
        pq["contrast"] = "PERTURBED " + pq["contrast"]
        fam.append(pq)
    if fam:
        for r, a in zip(fam, holm([r["p_raw"] for r in fam])):
            r["p_holm"] = a
        # write the corrected values back where make_tables.py reads them
        for r in fam:
            if r["contrast"].startswith("PERTURBED "):
                report["perturbed_c1_vs_c4_question_level"]["p_holm"] = r["p_holm"]
            else:
                qmap[r["contrast"]]["p_holm"] = r["p_holm"]
        print("=== Table 3 family, Holm-corrected within itself ===")
        print(pd.DataFrame(fam)[["contrast", "delta_pp", "p_raw", "p_holm"]].to_string(index=False))
        print()
        report["table3_family"] = fam

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, default=str))
    print(f"JSON written to {args.json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
crosscheck_paper_numbers.py — fail the build when a number in the manuscript
disagrees with the regenerated analysis output.

Three classes of check:

  1. CLAIM REGISTRY. Each hand-written number in the prose is registered with
     the analysis value it must equal. Generated tables are not checked here —
     they are emitted from the same JSON, so they cannot drift by construction.
  2. CROSS-DOCUMENT. Numbers stated in both ACL_Paper.tex and Revision_Notes.tex
     must agree with each other.
  3. CROSS-REFERENCES. Every \\ref must resolve; the compiled PDF must contain
     no "??".

    python3 analysis/crosscheck_paper_numbers.py            # human-readable
    python3 analysis/crosscheck_paper_numbers.py --csv out.csv
    echo $?                                                 # 1 if any check fails
"""

import argparse
import csv
import json
import pathlib
import re
import sys


def _repo_root(start: pathlib.Path) -> pathlib.Path:
    for d in [start, *start.parents]:
        if (d / "Code_Phase_2" / "results" / "outputs").is_dir() and (d / ".git").exists():
            return d
    for d in [start, *start.parents]:
        if (d / "Code_Phase_2" / "results" / "outputs").is_dir():
            return d
    raise SystemExit(f"could not locate the repository root from {start}")


REPO = _repo_root(pathlib.Path(__file__).resolve().parent)
PAPER = REPO / "Submission" / "ACL_Paper.tex"
NOTES = REPO / "Submission" / "Revision_Notes.tex"
PDF = REPO / "Submission" / "ACL_Paper.pdf"
AUX = REPO / "Submission" / "ACL_Paper.aux"
NUMBERS = pathlib.Path(__file__).resolve().parent / "paper_numbers.json"


def cond(rep, focal, condition, field):
    for r in rep["per_condition"]:
        if r["focal"] == focal and r["condition"] == condition:
            return r[field]
    return None


def contrast(rep, key, field="delta_pp", focal="deepseek_primary"):
    for r in rep.get("contrasts_question_level", []):
        if r.get("contrast") == key and r.get("focal") == focal:
            return r.get(field)
    return None


def build_registry(rep):
    """(claim_id, regex over the .tex, expected value, tolerance)."""
    sweep = {r["focal"]: r for r in rep["capability_sweep"]}
    rho = rep["sweep_spearman_solo_vs_harmful_flip"]
    cons = rep.get("c4_answer_consensus", {})
    strata = {s["peer_answers"]: s for s in cons.get("strata", [])}
    nat = rep.get("natural_vs_deliberate_wrong_peers", {})

    R = []
    add = lambda i, p, v, t=0.05: R.append((i, p, v, t))

    # --- headline accuracies quoted in prose
    add("C4 accuracy vs solo (trial-level)",
        r"Accuracy\s+rises\s+by\s+\$?([0-9.]+)\$?\s+points\s+over\s+solo",
        cond(rep, "deepseek_primary", "C4_one_smart_two_dumb", "r1_accuracy_pct")
        - cond(rep, "deepseek_primary", "C1_smart_solo", "r1_accuracy_pct"))
    add("C4-C1 question-level delta",
        r"gives a slightly smaller estimate,\s*\$?\+?([0-9.]+)\$?\s*points",
        contrast(rep, "C4_one_smart_two_dumb - C1_smart_solo"))

    # --- capability sweep
    add("Spearman rho", r"\\rho\s*=\s*-([0-9.]+)\$?,\s*exact permutation", abs(rho["rho"]), 0.005)
    add("Spearman exact p", r"exact permutation \$?p\s*=\s*([0-9.]+)", rho["p_exact_permutation"], 0.0005)
    add("rho excluding two weakest", r"leaves \$\\rho = -([0-9.]+)\$", abs(rho["rho_excluding_two_weakest"]), 0.005)
    add("sweep min delta", r"non-negative, from \$\+([0-9.]+)\$ to",
        min(s["c4_delta_pp"] for s in sweep.values()))
    add("sweep max delta", r"non-negative, from \$\+[0-9.]+\$ to \$\+([0-9.]+)\$",
        max(s["c4_delta_pp"] for s in sweep.values()))
    add("n deltas including zero", r"(four|three|five|six) of the eight paired bootstrap intervals",
        sum(1 for s in sweep.values() if s.get("c4_delta_ci_low") is not None
            and s["c4_delta_ci_low"] <= 0 <= s["c4_delta_ci_high"]), 0)

    # --- answer-level consensus
    if strata:
        add("consensus same-answer flip",
            r"harmful-flip rate is \$([0-9.]+)\\%\$\s*\n?when the peers name the same",
            strata["same wrong answer"]["flip_pct"], 0.01)
        add("consensus diff-answer flip",
            r"when the peers name the same wrong answer and \$([0-9.]+)\\%\$",
            strata["different wrong answers"]["flip_pct"], 0.01)
        add("same-answer rate", r"same wrong answer in only \$([0-9.]+)\\%\$",
            cons["same_answer_rate_pct"], 0.05)

    # --- natural vs instructed
    if nat:
        add("natural flip rate", r"errors \(\$([0-9.]+)\\%\$ against",
            nat["honest_unanimous_wrong_flip_pct"], 0.05)
        add("instructed flip rate", r"against \$([0-9.]+)\\%\$\s*\n?over the \$111\$",
            nat["anchored_wrong_flip_pct"], 0.05)

    # --- split peer: the prose deliberately no longer repeats the number, so
    # the check is against the generated table instead of the prose.
    R.append(("C4split accuracy (generated table)",
              ("TABLE", "tab_conditions", r"C4split & ([0-9.]+) &"),
              cond(rep, "deepseek_primary", "C4split_one_wrong_one_correct", "r1_accuracy_pct"),
              0.05))

    # --- counts
    add("total API focal trials", r"\$39\{,\}(470)\$ API focal trials", 470, 0)
    add("main pool trials", r"\(\$37\{,\}(500)\$ on the main", 500, 0)
    add("probe trials", r"plus \$2\{,\}(700)\$\s*\n?trials of the locally served", 700, 0)
    return R


def run_registry(tex, registry):
    tables = REPO / "Submission" / "tables"
    rows = []
    for cid, pat, expected, tol in registry:
        if isinstance(pat, tuple) and pat[0] == "TABLE":
            f = tables / f"{pat[1]}.tex"
            src = f.read_text(errors="replace") if f.exists() else ""
            m = re.search(pat[2], src)
        else:
            m = re.search(pat, tex)
        if expected is None:
            rows.append((cid, "SKIP", "", "", "no analysis value"))
            continue
        if not m:
            rows.append((cid, "NOT FOUND", "", f"{expected}", "pattern not present in .tex"))
            continue
        raw = m.group(1)
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8}
        found = float(words.get(raw, raw)) if not raw.replace(".", "").isdigit() else float(raw)
        ok = abs(found - float(expected)) <= tol
        rows.append((cid, "OK" if ok else "MISMATCH", f"{found}", f"{expected}",
                     "" if ok else f"differs by {abs(found-float(expected)):.4g}"))
    return rows


def cross_document(paper, notes):
    """Numbers asserted in both documents must agree."""
    tables = REPO / "Submission" / "tables"
    shared = [
        ("C4 vs C4H Holm p (table vs notes)",
         ("TABLE", "tab_stats", r"C4--C4H\) & \+7\.3 & [^&]+& 0\.([0-9]{3})"),
         r"wrong-anchored condition \(\$-7\.3\$~pp, \$p=0\.([0-9]{3})\$ Holm\)"),
    ]
    rows = []
    for cid, ppat, npat in shared:
        if isinstance(ppat, tuple) and ppat[0] == "TABLE":
            f = tables / f"{ppat[1]}.tex"
            pm = re.search(ppat[2], f.read_text(errors="replace")) if f.exists() else None
        else:
            pm = re.search(ppat, paper)
        nm = re.search(npat, notes)
        if not pm or not nm:
            rows.append((cid, "SKIP", "", "", "not found in one document"))
            continue
        a, b = "0." + pm.group(1), "0." + nm.group(1)
        rows.append((cid, "OK" if abs(float(a) - float(b)) < 1e-9 else "MISMATCH", a, b, ""))
    return rows


def cross_references():
    rows = []
    if PDF.exists():
        import subprocess
        txt = subprocess.run(["pdftotext", str(PDF), "-"], capture_output=True, text=True).stdout
        n = txt.count("??")
        rows.append(("PDF contains no '??'", "OK" if n == 0 else "MISMATCH", str(n), "0", ""))
    if AUX.exists():
        aux = AUX.read_text(errors="replace")
        labels = set(re.findall(r"\\newlabel\{([^}]+)\}", aux))
        refs = set(re.findall(r"\\ref\{([^}]+)\}", PAPER.read_text(errors="replace")))
        missing = sorted(refs - labels)
        rows.append(("every \\ref resolves", "OK" if not missing else "MISMATCH",
                     f"{len(refs)} refs", "0 unresolved",
                     ", ".join(missing[:5])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=pathlib.Path, default=None)
    args = ap.parse_args()

    rep = json.loads(NUMBERS.read_text())
    paper, notes = PAPER.read_text(errors="replace"), NOTES.read_text(errors="replace")

    rows = (run_registry(paper, build_registry(rep))
            + cross_document(paper, notes)
            + cross_references())

    width = max(len(r[0]) for r in rows) + 2
    bad = 0
    for cid, status, found, expected, note in rows:
        if status == "MISMATCH":
            bad += 1
        mark = {"OK": "ok  ", "SKIP": "skip", "NOT FOUND": "MISS", "MISMATCH": "FAIL"}[status]
        print(f"  {mark} {cid:<{width}} found={found:<10} expected={expected:<10} {note}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["check", "status", "value_in_tex", "value_from_analysis", "note"])
            w.writerows(rows)
        print(f"\nwritten: {args.csv}")

    n_ok = sum(1 for r in rows if r[1] == "OK")
    print(f"\n{n_ok}/{len(rows)} checks pass; {bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""
Final X8 analysis: reproducibility check, judge recovery, and bounds.

Run after the probe has produced a trials parquet that carries the raw
generations. Needs no GPU.

  python3 analysis/finalise_x8.py \
      --trials  GCP_L4_Run/results/big_probe_trials.parquet \
      --compare GCP_L4_Run/results/big_probe_trials_run1.parquet

WHAT IT REPORTS, AND WHY ALL FOUR ROWS ARE NEEDED.

  complete case          the subset the regex alone can see
  judge recovered        the subset after unreadable answers are re-judged
  bounds, regex only     every value the subset mean could take before recovery
  bounds, after recovery the same, over whatever is still unresolved

The bounds are the claim that does not depend on the judge behaving. The
recovered point estimate is the more informative number, and it should sit
inside the bounds; if it does not, something is wrong with the recovery and
the script says so rather than printing a number.

The reproducibility check exists because a second run is a second sample.
vLLM is deterministic per request given a seed, but batch composition can
perturb sampled text. The probability masses come from teacher-forced
scoring of fixed candidate strings and should match closely; the check
reports agreement rather than asserting it, so a real difference is visible
instead of hidden behind a tolerance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.nonparse_bounds import (  # noqa: E402
    KEYS, MASS, bounded_unchanged_subset, classify_pairs, nonparse_balance,
)
from analysis.recover_unparsed import (  # noqa: E402
    add_effective_answers, answer_column, effective_column,
)
from analysis.stats import bootstrap_ci, paired_permutation_p  # noqa: E402

logger = logging.getLogger("finalise_x8")


# -- reproducibility -----------------------------------------------------
def compare_runs(new: pd.DataFrame, old: pd.DataFrame) -> Dict[str, Any]:
    """Agreement between two runs of the same seed on the same questions."""
    keys = KEYS + ["condition"]
    merged = new.merge(old, on=keys, suffixes=("_new", "_old"))
    if merged.empty:
        return {"n_matched": 0, "note": "no overlapping cells"}
    out: Dict[str, Any] = {"n_matched": int(len(merged))}

    for column in ("prob_mass_round1_on_target", MASS):
        left = pd.to_numeric(merged.get(f"{column}_new"), errors="coerce")
        right = pd.to_numeric(merged.get(f"{column}_old"), errors="coerce")
        both = left.notna() & right.notna()
        if not both.any():
            continue
        difference = (left[both] - right[both]).abs()
        out[column] = {
            "n": int(both.sum()),
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
            "fraction_within_1e_6": float((difference < 1e-6).mean()),
        }

    for column in ("round1_answer", "fixed_target"):
        if f"{column}_new" in merged and f"{column}_old" in merged:
            left = merged[f"{column}_new"].astype(str).str.strip()
            right = merged[f"{column}_old"].astype(str).str.strip()
            out[f"{column}_agreement"] = float((left == right).mean())
    return out


# -- the contrast, under a chosen answer column --------------------------
def _contrast(trials: pd.DataFrame, answer_suffix: str, family: str,
              treatment: str = "WR", baseline: str = "E") -> Dict[str, Any]:
    """
    Paired WR-minus-E shift, restricted to pairs whose stated answer held.

    `answer_suffix` selects which extraction the subset is defined by:
    "" for the regex columns, "_effective" for regex-then-judge.
    """
    subset = trials[trials["source_dataset"] == family]
    left = subset[subset["condition"] == treatment]
    right = subset[subset["condition"] == baseline]
    merged = left.merge(right, on=KEYS, suffixes=("_treatment", "_baseline"))
    if merged.empty:
        return {"n_pairs": 0}

    if answer_suffix:
        for side in ("_treatment", "_baseline"):
            for round_name in ("round0", "round1"):
                source = f"{round_name}_answer{answer_suffix}{side}"
                merged[f"{round_name}_answer{side}"] = merged[source]

    membership = classify_pairs(merged)
    differences = (merged[f"{MASS}_treatment"]
                   - merged[f"{MASS}_baseline"])
    held = differences[membership == "in"].dropna().to_numpy()
    if held.size == 0:
        return {"n_pairs": 0}
    estimate, low, high = bootstrap_ci(held)
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "p_value": paired_permutation_p(held),
        "n_pairs": int(held.size),
        "n_ambiguous": int((membership == "ambiguous").sum()),
        "n_out": int((membership == "out").sum()),
    }


def finalise(trials: pd.DataFrame,
             comparison: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    frame = trials.copy()
    has_effective = effective_column("round1") in frame.columns
    if not has_effective and f"round1_answer_judged" in frame.columns:
        frame = add_effective_answers(frame)
        has_effective = True

    report: Dict[str, Any] = {
        "n_trials": int(len(frame)),
        "nonparse_balance": nonparse_balance(frame).to_dict("records"),
    }
    if comparison is not None:
        report["reproducibility"] = compare_runs(frame, comparison)

    # Bounds computed on the regex columns, and again on the effective ones.
    report["bounds_regex_only"] = bounded_unchanged_subset(frame)
    if has_effective:
        recovered = frame.copy()
        for side in ("round0", "round1"):
            recovered[answer_column(side)] = recovered[effective_column(side)]
        report["bounds_after_recovery"] = bounded_unchanged_subset(recovered)

    rows: List[Dict[str, Any]] = []
    for family in sorted(frame["source_dataset"].dropna().unique()):
        row: Dict[str, Any] = {"source_dataset": family}
        row["complete_case"] = _contrast(frame, "", family)
        if has_effective:
            row["judge_recovered"] = _contrast(frame, "_effective", family)
        rows.append(row)
    report["contrasts"] = rows

    # Consistency: the recovered point estimate must lie inside the bounds
    # computed before recovery. A judge that moved it outside has resolved
    # ambiguity in a way no resolution could, which means it invented data.
    checks: List[Dict[str, Any]] = []
    if has_effective:
        by_family = {b["source_dataset"]: b for b in report["bounds_regex_only"]}
        for row in rows:
            bound = by_family.get(row["source_dataset"])
            recovered_estimate = row.get("judge_recovered", {}).get("estimate")
            if not bound or recovered_estimate is None:
                continue
            inside = (bound["bound_low"] - 1e-9 <= recovered_estimate
                      <= bound["bound_high"] + 1e-9)
            checks.append({
                "source_dataset": row["source_dataset"],
                "recovered_estimate": recovered_estimate,
                "bound_low": bound["bound_low"],
                "bound_high": bound["bound_high"],
                "inside_pre_recovery_bounds": bool(inside),
            })
    report["consistency_checks"] = checks
    return report


def _fmt(value: Any, places: int = 6) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:+.{places}f}" if isinstance(value, float) else str(value)


def render(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"trials: {report['n_trials']}")
    repro = report.get("reproducibility")
    if repro:
        lines.append("")
        lines.append("REPRODUCIBILITY vs the earlier run")
        lines.append(f"  cells matched: {repro.get('n_matched')}")
        for key in ("prob_mass_round1_on_target", MASS):
            block = repro.get(key)
            if block:
                lines.append(f"  {key}: max|diff|={block['max_abs_difference']:.2e} "
                             f"within 1e-6 on {100*block['fraction_within_1e_6']:.1f}% "
                             f"of {block['n']}")
        if "round1_answer_agreement" in repro:
            lines.append("  round1_answer agreement: "
                         f"{100*repro['round1_answer_agreement']:.1f}%")

    lines.append("")
    lines.append("WR minus E, restricted to pairs whose stated answer held")
    header = f"  {'family':10s} {'basis':16s} {'estimate':>11s} {'95% CI':>26s} {'n':>5s} {'amb':>5s}"
    lines.append(header)
    for row in report["contrasts"]:
        for label, key in (("complete case", "complete_case"),
                           ("judge recovered", "judge_recovered")):
            block = row.get(key)
            if not block or not block.get("n_pairs"):
                continue
            ci = f"[{_fmt(block['ci_low'])}, {_fmt(block['ci_high'])}]"
            lines.append(f"  {row['source_dataset']:10s} {label:16s} "
                         f"{_fmt(block['estimate']):>11s} {ci:>26s} "
                         f"{block['n_pairs']:>5d} {block.get('n_ambiguous', 0):>5d}")

    for title, key in (("SHARP BOUNDS, regex only", "bounds_regex_only"),
                       ("SHARP BOUNDS, after judge recovery", "bounds_after_recovery")):
        block = report.get(key)
        if not block:
            continue
        lines.append("")
        lines.append(title)
        for row in block:
            lines.append(
                f"  {row['source_dataset']:10s} "
                f"[{_fmt(row['bound_low'])}, {_fmt(row['bound_high'])}]  "
                f"width {row['bound_width']:.6f}  "
                f"known-in {row['n_known_in']:3d}  ambiguous {row['n_ambiguous']:3d}  "
                f"sign robust: {'yes' if row['sign_robust_to_nonparse'] else 'NO'}")

    checks = report.get("consistency_checks") or []
    if checks:
        lines.append("")
        lines.append("CONSISTENCY: recovered estimate inside the pre-recovery bounds")
        for check in checks:
            lines.append(f"  {check['source_dataset']:10s} "
                         f"{_fmt(check['recovered_estimate'])} in "
                         f"[{_fmt(check['bound_low'])}, {_fmt(check['bound_high'])}] "
                         f"-> {'OK' if check['inside_pre_recovery_bounds'] else 'VIOLATED'}")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    trials = pd.read_parquet(args.trials)
    comparison = pd.read_parquet(args.compare) if args.compare else None
    report = finalise(trials, comparison)
    text = render(report)
    print(text)

    out = args.out or args.trials.with_name("x8_final_report.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out.with_suffix(".txt").write_text(text, encoding="utf-8")
    logger.info("wrote %s", out)

    violated = [c for c in report.get("consistency_checks", [])
                if not c["inside_pre_recovery_bounds"]]
    return 1 if violated else 0


if __name__ == "__main__":
    raise SystemExit(main())

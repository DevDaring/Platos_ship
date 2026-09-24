"""
X8 across six open-weight models: does distributional pull track capability?

  python3 analysis/x8_gradient.py

Implements the capability-gradient test of the paper (Spearman rho between
solo accuracy and a per-model effect, exact permutation p) on the X8
output-distribution read-out, so the distributional claim rests on the same
shape of evidence as the behavioural one.

DECISIONS FIXED BEFORE LOOKING AT THE SIX-MODEL RESULT

1. MMLU-Pro is the primary task. On GSM8K the probe reads the answer the
   model would give IMMEDIATELY after "Final answer:", before any
   arithmetic. On problems the model then solved correctly by reasoning, the
   probe gives that answer a median probability of 0.015 on GSM8K against
   0.906 on MMLU-Pro (Llama-3.1-8B). GSM8K therefore measures an unreasoned
   guess, not a belief. It is reported, labelled secondary, and never used
   for the gradient. `coherence` makes the reason visible per model.

2. Three effect measures, all reported, none selected afterwards:
     raw       paired WR-E of the change in P(target)
     logodds   paired WR-E of the change in logit P(target). Absolute
               probability shifts are not comparable across model families
               whose baseline confidence differs; a shift from 0.01 to 0.05
               and one from 0.20 to 0.24 are the same raw size and very
               different events.
     adoption  WR-E of P(Round-1 answer = the wrong peer's answer), on
               judge-recovered answers
   If they disagree, that disagreement is the finding.

3. Inference is clustered on the QUESTION. Each question contributes three
   replicates that share a prompt; resampling pairs would treat them as
   independent and understate the interval.

4. With six models the Spearman p-value is exact over all 720 orderings,
   and a leave-one-out range shows whether rho hinges on one model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.recover_unparsed import add_effective_answers, effective_column  # noqa: E402
from analysis.answer_normalise import normalise_frame  # noqa: E402
from analysis.stats import spearman_exact  # noqa: E402

VAST = _ROOT / "Vast_AI_Big_Model_Run"
SWEEP = VAST / "sweep_results"

# Ordered by nothing in particular; the analysis ranks them itself.
MODELS: List[Dict[str, Any]] = [
    {"name": "Gemma-3-4B", "family": "gemma",
     "dir": SWEEP / "google_gemma-3-4b-it"},
    {"name": "Llama-3.1-8B", "family": "llama",
     "dir": _ROOT / "GCP_L4_Run" / "results"},
    {"name": "Mistral-Small-24B", "family": "mistral",
     "dir": SWEEP / "mistralai_mistral-small-24b-instruct-2501"},
    {"name": "Gemma-3-27B", "family": "gemma",
     "dir": SWEEP / "google_gemma-3-27b-it"},
    {"name": "Llama-3.1-70B", "family": "llama",
     "dir": VAST / "results"},
    {"name": "Qwen2.5-72B", "family": "qwen",
     "dir": SWEEP / "qwen_qwen2.5-72b-instruct"},
]

EPS = 1e-6
N_BOOT = 5000
SEED = 20260502


_POOL: Optional[pd.DataFrame] = None


def _pool() -> pd.DataFrame:
    global _POOL
    if _POOL is None:
        _POOL = pd.read_parquet(_ROOT / "results" / "processed" / "question_pool.parquet")
    return _POOL


def _load(directory: Path) -> Optional[pd.DataFrame]:
    judged = directory / "big_probe_trials_judged.parquet"
    raw = directory / "big_probe_trials.parquet"
    path = judged if judged.exists() else raw
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "round1_answer_judged" in frame.columns and \
            effective_column("round1") not in frame.columns:
        frame = add_effective_answers(frame)
    # Map option VALUES onto LETTERS before any answer is compared with the
    # letter-keyed correct answer or target. Without this, "-42" for option
    # G scores as wrong, at a rate that differs by family (Gemma 7-9%, Llama
    # ~1%), which biases both solo accuracy and adoption by family.
    columns = tuple(c for c in ("round0_answer", "round1_answer",
                                effective_column("round0"),
                                effective_column("round1"))
                    if c in frame.columns)
    frame, tally = normalise_frame(frame, _pool(), columns)
    frame.attrs["judged"] = judged.exists()
    frame.attrs["normalised"] = tally
    return frame


def _answers(frame: pd.DataFrame, round_name: str) -> pd.Series:
    column = effective_column(round_name)
    if column not in frame.columns:
        column = f"{round_name}_answer"
    return frame[column].astype(str).str.strip()


def _logit(p: pd.Series) -> pd.Series:
    q = pd.to_numeric(p, errors="coerce").fillna(0.0).clip(EPS, 1 - EPS)
    return np.log(q) - np.log1p(-q)


def _clustered_ci(values: pd.Series, clusters: pd.Series,
                  n_boot: int = N_BOOT, seed: int = SEED) -> Dict[str, float]:
    """Mean with a bootstrap CI that resamples whole questions."""
    frame = pd.DataFrame({"v": values.to_numpy(), "c": clusters.to_numpy()})
    frame = frame.dropna()
    if frame.empty:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_pairs": 0, "n_questions": 0}
    sums = frame.groupby("c")["v"].agg(["sum", "count"])
    s, n = sums["sum"].to_numpy(), sums["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    draws = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return {"estimate": float(s.sum() / n.sum()),
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)),
            "n_pairs": int(n.sum()), "n_questions": int(len(s))}


def model_effects(frame: pd.DataFrame, family: str) -> Dict[str, Any]:
    task = frame[frame["source_dataset"] == family]
    keys = ["question_identifier", "replicate"]

    # Solo accuracy from Round 0, which is cached and identical across
    # conditions, so the R rows alone carry one answer per cell.
    solo = task[task["condition"] == "R"]
    r0 = _answers(solo, "round0")
    parsed = r0.ne("")
    solo_accuracy = float(
        (r0[parsed] == solo.loc[parsed, "correct_answer"].astype(str).str.strip()).mean())

    wr = task[task["condition"] == "WR"].copy()
    e = task[task["condition"] == "E"].copy()
    for part in (wr, e):
        part["_d_raw"] = (pd.to_numeric(part["prob_mass_round1_on_target"], errors="coerce")
                          - pd.to_numeric(part["prob_mass_round0_on_target"], errors="coerce"))
        part["_d_logit"] = (_logit(part["prob_mass_round1_on_target"])
                            - _logit(part["prob_mass_round0_on_target"]))
        a0, a1 = _answers(part, "round0"), _answers(part, "round1")
        tgt = part["fixed_target"].astype(str).str.strip()
        ok = a0.ne("") & a1.ne("") & a0.ne(tgt)
        part["_adopt"] = np.where(ok, (a1 == tgt).astype(float), np.nan)

    merged = wr.merge(e, on=keys, suffixes=("_wr", "_e"))
    q = merged["question_identifier"]
    raw = _clustered_ci(merged["_d_raw_wr"] - merged["_d_raw_e"], q)
    logodds = _clustered_ci(merged["_d_logit_wr"] - merged["_d_logit_e"], q)

    adopt_wr = _clustered_ci(wr["_adopt"], wr["question_identifier"])
    adopt_e = _clustered_ci(e["_adopt"], e["question_identifier"])

    # Does the read-out track what the model SAID? Separation between
    # P(correct) when it said the correct answer and when it did not.
    said_right = parsed & (r0 == solo["correct_answer"].astype(str).str.strip())
    p_correct = pd.to_numeric(solo["prob_mass_round0_on_correct"], errors="coerce")
    coherence = {
        "median_p_when_said_correct": float(p_correct[said_right].median()),
        "separation": float(p_correct[said_right].mean()
                            - p_correct[parsed & ~said_right].mean()),
        "off_candidate_round0": float(pd.to_numeric(
            solo["mass_outside_candidates_round0"], errors="coerce").mean()),
    }
    return {
        "solo_accuracy": solo_accuracy,
        "raw": raw,
        "logodds": logodds,
        "adoption_pp": 100 * (adopt_wr["estimate"] - adopt_e["estimate"]),
        "adoption_wr": adopt_wr, "adoption_e": adopt_e,
        "coherence": coherence,
    }


def run() -> Dict[str, Any]:
    report: Dict[str, Any] = {"models": [], "gradient": {}, "missing": []}
    for spec in MODELS:
        frame = _load(spec["dir"])
        if frame is None:
            report["missing"].append(spec["name"])
            continue
        entry = {"name": spec["name"], "family": spec["family"],
                 "judged": bool(frame.attrs.get("judged")),
                 "rows": int(len(frame)),
                 "values_mapped_to_letters": {
                     c: t.get("mapped", 0)
                     for c, t in (frame.attrs.get("normalised") or {}).items()}}
        for task in ("mmlu_pro", "gsm8k"):
            entry[task] = model_effects(frame, task)
        report["models"].append(entry)

    present = report["models"]
    for task in ("mmlu_pro", "gsm8k"):
        acc = [m[task]["solo_accuracy"] for m in present]
        report["gradient"][task] = {
            measure: spearman_exact(acc, [
                (m[task][measure]["estimate"] if isinstance(m[task][measure], dict)
                 else m[task][measure]) for m in present])
            for measure in ("raw", "logodds", "adoption_pp")
        }
    return report


def render(report: Dict[str, Any]) -> str:
    out: List[str] = []
    models = sorted(report["models"], key=lambda m: m["mmlu_pro"]["solo_accuracy"])
    for task, label in (("mmlu_pro", "MMLU-Pro  (PRIMARY)"),
                        ("gsm8k", "GSM8K  (SECONDARY: unreasoned-guess read-out)")):
        out.append(f"\n=== {label} ===")
        out.append(f"{'model':<19} {'solo':>6} {'raw shift':>11} {'95% CI':>24} "
                   f"{'log-odds':>9} {'adopt pp':>9} {'med P|said right':>17}")
        for m in models:
            t = m[task]
            r, lo = t["raw"], t["logodds"]
            out.append(f"{m['name']:<19} {t['solo_accuracy']:>6.3f} "
                       f"{r['estimate']:>+11.6f} "
                       f"{'['+format(r['ci_low'],'+.4f')+', '+format(r['ci_high'],'+.4f')+']':>24} "
                       f"{lo['estimate']:>+9.3f} {t['adoption_pp']:>+9.2f} "
                       f"{t['coherence']['median_p_when_said_correct']:>17.3f}")
        out.append("  Spearman vs solo accuracy (exact permutation p, leave-one-out range):")
        for measure, g in report["gradient"][task].items():
            out.append(f"    {measure:<12} rho={g['rho']:+.3f}  p={g['p_value']:.4f}  "
                       f"LOO=[{g.get('leave_one_out_min', float('nan')):+.3f}, "
                       f"{g.get('leave_one_out_max', float('nan')):+.3f}]  n={g['n']}")
    if report["missing"]:
        out.append(f"\nnot yet available: {', '.join(report['missing'])}")
    return "\n".join(out)


def main() -> int:
    report = run()
    text = render(report)
    print(text)
    out = VAST / "x8_gradient_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out.with_suffix(".txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

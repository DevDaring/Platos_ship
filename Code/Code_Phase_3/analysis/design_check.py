"""
design_check.py — every treatment must have its baseline, on the same models.

The single most consequential gap in the reviewed submission was not a missing
experiment but a missing *cell*. The re-answer control (C1R, here `R`) was run
for two focal models out of eight. Everything downstream followed from that:

  * the capability gradient could only be reported RAW, i.e. H under two wrong
    peers, with no way to subtract the revision churn that weak models show
    under any prompt. Reviewer oEjr's W2 — that weak models simply revise more
    — therefore had no answer;
  * "the effect is specific to confidently wrong peers" rested on one model;
  * Reviewer YVDD's W1 (Table 6 conditions not comparable) was literally true.

Running the analysis on the released logs shows it concretely: the
baseline-subtracted gradient resolves at n = 2, the raw one at n = 8.

So Phase 3 treats "each treatment condition has a matched baseline on exactly
the same focal models, questions and replicates" as a design invariant, checked
by this module rather than trusted. `run_all.py --list` runs it before anything
is spent, and `--analyse` runs it against what was actually collected.

Note on what does NOT help: raising the replicate count on the baseline. The
contrast is a question-level paired difference, so its width is governed by the
number of QUESTIONS, not by replicates per question. Resampling the released
DeepSeek data at 1-8 control replicates moves the 95% CI from 5.19 pp to
5.62 pp — flat, drifting slightly wider as marginal questions enter the pair
set. Subsampling questions moves it from 8.32 pp (120 questions) to 3.92 pp
(480). Spend budget on models and questions; not on replicates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import pandas as pd

logger = logging.getLogger("platos_ship3.design_check")

# A treatment is any revision condition that is not itself a baseline.
BASELINE_CONDITION = "R"

# Conditions that are baselines or references rather than treatments needing one.
NON_TREATMENT = {"R"}


@dataclass
class DesignFinding:
    severity: str            # "error" | "warning"
    experiment: str
    condition: str
    detail: str
    focal_models: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        models = f" [{', '.join(self.focal_models)}]" if self.focal_models else ""
        return (f"{self.severity.upper():<8} {self.experiment}/{self.condition}: "
                f"{self.detail}{models}")


def check_plan(experiment_config: Dict[str, Any],
               focal_specs: Dict[str, Any]) -> List[DesignFinding]:
    """
    Static check, before any money is spent.

    For every enabled experiment, each treatment condition must have the
    baseline `R` scheduled for the SAME focal models on the SAME pool — either
    inside that experiment or in another enabled experiment using the same pool.
    """
    from src.agents import resolve_focal_selector

    findings: List[DesignFinding] = []
    experiments = experiment_config.get("experiments", {})
    pools = experiment_config.get("pools", {})

    def covering_pools(pool: str) -> Set[str]:
        """A pool is covered by itself and by any pool it is a subset of."""
        chain, current = {pool}, pool
        while pools.get(current, {}).get("subset_of"):
            current = pools[current]["subset_of"]
            chain.add(current)
        return chain

    # Where is the baseline scheduled? pool -> set of focal models.
    baseline_coverage: Dict[str, Set[str]] = {}
    for name, spec in experiments.items():
        if not spec.get("enabled") or spec.get("offline"):
            continue
        if BASELINE_CONDITION not in (spec.get("conditions") or []):
            continue
        pool = spec.get("pool", "main300")
        models = set(resolve_focal_selector(spec.get("focal"), focal_specs))
        baseline_coverage.setdefault(pool, set()).update(models)

    for name, spec in experiments.items():
        if not spec.get("enabled") or spec.get("offline"):
            continue
        pool = spec.get("pool", "main300")
        models = set(resolve_focal_selector(spec.get("focal"), focal_specs))
        covered: Set[str] = set()
        for candidate in covering_pools(pool):
            covered |= baseline_coverage.get(candidate, set())

        for condition in spec.get("conditions") or []:
            if condition in NON_TREATMENT:
                continue
            missing = sorted(models - covered)
            if missing:
                findings.append(DesignFinding(
                    severity="error",
                    experiment=name,
                    condition=condition,
                    detail=(f"no '{BASELINE_CONDITION}' baseline scheduled on "
                            f"pool '{pool}' for {len(missing)} focal model(s)"),
                    focal_models=missing,
                ))

        if spec.get("round_sweep"):
            sweep_condition = spec["round_sweep"]["condition"]
            missing = sorted(models - covered)  # covered includes parent pools
            if missing:
                findings.append(DesignFinding(
                    severity="error", experiment=name,
                    condition=f"{sweep_condition}_round_sweep",
                    detail=f"no baseline on pool '{pool}'",
                    focal_models=missing,
                ))

    return findings


def check_collected(registry: pd.DataFrame,
                    protocol: str = "B") -> List[DesignFinding]:
    """
    Post-hoc check, against what was actually collected.

    Catches the case the static check cannot: an experiment was scheduled
    correctly but a cell failed to run (peer pool missing, provider outage,
    snapshot mismatch) and the matrix is quietly incomplete.
    """
    findings: List[DesignFinding] = []
    if registry.empty:
        return findings

    frame = registry[registry["protocol"] == protocol]
    if frame.empty:
        return findings

    for scope, scope_frame in frame.groupby("dataset_scope"):
        baseline = scope_frame[scope_frame["condition"] == BASELINE_CONDITION]
        baseline_models = set(baseline["focal_key"].dropna())

        for condition, condition_frame in scope_frame.groupby("condition"):
            if condition in NON_TREATMENT:
                continue
            models = set(condition_frame["focal_key"].dropna())
            missing = sorted(models - baseline_models)
            if missing:
                findings.append(DesignFinding(
                    severity="error", experiment=str(scope), condition=str(condition),
                    detail="treatment collected without a matched baseline",
                    focal_models=missing,
                ))

            # Same questions on both sides, per model.
            for model in sorted(models & baseline_models):
                treat_q = set(condition_frame[
                    condition_frame["focal_key"] == model]["question_identifier"])
                base_q = set(baseline[
                    baseline["focal_key"] == model]["question_identifier"])
                unmatched = treat_q - base_q
                if unmatched:
                    findings.append(DesignFinding(
                        severity="warning", experiment=str(scope),
                        condition=str(condition),
                        detail=(f"{len(unmatched)} question(s) present in the "
                                f"treatment but not in the baseline; they drop "
                                f"out of the paired contrast"),
                        focal_models=[model],
                    ))

        # Equal replication across models within a condition.
        for condition, condition_frame in scope_frame.groupby("condition"):
            replicates = condition_frame.groupby("focal_key")["replicate"].nunique()
            if replicates.nunique() > 1:
                findings.append(DesignFinding(
                    severity="error", experiment=str(scope), condition=str(condition),
                    detail=(f"unequal replication across focal models: "
                            f"{replicates.to_dict()}"),
                ))

    return findings


def report(findings: List[DesignFinding], context: str = "") -> bool:
    """Log the findings. Returns True when the design is complete."""
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]

    header = f"DESIGN CHECK{' — ' + context if context else ''}"
    print("\n" + "=" * 74)
    print(header)
    print("=" * 74)

    if not findings:
        print("  Every treatment condition has a matched baseline on the same")
        print("  focal models, questions and replicates.")
        return True

    for finding in errors + warnings:
        print(f"  {finding}")
    print(f"\n  {len(errors)} error(s), {len(warnings)} warning(s).")
    if errors:
        print("  An error means a treatment cannot be baseline-corrected — the")
        print("  capability gradient would fall back to the raw form, which is")
        print("  exactly the gap Reviewer oEjr (W2) and YVDD (W1) identified.")
    return not errors


def baseline_coverage_table(registry: pd.DataFrame,
                            protocol: str = "B") -> pd.DataFrame:
    """Per model: which conditions exist, and whether the baseline is there."""
    if registry.empty:
        return pd.DataFrame()
    frame = registry[registry["protocol"] == protocol]
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (scope, model), group in frame.groupby(["dataset_scope", "focal_key"]):
        conditions = sorted(group["condition"].dropna().unique())
        rows.append({
            "dataset_scope": scope,
            "focal_key": model,
            "has_baseline_R": BASELINE_CONDITION in conditions,
            "n_treatments": len([c for c in conditions if c not in NON_TREATMENT]),
            "conditions": ", ".join(conditions),
            "n_questions": int(group["question_identifier"].nunique()),
            "n_replicates": int(group["replicate"].nunique()),
        })
    return pd.DataFrame(rows).sort_values(["dataset_scope", "focal_key"])

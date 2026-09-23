"""
registry.py — one table behind every number in the paper.

Phase-2 defect (next_plan.md §2.4): `phase2_analyzer.py` grouped by (focal,
condition) with no dataset scope, so the perturbed-GSM8K trials — which reuse
the condition ids C1/C4 — were pooled with main-pool trials from other models.
The released `capability_sweep_analysis.json` therefore states DeepSeek solo
accuracy 0.2165 (the perturbed items) and Spearman rho -0.22, while the paper
states 0.762 and -0.95. Both cannot be right, and a reviewer who opens the
release sees the contradiction.

The registry keys every row by

    (protocol, dataset_scope, focal_key, focal_served_model,
     condition, question_identifier, replicate, round_index)

so a scope can never be pooled by accident. It carries BOTH protocols:

  * Protocol A — the released Phase-1/2 logs (stateless Round-1). Kept so the
    paper can show the capability gradient replicates under the published
    design as well as the repaired one.
  * Protocol B — Phase-3 (cached initial answer shown in every revision).

Exclusions applied here, once, and recorded in the returned report:
  * the 50-question GPT-4o-mini cross-validation subset (superseded);
  * the quarantined perturbed-GSM8K trials (invalid gold labels);
  * focal responses served by a checkpoint other than the pinned one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .metrics import summarise_by

logger = logging.getLogger("platos_ship3.registry")

# Legacy condition ids -> Protocol-B condition names, so both protocols can be
# read off one axis. C1 is the initial state, not a revision condition.
LEGACY_CONDITION_MAP = {
    "C1R_solo_reanswer": "R",
    "C2_three_smart": "E",
    "C2het_three_distinct_smart": "Ehet",
    "C3_two_smart_one_dumb": "WR1",
    "C4_one_smart_two_dumb": "WR",
    "C3H_two_smart_one_honest": "H1",
    "C4H_one_smart_two_honest": "H",
    "C4split_one_wrong_one_correct": "CR",
    "C5R_anchored_with_confidence_filter": "WRfilt",
    "C5H_honest_with_confidence_filter": "Hfilt",
    "C5_one_smart_two_dumb_confidence_weighted": "WRfilt_legacy_broken",
}

REGISTRY_COLUMNS = [
    "protocol", "dataset_scope", "experiment", "condition", "legacy_condition",
    "focal_key", "focal_served_model", "question_identifier", "source_dataset",
    "subject_category", "replicate", "round_index",
    "r0_is_correct", "is_correct", "extracted_answer", "r0_answer",
    "flip_correct_to_incorrect", "flip_incorrect_to_correct", "answer_changed",
    "peer_asserted_target", "adopted_peer_target", "n_peers_shown",
    "extracted_confidence", "answer_extraction_method",
    "total_input_tokens", "total_output_tokens",
]


def _scope_for_question(question_id: str) -> str:
    """Dataset scope from the question id prefix — never pool across these."""
    qid = str(question_id)
    if qid.startswith("gsm8k_perturbed"):
        return "perturbed_quarantined"
    if qid.startswith("gsmsym_"):
        return "gsm_symbolic"
    if qid.startswith("gsmorig_"):
        return "gsm8k_matched_original"
    return "main300"


def load_protocol_b(paths: Dict[str, str], project_root: Path) -> pd.DataFrame:
    """Read the Phase-3 revision log into registry shape."""
    path = Path(paths["revision_log_file"])
    if not path.is_absolute():
        path = project_root / paths["revision_log_file"]
    if not path.exists():
        logger.info("No Protocol-B revision log at %s yet.", path)
        return pd.DataFrame(columns=REGISTRY_COLUMNS)

    frame = pd.read_parquet(path)
    frame["protocol"] = "B"
    frame["legacy_condition"] = None
    frame["dataset_scope"] = frame["question_identifier"].map(_scope_for_question)
    return frame


def _legacy_focal_frame(trial_log: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce a legacy trial log to one row per (trial, round) for the focal agent,
    then pivot Round-0/Round-1 onto a single revision row.
    """
    focal = trial_log[trial_log["responding_agent_role"] == "smart_focal"].copy()
    if focal.empty:
        return pd.DataFrame()

    keys = ["trial_universal_unique_identifier"]
    wide = focal.pivot_table(
        index=keys + ["condition_identifier", "focal_smart_agent_name",
                      "question_identifier", "trial_replication_index"],
        columns="debate_round_index",
        values="extracted_answer_matches_ground_truth",
        aggfunc="first",
    ).reset_index()

    answers = focal.pivot_table(
        index=keys, columns="debate_round_index",
        values="extracted_final_answer", aggfunc="first",
    ).reset_index()
    answers.columns = [
        c if not isinstance(c, (int, float)) else f"answer_round{int(c)}"
        for c in answers.columns
    ]

    # Record every distinct served string for the trial, not just the last one:
    # a trial whose Round 0 came from the pinned snapshot and whose Round 1 came
    # from a fallback is still contaminated, and keeping only the last value
    # would hide half of them.
    served = (
        focal.sort_values("debate_round_index")
        .groupby(keys)["responding_agent_model_name"]
        .agg(lambda values: "|".join(sorted(set(str(v) for v in values))))
        .reset_index()
        .rename(columns={"responding_agent_model_name": "focal_served_model"})
    )

    merged = wide.merge(answers, on=keys, how="left").merge(served, on=keys, how="left")
    round_columns = [c for c in merged.columns if isinstance(c, (int, float))]
    rename = {c: f"correct_round{int(c)}" for c in round_columns}
    return merged.rename(columns=rename)


def load_protocol_a(paths: Dict[str, str], project_root: Path) -> pd.DataFrame:
    """
    Read the released Phase-1 and Phase-2 trial logs into registry shape.

    A legacy "revision" is Round 1; the initial state is that condition's own
    Round 0 (which is exactly the confound Protocol B removes — recorded here
    so the difference is visible rather than assumed).
    """
    frames: List[pd.DataFrame] = []
    for key in ("legacy_phase1_trial_log", "legacy_phase2_trial_log"):
        raw = paths.get(key)
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = (project_root / raw).resolve()
        if not path.exists():
            logger.info("Legacy log absent: %s", path)
            continue
        logger.info("Reading legacy log %s", path)
        frames.append(pd.read_parquet(path))

    if not frames:
        return pd.DataFrame(columns=REGISTRY_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    reduced = _legacy_focal_frame(combined)
    if reduced.empty:
        return pd.DataFrame(columns=REGISTRY_COLUMNS)

    has_round1 = "correct_round1" in reduced.columns
    out = pd.DataFrame(
        {
            "protocol": "A",
            "experiment": "legacy",
            "legacy_condition": reduced["condition_identifier"],
            "condition": reduced["condition_identifier"].map(LEGACY_CONDITION_MAP),
            "focal_key": reduced["focal_smart_agent_name"],
            "focal_served_model": reduced.get("focal_served_model"),
            "question_identifier": reduced["question_identifier"],
            "replicate": reduced["trial_replication_index"],
            "round_index": 1,
            "r0_is_correct": reduced["correct_round0"].astype("boolean"),
            "is_correct": (
                reduced["correct_round1"] if has_round1 else reduced["correct_round0"]
            ).astype("boolean"),
            "r0_answer": reduced.get("answer_round0"),
            "extracted_answer": (
                reduced.get("answer_round1") if has_round1
                else reduced.get("answer_round0")
            ),
        }
    )
    out["dataset_scope"] = out["question_identifier"].map(_scope_for_question)
    out["source_dataset"] = np.where(
        out["question_identifier"].astype(str).str.startswith("gsm"), "gsm8k", "mmlu_pro"
    )
    out["flip_correct_to_incorrect"] = (
        out["r0_is_correct"].fillna(False) & ~out["is_correct"].fillna(False))
    out["flip_incorrect_to_correct"] = (
        ~out["r0_is_correct"].fillna(False) & out["is_correct"].fillna(False))
    out["answer_changed"] = out["extracted_answer"].astype(str) != out["r0_answer"].astype(str)

    # C1 (solo) has no Round 1: it is the initial state, not a revision.
    out = out[out["legacy_condition"] != "C1_smart_solo"]
    return out


def build_registry(
    paths: Dict[str, str],
    project_root: Path,
    drop_quarantined: bool = True,
    drop_crossval50: bool = True,
    enforce_snapshots: bool = True,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build the unified registry and an exclusions report.

    The report is written into the paper's appendix verbatim, so every drop is
    declared rather than discovered.
    """
    protocol_b = load_protocol_b(paths, project_root)
    protocol_a = load_protocol_a(paths, project_root)

    frames = [f for f in (protocol_a, protocol_b) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=REGISTRY_COLUMNS), {"rows": 0}

    registry = pd.concat(frames, ignore_index=True)
    report: Dict[str, Any] = {"rows_before_exclusions": int(len(registry))}

    if drop_quarantined:
        mask = registry["dataset_scope"] == "perturbed_quarantined"
        report["dropped_perturbed_quarantined"] = int(mask.sum())
        report["dropped_perturbed_reason"] = (
            "gold labels invalid: every integer was scaled by 4, including "
            "percentages and fractions; see tools/quarantine_perturbed.py"
        )
        registry = registry[~mask]

    if drop_crossval50:
        # Phase 1 ran GPT-4o-mini on a 50-question subset later superseded by
        # the full 300-question run; pooling them double-counts those items.
        legacy_gpt = registry["focal_key"] == "openrouter_gpt4o_mini"
        report["dropped_superseded_crossval50"] = int(legacy_gpt.sum())
        registry = registry[~legacy_gpt]

    if enforce_snapshots:
        # Phase-2 LinkAPI fallback leak: 75 "gpt-4o-mini" responses came from
        # gpt-4.1-mini. Excluded from every reported number.
        served = registry["focal_served_model"].astype(str)
        leaked = (registry["focal_key"] == "gpt4o_mini") & served.str.contains(
            "4.1-mini", case=False, na=False)
        report["dropped_snapshot_mismatch"] = int(leaked.sum())
        report["snapshot_mismatch_detail"] = (
            "gpt4o_mini rows served by gpt-4.1-mini via the LinkAPI fallback chain"
        )
        registry = registry[~leaked]

    registry = registry.reset_index(drop=True)
    for column in REGISTRY_COLUMNS:
        if column not in registry.columns:
            registry[column] = pd.NA

    report["rows_after_exclusions"] = int(len(registry))
    report["protocols"] = registry["protocol"].value_counts().to_dict()
    report["dataset_scopes"] = registry["dataset_scope"].value_counts().to_dict()
    report["focal_models"] = sorted(registry["focal_key"].dropna().unique().tolist())
    logger.info("Registry: %d rows | %s", len(registry), report["protocols"])
    return registry[REGISTRY_COLUMNS + [
        c for c in registry.columns if c not in REGISTRY_COLUMNS]], report


def coverage_table(registry: pd.DataFrame) -> pd.DataFrame:
    """
    Condition x focal x protocol counts — the table Reviewer YVDD (W1) asked
    for. Any cell that is not identical across models is visible immediately.
    """
    if registry.empty:
        return pd.DataFrame()
    table = (
        registry.groupby(["protocol", "dataset_scope", "condition", "focal_key"])
        .agg(n_units=("is_correct", "size"),
             n_questions=("question_identifier", "nunique"),
             n_replicates=("replicate", "nunique"))
        .reset_index()
    )
    return table


def cell_metrics(registry: pd.DataFrame) -> pd.DataFrame:
    """Every rate for every (protocol, scope, condition, focal) cell."""
    if registry.empty:
        return pd.DataFrame()
    usable = registry.dropna(subset=["condition", "is_correct", "r0_is_correct"]).copy()
    usable["is_correct"] = usable["is_correct"].astype(bool)
    usable["r0_is_correct"] = usable["r0_is_correct"].astype(bool)
    return summarise_by(
        usable, ["protocol", "dataset_scope", "condition", "focal_key"]
    )

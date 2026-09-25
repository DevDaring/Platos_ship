"""
run_analysis.py — offline: logs in, every reported number out.

No API key, no GPU, no network. `python3 run_all.py --analyse` runs this, and
so does a clean checkout of the release, which is what makes the paper's
"regenerates every number" claim checkable rather than asserted.

Order:
    registry -> coverage -> cell metrics -> pre-registered contrasts
    -> capability gradient -> voting baseline -> retention gap
    -> safeguard -> tables -> figures -> paper_numbers.json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from . import PROJECT_ROOT  # noqa: F401  — importing this puts src/ on sys.path
from .contrasts import per_model_contrasts, run_family
from .gate import build_gate_report, distinct_messages, legacy_loudness_gate
from .metrics import accounting_identity_check
from .registry import build_registry, cell_metrics, coverage_table
from .voting import budget_matched_comparison, voting_baseline

logger = logging.getLogger("platos_ship3.run_analysis")


def _resolve(project_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (project_root / raw).resolve()


def _load_configs(project_root: Path) -> Dict[str, Any]:
    with open(project_root / "config" / "experiment.yaml") as handle:
        experiment = yaml.safe_load(handle)
    with open(project_root / "config" / "paths.yaml") as handle:
        paths = yaml.safe_load(handle)
    return {"experiment": experiment, "paths": paths}


def _peer_messages_with_correctness(
    project_root: Path, paths: Dict[str, str]
) -> pd.DataFrame:
    """
    Join the peer-message log to the honest bank so each message carries its
    own correctness — the retention gap needs P(retained | correct/wrong).
    """
    peer_path = _resolve(project_root, paths["peer_message_log_file"])
    if not peer_path.exists():
        return pd.DataFrame()
    peers = pd.read_parquet(peer_path)

    bank_path = _resolve(project_root, paths["honest_bank_file"])
    if bank_path.exists():
        bank = pd.read_parquet(bank_path)[
            ["message_text", "is_correct"]
        ].rename(columns={"message_text": "peer_text",
                          "is_correct": "peer_is_correct"})
        peers = peers.merge(bank.drop_duplicates("peer_text"), on="peer_text",
                            how="left")
    if "peer_is_correct" not in peers.columns:
        peers["peer_is_correct"] = pd.NA

    # Wrong-anchored and hedged peers are wrong by construction.
    anchored = peers["anchor_mode"].isin(["wrong", "hedged"])
    peers.loc[anchored, "peer_is_correct"] = False
    correct_anchored = peers["anchor_mode"] == "correct"
    peers.loc[correct_anchored, "peer_is_correct"] = True
    return peers


def _truncation_and_parsing(project_root: Path,
                            paths: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Per model and stage: how often the output hit the token cap and how often
    no answer could be read. A missing answer is scored wrong, so both rates
    are reported next to every result rather than folded into accuracy.
    """
    from src.call_guard import drop_failed_calls

    rows = []
    for stage, key in (("round0", "r0_cache_file"), ("revision", "revision_log_file")):
        path = _resolve(project_root, paths[key])
        if not path.exists():
            continue
        log = drop_failed_calls(pd.read_parquet(path), stage)
        for focal, group in log.groupby("focal_key"):
            finish = group.get("finish_reason")
            rows.append({
                "stage": stage, "focal_key": focal, "n": int(len(group)),
                "share_truncated": (float(finish.astype(str).eq("length").mean())
                                    if finish is not None else None),
                "share_unparsed": float(group["extracted_answer"].isna().mean()),
                "share_empty_text": float(group["raw_response_text"].fillna("")
                                          .str.strip().eq("").mean()),
            })
    return rows


def _filter_firing(project_root: Path, paths: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    How often the confidence filter actually removed a peer.

    WRfilt can only differ from WRconf on units where the filter fired. If it
    fires rarely, a null WRfilt-vs-WRconf contrast says little about the
    filter, so the firing rate is reported next to the contrast.
    """
    path = _resolve(project_root, paths["revision_log_file"])
    if not path.exists():
        return []
    from src.call_guard import drop_failed_calls

    log = drop_failed_calls(pd.read_parquet(path), "revision log")
    if "filter_enabled" not in log.columns:
        return []
    filtered = log[log["filter_enabled"].fillna(False).astype(bool)
                   & (log.get("round_index", 1) == 1)]
    rows = []
    for (condition, focal), group in filtered.groupby(["condition", "focal_key"]):
        before = pd.to_numeric(group["n_peers_before_filter"], errors="coerce")
        kept = pd.to_numeric(group["n_peers_retained"], errors="coerce")
        rows.append({
            "condition": condition, "focal_key": focal, "n_units": int(len(group)),
            "share_units_filter_dropped_any": float((kept < before).mean()),
            "share_units_filter_dropped_all": float((kept == 0).mean()),
            "mean_peers_retained": float(kept.mean()),
        })
    return rows


def run_full_analysis(project_root: Path) -> Dict[str, Any]:
    """Run everything and write the artefacts. Returns paper_numbers."""
    configs = _load_configs(project_root)
    experiment, paths = configs["experiment"], configs["paths"]
    analysis_config = experiment.get("analysis", {})
    output_dir = _resolve(project_root, paths["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── registry ──────────────────────────────────────────────────────────
    registry, exclusions = build_registry(paths, project_root)
    if registry.empty:
        logger.warning("Registry is empty — nothing to analyse yet.")
        return {"exclusions": exclusions}
    registry.to_parquet(_resolve(project_root, paths["registry_file"]), index=False)

    coverage = coverage_table(registry)
    coverage.to_parquet(output_dir / "coverage_table.parquet", index=False)

    # Post-hoc design check: did every treatment actually land with a matched
    # baseline on the same models and questions? The static check cannot see a
    # cell that failed mid-run.
    from .design_check import baseline_coverage_table, check_collected, report

    design_findings = check_collected(registry, protocol="B")
    design_ok = report(design_findings, context="collected Protocol-B data")
    baseline_table = baseline_coverage_table(registry, protocol="B")
    if not baseline_table.empty:
        baseline_table.to_parquet(output_dir / "baseline_coverage.parquet",
                                  index=False)

    metrics = cell_metrics(registry)
    metrics.to_parquet(_resolve(project_root, paths["metrics_file"]), index=False)

    # ── solo accuracy (the capability x-axis) ─────────────────────────────
    from src.r0_cache import load_r0_cache, solo_accuracy_by_focal

    r0 = load_r0_cache(_resolve(project_root, paths["r0_cache_file"]))
    solo_accuracy = solo_accuracy_by_focal(r0)
    if not solo_accuracy:
        # Fall back to the legacy solo condition so Protocol-A analysis still
        # runs before any Phase-3 call has been made.
        legacy = registry[(registry["protocol"] == "A")]
        if not legacy.empty:
            solo_accuracy = (
                legacy.groupby("focal_key")["r0_is_correct"].mean().to_dict()
            )

    # ── pre-registered contrast families ──────────────────────────────────
    all_contrasts: List[Dict[str, Any]] = []
    gradients: Dict[str, Any] = {}
    for family_name, specifications in analysis_config.get(
            "contrast_families", {}).items():
        results, family_gradients = run_family(
            registry, family_name, specifications, solo_accuracy,
            alpha=float(analysis_config.get("alpha", 0.05)),
            n_resamples=int(analysis_config.get("bootstrap_resamples", 5000)),
        )
        all_contrasts.extend(r.to_dict() for r in results)
        gradients.update(family_gradients)

    contrasts_frame = pd.DataFrame(all_contrasts)
    if not contrasts_frame.empty:
        contrasts_frame.to_parquet(
            _resolve(project_root, paths["contrasts_file"]), index=False)

    # Per-model contrasts for the main results table.
    per_model = per_model_contrasts(registry, "WR", "R", "accuracy_delta")
    per_model_harm = per_model_contrasts(registry, "WR", "R", "harmful_delta")
    if not per_model.empty:
        per_model.to_parquet(output_dir / "per_model_accuracy_contrasts.parquet",
                             index=False)
    if not per_model_harm.empty:
        per_model_harm.to_parquet(output_dir / "per_model_harmful_contrasts.parquet",
                                  index=False)

    for name, gradient in gradients.items():
        table = gradient.get("per_model")
        if isinstance(table, pd.DataFrame) and not table.empty:
            table.to_parquet(output_dir / f"gradient_{name}.parquet", index=False)

    # ── voting baseline (X5) ──────────────────────────────────────────────
    # Main pool only: X4's GSM-Symbolic Round-0 exists for some models only.
    from src.scopes import main_scope_only

    r0_main = main_scope_only(r0)
    voting = voting_baseline(r0_main) if not r0_main.empty else pd.DataFrame()
    if not voting.empty:
        voting.to_parquet(_resolve(project_root, paths["voting_file"]), index=False)
        budget = budget_matched_comparison(voting, metrics)
        budget.to_parquet(output_dir / "budget_matched_comparison.parquet", index=False)

    # ── retention gap (X7) ────────────────────────────────────────────────
    peers = _peer_messages_with_correctness(project_root, paths)
    peers = distinct_messages(peers)
    gate_report = build_gate_report(peers) if not peers.empty else pd.DataFrame()
    if not gate_report.empty:
        gate_report.to_parquet(
            _resolve(project_root, paths["gate_report_file"]), index=False)
    legacy_gate = (
        legacy_loudness_gate(peers) if not peers.empty else {}
    )

    # ── safeguard (X6) ────────────────────────────────────────────────────
    safeguard_table = pd.DataFrame()
    safeguard_path = _resolve(project_root, paths["safeguard_file"])
    revision_path = _resolve(project_root, paths["revision_log_file"])
    if safeguard_path.exists() and revision_path.exists():
        from .safeguard import score_policies, verifier_cost

        from src.call_guard import drop_failed_calls
        from .safeguard import x6_scoring_units

        verified = drop_failed_calls(pd.read_parquet(safeguard_path), "verifier")
        revisions = drop_failed_calls(pd.read_parquet(revision_path), "revision log")
        subset = x6_scoring_units(revisions, experiment, paths, project_root)
        # One table per condition: WR and H differ in how often the model
        # proposes a change and in whether the change is right, so a pooled
        # policy score would mostly report the WR/H mix.
        tables = []
        for (condition, focal), units in (
                [((c, "ALL"), g) for c, g in subset.groupby("condition")]
                + [(k, g) for k, g in subset.groupby(["condition", "focal_key"])]):
            table = score_policies(verified[verified["unit_id"].isin(units["unit_id"])],
                                   units)
            if not table.empty:
                tables.append(table.assign(condition=condition, focal_key=focal))
        safeguard_table = (pd.concat(tables, ignore_index=True)
                           if tables else pd.DataFrame())
        if not safeguard_table.empty:
            safeguard_table.to_parquet(
                output_dir / "safeguard_policies.parquet", index=False)
        cost = verifier_cost(verified[verified["unit_id"].isin(subset["unit_id"])],
                             subset)
    else:
        cost = {}

    # ── sanity: the accounting identity holds in every cell ───────────────
    identity_rows = []
    usable = registry.dropna(subset=["condition", "is_correct", "r0_is_correct"]).copy()
    usable["is_correct"] = usable["is_correct"].astype(bool)
    usable["r0_is_correct"] = usable["r0_is_correct"].astype(bool)
    for key, group in usable.groupby(["protocol", "condition", "focal_key"]):
        check = accounting_identity_check(group)
        identity_rows.append({"protocol": key[0], "condition": key[1],
                              "focal_key": key[2], **check})
    identity = pd.DataFrame(identity_rows)
    if not identity.empty:
        identity.to_parquet(output_dir / "accounting_identity_check.parquet",
                            index=False)

    # ── secondary: GEE ladder and worst-case parse bounds (plan §5) ───────
    from .secondary import adoption_excess, gee_condition_ladder, parse_failure_bounds

    gee = gee_condition_ladder(registry)
    if not gee.empty:
        gee.to_parquet(output_dir / "gee_condition_ladder.parquet", index=False)
    bounds = parse_failure_bounds(registry, analysis_config.get("contrast_families", {}))
    if not bounds.empty:
        bounds.to_parquet(output_dir / "parse_failure_bounds.parquet", index=False)

    # ── tables and figures ────────────────────────────────────────────────
    from .make_figures import make_all_figures
    from .make_tables import make_all_tables

    make_all_tables(project_root, registry, metrics, coverage, contrasts_frame,
                    gradients, voting, gate_report, safeguard_table)
    make_all_figures(project_root, registry, metrics, gradients, voting)

    # ── one JSON with every headline number ───────────────────────────────
    from src.store import write_json

    paper_numbers: Dict[str, Any] = {
        "exclusions": exclusions,
        "solo_accuracy_by_focal": solo_accuracy,
        "contrasts": all_contrasts,
        "capability_gradients": {
            name: {k: v for k, v in gradient.items() if k != "per_model"}
            for name, gradient in gradients.items()
        },
        "voting_baseline": voting.to_dict("records") if not voting.empty else [],
        "retention_gap": (gate_report.to_dict("records")
                          if not gate_report.empty else []),
        "legacy_loudness_gate": legacy_gate,
        "safeguard_policies": (safeguard_table.to_dict("records")
                               if not safeguard_table.empty else []),
        "verifier_cost": cost,
        "confidence_filter_firing": _filter_firing(project_root, paths),
        "truncation_and_parsing": _truncation_and_parsing(project_root, paths),
        "gee_condition_ladder": gee.to_dict("records") if not gee.empty else [],
        "parse_failure_bounds": bounds.to_dict("records") if not bounds.empty else [],
        "adoption_excess_exploratory": adoption_excess(registry, solo_accuracy),
        "accounting_identity_max_deviation": (
            float(identity["absolute_difference"].max()) if not identity.empty
            else None),
        "coverage_is_balanced": _coverage_is_balanced(coverage),
        "design_check": {
            "passed": bool(design_ok),
            "findings": [
                {"severity": f.severity, "scope": f.experiment,
                 "condition": f.condition, "detail": f.detail,
                 "focal_models": f.focal_models}
                for f in design_findings
            ],
        },
    }
    write_json(_resolve(project_root, paths["paper_numbers_file"]), paper_numbers)
    logger.info("Analysis complete -> %s", paths["paper_numbers_file"])
    return paper_numbers


def _coverage_is_balanced(coverage: pd.DataFrame) -> Dict[str, Any]:
    """
    Check the thing Reviewer YVDD objected to: do all focal models have the
    same questions and replicates in every Protocol-B condition?
    """
    if coverage.empty:
        return {"checked": False}
    protocol_b = coverage[coverage["protocol"] == "B"]
    if protocol_b.empty:
        return {"checked": False}
    issues = []
    for (scope, condition), group in protocol_b.groupby(["dataset_scope", "condition"]):
        if group["n_questions"].nunique() > 1 or group["n_replicates"].nunique() > 1:
            issues.append({
                "dataset_scope": scope,
                "condition": condition,
                "n_questions": sorted(group["n_questions"].unique().tolist()),
                "n_replicates": sorted(group["n_replicates"].unique().tolist()),
            })
    return {"checked": True, "balanced": not issues, "imbalances": issues}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_full_analysis(Path(__file__).resolve().parent.parent)

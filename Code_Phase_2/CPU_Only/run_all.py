#!/usr/bin/env python3
"""
run_all.py — Single entry point for Phase 2 (reviewer-driven follow-up).

    python3 run_all.py --list                 # show the experiment plan
    python3 run_all.py --dry-run              # 1 trial/condition smoke test, all wired experiments
    python3 run_all.py --p0                   # run every enabled P0 experiment (E1–E4)
    python3 run_all.py --experiments E1,E3    # run a specific subset
    python3 run_all.py --all                  # run every enabled experiment
    python3 run_all.py --analyse-only         # recompute metrics/gates/sweep from saved parquets

Design (matches the user's conventions):
  * single entry point, resume-capable checkpointing (TrialRunner is idempotent),
  * round-robin keys + robust answer extraction (inherited from Phase 1 core),
  * multi-provider routing so each model runs on its cheapest source (models.yaml).

Reviewer mapping — see Review_Fix.md and API_AND_RECHARGE.md.
"""

import os
import sys
import json
import argparse
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.resolve()


# ── logging ────────────────────────────────────────────────────────────────
def setup_logging(project_root: Path):
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = logging.FileHandler(logs_dir / "phase2_run.log")
    fh.setFormatter(fmt)
    api = logging.FileHandler(logs_dir / "api_failures.log")
    api.setLevel(logging.WARNING)
    api.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(ch)
    logging.getLogger("platos_ship.api_failures").addHandler(api)


logger = logging.getLogger("platos_ship.run_all")


def load_env(project_root: Path):
    """Load .env from Code_Phase_2/ (parent) first, then a local override."""
    parent_env = project_root.parent / ".env"
    if parent_env.exists():
        load_dotenv(parent_env)
        logger.info(f"Loaded env from {parent_env}")
    local_env = project_root / ".env"
    if local_env.exists():
        load_dotenv(local_env, override=True)
        logger.info(f"Loaded env override from {local_env}")


def load_configs(project_root: Path):
    with open(project_root / "config" / "experiment.yaml") as f:
        exp = yaml.safe_load(f)
    with open(project_root / "config" / "paths.yaml") as f:
        paths = yaml.safe_load(f)
    return exp, paths


# ── subset resolution ──────────────────────────────────────────────────────
def resolve_subset(name: str, questions_df: pd.DataFrame, exp: Dict, project_root: Path,
                   paths: Dict) -> pd.DataFrame:
    seed = exp["random_seed"]
    if name == "full300":
        return questions_df
    if name in ("crossval300",):
        return questions_df
    if name == "crossval50":
        n = min(exp["cross_model_validation_subset_size"], len(questions_df))
        return questions_df.sample(n=n, random_state=seed).reset_index(drop=True)
    if name == "mitigation100":
        return questions_df[questions_df["included_in_mitigation_subset"] == True].reset_index(drop=True)
    if name == "perturbed100":
        pp = paths["perturbed_gsm8k_pool_file"]
        if not Path(pp).is_absolute():
            pp = project_root / pp
        if not Path(pp).exists():
            raise FileNotFoundError("Perturbed GSM8K pool missing — run build for E6 first.")
        return pd.read_parquet(str(pp))
    raise ValueError(f"Unknown subset '{name}'")


def write_metadata(project_root: Path, paths: Dict, agents: Dict, exp: Dict, to_run: Dict):
    """
    Write Phase-2 provenance to experiment_metadata.json (symmetry with Phase 1:
    served model strings, providers, seed, and the experiments run).
    """
    def _describe(block):
        out = {}
        for k, ag in block.items():
            out[k] = {"provider": getattr(ag, "provider", "?"),
                      "model": getattr(ag, "model_name", "?")}
        return out

    md = {
        "phase": 2,
        "run_timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "random_seed_value": exp["random_seed"],
        "experiments_run": list(to_run.keys()),
        "smart_agents": _describe(agents["smart_agents"]),
        "dumb_agents": _describe(agents["dumb_agents"]),
        "sweep_focal_agents": _describe(agents["sweep_focal_agents"]),
        "heterogeneous_agents": _describe(agents["heterogeneous_agents"]),
        "note_deepseek_alias": "deepseek-chat alias retires 2026-07-24; pin deepseek-v4-flash for later runs.",
    }
    p = paths["experiment_metadata_file"]
    if not Path(p).is_absolute():
        p = project_root / p
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(str(p), "w") as f:
        json.dump(md, f, indent=2, default=str)
    logger.info(f"Provenance written to {p}")


# ── main orchestration ─────────────────────────────────────────────────────
def run(project_root: Path, selected: Optional[List[str]], priorities: Optional[List[str]],
        dry_run: bool, analyse_only: bool):
    from src.dataset_builder import build_question_pool
    from src.persona_generator import generate_all_personas
    from src.persona_validator import validate_and_regenerate
    from src.trial_runner import TrialRunner
    from src.phase2_agents import initialize_phase2_agents
    from src.metrics_calculator import compute_and_save_metrics
    from src.corrected_gate import run_corrected_gate
    from src.phase2_analyzer import run_phase2_analysis, run_phase2_statistics
    from src.perturbed_gsm8k import build_perturbed_gsm8k_pool

    exp, paths = load_configs(project_root)
    experiments = exp["experiments"]

    # Decide which experiments to run
    to_run = {}
    for eid, ecfg in experiments.items():
        short = eid.split("_")[0]  # "E1_solo_reanswer" -> "E1"
        if eid == "E9_mechanistic_probe":
            continue  # GPU-only; lives in GPU_Only/
        if analyse_only:
            continue
        if selected is not None and short not in selected and eid not in selected:
            continue
        if priorities is not None and ecfg.get("priority") not in priorities:
            continue
        if not ecfg.get("enabled", False) and selected is None:
            continue  # only auto-run enabled ones unless explicitly named
        to_run[eid] = ecfg

    need_sweep = any(c.get("focal") == "SWEEP" for c in to_run.values())
    need_het = any(c.get("focal") == "HET" for c in to_run.values())
    need_split = any("C4split_one_wrong_one_correct" in c.get("conditions", []) for c in to_run.values())
    need_perturbed = any(c.get("subset") == "perturbed100" for c in to_run.values())
    need_confidence_pool = any(
        "C5R_anchored_with_confidence_filter" in c.get("conditions", []) for c in to_run.values()
    )

    logger.info("=" * 66)
    logger.info(f"PHASE 2 — {'DRY RUN' if dry_run else 'FULL RUN'}")
    logger.info(f"Experiments queued: {list(to_run.keys()) or '(none — analyse only)'}")
    logger.info("=" * 66)

    # Build agents
    agents = initialize_phase2_agents(project_root, include_sweep=need_sweep, include_heterogeneous=need_het)
    smart_agents = dict(agents["smart_agents"])
    smart_agents.update(agents["sweep_focal_agents"])  # sweep focals resolvable by TrialRunner
    dumb_agents = agents["dumb_agents"]
    judge = agents["judge_cascade"]

    # Question pool + personas (probe/generation via a weak agent)
    weak_probe = next(iter(dumb_agents.values()))
    questions_df = build_question_pool(project_root, probe_agent=weak_probe, dry_run=dry_run)

    personas_df = generate_all_personas(project_root, weak_probe, dry_run=dry_run)
    personas_df = validate_and_regenerate(
        personas_df, weak_probe, questions_df,
        max_regeneration_attempts=1 if dry_run else 3, project_root=project_root,
    )

    correct_personas_df = None
    if need_split:
        correct_personas_df = generate_all_personas(
            project_root, weak_probe, dry_run=dry_run,
            anchor_mode="correct", output_path_key="correct_anchored_personas_file",
            questions_df=questions_df,
        )

    # E3 confidence pool — SEPARATE from the Phase-1-symmetric main pool
    confidence_personas_df = None
    if need_confidence_pool:
        confidence_personas_df = generate_all_personas(
            project_root, weak_probe, dry_run=dry_run,
            anchor_mode="wrong", output_path_key="confidence_personas_file",
            questions_df=questions_df, include_confidence_line_override=True,
        )
        confidence_personas_df = validate_and_regenerate(
            confidence_personas_df, weak_probe, questions_df,
            max_regeneration_attempts=1 if dry_run else 3, project_root=project_root,
        )

    if need_perturbed and not analyse_only:
        build_perturbed_gsm8k_pool(project_root, verify_agent=weak_probe, dry_run=dry_run)

    write_metadata(project_root, paths, agents, exp, to_run)

    runner = TrialRunner(
        project_root=project_root,
        smart_agents=smart_agents,
        dumb_agents=dumb_agents,
        judge_cascade=judge,
        personas_df=personas_df,
        questions_df=questions_df,
        dry_run=dry_run,
        heterogeneous_agents=agents["heterogeneous_agents"],
        correct_personas_df=correct_personas_df,
        confidence_personas_df=confidence_personas_df,
    )

    # ── run each experiment ──
    for eid, ecfg in to_run.items():
        logger.info("-" * 66)
        logger.info(f"{eid} [{ecfg.get('priority')}] — {ecfg.get('description', '')}")
        try:
            conditions = ecfg["conditions"]
            tpq = 1 if dry_run else ecfg.get("trials_per_question", 5)
            subset_df = resolve_subset(ecfg["subset"], questions_df, exp, project_root, paths)

            # Resolve focal list
            focal_spec = ecfg["focal"]
            if focal_spec == "SWEEP":
                focal_names = list(agents["sweep_focal_agents"].keys())
            elif focal_spec == "HET":
                het_keys = exp["phase2_conditions"]["C2het_three_distinct_smart"]["heterogeneous_smart_agent_keys"]
                focal_names = [het_keys[0]]  # label only; het path ignores focal
            else:
                focal_names = focal_spec

            for focal in focal_names:
                runner.run_conditions(
                    conditions=conditions,
                    focal_agent_name=focal,
                    question_filter=subset_df,
                    trials_per_question=tpq,
                    stage_name=f"{eid}:{focal}",
                )
        except Exception as e:
            logger.error(f"{eid} failed: {e}", exc_info=True)

    # ── analysis (always, even for --analyse-only) ──
    logger.info("=" * 66)
    logger.info("ANALYSIS")
    logger.info("=" * 66)
    for name, fn in [
        ("per-condition metrics", lambda: compute_and_save_metrics(project_root)),
        ("corrected calibration gate", lambda: run_corrected_gate(project_root)),
        ("phase2 sweep/contrast analysis", lambda: run_phase2_analysis(project_root)),
        ("phase2 paired McNemar statistics", lambda: run_phase2_statistics(project_root)),
    ]:
        try:
            fn()
            logger.info(f"  ok: {name}")
        except Exception as e:
            logger.warning(f"  skipped {name}: {e}")

    logger.info("PHASE 2 RUN COMPLETE")


def list_plan(project_root: Path):
    exp, _ = load_configs(project_root)
    print("\nPhase 2 experiment plan (config/experiment.yaml):\n")
    for eid, ecfg in exp["experiments"].items():
        flag = "on " if ecfg.get("enabled") else "off"
        print(f"  [{flag}] {ecfg.get('priority','?'):<3} {eid}")
        print(f"        {ecfg.get('description','')}")
        print(f"        focal={ecfg.get('focal')}  conditions={ecfg.get('conditions')}  "
              f"subset={ecfg.get('subset')}  reps={ecfg.get('trials_per_question')}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Plato's Ship — Phase 2 runner")
    ap.add_argument("--list", action="store_true", help="Print the experiment plan and exit")
    ap.add_argument("--dry-run", action="store_true", help="1 trial/condition smoke test")
    ap.add_argument("--all", action="store_true", help="Run every ENABLED experiment")
    ap.add_argument("--p0", action="store_true", help="Run enabled P0 experiments (E1–E4)")
    ap.add_argument("--p1", action="store_true", help="Run enabled P0+P1 experiments")
    ap.add_argument("--experiments", type=str, default=None,
                    help="Comma list, e.g. E1,E3 (overrides enabled flag)")
    ap.add_argument("--analyse-only", action="store_true", help="Recompute analysis from saved parquets")
    args = ap.parse_args()

    setup_logging(PROJECT_ROOT)
    load_env(PROJECT_ROOT)

    if args.list:
        list_plan(PROJECT_ROOT)
        return

    selected = None
    if args.experiments:
        selected = [s.strip() for s in args.experiments.split(",") if s.strip()]

    priorities = None
    if args.p0:
        priorities = ["P0"]
    elif args.p1:
        priorities = ["P0", "P1"]

    run(PROJECT_ROOT, selected=selected, priorities=priorities,
        dry_run=args.dry_run, analyse_only=args.analyse_only)


if __name__ == "__main__":
    main()

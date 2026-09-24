#!/usr/bin/env python3
"""
run_all.py — the single entry point for Phase 3.

    python3 run_all.py --list                  # show the plan, cost nothing
    python3 run_all.py --dry-run               # 1 unit per cell, all wiring
    python3 run_all.py --prepare               # pools only (hedged, honest, X4)
    python3 run_all.py --experiments X1,X2     # run named experiments
    python3 run_all.py --priority P0           # run everything at a priority
    python3 run_all.py --all                   # prepare + run + analyse
    python3 run_all.py --analyse               # offline: registry, contrasts,
                                               #   tables, figures (no API)

Resumable: every unit is checkpointed, so a killed run restarts where it
stopped and never pays for the same call twice.

Order matters and is enforced:
    artefacts -> R0 cache -> message banks -> revision conditions -> analysis
A revision condition cannot run before the initial answers it clones exist.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.call_guard import failure_counts                            # noqa: E402
from src.agents import build_agents, resolve_focal_selector           # noqa: E402
from src.honest_bank import build_honest_bank                          # noqa: E402
from src.hedged_personas import build_hedged_pool, sample_for_hand_audit  # noqa: E402
from src.peer_pools import load_pools                                  # noqa: E402
from src.r0_cache import build_r0_cache, load_r0_cache, r0_lookup, solo_accuracy_by_focal  # noqa: E402
from src.revision_runner import (                                      # noqa: E402
    REVISION_DEDUP_KEYS,
    load_prior_rounds,
    run_condition,
)
from src.snapshots import SnapshotAuditor                              # noqa: E402
from src.store import Checkpoint, IncrementalWriter, write_json        # noqa: E402

logger = logging.getLogger("platos_ship3.run_all")


# ──────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(log_dir / f"phase3_{time.strftime('%Y%m%d_%H%M%S')}.log",
                            encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def load_env(project_root: Path) -> None:
    """Load .env from Phase 3, then the repo root, then the parent repo."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv missing; relying on the ambient environment.")
        return
    for candidate in (project_root / ".env",
                      project_root.parent / ".env",
                      project_root.parent.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            logger.info("Loaded environment from %s", candidate)


def load_configs(project_root: Path) -> Dict[str, Any]:
    with open(project_root / "config" / "experiment.yaml") as handle:
        experiment = yaml.safe_load(handle)
    with open(project_root / "config" / "paths.yaml") as handle:
        paths = yaml.safe_load(handle)
    return {"experiment": experiment, "paths": paths}


def resolve(project_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (project_root / raw).resolve()


# ──────────────────────────────────────────────────────────────────────────
# Question pools
# ──────────────────────────────────────────────────────────────────────────

# --max-questions (smoke tests only). None in a real run.
MAX_QUESTIONS: Optional[int] = None


def _limit(pool: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Smoke-test subset: the first N questions that belong to BOTH the main and
    the mitigation pool, so X1 and the mitigation-pool experiments share them
    (X6's H needs honest messages for its questions). GSM-Symbolic: first N
    (its items carry the mitigation column too, all False).
    """
    if MAX_QUESTIONS is None:
        return pool
    if source != "gsm_symbolic" and "included_in_mitigation_subset" in pool.columns:
        pool = pool[pool["included_in_mitigation_subset"].astype(bool)]
    return pool.head(MAX_QUESTIONS).reset_index(drop=True)


def load_pool(
    pool_name: str, experiment: Dict[str, Any], paths: Dict[str, str],
    project_root: Path, master_seed: int,
) -> pd.DataFrame:
    """Return the question frame for a named pool, building X4's if needed."""
    pool = _limit(_load_pool(pool_name, experiment, paths, project_root, master_seed),
                  experiment["pools"][pool_name]["source"])
    if pool.empty:
        raise RuntimeError(f"question pool '{pool_name}' is empty; refusing to "
                           "run experiments on no questions")
    return pool


def _load_pool(
    pool_name: str, experiment: Dict[str, Any], paths: Dict[str, str],
    project_root: Path, master_seed: int,
) -> pd.DataFrame:
    spec = experiment["pools"][pool_name]

    if spec["source"] == "gsm_symbolic":
        from src.gsm_symbolic import audit_pool_answers, build_gsm_symbolic_pool

        pool_path = resolve(project_root, paths["gsm_symbolic_pool_file"])
        freshly_built = not pool_path.exists()
        pool = build_gsm_symbolic_pool(
            output_path=resolve(project_root, paths["gsm_symbolic_pool_file"]),
            master_seed=master_seed,
            hf_dataset=spec.get("hf_dataset", "apple/GSM-Symbolic"),
            hf_config=spec.get("hf_config", "main"),
            instance_index=int(spec.get("instance_index", 0)),
            n_questions=int(spec.get("n_questions", 100)),
            pair_with_originals=bool(spec.get("pair_with_originals", True)),
        )
        # Audit the gold answers only when the pool is BUILT, not every time it
        # is loaded. The audit writes a CSV into the shared processed/ folder,
        # so auditing on every load made each X4 process rewrite the same file
        # at once. Building happens once, in --prepare, before any shard runs.
        if freshly_built:
            audit_pool_answers(
                pool, sample_size=20,
                output_path=resolve(project_root, paths["processed_data_directory"])
                / "gsm_symbolic_gold_audit.csv",
            )
        return pool

    question_pool_path = resolve(project_root, paths["question_pool_file"])
    if not question_pool_path.exists():
        raise FileNotFoundError(
            f"Question pool missing at {question_pool_path}. "
            f"Run: python3 tools/fetch_artefacts.py"
        )
    pool = pd.read_parquet(question_pool_path)
    if spec.get("filter") == "included_in_mitigation_subset":
        pool = pool[pool["included_in_mitigation_subset"].astype(bool)]
    return pool.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────
# Plan
# ──────────────────────────────────────────────────────────────────────────

def selected_experiments(
    experiment: Dict[str, Any], names: Optional[List[str]],
    priorities: Optional[List[str]],
) -> Dict[str, Dict[str, Any]]:
    plan = {}
    for name, spec in experiment["experiments"].items():
        if names:
            if not any(name == n or name.startswith(f"{n}_") for n in names):
                continue
        else:
            if not spec.get("enabled", False):
                continue
            if priorities and spec.get("priority") not in priorities:
                continue
        plan[name] = spec
    return plan


def print_plan(experiment: Dict[str, Any], models_config: Dict[str, Any]) -> None:
    focal_specs = models_config.get("focal_agents", {})
    replicates = experiment["replicates_per_question"]
    print("\nPhase 3 experiment plan (Protocol B)\n" + "=" * 72)
    grand_total = 0
    extra_r0 = 0
    for name, spec in experiment["experiments"].items():
        focal_keys = resolve_focal_selector(spec.get("focal"), focal_specs)
        pool_name = spec.get("pool", "main300")
        pool_spec = experiment["pools"].get(pool_name, {})
        n_questions = pool_spec.get("n_questions", 0)
        # X4's pool holds each GSM-Symbolic item AND its matched original.
        if pool_spec.get("pair_with_originals"):
            n_questions *= 2
        if (spec.get("enabled") and not spec.get("offline")
                and pool_spec.get("source") == "gsm_symbolic"):
            extra_r0 += len(focal_keys) * n_questions * replicates
        conditions = spec.get("conditions", [])
        calls = (0 if spec.get("offline")
                 else len(focal_keys) * len(conditions) * n_questions * replicates)
        extra = ""
        if spec.get("round_sweep"):
            extra_rounds = sum(spec["round_sweep"]["rounds"])
            extra_calls = len(focal_keys) * n_questions * replicates * extra_rounds
            calls += extra_calls
            extra = f" (+{extra_calls:,} round-sweep)"
        grand_total += calls
        status = "on " if spec.get("enabled") else "OFF"
        print(f"\n[{status}] {name}   priority={spec.get('priority')}")
        print(f"      {spec.get('description', '').strip().splitlines()[0]}")
        print(f"      focal={spec.get('focal')} -> {len(focal_keys)} models"
              f" | pool={pool_name} ({n_questions} q x {replicates} reps)")
        print(f"      conditions={conditions}")
        print(f"      revision calls ~ {calls:,}{extra}")
        if spec.get("addresses"):
            print(f"      addresses: {', '.join(spec['addresses'])}")
    r0_calls = len(focal_specs) * experiment["pools"]["main300"]["n_questions"] * replicates
    print("\n" + "=" * 72)
    print(f"Round-0 cache (shared by every condition): ~{r0_calls:,} calls"
          f" + {extra_r0:,} on the X4 pool")
    print(f"Revision calls across enabled+disabled experiments: ~{grand_total:,}")
    print("Peer banks: honest ~1,800 | hedged rewrite ~1,500 (one-off)\n")


# ──────────────────────────────────────────────────────────────────────────
# Preparation
# ──────────────────────────────────────────────────────────────────────────

def prepare_pools(
    configs: Dict[str, Any], project_root: Path, agents: Dict[str, Any],
    questions: pd.DataFrame, master_seed: int, dry_run: bool, workers: int = 1,
) -> None:
    """Build the honest bank and the hedged pool (both one-off, then reused)."""
    experiment, paths = configs["experiment"], configs["paths"]
    replicates = int(experiment["replicates_per_question"])
    defaults = agents["models_config"].get("request_defaults", {})

    honest_path = resolve(project_root, paths["honest_bank_file"])
    honest_checkpoint = Checkpoint(
        resolve(project_root, paths["output_directory"]) / "checkpoint_honest.parquet")
    build_honest_bank(
        questions=questions,
        weak_agents=agents["weak_agents"],
        weak_specs=agents["weak_specs"],
        judge_cascade=agents["judge_cascade"],
        bank_path=honest_path,
        checkpoint=honest_checkpoint,
        replicates=replicates,
        temperature=float(defaults.get("weak_temperature", 0.9)),
        max_output_tokens=int(defaults.get("weak_max_output_tokens", 1024)),
        dry_run=dry_run,
        workers=workers,
    )

    # ── X4 inputs: the GSM-Symbolic pool and its wrong-anchored personas ──
    # Built here, once, before any shard runs: every X4 process reads them,
    # and a shard building them would write them concurrently.
    if any(spec.get("enabled") and spec.get("pool") == "gsm_symbolic100"
           for spec in experiment["experiments"].values()):
        from src.anchored_personas import generate_pool

        gsm_pool = load_pool("gsm_symbolic100", experiment, paths, project_root,
                             master_seed)
        generator_key = (agents["models_config"].get("persona_generator_agent")
                         or agents["models_config"].get("persona_rewrite_agent"))
        generator = agents["weak_agents"].get(generator_key)
        if generator is None:
            raise RuntimeError(
                f"persona generator '{generator_key}' unavailable; X4 cannot "
                "run without wrong-anchored personas for its GSM-Symbolic items")
        generator_spec = agents["weak_specs"].get(generator_key, {})
        generate_pool(
            questions=gsm_pool,
            agent=generator,
            output_path=resolve(project_root, paths["gsm_symbolic_personas_file"]),
            checkpoint=Checkpoint(resolve(project_root, paths["output_directory"])
                                  / "checkpoint_gsm_personas.parquet"),
            master_seed=master_seed,
            variants=int(experiment.get("persona_variants_per_question", 5)),
            anchor_mode="wrong",
            generator_name=str(generator_spec.get("model_slug", generator_key)),
            dry_run=dry_run,
            workers=workers,
        )

    anchored_path = resolve(project_root, paths["anchored_personas_file"])
    if not anchored_path.exists():
        logger.warning("Anchored persona pool missing (%s); skipping hedged pool. "
                       "Run tools/fetch_artefacts.py first.", anchored_path)
        return

    # Hedge only personas that may act as peers. Rewriting the ones that never
    # state an answer would carry that defect into the hedged pool.
    from src.anchored_personas import is_usable_persona

    anchored_all = pd.read_parquet(anchored_path)
    anchored = anchored_all[[is_usable_persona(r, "confident")
                             for r in anchored_all.to_dict("records")]]
    # Only questions that are run. WRh (X2) uses the main pool; the persona
    # file also holds personas for questions outside it, and under
    # --max-questions this keeps a smoke test from rewriting all 1,500.
    anchored = anchored[anchored["question_identifier"].isin(
        set(questions["question_identifier"]))]
    logger.info("Hedged pool source: %d of %d wrong-anchored personas usable.",
                len(anchored), len(anchored_all))
    rewrite_key = agents["models_config"].get("persona_rewrite_agent")
    rewrite_agent = agents["weak_agents"].get(rewrite_key)
    if rewrite_agent is None:
        logger.warning("Rewrite agent '%s' unavailable; skipping hedged pool.",
                       rewrite_key)
        return

    hedged_checkpoint = Checkpoint(
        resolve(project_root, paths["output_directory"]) / "checkpoint_hedged.parquet")
    pool = build_hedged_pool(
        anchored_personas=anchored,
        rewrite_agent=rewrite_agent,
        output_path=resolve(project_root, paths["hedged_personas_file"]),
        checkpoint=hedged_checkpoint,
        config=experiment["hedged_pool"],
        master_seed=master_seed,
        dry_run=dry_run,
        workers=workers,
    )
    sample_for_hand_audit(
        pool,
        n=int(experiment["hedged_pool"].get("hand_audit_sample_size", 50)),
        master_seed=master_seed,
        output_path=resolve(project_root, paths["processed_data_directory"])
        / "hedged_hand_audit.csv",
    )


# ──────────────────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────────────────────────────────

def run_experiments(
    plan: Dict[str, Dict[str, Any]], configs: Dict[str, Any],
    project_root: Path, agents: Dict[str, Any], auditor: SnapshotAuditor,
    master_seed: int, dry_run: bool, workers: int = 1,
) -> Dict[str, Any]:
    experiment, paths = configs["experiment"], configs["paths"]
    replicates = int(experiment["replicates_per_question"])
    defaults = agents["models_config"].get("request_defaults", {})
    focal_specs = agents["focal_specs"]

    checkpoint = Checkpoint(
        resolve(project_root, paths["completed_units_checkpoint_file"]))
    revision_writer = IncrementalWriter(
        resolve(project_root, paths["revision_log_file"]),
        flush_every=100, checkpoint=checkpoint)
    peer_writer = IncrementalWriter(
        resolve(project_root, paths["peer_message_log_file"]), flush_every=200)

    pools = load_pools(paths, project_root)
    r0 = load_r0_cache(resolve(project_root, paths["r0_cache_file"]))
    index = r0_lookup(r0)
    # Needed only to resume a multi-round condition correctly: it restores what
    # the focal model saw as its own previous answer.
    prior_rounds = load_prior_rounds(
        resolve(project_root, paths["revision_log_file"]))
    if prior_rounds:
        logger.info("Loaded %d prior rounds for multi-round resume.",
                    len(prior_rounds))

    summary: Dict[str, Any] = {}
    for name, spec in plan.items():
        if spec.get("offline"):
            logger.info("%s is offline-only; handled in the analysis stage.", name)
            continue

        questions = load_pool(spec.get("pool", "main300"), experiment, paths,
                              project_root, master_seed)
        focal_keys = resolve_focal_selector(spec.get("focal"), focal_specs)
        calls = 0

        for focal_key in focal_keys:
            agent = agents["focal_agents"].get(focal_key)
            if agent is None:
                logger.warning("Focal agent '%s' not built; skipping.", focal_key)
                continue

            for condition_name in spec.get("conditions", []):
                condition = experiment["conditions"][condition_name]
                logger.info("%s | %s | %s — starting.", name, focal_key, condition_name)
                calls += run_condition(
                    condition_name=condition_name,
                    condition=condition,
                    focal_key=focal_key,
                    focal_agent=agent,
                    focal_spec=focal_specs.get(focal_key, {}),
                    questions=questions,
                    r0_index=index,
                    pools=pools,
                    weak_specs=agents["weak_specs"],
                    weak_agents=agents["weak_agents"],
                    judge_cascade=agents["judge_cascade"],
                    auditor=auditor,
                    revision_writer=revision_writer,
                    peer_writer=peer_writer,
                    checkpoint=checkpoint,
                    master_seed=master_seed,
                    replicates=replicates,
                    rounds=1,
                    rounds_config=experiment.get("rounds", {}),
                    temperature=float(defaults.get("focal_temperature", 0.7)),
                    max_output_tokens=int(defaults.get("focal_max_output_tokens", 2048)),
                    experiment_name=name,
                    dry_run=dry_run,
                    workers=workers,
                    prior_rounds=prior_rounds,
                )

            # X3's round sweep: the same condition at 2 and 3 rounds.
            sweep = spec.get("round_sweep")
            if sweep:
                condition_name = sweep["condition"]
                condition = experiment["conditions"][condition_name]
                for n_rounds in sweep["rounds"]:
                    logger.info("%s | %s | %s @ %d rounds — starting.",
                                name, focal_key, condition_name, n_rounds)
                    calls += run_condition(
                        condition_name=f"{condition_name}_rounds{n_rounds}",
                        condition=condition,
                        focal_key=focal_key,
                        focal_agent=agent,
                        focal_spec=focal_specs.get(focal_key, {}),
                        questions=questions,
                        r0_index=index,
                        pools=pools,
                        weak_specs=agents["weak_specs"],
                        weak_agents=agents["weak_agents"],
                        judge_cascade=agents["judge_cascade"],
                        auditor=auditor,
                        revision_writer=revision_writer,
                        peer_writer=peer_writer,
                        checkpoint=checkpoint,
                        master_seed=master_seed,
                        replicates=replicates,
                        rounds=int(n_rounds),
                        rounds_config=experiment.get("rounds", {}),
                        temperature=float(defaults.get("focal_temperature", 0.7)),
                        max_output_tokens=int(
                            defaults.get("focal_max_output_tokens", 2048)),
                        experiment_name=name,
                        dry_run=dry_run,
                        workers=workers,
                        prior_rounds=prior_rounds,
                    )

        summary[name] = {"revision_calls": calls, "focal_models": focal_keys}
        logger.info("%s complete: %d revision calls.", name, calls)

    revision_writer.consolidate(dedup_on=REVISION_DEDUP_KEYS)
    peer_writer.consolidate(dedup_on=["unit_id", "peer_display_name",
                                      "shown_to_focal"])
    checkpoint.save()
    auditor.save(resolve(project_root, paths["snapshot_audit_file"]))
    return summary


def x6_focal_keys(spec: Dict[str, Any], focal_specs: Dict[str, Any],
                  verifier_spec: Dict[str, Any]) -> List[str]:
    """X6's focal models, minus any that is the verifier's own model."""
    keys = resolve_focal_selector(spec.get("focal", "TIER_X2"), focal_specs)
    verifier_slug = str((verifier_spec or {}).get("model_slug", ""))
    same = [k for k in keys
            if str(focal_specs.get(k, {}).get("model_slug", "")) == verifier_slug]
    if same:
        logger.warning("X6: %s excluded; a model cannot verify itself.", same)
    return [k for k in keys if k not in same]


def run_verification(
    configs: Dict[str, Any], project_root: Path, agents: Dict[str, Any],
    auditor: SnapshotAuditor, master_seed: int, dry_run: bool, workers: int = 1,
) -> None:
    """X6: verify proposed changes with an independent model."""
    from src.verifier import verify_changes

    experiment, paths = configs["experiment"], configs["paths"]
    spec = experiment["experiments"].get("X6_verification_safeguard", {})
    if not spec.get("enabled"):
        return
    if agents.get("verifier_agent") is None:
        logger.warning("No verifier agent configured; skipping X6.")
        return

    revisions_path = resolve(project_root, paths["revision_log_file"])
    if not revisions_path.exists():
        logger.warning("No revision log yet; run the experiments before X6.")
        return

    revisions = pd.read_parquet(revisions_path)
    # Only X6's own models. Selecting by condition alone verified every
    # model's WR and H rows, including Qwen-2.5-72B checked by Qwen-2.5-72B,
    # and in shard mode each shard verified whatever model it ran.
    x6_focal = x6_focal_keys(spec, agents["focal_specs"], agents["verifier_spec"])
    revisions = revisions[
        revisions["experiment"].isin(["X1_common_matrix", "X6_verification_safeguard"])
        & revisions["condition"].isin(spec.get("conditions", ["WR", "H"]))
        & revisions["focal_key"].isin(x6_focal)
    ]
    if revisions.empty:
        logger.info("X6: none of this process's models is an X6 model; "
                    "nothing to verify.")
        return
    questions = load_pool(spec.get("pool", "mitigation100"), experiment, paths,
                          project_root, master_seed)
    revisions = revisions[
        revisions["question_identifier"].isin(set(questions["question_identifier"]))
    ]

    checkpoint = Checkpoint(
        resolve(project_root, paths["output_directory"]) / "checkpoint_verifier.parquet")
    verify_changes(
        revisions=revisions,
        questions=questions,
        verifier_agent=agents["verifier_agent"],
        verifier_spec=agents["verifier_spec"],
        judge_cascade=agents["judge_cascade"],
        auditor=auditor,
        output_path=resolve(project_root, paths["safeguard_file"]),
        checkpoint=checkpoint,
        master_seed=master_seed,
        dry_run=dry_run,
        workers=workers,
    )


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 — Protocol B controlled revision study.")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="one unit per cell; exercises every code path")
    parser.add_argument("--prepare", action="store_true",
                        help="build the honest bank and hedged pool only")
    parser.add_argument("--experiments", type=str,
                        help="comma-separated experiment names, e.g. X1,X2")
    parser.add_argument("--priority", type=str,
                        help="comma-separated priorities, e.g. P0,P1")
    parser.add_argument("--analyse", action="store_true",
                        help="offline analysis only (no API calls)")
    parser.add_argument("--all", action="store_true",
                        help="prepare, run every enabled experiment, then analyse")
    parser.add_argument("--skip-r0", action="store_true",
                        help="assume the Round-0 cache is already complete")
    parser.add_argument("--ignore-design-check", action="store_true",
                        help="proceed even when a treatment has no matched "
                             "baseline (records the gap in run metadata)")
    parser.add_argument("--skip-unavailable", action="store_true",
                        help="development only: continue when a focal model's "
                             "provider keys are missing, recording the "
                             "omission in the run metadata")
    parser.add_argument("--focal", type=str,
                        help="comma-separated focal model keys; restrict every "
                             "experiment to these models")
    parser.add_argument("--shard", type=str,
                        help="run as an isolated shard: every output, "
                             "checkpoint and log goes to a private folder. "
                             "Requires --focal. Used to run one process per "
                             "model in parallel; see src/sharding.py")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="SMOKE TEST ONLY: run every experiment on the first "
                             "N questions. Recorded in metadata; the merge "
                             "refuses to mix it with a full run")
    parser.add_argument("--workers", type=int, default=1,
                        help="units run concurrently inside this process "
                             "(threads; see src/concurrency.py). 1 = sequential")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configs = load_configs(PROJECT_ROOT)
    global MAX_QUESTIONS
    MAX_QUESTIONS = args.max_questions

    # ── Shard mode: isolate every write BEFORE anything is written ────────
    # Redirected before logging is configured, so even the log file is
    # private. See src/sharding.py for why each guard exists.
    from src import sharding

    focal_filter = ([k.strip() for k in args.focal.split(",") if k.strip()]
                    if args.focal else None)
    if args.shard:
        if not focal_filter:
            raise sharding.ShardError("--shard requires --focal")
        if args.prepare or args.all:
            raise sharding.ShardError(
                "--prepare/--all build SHARED inputs, which must happen once, "
                "sequentially, before any shard starts")
        if args.analyse:
            raise sharding.ShardError(
                "--analyse reads every model; run it once after tools/merge_shards.py")
        original_paths = configs["paths"]
        configs["paths"] = sharding.shard_paths(original_paths, args.shard)
        sharding.assert_isolated(original_paths, configs["paths"], args.shard)

    experiment, paths = configs["experiment"], configs["paths"]
    setup_logging(resolve(PROJECT_ROOT, paths["logs_directory"]), args.verbose)
    load_env(PROJECT_ROOT)
    if args.shard:
        logger.info("SHARD %s | focal=%s | outputs -> %s", args.shard,
                    focal_filter, paths["output_directory"])

    master_seed = int(experiment["random_seed"])

    if args.list:
        from src.agents import load_models_config

        models_config = load_models_config(PROJECT_ROOT)
        print_plan(experiment, models_config)
        # Static design check: every treatment must have its baseline on the
        # same models, before a single call is paid for.
        from analysis.design_check import check_plan, report

        report(check_plan(experiment, models_config.get("focal_agents", {})),
               context="planned experiments")
        return 0

    if args.analyse:
        from analysis.run_analysis import run_full_analysis

        run_full_analysis(PROJECT_ROOT)
        return 0

    names = [n.strip() for n in args.experiments.split(",")] if args.experiments else None
    priorities = [p.strip() for p in args.priority.split(",")] if args.priority else None
    plan = selected_experiments(experiment, names, priorities)
    if not plan and not args.prepare:
        logger.error("No experiments selected. Try --list.")
        return 1

    # Refuse to spend on a design that cannot be baseline-corrected.
    from src.agents import load_models_config as _load_models
    from analysis.design_check import check_plan, report

    if not report(check_plan({**experiment, "experiments": plan},
                             _load_models(PROJECT_ROOT).get("focal_agents", {})),
                  context="selected experiments"):
        logger.error(
            "Design check failed: a treatment condition has no matched "
            "baseline. Add 'R' for the affected models, or pass "
            "--ignore-design-check if this is deliberate."
        )
        if not args.ignore_design_check:
            return 1

    # Which focal models do we actually need?
    from src.agents import load_models_config

    focal_specs_all = sharding.restrict_focal_specs(
        load_models_config(PROJECT_ROOT).get("focal_agents", {}), focal_filter)
    needed: set[str] = set()
    for spec in plan.values():
        needed.update(resolve_focal_selector(spec.get("focal"), focal_specs_all))
    if args.prepare or args.all:
        needed.update(focal_specs_all)
    if focal_filter and not needed:
        logger.error("None of %s takes part in the selected experiments.", focal_filter)
        return 1

    agents = build_agents(
        PROJECT_ROOT,
        focal_keys=sorted(needed) or None,
        need_weak=True,
        need_judge=True,
        need_verifier=any(n.startswith("X6") for n in plan) or args.all,
        skip_unavailable=args.skip_unavailable,
    )
    # Restrict to this process's models. An experiment's selector then
    # resolves to exactly this process's share: "ALL" becomes these models,
    # and "TIER_X2" becomes them only if they belong to that tier.
    agents["focal_specs"] = sharding.restrict_focal_specs(
        agents["focal_specs"], focal_filter)
    agents["focal_agents"] = {k: v for k, v in agents["focal_agents"].items()
                              if k in agents["focal_specs"]}
    if agents["unavailable_focal"]:
        logger.warning(
            "Running with an INCOMPLETE focal set; %d model(s) omitted: %s",
            len(agents["unavailable_focal"]), sorted(agents["unavailable_focal"]),
        )

    # Exclusive lock on the output root, in EVERY mode. Two processes on one
    # root lose data (see src/sharding.py); that includes two ordinary full
    # runs started by accident, not only two shards.
    import atexit
    run_lock = sharding.ShardLock(resolve(PROJECT_ROOT, paths["output_directory"]))
    run_lock.acquire()
    atexit.register(run_lock.release)

    if args.shard:
        # Every input a shard reads must already exist: a shard building a
        # shared input would write it concurrently with the other seven.
        required = ["question_pool_file", "anchored_personas_file",
                    "correct_anchored_personas_file", "honest_bank_file",
                    "hedged_personas_file"]
        if any(spec.get("pool") == "gsm_symbolic100" for spec in plan.values()):
            required += ["gsm_symbolic_pool_file", "gsm_symbolic_personas_file"]
        if any(experiment["conditions"].get(c, {}).get("peer_source") == "anchored_confidence"
               for spec in plan.values() for c in spec.get("conditions", [])):
            required += ["confidence_personas_file"]
        absent = sharding.missing_shared_inputs(paths, PROJECT_ROOT, required)
        if absent:
            raise sharding.ShardError(
                "shared inputs missing; run `python3 run_all.py --prepare` once "
                f"before starting any shard: {absent}")
    input_hashes = sharding.hash_shared_inputs(paths, PROJECT_ROOT)
    auditor = SnapshotAuditor()

    main_questions = load_pool("main300", experiment, paths, PROJECT_ROOT, master_seed)

    if args.prepare or args.all:
        logger.info("=== Stage: prepare message banks ===")
        prepare_pools(configs, PROJECT_ROOT, agents, main_questions,
                      master_seed, args.dry_run, workers=args.workers)
        if args.prepare and not args.all:
            return _exit_status()

    if not args.skip_r0:
        logger.info("=== Stage 0: Round-0 cache ===")
        r0_checkpoint = Checkpoint(
            resolve(PROJECT_ROOT, paths["output_directory"]) / "checkpoint_r0.parquet")
        # Round-0 only for the pools each model actually runs. Previously every
        # model got Round-0 on every pool in the plan, so Gemma-3-4B paid for
        # 300 GSM-Symbolic first answers no Gemma-3-4B experiment reads. An
        # answer nobody reads cannot change a result, so this is free and safe.
        pool_to_focals: Dict[str, set] = {}
        for spec in plan.values():
            if spec.get("offline"):
                continue
            for focal_key in resolve_focal_selector(spec.get("focal"),
                                                    agents["focal_specs"]):
                if focal_key in agents["focal_agents"]:
                    pool_to_focals.setdefault(
                        spec.get("pool", "main300"), set()).add(focal_key)
        for pool_name in sorted(pool_to_focals):
            questions = load_pool(pool_name, experiment, paths, PROJECT_ROOT,
                                  master_seed)
            focal_for_pool = sorted(pool_to_focals[pool_name])
            logger.info("Round-0 for pool %s: %s", pool_name, focal_for_pool)
            build_r0_cache(
                questions=questions,
                focal_keys=focal_for_pool,
                focal_agents=agents["focal_agents"],
                focal_specs=agents["focal_specs"],
                judge_cascade=agents["judge_cascade"],
                auditor=auditor,
                cache_path=resolve(PROJECT_ROOT, paths["r0_cache_file"]),
                checkpoint=r0_checkpoint,
                replicates=int(experiment["replicates_per_question"]),
                temperature=float(experiment["r0_cache"]["temperature"]),
                max_output_tokens=int(experiment["r0_cache"]["max_output_tokens"]),
                dry_run=args.dry_run,
                workers=args.workers,
            )

    logger.info("=== Stage: revision conditions ===")
    summary = run_experiments(plan, configs, PROJECT_ROOT, agents, auditor,
                              master_seed, args.dry_run, workers=args.workers)

    if any(n.startswith("X6") for n in plan) or args.all:
        logger.info("=== Stage: X6 verification safeguard ===")
        run_verification(configs, PROJECT_ROOT, agents, auditor, master_seed,
                         args.dry_run, workers=args.workers)

    r0 = load_r0_cache(resolve(PROJECT_ROOT, paths["r0_cache_file"]))
    write_json(
        resolve(PROJECT_ROOT, paths["experiment_metadata_file"]),
        {
            "phase": 3,
            "protocol": experiment["protocol_label"],
            "run_timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
            "random_seed": master_seed,
            "seed_derivation": "zlib.crc32 (process-stable); see src/seeding.py",
            "replicates_per_question": experiment["replicates_per_question"],
            "experiments_run": sorted(plan),
            "experiment_summary": summary,
            "focal_models": {
                key: {"model_slug": spec.get("model_slug"),
                      "expected_served_prefix": spec.get("expected_served_prefix"),
                      "paper_name": spec.get("paper_name")}
                for key, spec in agents["focal_specs"].items()
            },
            "solo_accuracy_by_focal": solo_accuracy_by_focal(r0),
            "snapshot_audit": auditor.summary(),
            # An omitted focal model means the common matrix is incomplete,
            # which is the defect this phase exists to fix. Recorded here so it
            # cannot be forgotten between running and writing.
            "unavailable_focal_models": agents["unavailable_focal"],
            "focal_set_is_complete": not agents["unavailable_focal"],
            "dry_run": bool(args.dry_run),
            "workers": int(args.workers),
            "max_questions": args.max_questions,
            # Shard identity and the SHA-256 of every shared input this
            # process read. tools/merge_shards.py refuses to combine shards
            # that did not all read byte-identical inputs.
            "shard": args.shard,
            "focal_filter": focal_filter,
            "shared_input_sha256": input_hashes,
            # Calls that failed at the provider were NOT recorded and will be
            # retried by the next run of the same command. Non-empty means
            # this run is incomplete.
            "failed_calls_not_recorded": failure_counts(),
        },
    )

    if args.all:
        logger.info("=== Stage: analysis ===")
        from analysis.run_analysis import run_full_analysis

        run_full_analysis(PROJECT_ROOT)

    logger.info("Phase 3 run complete.")
    return _exit_status()


EXIT_INCOMPLETE = 3


def _exit_status() -> int:
    """
    0 when every call succeeded; EXIT_INCOMPLETE when some failed and were
    left unrecorded. The launcher re-runs the same command until it gets 0,
    and each re-run pays only for the missing units.
    """
    failures = failure_counts()
    if failures:
        logger.warning("INCOMPLETE: %d failed calls not recorded %s; re-run "
                       "the same command to retry them.",
                       sum(failures.values()), failures)
        return EXIT_INCOMPLETE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
